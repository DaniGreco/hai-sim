"""MedGemma as an AI reader — a generative model wearing the same contract as the CNN.

MedGemma is a medical vision-language model (Google, Gemma 3 + a SigLIP image encoder
pre-trained on medical images). Unlike `TorchXRayVisionAI`, which was trained to emit one
probability per pathology, this one was trained to emit *text*. Everything awkward about
this module follows from that single difference; see `hai_sim.models.readout` for how the
gap is bridged and what it costs.

Two contracts are implemented here, deliberately:

- `DiagnosticActor`, so the simulator can put this model at the AI station in place of
  the CNN with nothing else changing;
- `VisionLanguageBackend`, so the readouts can drive it without knowing about
  transformers — which is what lets every readout be unit-tested against a fake.

**Prompt ensembling.** Ask the same image K different ways and the K answers form an
ensemble: where the wordings agree the model is genuinely sure, where a paraphrase flips
the answer it is not. Those K vectors go straight into the existing `EnsembleUQ`, so the
generative model gets the same total / aleatoric / epistemic decomposition as the Deep
Ensemble — no new uncertainty code, and the two backends stay comparable.

**Cost.** On a CPU one forward pass of the 4B model takes tens of seconds, so a K-prompt
ensemble over a few thousand images is measured in days. Build an `InferenceCache` once
(``scripts/build_inference_cache.py``) and replay it; that is the supported workflow, not
an optimisation.

``transformers`` is imported lazily, so this module imports fine without it — the error
only appears if you actually try to load weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Diagnosis, Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.models.readout import (
    ContinuationScores,
    LabelReadout,
    TeacherForcedLogprobReadout,
    resolve_prompt_set,
)
from hai_sim.uq.methods import EnsembleUQ, EntropyUQ

if TYPE_CHECKING:
    from hai_sim.core.protocols import UQMethod

DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"
"""MedGemma 1.5, 4B, instruction-tuned — the only size 1.5 was released in.

``google/medgemma-27b-it`` also exists but belongs to **MedGemma 1**, an earlier
generation. Comparing the two therefore varies size *and* generation at once; say so
rather than presenting it as a scaling result.
"""

_LICENCE_HINT = (
    "MedGemma is gated: accept the Health AI Developer Foundations terms at "
    "https://huggingface.co/{model_id} while signed in, then authenticate locally "
    "with `huggingface-cli login`."
)


class MedGemmaAI(BaseDiagnosticActor):
    """A medical vision-language model serving as the AI reader."""

    def __init__(
        self,
        schema: LabelSchema,
        model_id: str = DEFAULT_MODEL_ID,
        readout: LabelReadout | None = None,
        prompt_set: str = "single",
        uq: UQMethod | None = None,
        decision_threshold: float = 0.5,
        service_time: float = 30.0,
        device: str = "cpu",
        dtype: str = "auto",
        name: str | None = None,
        load_now: bool = True,
    ) -> None:
        self.prompts = resolve_prompt_set(prompt_set)
        resolved = name if name is not None else f"MedGemmaAI({model_id}, {prompt_set})"
        super().__init__(name=resolved, service_time=service_time)

        self.schema = schema
        self.model_id = model_id
        self.readout: LabelReadout = (
            readout if readout is not None else TeacherForcedLogprobReadout()
        )
        self.prompt_set = prompt_set
        # One prompt is a single member, so entropy of its own probability is all the
        # uncertainty there is; several prompts make member disagreement available.
        self.uq: UQMethod = uq or (EntropyUQ() if len(self.prompts) == 1 else EnsembleUQ())
        self.decision_threshold = decision_threshold
        self.device = device
        self.dtype = dtype

        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        if load_now:
            self.load()

    # ── Loading ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Fetch weights and processor. Separated from ``__init__`` so the actor can be
        constructed (and its wiring tested) without a multi-gigabyte download."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "MedGemmaAI needs transformers and torch: "
                'uv pip install -e ".[models]"'
            ) from exc

        self._torch = torch
        dtype = self._resolve_dtype(torch)
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(self.device)
        except OSError as exc:  # gated repo, or no network
            raise OSError(
                f"could not load {self.model_id!r}. "
                + _LICENCE_HINT.format(model_id=self.model_id)
            ) from exc
        self._model.eval()

    def _resolve_dtype(self, torch: Any) -> Any:
        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        # bfloat16 is the model card's recommendation, but CPU support for it is uneven
        # and silently falls back to slow emulation on older cores; float32 is the safe
        # default off-GPU, at 2x the memory.
        return torch.bfloat16 if self.device != "cpu" else torch.float32

    def _require_loaded(self) -> None:
        if self._model is None or self._processor is None:
            raise RuntimeError("MedGemmaAI weights are not loaded; call .load() first")

    # ── DiagnosticActor ─────────────────────────────────────────────────────────

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        member_probs = self.member_probabilities(images[0])
        probs, uncertainty, uq_extra = self.uq.estimate(member_probs)
        probs = np.nan_to_num(np.asarray(probs, dtype=float), nan=0.0)
        uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=float), nan=0.0)
        labels = (probs >= self.decision_threshold).astype(int)
        extra: dict[str, Any] = {
            "uq_method": self.uq.name,
            "readout": self.readout.name,
            "model_id": self.model_id,
            "n_prompts": len(self.prompts),
            **uq_extra,
        }
        return ModelOutput(
            diagnosis=Diagnosis(
                labels=tuple(int(x) for x in labels),
                confidence=tuple(float(x) for x in probs),
            ),
            uncertainty=tuple(float(x) for x in uncertainty),
            extra=extra,
        )

    @property
    def member_names(self) -> tuple[str, ...]:
        """Member identities = the prompt paraphrases, in the order they are run.

        Same contract as the Deep Ensemble's weight sets, which is what lets one cache
        format hold either backend.
        """
        return tuple(f"{self.prompt_set}#{i}" for i in range(len(self.prompts)))

    def member_probabilities_batch(self, images: Sequence[Image]) -> np.ndarray:
        """``(n_images, n_members, n_labels)`` — a plain loop, not real batching.

        Batching a vision-language model would mean padding several 256-token image
        prefixes together, which on a CPU buys nothing: the forward pass is already
        memory-bound. The loop is honest about that, and keeps the builder's per-image
        timing meaningful.
        """
        return np.stack([self.member_probabilities(img) for img in images], axis=0)

    def member_probabilities(self, image: Image) -> np.ndarray:
        """``(n_prompts, n_labels)`` — one row per prompt paraphrase.

        Same shape and meaning as `TorchXRayVisionAI.member_probabilities`, which is what
        lets a prompt ensemble and a Deep Ensemble share every downstream consumer.
        """
        self._require_loaded()
        pil = to_pil(image)
        rows = [
            self.readout.probabilities(self, pil, self.schema.labels, prefix)
            for prefix in self.prompts
        ]
        return np.stack(rows, axis=0)

    # ── VisionLanguageBackend ───────────────────────────────────────────────────

    def _prompt_ids(self, image: Any, prompt: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    def score_continuation(
        self,
        image: Any,
        prompt: str,
        continuation: str,
        candidates: Sequence[str],
    ) -> ContinuationScores:
        """One forward pass over prompt + forced continuation; score the candidates.

        Teacher forcing rather than generation: we already know what we want the model to
        fill in, and we only care what probability it assigns there. That costs one pass
        instead of a decoding loop, and cannot fail to produce an answer for every label.
        """
        self._require_loaded()
        torch = self._torch
        tokenizer = self._processor.tokenizer

        encoded = tokenizer(
            continuation, add_special_tokens=False, return_offsets_mapping=True
        )
        cont_ids = list(encoded["input_ids"])
        token_texts = _contiguous_slices(continuation, list(encoded["offset_mapping"]))

        prompt_inputs = self._prompt_ids(image, prompt)
        prompt_ids = prompt_inputs["input_ids"]
        cont_tensor = torch.tensor([cont_ids], device=prompt_ids.device)
        full_ids = torch.cat([prompt_ids, cont_tensor], dim=1)

        forward_kwargs = {
            k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")
        }
        attention = torch.ones_like(full_ids)
        with torch.inference_mode():
            out = self._model(input_ids=full_ids, attention_mask=attention, **forward_kwargs)

        # logits[:, t] predicts token t+1, so the distribution *at* continuation position
        # i sits at index (len(prompt) - 1 + i).
        start = prompt_ids.shape[1] - 1
        window = out.logits[0, start : start + len(cont_ids), :]
        logprobs = torch.log_softmax(window.float(), dim=-1).cpu().numpy()

        kept, columns = _single_token_candidates(tokenizer, candidates)
        table = logprobs[:, columns] if columns else np.zeros((len(cont_ids), 0))
        return ContinuationScores(
            token_texts=tuple(token_texts),
            candidates=tuple(kept),
            candidate_logprobs=np.asarray(table, dtype=float),
        )

    def generate(self, image: Any, prompt: str, max_new_tokens: int) -> str:
        self._require_loaded()
        torch = self._torch
        inputs = self._prompt_ids(image, prompt)
        with torch.inference_mode():
            produced = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = produced[0][inputs["input_ids"].shape[-1] :]
        text: str = self._processor.decode(new_tokens, skip_special_tokens=True)
        return text


# ── Helpers, kept module-level so they are testable without weights ─────────────


def _contiguous_slices(text: str, offsets: Sequence[tuple[int, int]]) -> list[str]:
    """Split ``text`` into per-token slices that tile it exactly, with no gaps.

    Tokenizer offset maps may skip characters (leading spaces, mostly), which would make
    the concatenated tokens shorter than the text and shift every later character offset.
    Extend each token backwards to swallow whatever preceded it so the slices tile the
    string exactly — the answer-position lookup depends on that.
    """
    slices: list[str] = []
    cursor = 0
    for _start, end in offsets:
        slices.append(text[cursor:end])
        cursor = end
    if cursor < len(text) and slices:
        slices[-1] += text[cursor:]
    return slices


def _single_token_candidates(
    tokenizer: Any, candidates: Sequence[str]
) -> tuple[list[str], list[int]]:
    """Keep only candidates this tokenizer encodes as exactly one token.

    A multi-token spelling has no single position to read a probability from, so scoring
    it here would be meaningless. Dropping it is safe because the readout supplies several
    spellings of each answer and takes the best surviving one.
    """
    kept: list[str] = []
    columns: list[int] = []
    for spelling in candidates:
        ids = tokenizer.encode(spelling, add_special_tokens=False)
        if len(ids) == 1:
            kept.append(spelling)
            columns.append(int(ids[0]))
    return kept, columns


def to_pil(image: Image) -> Any:
    """Framework image array → a PIL RGB image the processor can consume.

    Datasets hand us model-ready arrays in whatever range their own pipeline uses —
    TorchXRayVision, for instance, works in ``[-1024, 1024]`` with a leading channel axis.
    Rescale from the array's own min/max rather than assuming a range: getting this wrong
    produces a uniformly grey image that the model will still cheerfully report on.
    """
    try:
        from PIL import Image as PILImage
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("MedGemmaAI needs pillow (installed with the [models] extra)") from exc

    array = np.asarray(image, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in (1, 3):  # channel-first → channel-last
        array = np.moveaxis(array, 0, -1)
        array = np.squeeze(array)
    if array.ndim != 2 and not (array.ndim == 3 and array.shape[-1] == 3):
        raise ValueError(f"expected a 2-D image or an RGB image, got shape {array.shape}")

    low, high = float(np.min(array)), float(np.max(array))
    scaled = np.zeros_like(array) if high <= low else (array - low) / (high - low)
    as_uint8 = (scaled * 255.0).round().astype(np.uint8)
    return PILImage.fromarray(as_uint8).convert("RGB")
