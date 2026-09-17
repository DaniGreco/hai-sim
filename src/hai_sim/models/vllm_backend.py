"""MedGemma served by a vLLM endpoint — the HTTP twin of `MedGemmaAI`.

Where `MedGemmaAI` loads the weights in this process and reads logits directly, this talks
to a **running vLLM server** (`/v1/chat/completions`, OpenAI-compatible) over HTTP. It wears
the same two contracts, so nothing downstream changes:

- `VisionLanguageBackend` — the readouts drive it unchanged (tested against a fake here);
- `CacheableModel` (via `VLLMMedGemmaAI`) — `materialize_cache` builds the same per-model
  cells, addressed by the same `<dataset>__<model>` stem.

The point is **deployment, not a new model**: use this when the GPU lives in a cluster (the
k3s node's vLLM pods) rather than in this process, which also means the 27B fits on one
96 GB card with no in-process sharding.

**Only `independent_question` is supported over HTTP.** That readout needs one probability
per label, read off the *first generated token's* logprobs — which the chat API exposes
directly (`max_tokens=1` + `logprobs` + `top_logprobs`). The teacher-forced readout instead
needs per-position logits over a forced multi-token continuation, which the chat API does
not expose; `score_continuation` therefore raises. That readout is superseded anyway
(see `readout.py`), so this is a limitation only on paper.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from hai_sim.core.base import BaseDiagnosticActor
from hai_sim.core.types import Image, LabelSchema, ModelOutput, PatientContext
from hai_sim.models._finalize import model_output_from_uq
from hai_sim.models.medgemma import to_pil
from hai_sim.models.readout import (
    BranchScores,
    IndependentQuestionReadout,
    LabelReadout,
    resolve_prompt_set,
)
from hai_sim.uq.methods import EnsembleUQ, EntropyUQ

# SentencePiece marks a word-initial subword with U+2581 ("▁"). vLLM may hand the token
# back with that marker where a real leading space is meant; the readout's yes/no spellings
# use an actual space (" yes"), so normalise the marker to a space before matching — miss
# this and every word-initial answer token fails to match and every P(yes) comes back NaN.
_SP_UNDERLINE = "▁"

PostFn = Callable[[str, dict[str, Any]], dict[str, Any]]
"""POST a JSON body to a URL, return the parsed JSON. Injectable so the backend is testable
without a live server (the default posts with httpx)."""


def _png_data_uri(image: Any) -> str:
    """A framework image (array or PIL) → a ``data:image/png;base64,...`` URI for the API.

    Reuses `medgemma.to_pil`, so a vLLM run sees pixels prepared **identically** to the
    in-process backend (same min/max rescale, same grey→RGB), which is what keeps the two
    deployments comparable rather than measuring a preprocessing difference.
    """
    try:
        from PIL import Image as PILImage
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("VLLMBackend needs pillow (installed with the [models] extra)") from exc

    pil = image if isinstance(image, PILImage.Image) else to_pil(image)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class VLLMBackend:
    """A `VisionLanguageBackend` over a vLLM OpenAI-compatible endpoint.

    ``endpoint`` is the base URL up to and including ``/v1`` (e.g.
    ``http://localhost:30840/v1``); ``model_name`` is the server's ``--served-model-name``
    (e.g. ``medgemma-4b``), which the request's ``model`` field must match.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        *,
        concurrency: int = 8,
        top_logprobs: int = 20,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        post: PostFn | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        # Questions about one image are independent, so fire several at once to fill the
        # server's batch (vLLM's --max-num-seqs). Keep this <= that, and modest, since the
        # GPU may be shared with production.
        self.concurrency = max(1, int(concurrency))
        self.top_logprobs = int(top_logprobs)
        self.timeout = float(timeout)
        # Retry transient failures (connection refused while a pod is still starting, a 5xx
        # while the model loads) so a day-long build survives a blip; a 4xx fails fast.
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = float(retry_backoff)
        self._post: PostFn = post if post is not None else self._httpx_post
        self._client: Any = None

    # ── VisionLanguageBackend ────────────────────────────────────────────────────

    def score_branches(
        self,
        image: Any,
        prompt: str,
        branches: Sequence[str],
        candidates: Sequence[str],
    ) -> BranchScores:
        """One chat call per branch; read the candidates off the first generated token.

        The image and the instruction prefix are identical across branches, so vLLM's
        automatic prefix caching reuses their KV across the per-question calls — the HTTP
        analogue of the in-process backend's "encode once, rewind" trick, without needing
        server state we cannot reach.
        """
        cands = tuple(candidates)
        branch_list = list(branches)
        if not branch_list:
            return BranchScores((), cands, np.zeros((0, len(cands)), dtype=float))

        data_uri = _png_data_uri(image)  # encode the image once, reuse for every branch

        def one(branch: str) -> np.ndarray:
            top = self._first_token_logprobs(data_uri, prompt + branch)
            # Absent spelling → -inf, exactly what the local backend returns for a candidate
            # this tokenizer never produces; the readout takes the best surviving spelling.
            return np.array([top.get(c, -np.inf) for c in cands], dtype=float)

        rows = self._map(one, branch_list)
        return BranchScores(tuple(branch_list), cands, np.stack(rows, axis=0))

    def score_continuation(
        self,
        image: Any,
        prompt: str,
        continuation: str,
        candidates: Sequence[str],
    ) -> Any:
        raise NotImplementedError(
            "the vLLM HTTP backend supports only the 'independent_question' readout; "
            "'teacher_forced_logprob' needs per-position logits over a forced continuation "
            "that the chat API does not expose (and that readout is superseded — see readout.py)"
        )

    def generate(self, image: Any, prompt: str, max_new_tokens: int) -> str:
        data_uri = _png_data_uri(image)
        resp = self._chat(data_uri, prompt, max_tokens=int(max_new_tokens), want_logprobs=False)
        return str(resp["choices"][0]["message"]["content"])

    # ── HTTP ─────────────────────────────────────────────────────────────────────

    def _first_token_logprobs(self, data_uri: str, text: str) -> dict[str, float]:
        """``{token_text: logprob}`` for the first generated token, spellings normalised."""
        resp = self._chat(data_uri, text, max_tokens=1, want_logprobs=True)
        choice = resp["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        if not content:
            return {}
        out: dict[str, float] = {}
        for entry in content[0].get("top_logprobs", []):
            token = str(entry["token"]).replace(_SP_UNDERLINE, " ")
            logprob = float(entry["logprob"])
            # After normalising "▁"→" " two raw tokens can map to one spelling; keep the best.
            if token not in out or logprob > out[token]:
                out[token] = logprob
        return out

    def _chat(
        self, data_uri: str, text: str, *, max_tokens: int, want_logprobs: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if want_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        return self._post(f"{self.endpoint}/chat/completions", body)

    def _httpx_post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        import time

        import httpx

        if self._client is None:
            # One pooled client for the whole build: keep-alive across the ~1M requests
            # instead of a TCP + TLS handshake per call.
            self._client = httpx.Client(timeout=self.timeout)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(url, json=body)
            except httpx.TransportError:
                # Connection refused / reset — a pod still starting, or a network flap.
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff * 2**attempt)
                continue
            if response.status_code >= 500 and attempt < self.max_retries:
                time.sleep(self.retry_backoff * 2**attempt)  # transient server-side; retry
                continue
            if response.status_code >= 400:
                # 4xx, or 5xx with retries exhausted: surface vLLM's own message (an
                # unsupported arch/param on the first live run says why in the body), not a
                # bare status code. See the TA note about the first run being unvalidated.
                raise RuntimeError(
                    f"vLLM returned {response.status_code} for {url}: {response.text[:500]}"
                )
            result: dict[str, Any] = response.json()
            return result
        raise RuntimeError("unreachable: retry loop exited without returning")  # pragma: no cover

    def _map(self, fn: Callable[[str], np.ndarray], items: list[str]) -> list[np.ndarray]:
        if self.concurrency == 1 or len(items) <= 1:
            return [fn(item) for item in items]
        # ThreadPoolExecutor.map preserves input order, so row i stays branch i.
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(fn, items))


class VLLMMedGemmaAI(BaseDiagnosticActor):
    """MedGemma via a vLLM endpoint, as a `DiagnosticActor` **and** `CacheableModel`.

    Mirrors `MedGemmaAI`'s actor surface (`member_probabilities` / `member_names` /
    `diagnose`) but delegates the per-image scoring to a `VLLMBackend`, so the same readout,
    UQ decomposition and cache builder all work unchanged. The prompt paraphrases of
    ``prompt_set`` are the ensemble members, exactly as for the in-process backend.
    """

    def __init__(
        self,
        schema: LabelSchema,
        backend: VLLMBackend,
        *,
        readout: LabelReadout | None = None,
        prompt_set: str = "single",
        uq: Any | None = None,
        decision_threshold: float = 0.5,
        service_time: float = 30.0,
        name: str | None = None,
    ) -> None:
        self.prompts = resolve_prompt_set(prompt_set)
        resolved = (
            name if name is not None else f"VLLMMedGemmaAI({backend.model_name}, {prompt_set})"
        )
        super().__init__(name=resolved, service_time=service_time)

        self.schema = schema
        self.backend = backend
        self.readout: LabelReadout = (
            readout if readout is not None else IndependentQuestionReadout()
        )
        self.prompt_set = prompt_set
        self.uq = uq or (EntropyUQ() if len(self.prompts) == 1 else EnsembleUQ())
        self.decision_threshold = decision_threshold

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prompt_set}#{i}" for i in range(len(self.prompts)))

    def member_probabilities(self, image: Image) -> np.ndarray:
        """``(n_prompts, n_labels)`` — one row per prompt paraphrase, via the endpoint."""
        rows = [
            self.readout.probabilities(self.backend, image, self.schema.labels, prefix)
            for prefix in self.prompts
        ]
        return np.stack(rows, axis=0)

    def member_probabilities_batch(self, images: Sequence[Image]) -> np.ndarray:
        """``(n_images, n_members, n_labels)`` — a plain loop; the concurrency lives inside
        the backend (one image's questions fan out), which is where the batch actually is."""
        return np.stack([self.member_probabilities(img) for img in images], axis=0)

    def diagnose(
        self,
        images: Sequence[Image],
        *,
        context: PatientContext | None = None,
        ai_hint: ModelOutput | None = None,
    ) -> ModelOutput:
        member_probs = self.member_probabilities(images[0])
        return model_output_from_uq(
            self.uq,
            member_probs,
            self.decision_threshold,
            extra={
                "readout": self.readout.name,
                "model_id": self.backend.model_name,
                "n_prompts": len(self.prompts),
                "backend": "vllm",
            },
        )
