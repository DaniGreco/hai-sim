"""The vLLM HTTP backend, exercised against a fake POST so no server is needed.

The readout logic is already covered in ``test_medgemma_readout.py`` against a fake
`VisionLanguageBackend`; here we test the *other* half — that `VLLMBackend` turns a vLLM
chat/completions response into exactly the `BranchScores` the readout expects, and that the
`independent_question` readout stacked on top recovers the P(yes) the server implied.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hai_sim.analysis.inference_cache import CacheableModel
from hai_sim.core.types import LabelSchema
from hai_sim.models.readout import IndependentQuestionReadout
from hai_sim.models.vllm_backend import VLLMBackend, VLLMMedGemmaAI, _png_data_uri

_IMG = np.zeros((1, 8, 8), dtype=np.float32)


class FakePost:
    """Stands in for one vLLM chat/completions call.

    Reads the question text out of the request body and returns a first-token
    ``top_logprobs`` table whose yes/no split encodes a per-finding P(yes) — so a whole
    `score_branches` call can be checked against known numbers. ``yes_token`` / ``no_token``
    let a test choose the spelling the server returns (incl. the SentencePiece marker).
    """

    def __init__(self, p_yes: dict[str, float], *, yes_token: str = "yes", no_token: str = "no"):
        self.p_yes = p_yes
        self.yes_token = yes_token
        self.no_token = no_token
        self.calls: list[dict] = []

    def __call__(self, url: str, body: dict) -> dict:
        self.calls.append(body)
        text = body["messages"][0]["content"][0]["text"]
        p = next((v for kw, v in self.p_yes.items() if kw in text), 0.5)
        top = [
            {"token": self.yes_token, "logprob": math.log(max(p, 1e-12))},
            {"token": self.no_token, "logprob": math.log(max(1.0 - p, 1e-12))},
            {"token": "maybe", "logprob": math.log(1e-6)},  # distractor in the top-k
        ]
        return {
            "choices": [{"logprobs": {"content": [{"token": self.yes_token, "top_logprobs": top}]}}]
        }


def _backend(post: FakePost, **kw) -> VLLMBackend:
    return VLLMBackend("http://node:30840/v1", "medgemma-4b", post=post, **kw)


def test_png_data_uri_from_array_and_pil():
    uri = _png_data_uri(_IMG)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > len("data:image/png;base64,")
    from PIL import Image as PILImage

    again = _png_data_uri(PILImage.new("RGB", (4, 4)))
    assert again.startswith("data:image/png;base64,")


def test_score_branches_reads_pyes_per_label():
    post = FakePost({"pleural effusion": 0.8, "pneumothorax": 0.1, "pulmonary nodule": 0.5})
    schema = LabelSchema(labels=("Effusion", "Pneumothorax", "Nodule"))
    probs = IndependentQuestionReadout().probabilities(
        _backend(post, concurrency=1), _IMG, schema.labels
    )
    assert probs == pytest.approx([0.8, 0.1, 0.5], abs=1e-6)
    # one call per label, each carrying the image and asking a single-token answer
    assert len(post.calls) == 3
    assert post.calls[0]["max_tokens"] == 1 and post.calls[0]["top_logprobs"] == 20
    assert post.calls[0]["messages"][0]["content"][1]["type"] == "image_url"


def test_sentencepiece_marker_is_normalised():
    # vLLM hands back word-initial answers marked with "▁"; without normalisation the
    # readout would match neither spelling and return NaN.
    post = FakePost({"pleural effusion": 0.7}, yes_token="▁yes", no_token="▁no")
    schema = LabelSchema(labels=("Effusion",))
    probs = IndependentQuestionReadout().probabilities(
        _backend(post, concurrency=1), _IMG, schema.labels
    )
    assert probs[0] == pytest.approx(0.7, abs=1e-6)


def test_missing_yes_and_no_is_nan():
    # A response whose top-k has neither yes nor no ⇒ no opinion ⇒ NaN, not a silent 0.5.
    class NoAnswer(FakePost):
        def __call__(self, url, body):
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "the",
                                    "top_logprobs": [
                                        {"token": "the", "logprob": -0.1},
                                        {"token": "a", "logprob": -2.0},
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }

    schema = LabelSchema(labels=("Effusion",))
    probs = IndependentQuestionReadout().probabilities(_backend(NoAnswer({})), _IMG, schema.labels)
    assert math.isnan(probs[0])


def test_concurrency_preserves_branch_order():
    post = FakePost({"pleural effusion": 0.9, "pneumothorax": 0.2, "pulmonary nodule": 0.6})
    schema = LabelSchema(labels=("Effusion", "Pneumothorax", "Nodule"))
    probs = IndependentQuestionReadout().probabilities(
        _backend(post, concurrency=4), _IMG, schema.labels
    )
    assert probs == pytest.approx([0.9, 0.2, 0.6], abs=1e-6)


def test_score_continuation_unsupported():
    with pytest.raises(NotImplementedError, match="independent_question"):
        _backend(FakePost({})).score_continuation(_IMG, "p", "c", ("yes", "no"))


def test_generate_returns_text():
    def post(url, body):
        assert "logprobs" not in body
        return {"choices": [{"message": {"content": "no acute findings"}}]}

    backend = VLLMBackend("http://node:30840/v1", "medgemma-4b", post=post)
    assert backend.generate(_IMG, "describe", 16) == "no acute findings"


def test_actor_is_cacheable_and_shapes():
    post = FakePost({"pleural effusion": 0.8, "pneumothorax": 0.1})
    schema = LabelSchema(labels=("Effusion", "Pneumothorax"))
    actor = VLLMMedGemmaAI(schema, _backend(post), prompt_set="single")

    assert isinstance(actor, CacheableModel)
    assert actor.member_names == ("single#0",)
    assert actor.member_probabilities(_IMG).shape == (1, 2)  # (n_members, n_labels)
    batch = actor.member_probabilities_batch([_IMG, _IMG])
    assert batch.shape == (2, 1, 2)
    assert batch[0, 0] == pytest.approx([0.8, 0.1], abs=1e-6)
