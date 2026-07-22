"""Five answers to "how hard is this case?", behind one contract.

Difficulty is the latent ``delta`` of ``../TESI.md`` §8.1: the per-case quantity that
drives *both* the AI's error and the reader's, and therefore the only mechanism by
which their errors can be made to correlate. §10 declares openly that radiology has
no canonical model of case difficulty and that the ``Beta(2, 5)`` prior used in the
design is a modelling choice rather than a derived result. The honest response to "no
canonical model" is not to pick one and hide it — it is to make the choice a
configuration line and report the sensitivity.

The five, ordered by how much they claim:

``ConstantDifficulty``
    Every case equally hard. The null model: run it to show that an effect you
    attribute to difficulty disappears when difficulty is switched off.

``PriorDifficulty``
    A fresh ``Beta(a, b)`` draw per case — the §8.1 placeholder, and the *only* one
    here that is unrelated to the image. Useful precisely because it is: it gives a
    reader case-to-case variation with zero coupling to the AI, i.e. the ``rho = 0``
    end of the sweep, by construction rather than by hoping.

``UncertaintyDifficulty``
    The AI's own predictive uncertainty. §8.1 calls this the natural observable proxy
    for difficulty, and §10 names it as the escape route from the arbitrary prior:
    with it the human-AI error correlation is **induced from real model behaviour**
    instead of imposed by a parameter.

``ImageQualityDifficulty``
    Blur/contrast of the pixels. Deliberately *not* about the finding: it captures
    the fraction of difficulty that is acquisition rather than pathology, which is
    the part a reader and a model plausibly share for wholly mundane reasons.

``TableDifficulty``
    A precomputed score per image, looked up by ``example_id``. This is the general
    case and the reason the seam pays for itself early: **inter-reader disagreement**
    (a case is hard exactly when experts disagree — the most defensible operational
    definition available, and computable today from the CheXpert per-reader reads)
    and **a future learned complexity model** are the same object as far as the
    framework is concerned, namely a column of numbers keyed by image. Neither needs
    new library code, and the first is the natural validation target for the second.

All five return a scalar by default; ``UncertaintyDifficulty`` can return a per-label
vector, because a case really can be hard for nodules and easy for cardiomegaly.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hai_sim.core.types import ModelOutput, Patient, PatientContext

Aggregator = str  # "mean" | "max" | "none" (per-label)

# Binary entropy of a Bernoulli peaks at ln 2 nats; standard deviation of
# probabilities in [0, 1] peaks at 0.5. Dividing by these puts every measure on a
# comparable 0-1 scale, so `rho` sweeps mean the same thing across UQ measures.
_MEASURE_MAX: Mapping[str, float] = {
    "predictive_entropy": float(np.log(2.0)),
    "mean_entropy": float(np.log(2.0)),
    "mutual_information": float(np.log(2.0)),
    "entropy": float(np.log(2.0)),
    "std": 0.5,
}


def _aggregate(values: np.ndarray, aggregator: Aggregator) -> float | np.ndarray:
    """Collapse a per-label vector, ignoring labels no member covered (NaN)."""
    if aggregator == "none":
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if aggregator == "max":
        return float(np.max(finite))
    if aggregator == "mean":
        return float(np.mean(finite))
    raise ValueError(f"unknown difficulty aggregator: {aggregator!r}")


class ConstantDifficulty:
    """Every case equally hard. The null model against which the others are read."""

    def __init__(self, value: float = 0.5) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"difficulty value must be in [0, 1], got {value}")
        self.value = float(value)
        self.name = f"constant:{value:g}"

    def score(
        self,
        *,
        patient: Patient,
        ai_output: ModelOutput | None,
        context: PatientContext,
    ) -> float | np.ndarray:
        return self.value


class PriorDifficulty:
    """A ``Beta(a, b)`` draw per case — §8.1's placeholder, and the rho = 0 control.

    ``Beta(2, 5)`` is the design default: right-skewed, most cases easy, a thin tail
    of hard ones. It is **not** derived from radiology data (§10, hole 2) and must be
    reported as an assumption. Its value is as a control: because the draw ignores
    the image entirely, a reader driven by it errs independently of the AI whatever
    the coupling parameter says, which is the baseline any complementarity claim has
    to beat.

    Draws are keyed by ``example_id`` when the patient carries one, so the same image
    is equally hard every time it is sampled — a property the study layer's paired
    contrasts depend on. Synthetic patients without an image identity fall back to a
    fresh draw per patient.
    """

    def __init__(
        self,
        a: float = 2.0,
        b: float = 5.0,
        rng: np.random.Generator | None = None,
        seed: int = 0,
    ) -> None:
        if a <= 0 or b <= 0:
            raise ValueError(f"Beta parameters must be > 0, got a={a}, b={b}")
        self.a = float(a)
        self.b = float(b)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.seed = int(seed)
        self.name = f"prior:beta({a:g},{b:g})"
        self._cache: dict[str, float] = {}

    def score(
        self,
        *,
        patient: Patient,
        ai_output: ModelOutput | None,
        context: PatientContext,
    ) -> float | np.ndarray:
        key = context.example_id
        if key is None:
            return float(self.rng.beta(self.a, self.b))
        hit = self._cache.get(key)
        if hit is None:
            # Derive this image's draw from its identity, not from call order, so the
            # value survives resampling, reordering and a different number of workers.
            stream = np.random.default_rng(
                [self.seed, *(ord(c) for c in key[-16:])]
            )
            hit = float(stream.beta(self.a, self.b))
            self._cache[key] = hit
        return hit


class UncertaintyDifficulty:
    """The AI's predictive uncertainty as the observable proxy for difficulty.

    ``../TESI.md`` §8.1: *"per l'AI reale, la sua incertezza predittiva è il proxy
    osservabile naturale della difficoltà -> l'accoppiamento si può indurre invece che
    imporre"*. This is that sentence as code, and using it means the human-AI error
    correlation reported by a study is a **measurement of a real model's behaviour**
    rather than a number typed into a config.

    ``source`` picks which measure: ``uncertainty`` is whatever the run's `UQMethod`
    selected, or name a specific one (``mutual_information`` = epistemic,
    ``mean_entropy`` = aleatoric) to read it out of the AI's observation channel.
    Choosing between them is itself a research question — epistemic uncertainty says
    "unlike my training data", aleatoric says "ambiguous image", and only the second
    is obviously something a human reader would also struggle with.
    """

    def __init__(
        self,
        source: str = "uncertainty",
        aggregator: Aggregator = "mean",
        scale: float | None = None,
    ) -> None:
        self.source = source
        self.aggregator = aggregator
        self.scale = scale
        self.name = f"uncertainty:{source}/{aggregator}"

    def _raw(self, ai_output: ModelOutput) -> np.ndarray:
        if self.source in ("uncertainty", ""):
            return np.asarray(ai_output.uncertainty, dtype=float)
        value = ai_output.extra.get(self.source)
        if value is None:
            raise ValueError(
                f"difficulty source {self.source!r} is not in the AI's observation "
                f"channel; available: {sorted(ai_output.extra)}. A named measure needs "
                "an ensemble UQ method (ai.uq: ensemble) to be produced at all."
            )
        return np.asarray(value, dtype=float)

    def score(
        self,
        *,
        patient: Patient,
        ai_output: ModelOutput | None,
        context: PatientContext,
    ) -> float | np.ndarray:
        if ai_output is None:
            return 0.0
        raw = self._raw(ai_output)
        span = self.scale if self.scale is not None else _MEASURE_MAX.get(self.source)
        if span is None:
            span = _MEASURE_MAX["entropy"]
        normalised = np.clip(raw / span, 0.0, 1.0)
        return _aggregate(normalised, self.aggregator)


class ImageQualityDifficulty:
    """Difficulty from the pixels alone: a blurrier image is a harder read.

    Uses the same two cheap proxies as `ImageQualitySeverity` — Laplacian variance
    (focus) and intensity spread (contrast) — inverted, since *low* variance means
    *high* difficulty. Deliberately blind to the finding: this isolates the share of
    difficulty that comes from acquisition rather than pathology, which is a
    plausible common cause of human and machine error that owes nothing to the
    clinical content.

    ``reference`` sets the quality at which difficulty is 0.5; leave it to have the
    model self-scale against the first images it sees (reported in ``name`` at the
    end of a run is not possible, so pass an explicit value for a reproducible study).
    """

    def __init__(
        self,
        metric: str = "laplacian_var",
        reference: float | None = None,
    ) -> None:
        if metric not in ("laplacian_var", "intensity_std"):
            raise ValueError(f"unknown image quality metric: {metric!r}")
        self.metric = metric
        self.reference = reference
        self.name = f"image_quality:{metric}"
        self._running: list[float] = []

    def _quality(self, image: np.ndarray) -> float:
        arr = np.asarray(image, dtype=float)
        if arr.size <= 1:
            return 0.0
        if self.metric == "intensity_std":
            return float(np.std(arr))
        flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
        if flat.ndim != 2 or min(flat.shape) < 3:
            return float(np.std(arr))
        # Discrete Laplacian: the classic no-reference focus measure.
        lap = (
            -4.0 * flat[1:-1, 1:-1]
            + flat[:-2, 1:-1]
            + flat[2:, 1:-1]
            + flat[1:-1, :-2]
            + flat[1:-1, 2:]
        )
        return float(np.var(lap))

    def score(
        self,
        *,
        patient: Patient,
        ai_output: ModelOutput | None,
        context: PatientContext,
    ) -> float | np.ndarray:
        if not patient.images:
            return 0.0
        quality = self._quality(patient.images[0])
        reference = self.reference
        if reference is None:
            self._running.append(quality)
            reference = float(np.median(self._running))
        if reference <= 0.0:
            return 0.0
        # Sharper than reference -> below 0.5; blurrier -> above. Bounded, monotone.
        return float(np.clip(reference / (reference + quality), 0.0, 1.0))


class TableDifficulty:
    """A precomputed difficulty per image, looked up by ``example_id``.

    The general case, and the one that makes this seam worth building before any
    sophisticated difficulty model exists. Two very different things reduce to it:

    - **Inter-reader disagreement.** A case is hard exactly when expert readers
      disagree about it. This is the most defensible operational definition on offer
      and it is computable today — the CheXpert test set ships eight radiologists'
      reads of 500 studies. Build the column with
      `hai_sim.difficulty.disagreement_table`.
    - **A learned complexity model.** Whatever predicts difficulty from the image in
      future, its output is a column of numbers keyed by image id. It plugs in here
      with no library change, and it is validated against the disagreement column —
      which is a research question in its own right, and one nobody has to write new
      framework code to ask.

    ``default`` is used for images missing from the table. Set ``strict=True`` to
    fail instead, which is what you want when the table is meant to cover the whole
    stream and a silent default would quietly flatten the difficulty distribution.
    """

    def __init__(
        self,
        table: Mapping[str, float],
        default: float = 0.5,
        strict: bool = False,
        name: str = "table",
    ) -> None:
        self.table = {str(k): float(v) for k, v in table.items()}
        self.default = float(default)
        self.strict = strict
        self.name = f"{name}:n={len(self.table)}"

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        id_column: str = "example_id",
        value_column: str = "difficulty",
        default: float = 0.5,
        strict: bool = False,
    ) -> TableDifficulty:
        """Load a two-column table. Any producer that writes a CSV can drive this."""
        rows: dict[str, float] = {}
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {id_column, value_column} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing column(s) {sorted(missing)}; "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                raw = row[value_column]
                if raw is None or raw == "":
                    continue
                rows[row[id_column]] = float(raw)
        return cls(rows, default=default, strict=strict, name=Path(path).stem)

    def score(
        self,
        *,
        patient: Patient,
        ai_output: ModelOutput | None,
        context: PatientContext,
    ) -> float | np.ndarray:
        key = context.example_id
        if key is not None and key in self.table:
            return self.table[key]
        if self.strict:
            raise KeyError(
                f"difficulty table has no entry for example_id={key!r}. The table must "
                "cover every image in the stream, or pass strict=False to accept the "
                "default (which flattens the difficulty distribution where it applies)."
            )
        return self.default


def disagreement_table(
    reader_labels: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    label_indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Turn per-reader reads into a difficulty column: how much the experts disagreed.

    ``reader_labels`` maps ``example_id -> reader_id -> label vector`` in the usual
    ``{1, 0, -1}`` convention. For each image, difficulty is the mean over labels of
    the *normalised disagreement* among readers who annotated that label: 0.0 when
    they were unanimous, 1.0 at an even split. Labels nobody annotated are skipped;
    images where nothing was annotated get 0.0.

    This is the operational definition of "hard case" that ``../TESI.md`` §4.8
    already settled on for *error* — a read that diverges from the adjudicated
    consensus — pointed at difficulty instead: rather than asking who was wrong, ask
    how much the panel split. It needs no model, no training and no new data.
    """
    out: dict[str, float] = {}
    for example_id, per_reader in reader_labels.items():
        votes = np.asarray([list(v) for v in per_reader.values()], dtype=float)
        if votes.size == 0:
            out[str(example_id)] = 0.0
            continue
        if label_indices is not None:
            votes = votes[:, list(label_indices)]
        scores: list[float] = []
        for column in votes.T:
            annotated = column[column >= 0]
            if annotated.size < 2:
                continue
            share = float(np.mean(annotated))
            # Gini impurity of the vote, 2p(1-p), rescaled by 2 so the range is the
            # full [0, 1]: 0 when the panel was unanimous, 1 at an even split.
            scores.append(4.0 * share * (1.0 - share))
        out[str(example_id)] = float(np.mean(scores)) if scores else 0.0
    return out
