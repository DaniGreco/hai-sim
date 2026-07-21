"""Named doctor-persona library: capability x visibility x trust presets.

A 2 (capability: senior/junior) x 3 (solo / trusting / skeptical) grid — the natural
sweep for the RQ3 "how do personas change team outcome & appropriate reliance?" study.

Capability = per-finding sensitivity/specificity (senior 0.95/0.95, junior 0.85/0.88).
Trust = the Weight of Advice (`weight_ai`) on a linear pool:

    skeptical:  senior 0.20 < junior 0.40      trusting: senior 0.45 < junior 0.65

**What is anchored and what is not** — the two axes differ sharply, and papers should say
so (see ../TESI.md §6.7-bis):

- `weight_ai` default 0.39 = the pooled mean WOA of Bailey et al. 2023 (346 effects,
  N~17,000). Anchored, but that meta-analysis covers *generic* numeric-estimate
  judge-advisor tasks, not clinical reading.
- **senior < junior is anchored only as an ordering, and on a different construct.**
  Gaube 2021 (265 physicians, 8 CXR cases): 27.54% of radiologists vs 41.73% of IM/EM
  physicians are *always* susceptible (always wrong when the advice is inaccurate), and
  28.26% vs 17.32% refute *every* incorrect advice. Those are per-physician
  classifications, NOT advice-uptake rates. Same direction in Dratsch 2023 and Logg 2019.
- **No clinical source measures WOA at all**, and Gaube cannot: participants see the X-ray
  and the advice together, so there is no pre-advice opinion to compute
  (final - initial)/(advice - initial) from. Gaube also found the advice *source*
  (AI vs human) did **not** affect decisions — algorithmic aversion showed up only in
  quality ratings, and only among experts. So it licenses no AI-specific decision weight.
- The four magnitudes (0.20/0.45/0.40/0.65) are modelling choices that respect the
  ordering around 0.39. Not measured.
- The sensitivity/specificity values are **modelling choices, not calibrated** — and they
  are optimistic: the radiology literature reports ~0.82 baseline sensitivity, 54% on
  subtle vs 87% on obvious findings, and a 32-63% nodule-reporting range across readers.

Calibrating capability on real per-reader, per-finding reads (CheXpert reads vs the
5-vote majority) is the planned fix; it unlocks `capability.source: reader`.
"""

from __future__ import annotations

from hai_sim.config.scenario import CapabilityConfig, PersonaConfig, TrustConfig

_SENIOR = (0.95, 0.95)
_JUNIOR = (0.85, 0.88)


def _cap(sens_spec: tuple[float, float]) -> CapabilityConfig:
    return CapabilityConfig(
        kind="fixed_accuracy", sensitivity=sens_spec[0], specificity=sens_spec[1]
    )


def _woa(weight_ai: float) -> TrustConfig:
    return TrustConfig(kind="linear_pool", weight_ai=weight_ai)


PERSONA_LIBRARY: dict[str, PersonaConfig] = {
    "solo_senior": PersonaConfig(name="solo_senior", capability=_cap(_SENIOR), sees_ai=False),
    "solo_junior": PersonaConfig(name="solo_junior", capability=_cap(_JUNIOR), sees_ai=False),
    "senior_skeptical": PersonaConfig(
        name="senior_skeptical", capability=_cap(_SENIOR), sees_ai=True, trust=_woa(0.20)
    ),
    "senior_trusting": PersonaConfig(
        name="senior_trusting", capability=_cap(_SENIOR), sees_ai=True, trust=_woa(0.45)
    ),
    "junior_skeptical": PersonaConfig(
        name="junior_skeptical", capability=_cap(_JUNIOR), sees_ai=True, trust=_woa(0.40)
    ),
    "junior_trusting": PersonaConfig(
        name="junior_trusting", capability=_cap(_JUNIOR), sees_ai=True, trust=_woa(0.65)
    ),
}


def resolve_persona(persona: str | PersonaConfig) -> PersonaConfig:
    """A persona is either a library name or an inline `PersonaConfig`."""
    if isinstance(persona, str):
        if persona not in PERSONA_LIBRARY:
            raise ValueError(
                f"unknown persona {persona!r}; library: {tuple(PERSONA_LIBRARY)}"
            )
        return PERSONA_LIBRARY[persona]
    return persona
