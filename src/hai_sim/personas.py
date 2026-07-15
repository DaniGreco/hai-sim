"""Named doctor-persona library: capability x visibility x trust presets.

A 2 (capability: senior/junior) x 3 (solo / trusting / skeptical) grid — the natural
sweep for the RQ3 "how do personas change team outcome & appropriate reliance?" study.

Capability = per-finding sensitivity/specificity (senior 0.95/0.95, junior 0.85/0.88).
Trust = the Weight of Advice (`weight_ai`) on a linear pool, anchored to the advice-taking
literature: pooled mean WOA ~= 0.39 (Bailey 2023). Experts discount more than novices
(Logg 2019; Tschandl 2020; clinical uptake of a *wrong* AI ~25-36% experts vs 42-80%
novices, Gaube 2021 / Dratsch 2023), so senior < junior at each attitude:

    skeptical:  senior 0.20 < junior 0.40      trusting: senior 0.45 < junior 0.65
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
