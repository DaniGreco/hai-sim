"""Offline analysis of model behaviour, decoupled from the SimPy engine.

These tools operate on a one-shot inference cache (see ``inference_cache``) rather
than on simulation runs: the uncertainty-vs-error relationship is a property of
*model + UQ + dataset + threshold*, independent of queueing, the doctor twin or
arrival timing. Caching the raw per-member probabilities once lets every UQ
measure / threshold / aggregation / error definition be re-derived for free.
"""
