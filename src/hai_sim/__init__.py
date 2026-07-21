"""hai_sim — a playground for testing an AI placed *in series* with a doctor.

AI = artificial intelligence; here it always means the automated reader (a chest X-ray
classifier), never a person.

WHAT THIS IS
------------
A hospital reading room where an AI model and a simulated radiologist read the same
chest X-rays, one after the other, and everything that happens is measured. You change
one piece — the model, the doctor's skill, how much the doctor trusts the model, the
order the queue is worked through — rerun, and see what moved.

It is a *playground*, not a product: every stage is swappable from a YAML file, and no
stage needs to know what you swapped elsewhere.

THE ARRANGEMENT — why "in series"
---------------------------------
    patients arrive  →  [ AI reads ]  →  queue reordered  →  [ doctor reads ]  →  measured
                             │               by the AI's         sees the AI's
                             │              uncertainty          opinion as a hint
                             └────────────── same contract ──────────────┘

The AI goes **first** and does two distinct jobs, which the design keeps separate
because they can be studied independently:

1. **It reorders the queue.** Its uncertainty becomes a triage key, so the cases it
   finds hardest can be put in front of the doctor first. This only changes anything
   when patients arrive faster than they can be read — with an idle queue there is
   nothing to reorder.
2. **It advises the doctor.** Its opinion reaches the doctor as a hint, and a swappable
   *integration policy* decides how much weight that hint carries.

The doctor always forms its own opinion **before** the AI's is folded in, so both the
solo and the aided verdict exist for every patient. That is what makes it possible to
ask not just "was the team right?" but "did the doctor take the advice when it should
have?" — a question a real two-stage reader study needs a costly protocol to answer.

THE SEAMS — where you plug your own thing in
--------------------------------------------
Every one of these is a Protocol in ``core/protocols.py``: implement the methods and it
fits, no inheritance needed.

    DiagnosticActor     an AI model, or a doctor — anything that reads images
    SortStrategy        the triage rule: what order the doctor works the queue in
    IntegrationPolicy   how much the doctor trusts the AI, and under what conditions
    UQMethod            how per-prediction uncertainty is derived
    SeverityModel       a per-patient clinical-weight signal
    Dataset             where images and ground truth come from
    KPIBundle           a metric — including one written in your own file
    Grouping            a conditioning axis (per severity band, per pathology, ...)

WHERE TO START
--------------
    hai-sim run configs/examples/v0_smoke_hybrid.yaml     a complete run in one second
    GLOSSARY.md                                           every acronym, defined once
    README.md                                             every knob and every metric
    ../TESI.md                                            why any of it is arranged so

Reading the code in pipeline order: ``sim/patient_generator.py`` → ``models/`` →
``sort/`` → ``doctors/`` → ``metrics/``, with ``sim/engine.py`` as the spine that runs
them and ``factory.py`` as the bridge from YAML to objects.

A NOTE ON REPRODUCIBILITY
-------------------------
One seed per run is split into independent per-module streams, so changing the doctor
leaves the patient sequence untouched. Two variants at the same seed therefore meet the
same patients in the same order, and comparing them seed by seed cancels the sampling
noise instead of averaging over it. Without that property, comparing two runs would
mostly measure luck.
"""

__version__ = "0.0.1"
