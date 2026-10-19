"""HaiSIM (package ``hai_sim``) — a playground for testing an AI placed *in series* with a doctor.

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

    ArrivalProcess      when patients arrive — kept apart from which patient arrives
    Dataset             where images and ground truth come from
    DiagnosticActor     an AI model, or a doctor — anything that reads images
    UQMethod            how per-prediction uncertainty is derived
    Calibrator          whether that probability can be believed
    SortStrategy        the triage rule: what order the doctor works the queue in
    SeverityModel       how much this case matters clinically
    DifficultyModel     how easy this case is to get wrong
    FatigueCurve        how worn the reading station is, right now
    ServiceTimeModel    how long one read takes
    IntegrationPolicy   how much the doctor trusts the AI, and under what conditions
    KPIBundle           a metric — including one written in your own file
    Grouping            a conditioning axis (per severity band, per pathology, ...)

A seam is worth having when you can name three implementations you would actually run
*and compare in the same study*. That test is why a cost model is not on this list — a
cost matrix is already a KPIBundle — and why station topology is not either: there is
only one arrangement under test here.

THE SIGNAL PLANE — what the engine computes between the two stations
--------------------------------------------------------------------
Severity, difficulty and fatigue are not produced by either reader. The engine computes
them after the AI has spoken and hands them to the doctor in ``context.signals``, and to
the telemetry as ``sig.*`` so any metric can be conditioned on them.

Difficulty is the one that matters most. A simulated reader that flips labels at a fixed
rate draws its errors from a coin that has never seen the image, so its mistakes are
independent of the AI's *by construction* — no scenario can make them otherwise. Wire a
difficulty model to the AI's own uncertainty and give the reader a ``difficulty_slope``,
and the two start failing on the same cases. That correlation is the quantity the whole
complementarity argument turns on, and it is a knob only once this plane exists.

WHERE TO START
--------------
    hai-sim run configs/examples/quickstart.yaml          one run, in seconds
    hai-sim capture configs/studies/capture.yaml          RQ1 and RQ2 of the paper
    hai-sim study configs/studies/queue_load_medgemma27b.yaml   RQ3
    hai-sim study configs/studies/team_effect_medgemma27b.yaml  RQ4
    docs/scenarios.md                                     every knob
    docs/outputs.md                                       every metric
    docs/glossary.md                                      every acronym, defined once

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

Arrivals get their own stream, separate from patient sampling, for the same reason: a
comparison of two arrival processes must be *only* that. If the image sequence moved
too, any difference in outcome would confound the schedule with the case mix.
"""

__version__ = "0.0.1"
