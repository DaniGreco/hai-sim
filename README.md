# hai_sim — Human-AI Diagnostic Workflow Simulator

Modular SimPy-based simulator of a hybrid human-AI diagnostic department.
Part of the master thesis of Daniele Greco (UniBg).

## Pipeline

```
PatientGenerator → input_queue → ModelStation → triaged_store (PriorityStore)
                                                     ↓
                                              DoctorStation → MetricsCollector
```

All actors (AI model, doctor digital twin) implement the same `DiagnosticActor`
protocol: `diagnose(images, *, context, ai_hint) → ModelOutput`. Swap any
implementation via the YAML scenario config.

## v0 scope

- Mocks only: `RandomMockAI`, `RandomMockDoctor`, `RandomPatientGenerator`
- Two sort strategies: `RandomSort`, `UncertaintyDescSort`
- Metrics: TTFE, MTTF, mean-time-to-FN, F1 (per-actor + hybrid)
- No fatigue model, no X-ray machine, no real CNN

## Quickstart

```bash
pip install -e .[dev]
hai-sim run configs/examples/v0_smoke.yaml
```

Output goes to `out/run_<ts>.csv` + `out/summary_<ts>.json`.
