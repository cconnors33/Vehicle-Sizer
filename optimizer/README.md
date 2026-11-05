# Optimizer

`core.py` defines evaluation, solver dispatch, and the `optimize()`
API. `coordinates.py` handles design vectors and `objectives.py` handles scoring.
`core.ProgressTracker` formats console progress for both solvers and process-worker runs.
`workers.py` supervises process workers, candidate timeouts, recycling, and
resumption. `timing.py` supplies optional nested CPU/wall profiling.

Run from the project root:

```bash
.venv/bin/python -m optimizer Configs/optimizer_epump.yaml --output outputs/epump_search_01 --workers 4 --max-evaluations 100
```

Use `--profile` to save candidate timings, or `--resume` to replay saved results
and continue with the original settings, budget, and source files. The default
is serial evaluation; worker controls can also be specified in the search config.

Python callers use `from optimizer.core import optimize`. The worker entry point
is `python -m optimizer.workers`; the coordinator launches it automatically.
Source fingerprints include the packages used by evaluations and exclude `examples`.

When a vehicle section contains `masses.counterweight`, the optimizer registers
`counterweight_mass`. Supply its kg range in `bounds`, e.g. `[0.0, 20.0]`; equal
bounds such as `[10.0, 10.0]` hold it fixed, and `[0.0, 0.0]` disables the mass.
Exactly one section may contain this mass entry. Its section determines placement;
the mass remains distributed through that section using the existing vehicle model.
Running main directly uses the mass in the vehicle config, not optimizer bounds.
Use a new output directory when changing the variable set or bounds; resume retains
the original search settings and checks source fingerprints.

Candidate records in `evaluations.jsonl`, worker `result.json`, and `best.json`
include `gg_sizing` keyed by GG name. Each entry reports required shaft power [W],
turbine inlet/outlet and upstream feed pressures [Pa absolute], oxidizer/fuel
mixture ratio, and total/fuel/oxidizer mass flows [kg/s]. The record is empty when
there is no GG or GG sizing was unavailable. Available sizing is retained for
later preflight rejections and partial flight results.

`Evaluator.evaluate(candidate, progress=...)` decodes search coordinates once and
returns the resolved configuration, `SimResult`, and scored record. The serial
optimizer callable owns its budget, incumbent, and JSONL log; workers publish the
returned record directly. Each evaluation saves `candidate.yaml` before simulation
so native failures retain their inputs. Workers reuse property/aero models and
retain process deadlines, recycling, source checks, and resume support.

The final result is the best design accepted during the search. Its configuration
and evaluation record are saved as `best.yaml` and `best.json`; no additional
simulation runs after the search. `summary.json` reports search acceptance.

Simulation owns propulsion cleanup and returns recovered solver attempts in
`SimResult.fluid_retries`, including partial flight results. Evaluation records expose
these as `recovered_fluid_retries`; load peaks and failure classification also use the
same record construction in serial and worker runs. Per-worker `evaluations.jsonl`
and `best.*` files are no longer produced; `result.json` is the persisted worker
record and the coordinator writes the search-wide log and winner.

Search preparation validates soft-penalty settings once; scoring checks changing
measurements. `candidate_score()` returns `(score, violations, soft_penalties)` so
the evaluation record reuses the exact penalties used in the objective. Constraint
scoring consumes `ConstraintRecord` directly; there is no raw-margin fallback.

## SLSQP local refinement

Select the solver with `algorithm: differential_evolution` (the default) or
`algorithm: slsqp`, or override it with `--algorithm slsqp`. There is one solver
selector. SLSQP settings are:

```yaml
algorithm: slsqp
slsqp:
  maxiter: 100
  ftol: 1.0e-6
  gradient_step: 0.001
  probe_retries: 4       # Step halvings after trying both directions
  min_step: 1.0e-6       # Minimum normalized probe step / restart radius
  max_restarts: 5        # Restarts after unusable optimization steps
```

SLSQP starts from `base_config` after `evaluation_overrides`. The initial design
must be accepted and representable within the configured bounds, including fixed
values and conditional geometry. To refine a previous winner, select its
`best.yaml` as `base_config` and use compatible bounds and overrides. The encoder
checks that decoding reproduces the requested physical values; it never clips an
initial design. An invalid start is reported in `summary.json`.

`core.py` owns setup, `run_de()` / `run_slsqp()`, evaluation bookkeeping, and
final reporting. `coordinates.py` owns variable paths, conditional encoding/decoding,
and normalization of free coordinates to [0, 1]. `objectives.py` exposes
`objective()` (normalized mass plus soft penalties), `ranked_score()` (DE's
existing feasibility bands), and `constraint_margins()` (ordered required margins,
normalized by the configured constraint/unit scales, feasible when nonnegative).
`candidate_score` remains available from `core.py` for existing callers.

SLSQP uses `objective()` directly and supplies hard margins as explicit inequality
constraints. The initial accepted evaluation establishes the required constraint
set. Completed infeasible trials remain usable when all measurements are present.
For classified nonfatal domain/numerical failures and preflight rejections,
finite differences try the opposite direction and progressively halve the probe.
Objective gradients and constraint Jacobians share the same successful probes.
If a full optimization trial fails, the solver restarts from the last usable major
iterate in a smaller box: half the smaller of the previous radius and failed step's
maximum normalized displacement. Restarts retain the global evaluation budget and
iteration count. The box remains restricted for the remainder of the run, so
convergence after recovery describes that smaller neighborhood, not the original
full search domain. This is a bounded recovery heuristic, not a guarantee of progress.

`recovery.jsonl` records failed probes and restarts, including the variable/step or
new radius and anchor. Failed evaluations retain null objective values and never
become accepted winners. Initial failures, unknown/missing measurements, changing
required constraints, exhausted recovery limits, fatal simulation errors, and the
existing failure monitor still stop the search. Normal local stops retain the best
accepted design as the final result.

`gradient_step` is an absolute finite-difference step in normalized coordinates;
`ftol` is SLSQP's convergence tolerance. These defaults require calibration to the
simulation's numerical resolution. Conditional intervals and flight events may
introduce kinks. Objective and constraint callbacks share a bounded cache of exact
candidate evaluations. Every actual search evaluation, including the initial design
and numerical-gradient probes, counts against `max_evaluations`. A fully fixed
design is evaluated once without running SLSQP.

SLSQP is serial-only in this version (`workers: 1`, no resume or profiling).
Its evaluation records add `objective_value`, `constraint_names`,
`normalized_constraint_margins`, and `slsqp_error`; `score` retains the DE ranking
for reporting. The summary distinguishes solver success from design acceptance.

## Progress output

DE and SLSQP use the same `ProgressTracker`, including process-worker DE runs.
Each completed simulation prints the same fields: completed count/budget, candidate
ID, ranked score, best accepted score, apogee, mass, simulated duration, evaluation
wall time, total wall time, average wall time per completion, acceptance, and
termination. Missing measurements display as `N/A`. Worker completions may arrive
out of order; the completion count still increases monotonically and the candidate
ID identifies the corresponding record. Total and average wall times describe the
current invocation, including any replay of cached worker results.

SLSQP cache hits do not add progress lines or consume evaluations. Recovery events
use the same tracker without increasing the completion count. The displayed score
uses the shared feasibility ranking for both solvers; SLSQP's raw objective remains
available as `objective_value` in evaluation records.
