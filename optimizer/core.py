"""Bounded launch-mass search. Physical units in; accepted designs alone can win."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import json
from math import isfinite
from pathlib import Path
import time
import traceback
from collections import Counter, deque
from diagnostics.errors import failure_details, SearchFailureLimit, LookupBoundsError, UnsupportedPhaseChangeError

from scipy.optimize import minimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import yaml
import numpy as np

from AeroTables import DragModel
from AeroTables.dragmodel import _end as aero_endpoint
from Configs.loader import load_config
from diagnostics.constraints import ConstraintRecord, configured_limits
from diagnostics.errors import EvaluationFailure, GeometryError
from simulation import project_path, simulate
from FluidTables.PropertyModels import property_sources, TableCombustionPropertySource
from simulation_types import SimResult
from optimizer.coordinates import (CoordinateSpace, INCH, AERO_PATHS, DIMENSIONLESS,
                                   pump_roles, variable_paths, get_path, set_path,
                                   pressurant_inner_radius)
from optimizer.objectives import (candidate_class, soft_penalties, objective,
                                  ranked_score as candidate_score, constraint_margins)


def aero_bounds(model):
    """Independent box bounds in SI; coupled bounds remain candidate checks."""
    def endpoint(value, index):
        if isinstance(value, str):
            return endpoint(model.ranges[value][index], index)
        if isinstance(value, (list, tuple)):
            return float(value[0]) * endpoint(value[1], index)
        return float(value)
    return {name: tuple(endpoint(v, i) * (1 if name in DIMENSIONLESS else INCH)
                        for i, v in enumerate(model.ranges[name])) for name in AERO_PATHS}


def rejection(records):
    return SimResult(termination="infeasible_initial_design",
                     constraint_records=records)


def domain_records(candidate, model):
    records = {}
    for name, limits in model.ranges.items():
        low, high = [aero_endpoint(v, candidate) for v in limits]
        factor = 1 if name in DIMENSIONLESS else INCH
        margin = min(candidate[name] - low, high - candidate[name])
        # Match DragModel.check's boundary tolerance after SI/inch conversion.
        if -1e-9 <= margin < 0:
            margin = 0.0
        records[f"aero.{name}"] = ConstraintRecord(
            margin * factor,
            "AeroTables", "sizing", "1" if name in DIMENSIONLESS else "m")
    return records


class EvaluationBudget(Exception):
    pass


class FailureMonitor:
    """Candidate-ordered circuit breaker; physical/domain rejects are not solver failures."""
    def __init__(self, policy=None):
        policy = policy or {}
        self.consecutive_limit = policy.get('consecutive_limit', 5)
        self.window = policy.get('window', 50)
        self.rate_limit = policy.get('rate_limit', .2)
        self.repeated_limit = policy.get('repeated_limit', 3)
        for value in (self.consecutive_limit, self.window, self.repeated_limit):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Failure monitor limits must be positive integers')
        if not 0 < self.rate_limit <= 1:
            raise ValueError('Failure rate limit must be in (0, 1]')
        self.recent = deque(maxlen=self.window)
        self.consecutive = 0
        self.fingerprints = Counter()

    def observe(self, row):
        detail = row.get('failure_details') or {}
        kind = detail.get('kind', row.get('termination'))
        native_attempts = [attempt for attempt in row.get('worker_attempts', [])
                           if attempt['kind'] == 'native_crash']
        if native_attempts:
            kind = 'native_crash'
        failed = kind in {'solver_nonconvergence', 'residual_acceptance_failure',
                          'trial_domain_failure', 'native_crash', 'wall_timeout',
                          'worker_error', 'unexpected_error'}
        self.recent.append(failed)
        self.consecutive = self.consecutive + 1 if failed else 0
        if kind in {'unexpected_error', 'worker_error', 'native_crash'}:
            signature = detail.get('fingerprint', kind)
            if native_attempts:
                signature = f"native_crash:{native_attempts[-1]['exit_code']}"
            self.fingerprints[signature] += 1
            if self.fingerprints[signature] >= self.repeated_limit:
                raise SearchFailureLimit(f'Repeated {kind}: {signature}')
        if self.consecutive >= self.consecutive_limit:
            raise SearchFailureLimit(f'{self.consecutive} consecutive numerical/worker failures')
        if len(self.recent) == self.window and sum(self.recent) / self.window > self.rate_limit:
            raise SearchFailureLimit(f'Numerical/worker failure rate exceeds {self.rate_limit:.0%} in last {self.window} candidates')


class GenerationBudgetSolver(DifferentialEvolutionSolver):
    """Run to maxiter/evaluation budget; equal rejection scores are not convergence."""

    def converged(self):
        return False


def measurement(value, spec, unit=''):
    if value is None or not isfinite(value):
        return 'N/A'
    return f'{value:{spec}}{unit}'


class ProgressTracker:
    """track progress!"""
    def __init__(self, budget):
        self.budget = budget
        self.completed = 0
        self.best_score = None
        self.started = time.perf_counter()

    def update(self, record):
        self.completed += 1
        score = record['score']
        if record['accepted'] and (self.best_score is None or score < self.best_score):
            self.best_score = score
        elapsed = time.perf_counter() - self.started
        candidate = record.get('evaluation', record.get('index', self.completed))
        wall = record.get('wall_seconds', record.get('elapsed_seconds'))
        print(
            f"Run {self.completed}/{self.budget} | candidate={candidate}"
            f" | score={measurement(score, '.6g')} | best={measurement(self.best_score, '.6g')}"
            f" | apogee={measurement(record.get('apogee'), ',.1f', ' m')}"
            f" | mass={measurement(record.get('mass'), '.2f', ' kg')}"
            f" | sim={measurement(record.get('simulation_seconds'), '.2f', ' s')}"
            f" | wall={measurement(wall, '.2f', ' s')}"
            f" | total wall={elapsed:.1f} s | avg={elapsed / self.completed:.2f} s/run"
            f" | accepted={record['accepted']} | {record['termination']}",
            flush=True,
        )

    def event(self, kind, details):
        """Report extra solver events without counting another evaluation."""
        print(f"Search {kind} after {self.completed}/{self.budget} evaluations: {details}", flush=True)


class Evaluator:
    def __init__(self, cfg, settings, model, pure, combustion, output):
        self.cfg, self.settings = deepcopy(cfg), settings
        self.model, self.pure, self.combustion = model, pure, combustion
        if isinstance(combustion, TableCombustionPropertySource):
            low, high = combustion.design_exit_pressure_bounds(settings['bounds']['chamber_pressure'],
                                                               settings['bounds']['mixture_ratio'])
            if not low <= settings['bounds']['exit_pressure'][0] <= settings['bounds']['exit_pressure'][1] <= high:
                raise ValueError(f'Exit-pressure bounds must lie in the common table sizing interval [{low}, {high}] Pa')
        self.output = Path(output)
        self.coordinates = CoordinateSpace(cfg, settings, model, pure, combustion)
        self.paths = self.coordinates.paths
        self.names = self.coordinates.names
        self.bounds = self.coordinates.bounds
        self.count = 0
        self.best = None
        self.progress = ProgressTracker(settings["max_evaluations"])
        self.failure_monitor = FailureMonitor(settings.get('failure_policy'))

    def decode(self, values):
        return self.coordinates.decode(values)

    def evaluate(self, candidate, *, progress=None):
        """Return (resolved config, simulation result, record), without search bookkeeping.

        Candidates are search coordinates decoded into physical configurations.
        """
        started = time.perf_counter()
        cfg, failure, details = None, None, None
        try:
            cfg = self.decode(candidate)
            (self.output / "candidate.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            result = simulate(cfg, pure_properties=self.pure, combustion_properties=self.combustion,
                              aero_model=self.model, assess_constraints=True,
                              compute_loads=False, progress=progress)
        except GeometryError as error:
            result = rejection({k: ConstraintRecord(v, "optimizer", "sizing", "m") for k, v in error.constraints.items()})
        except Exception as error:
            phase = getattr(error, 'phase', 'sizing')
            details = failure_details(error, phase=phase)
            failure = traceback.format_exc()
            suffix = f"_{self.count + 1:04d}"
            (self.output / f"failed{suffix}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (self.output / f"failure{suffix}.txt").write_text(failure)
            if (details['fatal'] or not isinstance(
                    error, (EvaluationFailure, LookupBoundsError, UnsupportedPhaseChangeError))):
                if isinstance(error, EvaluationFailure):
                    raise
                raise EvaluationFailure("optimizer evaluation", cfg) from error
            result = deepcopy(getattr(error, 'partial_result', None) or SimResult())
            result.apogee = None
            result.termination = details['kind']
        score, contributions, penalties = candidate_score(result, self.settings)
        variables = ({name: get_path(cfg, path) for name, path in self.paths.items()} if cfg is not None else {})
        entry = dict(variables=variables,
                     search_coordinates=list(map(float, candidate)), score_class=candidate_class(result), failure=failure,
                     failure_details=details,
                     battery_sizing=result.battery_sizing,
                     recovered_fluid_retries=result.fluid_retries, load_peaks=result.load_peaks,
                     gg_sizing={name: {key: sizing[key] for key in (
                         "required_shaft_power_w", "inlet_pressure_pa", "outlet_pressure_pa",
                         "upstream_pressures_pa", "mixture_ratio", "design_mdot",
                         "fuel_mdot", "oxidizer_mdot")}
                         for name, sizing in result.gg_sizing.items()},
                     pressure_tracking=result.pressure_tracking,
                     min_stability_length_fraction=result.min_stability_length_fraction,
                     max_stability_length_fraction=result.max_stability_length_fraction,
                     soft_penalties=penalties,
                     score=score, accepted=result.accepted, mass=result.initial_mass,
                     termination=result.termination, warnings=result.warnings,
                     apogee=result.apogee, dry_mass=result.dry_mass,
                     burn_duration=result.burn_duration, violations=contributions,
                     simulation_seconds=result.final_time,
                     constraints={k: asdict(v) for k, v in result.constraint_records.items()},
                     elapsed_seconds=time.perf_counter() - started)
        if self.settings.get('algorithm') == 'slsqp':
            entry.update(objective_value=None, constraint_names=[], normalized_constraint_margins=[],
                         slsqp_error=None)
            try:
                if candidate_class(result) not in ('feasible', 'completed_infeasible'):
                    raise ValueError(f"SLSQP requires a fully assessed flight: {result.termination}")
                if not result.accepted:
                    penalties = soft_penalties(result, self.settings)
                entry['soft_penalties'] = penalties
                entry['objective_value'] = objective(result, self.settings, penalties=penalties)
                names = sorted(key for key, record in result.constraint_records.items() if record.required)
                entry['constraint_names'] = names
                entry['normalized_constraint_margins'] = constraint_margins(result, self.settings, names)
            except ValueError as error:
                entry['slsqp_error'] = str(error)
        return cfg, result, entry

    def __call__(self, values):
        return self.evaluate_candidate(values)[2]['score']

    def evaluate_candidate(self, values):
        """Budgeted evaluation returning config, SimResult, and persisted record."""
        if self.count >= self.settings["max_evaluations"]:
            raise EvaluationBudget
        cfg, result, entry = self.evaluate(values)
        self.count += 1
        entry['evaluation'] = self.count
        score = entry['score']
        with (self.output / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        if result.accepted and (self.best is None or score < self.best["score"]):
            self.best = entry
            (self.output / "best.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (self.output / "best.json").write_text(json.dumps(entry, indent=2, allow_nan=False))
        self.progress.update(entry)
        try:
            self.failure_monitor.observe(entry)
        except SearchFailureLimit as error:
            (self.output / 'stopped.json').write_text(json.dumps(dict(
                reason=str(error), evaluations=self.count, last_failure=entry['failure_details']), indent=2))
            raise
        return cfg, result, entry


def prepare(settings, model):
    settings = deepcopy(settings)
    algorithm = settings.get('algorithm', 'differential_evolution')
    if algorithm not in ('differential_evolution', 'slsqp'):
        raise ValueError(f"Unknown optimizer algorithm: {algorithm}")
    if algorithm == 'slsqp':
        options = dict(maxiter=100, ftol=1e-6, gradient_step=1e-3,
                       max_restarts=5, probe_retries=4, min_step=1e-6)
        supplied = settings.get('slsqp', {})
        if not isinstance(supplied, dict):
            raise ValueError("slsqp must be a YAML mapping; indent maxiter, ftol, and gradient_step under slsqp:")
        if set(supplied) - set(options):
            raise ValueError(f"Unknown slsqp options: {sorted(set(supplied) - set(options))}")
        options.update(supplied)
        value = options['maxiter']
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("slsqp.maxiter must be a positive integer")
        for name in ('max_restarts', 'probe_retries'):
            value = options[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"slsqp.{name} must be a nonnegative integer")
        for name in ('ftol', 'gradient_step', 'min_step'):
            if isinstance(options[name], bool) or not isfinite(options[name]) or options[name] <= 0:
                raise ValueError(f"slsqp.{name} must be finite and positive")
        if options['gradient_step'] > 1:
            raise ValueError("slsqp.gradient_step must not exceed the normalized range of 1")
        if options['min_step'] > options['gradient_step']:
            raise ValueError("slsqp.min_step must not exceed gradient_step")
        settings['slsqp'] = options
    cfg = load_config(project_path(settings["base_config"]))
    pumps = pump_roles(cfg)
    ids = settings["tank_ids"]
    if set(ids) != {"oxidizer", "fuel", "pressurant"} or len(set(ids.values())) != 3:
        raise ValueError("Optimizer tank roles must reference three distinct tanks")
    for role, tank_id in ids.items():
        expected = "pressurant" if role == "pressurant" else "propellant"
        if cfg["tanks"][tank_id]["type"] != expected:
            raise ValueError(f"{role} must reference a {expected} tank")
    if len(set(pumps.values())) != len(pumps):
        raise ValueError("Oxidizer and fuel must use distinct pumps")
    copv = cfg["tanks"][ids["pressurant"]]
    copv.pop("mass", None)
    states = cfg["prop_system"].get("initial_conditions", {})
    for role in ("oxidizer", "fuel"):
        states[ids[role]].pop("P", None)
    cfg.setdefault('constraints', {})
    for path, value in settings.get("evaluation_overrides", {}).items():
        set_path(cfg, path, value)
    limits = configured_limits(cfg)
    for key in settings.get('hard_constraints', []):
        if key in limits:
            continue
        parts = key.split('.')
        field = {'Tmin': 'min_temperature', 'Pmin': 'min_pressure',
                 'wall_min': 'min_wall_thickness'}.get(parts[-1])
        if len(parts) != 3 or parts[0] != 'tank' or field is None or field not in cfg.get('tanks', {}).get(parts[1], {}):
            raise ValueError(f'Hard constraint {key!r} requires a limit in evaluation_overrides or the base config')
    tracking = settings.get('soft_penalties', {}).get('tank_pressure_tracking', {})
    for tank in tracking:
        if cfg.get('tanks', {}).get(tank, {}).get('type') != 'propellant':
            raise ValueError(f'Pressure tracking requires a propellant tank: {tank}')
    if 'max_q' in settings.get('soft_penalties', {}) and 'max_q' in limits:
        raise ValueError('max_q cannot be both a hard constraint and a soft preference in this policy')
    policy = settings.get('soft_penalties', {})
    if set(policy) - {'tank_pressure_tracking', 'max_q', 'stability_length_fraction', 'downrange'}:
        raise ValueError('Unknown soft penalty')
    if 'downrange' in policy:
        spec = policy['downrange']
        if any(not isfinite(spec[k]) or spec[k] < 0 for k in ('weight', 'target_m')) or not isfinite(spec['scale_m']) or spec['scale_m'] <= 0:
            raise ValueError('Invalid downrange penalty')
    for spec in tracking.values():
        for key in ('weight', 'pressure_scale_pa', 'time_scale_s'):
            value = spec[key]
            if not isfinite(value) or (value < 0 if key == 'weight' else value <= 0):
                raise ValueError(f'Invalid pressure penalty {key}')
    if 'max_q' in policy:
        spec = policy['max_q']
        if any(not isfinite(spec[k]) or spec[k] < 0 for k in ('weight', 'target_pa')) or not isfinite(spec['scale_pa']) or spec['scale_pa'] <= 0:
            raise ValueError('Invalid max-Q penalty')
    if 'stability_length_fraction' in policy:
        lower, upper, scale, weight = (policy['stability_length_fraction'][k]
                                      for k in ('lower_target', 'upper_target', 'scale', 'weight'))
        if (not all(isfinite(v) for v in (lower, upper, scale, weight))
                or not 0 <= lower < upper or scale <= 0 or weight < 0):
            raise ValueError('Invalid stability length-fraction penalty')
    if not {"goal_apogee", "max_burn_duration"} <= limits.keys():
        raise ValueError("Search requires goal_apogee and max_burn_duration")
    if cfg["environment"]["max_altitude"] <= limits["goal_apogee"]:
        raise ValueError("Atmosphere max_altitude must exceed goal_apogee")
    paths = variable_paths(cfg, ids)
    if set(settings["bounds"]) != set(paths):
        raise ValueError(f"Bounds must contain exactly: {list(paths)}")
    deck = aero_bounds(model)
    for name, bounds in settings["bounds"].items():
        if len(bounds) != 2:
            raise ValueError(f"{name} bounds require [lower, upper]")
        low, high = map(float, bounds)
        if (not all(map(isfinite, (low, high))) or low > high
                or (low < 0 if name == "counterweight_mass" else low <= 0)):
            raise ValueError(f"Invalid bounds for {name}")
        if name in deck and (low < deck[name][0] - 1e-12 or high > deck[name][1] + 1e-12):
            raise ValueError(f"{name} bounds exceed aero deck {deck[name]}")
        get_path(cfg, paths[name])
        settings["bounds"][name] = (low, high)
    if settings['bounds']['exit_pressure'][1] >= settings['bounds']['chamber_pressure'][0]:
        raise ValueError('Exit-pressure upper bound must be below the chamber-pressure lower bound')
    if not isinstance(settings.get("conditional_geometry", False), bool):
        raise ValueError("conditional_geometry must be boolean")
    for name in ("reference_mass",):
        if not isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (("max_evaluations", "maxiter", "popsize") if algorithm == "differential_evolution" else ("max_evaluations",)):
        if isinstance(settings[name], bool) or int(settings[name]) != settings[name] or settings[name] < (0 if name == "maxiter" else 1):
            raise ValueError(f"{name} must be an integer in range")
    for scale in (*settings["constraint_scales"].values(), *settings["unit_scales"].values()):
        if not isfinite(scale) or scale <= 0:
            raise ValueError("Constraint scales must be finite and positive")
    return cfg, settings


class LocalEvaluationError(Exception):
    """A candidate cannot supply the objective/constraint vector required by SLSQP."""


def run_de(evaluator, settings):
    initial = [float(get_path(evaluator.cfg, path)) for path in evaluator.paths.values()]
    x0 = initial if not settings.get("conditional_geometry") and all(lo <= x <= hi for x, (lo, hi) in zip(initial, evaluator.bounds)) else None
    with GenerationBudgetSolver(evaluator, evaluator.bounds, rng=settings["seed"],
                                popsize=settings["popsize"], maxiter=settings["maxiter"],
                                x0=x0, polish=False, workers=1) as solver:
        solver.solve()
    return dict(stop="generation budget exhausted")


class UnusableTrial(LocalEvaluationError):
    """Recoverable physical/numerical rejection, with the normalized trial point."""
    def __init__(self, message, point):
        super().__init__(message)
        self.point = np.asarray(point, dtype=float)


def run_slsqp(evaluator, settings):
    """Refine a valid design, shrinking probes and restarting on unusable steps."""
    space = evaluator.coordinates
    try:
        z0 = space.normalize(space.encode(evaluator.cfg))
    except (ValueError, GeometryError) as error:
        raise LocalEvaluationError(f"Invalid SLSQP initial design: {error}") from error
    names = None
    options = settings['slsqp']

    @lru_cache(maxsize=max(32, 2 * (len(z0) + 1)))
    def record(z):
        # Cache failures too: revisiting an identical failed point needs no simulation.
        return evaluator.evaluate_candidate(space.expand(z))[2]

    def evaluate(z):
        entry = record(tuple(z))
        if entry['slsqp_error']:
            details = entry.get('failure_details') or {}
            recoverable = (entry['score_class'] == 'preflight_rejected' or
                           (details.get('fatal') is False and details.get('kind') in {
                               'trial_domain_failure', 'solver_nonconvergence',
                               'residual_acceptance_failure'}))
            if recoverable:
                raise UnusableTrial(entry['slsqp_error'], z)
            raise LocalEvaluationError(entry['slsqp_error'])
        if names is not None and entry['constraint_names'] != names:
            raise LocalEvaluationError("Required constraint set changed during SLSQP")
        return np.array([entry['objective_value'], *entry['normalized_constraint_margins']])

    evaluate(z0)  # Initial failures are never recovered by searching from an unknown state.
    initial = record(tuple(z0))
    if not initial['accepted']:
        raise LocalEvaluationError("SLSQP requires an accepted initial design")
    names = initial['constraint_names']
    if not z0:
        return dict(stop="all variables fixed", solver_success=True, iterations=0, restarts=0)

    anchor = np.asarray(z0)
    lower, upper = np.zeros(len(z0)), np.ones(len(z0))
    radius, iterations, restarts = 1., 0, 0

    def log_recovery(kind, error, **fields):
        event = dict(kind=kind, evaluation=evaluator.count, reason=str(error),
                     trial=error.point.tolist(), **fields)
        with (evaluator.output / 'recovery.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        evaluator.progress.event(kind, fields)

    while True:
        # Share numerical derivatives between the objective and all constraints.
        @lru_cache(maxsize=1)
        def derivatives(point):
            z = np.asarray(point)
            baseline = evaluate(z)
            columns = []
            for i in range(len(z)):
                column = None
                for retry in range(options['probe_retries'] + 1):
                    step = options['gradient_step'] * .5 ** retry
                    # Prefer a full forward step, then try its opposite.
                    direction = 1 if upper[i] - z[i] >= step else -1
                    for sign in (direction, -direction):
                        trial = z.copy()
                        trial[i] = np.clip(z[i] + sign * step, lower[i], upper[i])
                        delta = trial[i] - z[i]
                        if abs(delta) < options['min_step'] * (1 - 1e-10):
                            continue
                        try:
                            column = (evaluate(trial) - baseline) / delta
                        except UnusableTrial as error:
                            log_recovery('probe_retry', error, variable=space.names[space.free[i]],
                                         step=float(delta), retry=retry)
                            continue
                        break
                    if column is not None:
                        break
                if column is None:
                    raise LocalEvaluationError(f"No usable derivative probe for {space.names[space.free[i]]}")
                columns.append(column)
            return np.column_stack(columns)

        def iteration_finished(z):
            nonlocal anchor, iterations
            evaluate(z)
            anchor = np.asarray(z).copy()
            iterations += 1

        try:
            result = minimize(
                lambda z: evaluate(z)[0], anchor, method='SLSQP',
                jac=lambda z: derivatives(tuple(z))[0],
                bounds=list(zip(lower, upper)),
                constraints={'type': 'ineq', 'fun': lambda z: evaluate(z)[1:],
                             'jac': lambda z: derivatives(tuple(z))[1:]},
                callback=iteration_finished,
                options=dict(maxiter=max(1, options['maxiter'] - iterations), ftol=options['ftol']))
            return dict(stop=str(result.message), solver_success=bool(result.success),
                        solver_status=int(result.status), iterations=iterations, restarts=restarts,
                        recovery_radius=radius if restarts else None)
        except UnusableTrial as error:
            if restarts >= options['max_restarts'] or iterations >= options['maxiter']:
                raise LocalEvaluationError(f"SLSQP recovery limit reached: {error}") from error
            radius = .5 * min(radius, float(np.max(np.abs(error.point - anchor))))
            if radius < options['min_step']:
                raise LocalEvaluationError(f"SLSQP recovery radius below min_step: {error}") from error
            restarts += 1
            lower, upper = np.maximum(0., anchor - radius), np.minimum(1., anchor + radius)
            log_recovery('restart', error, restart=restarts, radius=radius, anchor=anchor.tolist())


def optimize(settings, output, *, workers=None, timeout=None, candidates_per_worker=None,
             resume=False, profile=False):
    if settings.get('algorithm', 'differential_evolution') not in ('differential_evolution', 'slsqp'):
        raise ValueError(f"Unknown optimizer algorithm: {settings['algorithm']}")
    workers = settings.get('workers', 1) if workers is None else workers
    timeout = settings.get('worker_timeout', 300.) if timeout is None else timeout
    candidates_per_worker = (settings.get('candidates_per_worker', 3)
                             if candidates_per_worker is None else candidates_per_worker)
    for name, value in (('workers', workers), ('candidates_per_worker', candidates_per_worker)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError('worker timeout must be finite and positive')
    if settings.get('algorithm') == 'slsqp' and (workers > 1 or resume or profile):
        raise ValueError("SLSQP supports serial execution only; workers, resume, and profile are unavailable")
    if workers > 1 or resume or profile:
        from optimizer.workers import optimize_parallel
        return optimize_parallel(settings, output, workers=workers, timeout=timeout,
                                 candidates_per_worker=candidates_per_worker,
                                 resume=resume, profile=profile)
    base = load_config(project_path(settings["base_config"]))
    model = DragModel(project_path(base["aero"]["model"]))
    cfg, settings = prepare(settings, model)
    pure, combustion = property_sources(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "search.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    (output / "base.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    evaluator = Evaluator(cfg, settings, model, pure, combustion, output)
    try:
        outcome = (run_slsqp(evaluator, settings) if settings.get('algorithm') == 'slsqp'
                   else run_de(evaluator, settings))
    except EvaluationBudget:
        outcome = dict(stop="evaluation budget exhausted")
        if settings.get("algorithm") == "slsqp":
            outcome["solver_success"] = False
    except LocalEvaluationError as error:
        outcome = dict(stop=str(error), solver_success=False)
    summary = dict(evaluations=evaluator.count, **outcome,
                   accepted=evaluator.best is not None,
                   search_accepted=evaluator.best is not None,
                   best_mass=None if evaluator.best is None else evaluator.best["mass"])
    if settings.get("algorithm") == "slsqp":
        summary["algorithm"] = "slsqp"
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="Configs/optimizer_pressure_fed.yaml")
    parser.add_argument("--output", required=True, help="New directory for this search")
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--algorithm", choices=("differential_evolution", "slsqp"))
    parser.add_argument('--workers', type=int, help='Concurrent candidate processes (default: config workers or 1)')
    parser.add_argument('--timeout', type=float, help='Wall-time limit in seconds per worker candidate (default: 300)')
    parser.add_argument('--candidates-per-worker', type=int, help='Recycle after this many candidates (default: 3)')
    parser.add_argument('--resume', action='store_true', help='Replay saved results and continue a process-worker search')
    parser.add_argument('--profile', action='store_true', help='Collect per-candidate timings in worker processes')
    args = parser.parse_args()
    settings = load_config(args.config)
    if args.algorithm is not None:
        settings["algorithm"] = args.algorithm
    if args.max_evaluations is not None:
        settings["max_evaluations"] = args.max_evaluations
    summary = optimize(settings, args.output, workers=args.workers, timeout=args.timeout,
                       candidates_per_worker=args.candidates_per_worker,
                       resume=args.resume, profile=args.profile)
    print(json.dumps(summary, indent=2))
    if not summary["accepted"]:
        print("No accepted design found.")
