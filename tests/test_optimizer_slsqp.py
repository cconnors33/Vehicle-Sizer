"""Local search contracts using real coordinates and a deterministic flight surrogate."""
from copy import deepcopy
import json

import numpy as np
import pytest

import optimizer.core as opt
from optimizer.objectives import objective, constraint_margins
from diagnostics.constraints import ConstraintRecord, finalize
from simulation_types import SimResult


def flight(x=0.5):
    return finalize(SimResult(initial_mass=100 + (x - .2)**2 * 100,
                              apogee=150000, burn_complete=True, burn_duration=40,
                              constraint_records={'design': ConstraintRecord(x - .4, 'test', 'sizing', '1')}),
                    {'goal_apogee': 130000, 'max_burn_duration': 75})


@pytest.fixture
def local_setup(tmp_path):
    settings = opt.load_config('Configs/optimizer_epump.yaml')
    settings.update(algorithm='slsqp', conditional_geometry=False, max_evaluations=100)
    settings.pop('soft_penalties', None)
    model = opt.DragModel()
    cfg, settings = opt.prepare(settings, model)
    # A representable initial config with only sweep free.
    for name, path in opt.variable_paths(cfg, settings['tank_ids']).items():
        lo, hi = settings['bounds'][name]
        value = (lo + hi) / 2
        opt.set_path(cfg, path, value)
        settings['bounds'][name] = (value, value)
    settings['bounds']['sweep_fraction'] = (0., 1.)
    cfg['fin_can']['sweep_fraction'] = .5
    return cfg, settings, model


def test_shared_objective_and_ordered_constraints(local_setup):
    _, settings, _ = local_setup
    good, bad = flight(.5), flight(.3)
    assert good.accepted and not bad.accepted
    assert objective(bad, settings) < objective(good, settings)
    value = objective(good, settings)
    assert opt.candidate_score(good, settings)[0] == pytest.approx(value / (1 + value))
    names = ['max_burn_duration', 'design', 'goal_apogee']
    assert constraint_margins(bad, settings, names)[1] == pytest.approx(-.1)
    with pytest.raises(ValueError, match='set changed'):
        constraint_margins(good, settings, names[:-1])


def test_slsqp_constrained_solution_and_cache(local_setup, tmp_path, monkeypatch):
    cfg, settings, model = local_setup
    calls = []
    def simulate(cfg, **kwargs):
        x = cfg['fin_can']['sweep_fraction']
        calls.append(x)
        return flight(x)
    monkeypatch.setattr(opt, 'simulate', simulate)
    evaluator = opt.Evaluator(cfg, settings, model, None, None, tmp_path)
    outcome = opt.run_slsqp(evaluator, settings)
    assert outcome['solver_success']
    rows = [json.loads(line) for line in (tmp_path/'evaluations.jsonl').read_text().splitlines()]
    assert len(calls) == evaluator.count == len(rows)
    assert calls.count(.5) == 1  # Initial, objective, and constraints share one simulation.
    assert len(calls) == len(set(calls))
    assert evaluator.best['variables']['sweep_fraction'] == pytest.approx(.4, abs=.002)
    assert all(row['objective_value'] is not None for row in rows)


def test_budget_and_fixed_design(local_setup, tmp_path, monkeypatch):
    cfg, settings, model = local_setup
    monkeypatch.setattr(opt, 'simulate', lambda *args, **kwargs: flight())
    settings['max_evaluations'] = 1
    evaluator = opt.Evaluator(cfg, settings, model, None, None, tmp_path)
    with pytest.raises(opt.EvaluationBudget):
        opt.run_slsqp(evaluator, settings)
    assert evaluator.count == 1
    settings['bounds']['sweep_fraction'] = (.5, .5)
    evaluator = opt.Evaluator(cfg, settings, model, None, None, tmp_path)
    assert opt.run_slsqp(evaluator, settings)['stop'] == 'all variables fixed'
    assert evaluator.count == 1


@pytest.mark.parametrize('failure', ['incomplete', 'changed', 'missing'])
def test_unusable_trial_stops_after_logging(local_setup, tmp_path, monkeypatch, failure):
    cfg, settings, model = local_setup
    results = [flight(), flight()]
    if failure == 'incomplete':
        results[1] = SimResult()
    elif failure == 'changed':
        results[1].constraint_records['new'] = ConstraintRecord(1., 'test', 'sizing', '1')
    else:
        results[1].constraint_records['design'] = ConstraintRecord(None, 'test', 'sizing', '1')
    monkeypatch.setattr(opt, 'simulate', lambda *args, **kwargs: results.pop(0))
    evaluator = opt.Evaluator(cfg, settings, model, None, None, tmp_path)
    with pytest.raises(opt.LocalEvaluationError):
        opt.run_slsqp(evaluator, settings)
    assert evaluator.count == 2
    assert len((tmp_path/'evaluations.jsonl').read_text().splitlines()) == 2
    assert evaluator.best is not None


def test_normalization_and_initial_bounds(local_setup, tmp_path):
    cfg, settings, model = local_setup
    space = opt.CoordinateSpace(cfg, settings, model, None, None)
    x = space.encode(cfg)
    assert space.normalize(x) == [.5]
    assert space.expand([.5]) == x
    cfg['fin_can']['sweep_fraction'] = 2
    with pytest.raises(ValueError, match='sweep_fraction'):
        space.encode(cfg)


def test_conditional_round_trip(tmp_path):
    settings = opt.load_config('Configs/optimizer_epump.yaml')
    settings['conditional_geometry'] = True
    model = opt.DragModel()
    cfg, settings = opt.prepare(settings, model)
    pure, combustion = opt.property_sources(cfg)
    space = opt.CoordinateSpace(cfg, settings, model, pure, combustion)
    before = deepcopy(cfg)
    for z in [np.zeros(len(space.free)), np.ones(len(space.free)),
              *np.random.default_rng(10).random((5, len(space.free)))]:
        design = space.decode(space.expand(z))
        recovered = space.decode(space.encode(design))
        for path in space.paths.values():
            assert opt.get_path(recovered, path) == pytest.approx(opt.get_path(design, path))
    assert cfg == before


@pytest.mark.parametrize('controls', [{'workers': 2}, {'resume': True}, {'profile': True}])
def test_worker_options_rejected_early(tmp_path, controls):
    with pytest.raises(ValueError, match='serial'):
        opt.optimize({'algorithm': 'slsqp'}, tmp_path/'run', **controls)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('budget', [1, 30])
def test_optimize_dispatch_returns_search_incumbent(local_setup, tmp_path, monkeypatch, budget):
    cfg, settings, model = local_setup
    settings['max_evaluations'] = budget
    calls = []
    def simulate(candidate, **kwargs):
        calls.append((candidate['simulation']['dt'], kwargs['compute_loads']))
        return flight(candidate['fin_can']['sweep_fraction'])
    monkeypatch.setattr(opt, 'prepare', lambda *args: (deepcopy(cfg), deepcopy(settings)))
    monkeypatch.setattr(opt, 'property_sources', lambda cfg: (None, None))
    monkeypatch.setattr(opt, 'simulate', simulate)
    summary = opt.optimize(settings, tmp_path/'run')
    assert summary['accepted']
    assert 'verified' not in summary
    assert len(calls) == summary['evaluations']
    assert all(call == (cfg['simulation']['dt'], False) for call in calls)
    if budget == 1:
        assert summary['stop'] == 'evaluation budget exhausted'
        assert not summary['solver_success']
    assert (tmp_path/'run'/'best.yaml').exists()
    assert not (tmp_path/'run'/'verified.yaml').exists()


def test_unaccepted_start_stops(local_setup, tmp_path, monkeypatch):
    cfg, settings, model = local_setup
    monkeypatch.setattr(opt, 'simulate', lambda *args, **kwargs: flight(.3))
    evaluator = opt.Evaluator(cfg, settings, model, None, None, tmp_path)
    with pytest.raises(opt.LocalEvaluationError, match='accepted initial'):
        opt.run_slsqp(evaluator, settings)
    assert evaluator.count == 1 and evaluator.best is None


def test_slsqp_does_not_require_de_options():
    settings = opt.load_config('Configs/optimizer_epump.yaml')
    settings['algorithm'] = 'slsqp'
    for name in ('maxiter', 'popsize', 'seed'):
        settings.pop(name)
    _, prepared = opt.prepare(settings, opt.DragModel())
    assert prepared['slsqp']['maxiter'] == 100
    settings['slsqp'] = {'gradient_step': 0}
    with pytest.raises(ValueError, match='gradient_step'):
        opt.prepare(settings, opt.DragModel())


def recoverable_evaluator(local_setup, tmp_path, monkeypatch, fails):
    """Emulate a nonfatal aero-domain failure while retaining real bookkeeping."""
    cfg, settings, model = local_setup
    def simulate(candidate, **kwargs):
        x = candidate['fin_can']['sweep_fraction']
        return SimResult(termination='trial_domain_failure') if fails(x) else flight(x)
    monkeypatch.setattr(opt, 'simulate', simulate)
    evaluator = opt.Evaluator(cfg, settings, model, None, None, tmp_path)
    original = evaluator.evaluate
    def evaluate(*args, **kwargs):
        cfg, result, row = original(*args, **kwargs)
        if result.termination == 'trial_domain_failure':
            row['failure_details'] = dict(kind='trial_domain_failure', fatal=False)
        return cfg, result, row
    monkeypatch.setattr(evaluator, 'evaluate', evaluate)
    return evaluator, settings


def test_failed_design_step_restarts_in_smaller_neighborhood(local_setup, tmp_path, monkeypatch):
    evaluator, settings = recoverable_evaluator(local_setup, tmp_path, monkeypatch, lambda x: x < .44)
    report = opt.run_slsqp(evaluator, settings)
    assert report['solver_success'] and report['restarts'] == 1
    assert evaluator.best['variables']['sweep_fraction'] == pytest.approx(.45, abs=.002)
    events = [json.loads(s) for s in (tmp_path/'recovery.jsonl').read_text().splitlines()]
    assert events[0]['kind'] == 'restart'
    assert events[0]['radius'] == pytest.approx(.05, abs=.002)
    rows = [json.loads(s) for s in (tmp_path/'evaluations.jsonl').read_text().splitlines()]
    failed = next(i for i, row in enumerate(rows) if row['slsqp_error'])
    assert rows[failed]['objective_value'] is None
    assert any(row['accepted'] for row in rows[failed+1:])


def test_failed_probe_uses_other_direction(local_setup, tmp_path, monkeypatch):
    evaluator, settings = recoverable_evaluator(local_setup, tmp_path, monkeypatch, lambda x: x > .5)
    report = opt.run_slsqp(evaluator, settings)
    assert report['solver_success'] and report['restarts'] == 0
    assert evaluator.best['variables']['sweep_fraction'] == pytest.approx(.4, abs=.002)
    events = [json.loads(s) for s in (tmp_path/'recovery.jsonl').read_text().splitlines()]
    assert events[0]['kind'] == 'probe_retry'


def test_failed_probes_shrink_and_respect_budget(local_setup, tmp_path, monkeypatch):
    evaluator, settings = recoverable_evaluator(local_setup, tmp_path, monkeypatch,
                                               lambda x: abs(x - .5) > .0006)
    settings['max_evaluations'] = 4
    with pytest.raises(opt.EvaluationBudget):
        opt.run_slsqp(evaluator, settings)
    rows = [json.loads(s) for s in (tmp_path/'evaluations.jsonl').read_text().splitlines()]
    assert evaluator.count == 4
    assert rows[-1]['accepted']
    assert rows[-1]['variables']['sweep_fraction'] == pytest.approx(.5005)


def test_restart_limit_is_bounded(local_setup, tmp_path, monkeypatch):
    evaluator, settings = recoverable_evaluator(local_setup, tmp_path, monkeypatch, lambda x: x < .49)
    settings['slsqp']['max_restarts'] = 0
    with pytest.raises(opt.LocalEvaluationError, match='recovery limit'):
        opt.run_slsqp(evaluator, settings)
    assert evaluator.best is not None


def test_initial_domain_failure_is_not_restarted(local_setup, tmp_path, monkeypatch):
    evaluator, settings = recoverable_evaluator(local_setup, tmp_path, monkeypatch, lambda x: True)
    with pytest.raises(opt.UnusableTrial):
        opt.run_slsqp(evaluator, settings)
    assert evaluator.count == 1
    assert not (tmp_path/'recovery.jsonl').exists()


def test_progress_matches_serial_and_worker_records(capsys, monkeypatch):
    from optimizer.core import ProgressTracker
    monkeypatch.setattr('optimizer.core.time.perf_counter', lambda: 100.)
    common = dict(score=.5, accepted=True, apogee=150000., mass=500.,
                  simulation_seconds=180., termination='apogee')
    serial = ProgressTracker(10)
    serial.update(dict(common, evaluation=1, elapsed_seconds=2.))
    serial_output = capsys.readouterr().out
    worker = ProgressTracker(10)
    worker.update(dict(common, index=1, wall_seconds=2.))
    assert capsys.readouterr().out == serial_output


def test_progress_counts_completions_and_keeps_accepted_best(capsys):
    from optimizer.core import ProgressTracker
    progress = ProgressTracker(10)
    progress.update(dict(index=7, score=.5, accepted=True, termination='apogee'))
    progress.update(dict(index=2, score=.1, accepted=False, termination='wall_timeout'))
    progress.event('restart', {'radius': .1})
    lines = capsys.readouterr().out.splitlines()
    assert 'Run 1/10 | candidate=7' in lines[0]
    assert 'Run 2/10 | candidate=2' in lines[1]
    assert 'best=0.5' in lines[1]
    assert 'sim=N/A' in lines[1]
    assert 'after 2/10 evaluations' in lines[2]
    assert progress.completed == 2
