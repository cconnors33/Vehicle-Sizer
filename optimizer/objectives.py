"""Shared engineering objective and solver-specific constraint scoring."""
from math import isfinite, nextafter


def candidate_class(result):
    """Missing/unresolved physics must never outrank a fully assessed flight."""
    if result.accepted:
        return "feasible"
    if result.termination == "infeasible_initial_design":
        return "preflight_rejected"
    if (not result.completed or not result.burn_complete
            or not {"goal_apogee", "max_burn_duration"} <= result.constraint_records.keys()
            or any(r.required and r.margin is None for r in result.constraint_records.values())):
        return "unresolved"
    return "completed_infeasible"


def soft_penalties(result, settings):
    """Soft preferences never change feasibility; missing measurements cannot pass."""
    policy = settings.get('soft_penalties', {})
    penalties = {}
    if 'downrange' in policy:
        spec = policy['downrange']
        if not result.apogee_reached or not isfinite(result.final_x):
            raise ValueError('Downrange penalty requires a completed flight and finite displacement')
        penalties['downrange'] = spec['weight'] * max(0., abs(result.final_x) - spec['target_m']) / spec['scale_m']
    for tank, spec in policy.get('tank_pressure_tracking', {}).items():
        record = result.pressure_tracking.get(tank)
        if record is None or not isfinite(record['duration_s']) or record['duration_s'] <= 0:
            raise ValueError(f'Missing powered pressure tracking for {tank}')
        integral = record['integral_pa_s']
        if not isfinite(integral) or integral < 0:
            raise ValueError('Invalid pressure error integral')
        penalties[f'tank.{tank}.pressure_tracking'] = spec['weight'] * integral / (spec['pressure_scale_pa'] * spec['time_scale_s'])
    if 'max_q' in policy:
        spec = policy['max_q']
        if not isfinite(result.max_q) or result.max_q < 0:
            raise ValueError('Invalid maximum dynamic pressure')
        penalties['max_q'] = spec['weight'] * max(0., result.max_q - spec['target_pa']) / spec['scale_pa']
    if 'stability_length_fraction' in policy:
        spec = policy['stability_length_fraction']
        lower, upper, scale, weight = (spec[k] for k in ('lower_target', 'upper_target', 'scale', 'weight'))
        minimum, maximum = result.min_stability_length_fraction, result.max_stability_length_fraction
        if (minimum is None or maximum is None or not all(isfinite(v) for v in (minimum, maximum))
                or minimum > maximum):
            raise ValueError('Missing or invalid stability length-fraction extrema')
        penalties['stability_length_fraction'] = weight * (max(0., lower - minimum) + max(0., maximum - upper)) / scale
    return penalties


def ranked_score(result, settings):
    """Return score, violations and penalties using prepared search settings."""
    category = candidate_class(result)
    if category == "feasible":
        penalties = soft_penalties(result, settings)
        value = objective(result, settings, penalties=penalties)
        return min(value / (1. + value), nextafter(1., 0.)), {}, penalties
    contributions = {}
    for key, record in result.constraint_records.items():
        if not record.required or record.margin is None:
            continue
        scale = settings["constraint_scales"].get(key, settings["unit_scales"].get(record.units))
        if scale is None or not isfinite(scale) or scale <= 0:
            raise ValueError(f"Missing positive violation scale for {key} ({record.units})")
        if record.margin < 0:
            contributions[key] = -record.margin / scale
    if category == "unresolved":
        return 3., contributions, {}
    offset = 2. if category == "preflight_rejected" else 1.
    violation = sum(contributions.values())
    score = offset + (1. - 1. / (1. + violation))
    return min(score, nextafter(offset + 1., offset)), contributions, {}


def objective(result, settings, *, penalties=None):
    """Normalized launch mass plus soft preferences, including infeasible flights."""
    mass = result.initial_mass
    if mass is None or not isfinite(mass) or mass <= 0:
        raise ValueError("Candidate has no finite positive launch mass")
    if penalties is None:
        penalties = soft_penalties(result, settings)
    value = mass / settings['reference_mass'] + sum(penalties.values())
    if not isfinite(value):
        raise ValueError("Objective must be finite")
    return value


def constraint_margins(result, settings, names):
    """Fixed-order required margins, normalized so nonnegative is feasible."""
    required = {key for key, record in result.constraint_records.items() if record.required}
    if required != set(names):
        raise ValueError("Required constraint set changed during SLSQP")
    margins = []
    for name in names:
        record = result.constraint_records[name]
        if record.margin is None or not isfinite(record.margin):
            raise ValueError(f"Missing finite constraint margin: {name}")
        scale = settings['constraint_scales'].get(name, settings['unit_scales'].get(record.units))
        if scale is None or not isfinite(scale) or scale <= 0:
            raise ValueError(f"Missing positive violation scale for {name} ({record.units})")
        margins.append(record.margin / scale)
    return margins
