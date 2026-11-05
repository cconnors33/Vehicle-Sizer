"""Physical configurations, conditional search coordinates, and normalized vectors."""
from copy import deepcopy
from math import isfinite, isclose, pi, sqrt

from AeroTables.dragmodel import _end as aero_endpoint
from diagnostics.errors import GeometryError
from Fluids.helpers.templates import bind_template, read_template
from Fluids.design import prepare_design_inputs
from Vehicle.sections.PressTank import PressTank
from Vehicle.utils.geometry import pressure_wall_thickness, tank_capacity

INCH = 0.0254
AERO_PATHS = {
    "omld": "vehicle.OMLD", "fineness": "nosecone.fineness_ratio",
    "boattail_aft": "fin_can.boattail_aft_diameter",
    "boattail_length": "fin_can.boattail_length", "span": "fin_can.span",
    "root": "fin_can.root_chord", "tip": "fin_can.tip_chord",
    "sweep_fraction": "fin_can.sweep_fraction", "thickness": "fin_can.fin_thickness",
}
DIMENSIONLESS = {"fineness", "sweep_fraction"}


def pump_roles(cfg):
    """Find optimizer pump variables from the same wiring used by PropSystem."""
    template = read_template(cfg)
    if not any(branch["component"] == "pump" for branch in template["branches"].values()):
        return {}  # Pressure-fed CdA inputs are resolved later during design preparation.
    template = bind_template(dict(cfg, prop_system=dict(cfg["prop_system"], template=template)))
    roles = {}
    for branch in template['branches'].values():
        if branch['component'] != 'pump':
            continue
        role = template['circuits'][branch['circuit']]['prop']
        if role not in ('oxidizer', 'fuel') or role in roles:
            raise ValueError('Optimizer supports at most one pump per propellant role')
        roles[role] = branch['pump_id']
    return roles


def variable_paths(cfg, tank_ids):
    paths = dict(AERO_PATHS)
    paths.update(chamber_pressure="prop_system.Pc_target", mixture_ratio="prop_system.MR_target",
                 thrust="prop_system.thrust_target", exit_pressure="engine.exit_pressure")
    pumps = pump_roles(cfg)
    for role in ("oxidizer", "fuel"):
        if role in pumps:
            pump = pumps[role]
            paths[f"{role}_pump_head"] = f"prop_system.pumps.{pump}.pressure_rise_pa"
        paths[f"{role}_mass"] = f"tanks.{tank_ids[role]}.propellant_mass"
    paths.update(copv_pressure=f"tanks.{tank_ids['pressurant']}.design_pressure",
                 copv_volume=f"tanks.{tank_ids['pressurant']}.volume")
    for i, section in enumerate(cfg["vehicle"]["sections"]):
        if "counterweight" in section.get("masses", {}):
            if "counterweight_mass" in paths:
                raise ValueError("Specify only one optimizer counterweight")
            paths["counterweight_mass"] = f"vehicle.sections.{i}.masses.counterweight"
    return paths


def get_path(cfg, path):
    for key in path.split("."):
        cfg = cfg[int(key) if isinstance(cfg, list) else key]
    return cfg


def set_path(cfg, path, value):
    *parents, key = path.split(".")
    for parent in parents:
        cfg = cfg[int(parent) if isinstance(cfg, list) else parent]
    cfg[int(key) if isinstance(cfg, list) else key] = value


def pressurant_inner_radius(cfg, tank):
    """Match the selected vessel's wall model before checking its capacity."""
    diameter = float(tank["outer_diameter"])
    wall, _ = PressTank.wall_sizing(cfg, tank)
    radius = diameter / 2 - wall
    if not isfinite(radius) or radius <= 0 or tank["ellipse_ratio"] <= 1:
        raise ValueError("Invalid pressurant diameter, sized wall or ellipse ratio")
    return radius


class CoordinateSpace:
    def __init__(self, cfg, settings, model, pure, combustion):
        self.cfg, self.settings = deepcopy(cfg), settings
        self.model, self.pure, self.combustion = model, pure, combustion
        self.paths = variable_paths(cfg, settings['tank_ids'])
        self.names = list(self.paths)
        self.bounds = [settings['bounds'][name] for name in self.names]
        self.free = [i for i, (lo, hi) in enumerate(self.bounds) if lo < hi]

    def normalize(self, values):
        if len(values) != len(self.bounds):
            raise ValueError("Wrong design vector length")
        for name, value, (lo, hi) in zip(self.names, values, self.bounds):
            if not isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Candidate {name} is outside configured bounds")
        return [(values[i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0])
                for i in self.free]

    def expand(self, normalized):
        if len(normalized) != len(self.free):
            raise ValueError("Wrong normalized vector length")
        values = [lo for lo, hi in self.bounds]
        for i, fraction in zip(self.free, normalized):
            if not isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Normalized coordinates must lie in [0, 1]")
            lo, hi = self.bounds[i]
            values[i] = lo + float(fraction) * (hi - lo)
        return values

    def encode(self, cfg):
        """Represent the requested physical design exactly; never clip an initial design."""
        values = [float(get_path(cfg, path)) for path in self.paths.values()]
        self.normalize(values)  # Physical values must also respect user bounds.
        if self.settings.get('conditional_geometry', False):
            values = self.condition_geometry(deepcopy(cfg), values, encode=True)
        decoded = self.decode(values)
        for name, path in self.paths.items():
            if not isclose(get_path(decoded, path), get_path(cfg, path), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Initial {name} cannot be represented by conditional coordinates")
        return values

    def decode(self, values):
        if len(values) != len(self.paths):
            raise ValueError("Wrong design vector length")
        cfg = deepcopy(self.cfg)
        for value, (name, path), (low, high) in zip(values, self.paths.items(), self.bounds):
            if not isfinite(value) or not low <= value <= high:
                raise ValueError(f"Candidate {name} is outside configured bounds")
            set_path(cfg, path, float(value))
        if self.settings.get("conditional_geometry", False):
            return self.condition_geometry(cfg, values)
        return cfg

    def condition_geometry(self, cfg, values, *, encode=False):
        """Map each box coordinate into its candidate-dependent physical interval.

        User bounds remain hard limits. Empty intersections are explicit design
        rejections, never clipping a derived nozzle or overriding a user bound.
        """
        coordinates = dict(zip(self.names, values))
        def assign(name, minimum=-float("inf"), maximum=float("inf")):
            lower, upper = self.settings["bounds"][name]
            lo, hi = max(lower, minimum), min(upper, maximum)
            if lo > hi:
                raise GeometryError({f"geometry.bounds.{name}": hi - lo})
            if encode:
                value = get_path(cfg, self.paths[name])
                if not lo <= value <= hi:
                    raise ValueError(f"Initial {name}={value} is outside conditional interval [{lo}, {hi}]")
                fraction = (value - lo) / (hi - lo) if hi > lo else 0.
                coordinates[name] = lower + fraction * (upper - lower)
            else:
                fraction = (coordinates[name] - lower) / (upper - lower) if upper > lower else 0.
                value = float(lo + fraction * (hi - lo))
            set_path(cfg, self.paths[name], value)
            return value

        clearance = float(cfg.get("geometry", {}).get("radial_clearance", 0.))
        fin, engine, prop = cfg["fin_can"], cfg["engine"], cfg["prop_system"]
        copv = cfg["tanks"][self.settings["tank_ids"]["pressurant"]]
        margin = 2 * (fin["boattail_wall_thickness"] + clearance) + 1e-9
        # The nozzle stays derived from the same CEA design calculation.
        epsilon = self.combustion.expansion_ratio(prop["Pc_target"], prop["MR_target"], engine["exit_pressure"])
        design = self.combustion.evaluate(chamber_pressure=prop["Pc_target"], mixture_ratio=prop["MR_target"],
            ambient_pressure=engine["exit_pressure"], expansion_ratio=epsilon,
            cstar_efficiency=engine["cstar_efficiency"], cf_efficiency=engine["cf_efficiency"])
        thrust_per_area = prop["Pc_target"] * design.Cf / epsilon
        exit_low = float(self.model.ranges["exit"][0]) * INCH
        aft_max = min(self.settings["bounds"]["boattail_aft"][1], self.settings["bounds"]["omld"][1])
        if aft_max <= margin:
            raise GeometryError({"engine.nozzle": aft_max - margin})
        thrust = assign("thrust", pi * exit_low**2 / 4 * thrust_per_area,
                        pi * (aft_max - margin)**2 / 4 * thrust_per_area)
        exit_diameter = sqrt(4 * thrust / thrust_per_area / pi)
        diameter = assign("omld", max(exit_diameter + margin,
            self.settings["bounds"]["boattail_aft"][0],
            copv["outer_diameter"] + 2 * (clearance) + 1e-9))
        # Span remains a physical length. Its permitted interval moves with D;
        # intersect it with the user's bounds before mapping the search coordinate.
        span_low, span_high = (aero_endpoint(bound, {"omld": diameter / INCH}) * INCH
                               for bound in self.model.ranges["span"])
        assign("span", span_low, span_high)
        length = assign("boattail_length", max(engine["length"], self.settings["bounds"]["root"][0]))
        aft_min = exit_diameter + margin
        if engine.get("envelope_diameter") is not None:
            envelope = engine["envelope_diameter"] + margin
            if diameter < envelope:
                raise GeometryError({"engine.envelope": diameter - envelope})
            aft_min = max(aft_min, diameter + (envelope - diameter) * length / engine["length"])
        assign("boattail_aft", aft_min, diameter)
        root = assign("root", maximum=length)
        assign("tip", maximum=root)
        radius = pressurant_inner_radius(cfg, copv)
        if copv.get("length") is None:
            assign("copv_volume", 4 * pi * radius**3 / (3 * copv["ellipse_ratio"]) + pi * radius**2 * 1e-8)

        if pump_roles(cfg):
            for role, leg in (("oxidizer", "ox"), ("fuel", "fuel")):
                outlet = prop["Pc_target"] * (1 + prop[f"{leg}_inj_stiffness"]) + prop[f"{leg}_inj_pumpout_dp"]
                assign(f"{role}_pump_head", maximum=outlet - 1.)
        # Match PropTank's pressure-sized wall and usable ellipsoidal-head volume.
        _, states, wall_pressures = prepare_design_inputs(cfg, fluid_properties=self.pure, combustion_properties=self.combustion)
        for role in ("oxidizer", "fuel"):
            tank_id = self.settings["tank_ids"][role]
            tank, state = cfg["tanks"][tank_id], states[tank_id]
            pressure = wall_pressures[tank_id]
            allowance = float(cfg["advanced"].get("tank_wall_thickness_allowance", 0.))
            if not isfinite(allowance) or allowance < 0:
                raise ValueError("Tank wall thickness allowance must be finite and nonnegative")
            wall = tank.get("wall_thickness")
            if wall is None:
                wall = pressure_wall_thickness(pressure, diameter, cfg["advanced"]["tank_pressure_fos"],
                                               cfg["advanced"]["weld_allowable"]) + allowance
            radius, tube = diameter / 2 - wall, tank["passthrough_diameter"] / 2
            if radius <= tube + clearance:
                raise GeometryError({f"{tank_id}.passthrough": radius - tube - clearance})
            head, cylinder_area = tank_capacity(radius, tube, tank["ellipse_ratio"])
            rho = self.pure.state_pt(state["fluid"], state["P"], state["T"]).rho
            assign(f"{role}_mass", (head + cylinder_area * 1e-8) * rho / tank["ullage_factor"])
        return [coordinates[name] for name in self.names] if encode else cfg

