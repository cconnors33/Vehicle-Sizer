"""Equivalence checks for the benchmark-only explicit-volume tank."""
import numpy as np
import pytest

from examples.tank_volume_experiment import ExplicitVolumeTank
from test_fluid_nodes import wet_tank


@pytest.mark.parametrize('rates', [(-.2, .01, -1e5, 2e3), (.1, -.02, 3e4, -1e3)])
def test_explicit_volume_balances_and_regulator_pressure_rate_match_A(rates):
    a, values = wet_tank()
    c = ExplicitVolumeTank(a.id, a.definition, fluid_properties=a.fluid_properties)
    sa = a.evaluate(values, axial_specific_force=20.)
    derivatives = dict(zip(a.differential_variable_names, rates))
    w = sa.evaluation_data['volume_gradient'] @ np.array(rates)
    sc = c.evaluate(dict(values, V_liq=values['m_liq']/sa.fluids['water']['rho']),
                    axial_specific_force=20.)
    dc = dict(derivatives, V_liq=w)
    np.testing.assert_allclose(c.residual(sc, dc)[:4], a.residual(sa, derivatives)[:4])
    assert c.pressure_rate(sc, dc) == pytest.approx(a.pressure_rate(sa, derivatives), rel=1e-12)
    np.testing.assert_allclose(c.residual(sc, dc)[4:], 0., atol=1e-8)
    assert sc.properties == sa.properties
    c.reduced = True
    reduced = c.evaluate(sc.trial_values)
    np.testing.assert_allclose(c.residual(reduced, derivatives), c.residual(reduced, dc))


def test_C_dryout_discards_volume_unknown_and_preserves_gas_inventory():
    a, _ = wet_tank()
    c = ExplicitVolumeTank(a.id, a.definition, fluid_properties=a.fluid_properties)
    values = c.initial_values()
    values.update(m_liq=c.dryout_threshold, U_liq=.01)
    gas = {k: values[v] for k, v in dict(m='m_ull', U='U_ull', P='P', T='T_ull').items()}
    result = c.apply_event('dryout', values, 1.)
    assert values == gas
    assert c.variable_names == ('m', 'U', 'P', 'T')
    assert result['numerical_remainder']['mass'] == c.dryout_threshold


def test_C_network_preview_coloring_and_dryout_match_A(monkeypatch):
    pytest.importorskip('sundials4py')
    from examples.tank_volume_experiment import install
    from Fluids.FluidNetwork import FluidNetwork
    from test_fluid_network import propulsion

    options = dict(stop_at_shutdown=True,
                   tolerances={'max_step': .05, 'verify_jacobian': True})
    with propulsion(**options) as network:
        network.initialize()
        reference = network.update(3.)
    # Register original methods for automatic restoration after process-local install.
    for method in ('_make_node', '_build_layout', '_derivatives', '_new_session', '_accept'):
        monkeypatch.setattr(FluidNetwork, method, getattr(FluidNetwork, method))
    install()
    with propulsion(**options) as network:
        network.initialize()
        assert network.session.structured_verified
        y = network.y.copy()
        preview = network.update(.01, commit=False)
        np.testing.assert_array_equal(network.y, y)
        actual = network.update(.01)
        assert preview['node']['ox']['P'] == pytest.approx(actual['node']['ox']['P'])
        result = network.update(2.99)
        assert network.frozen
        assert result['node']['chamber']['mode'] == 'shutdown'
        dry = [event['time_s'] for event in result['events'] if event['name'] == 'dryout']
        reference_dry = [event['time_s'] for event in reference['events'] if event['name'] == 'dryout']
        np.testing.assert_allclose(dry, reference_dry, rtol=2e-5)
