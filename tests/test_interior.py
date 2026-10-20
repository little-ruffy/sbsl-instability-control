from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from core import interior as it
from core import keller_miksis as km
from core import plasma_core as pc
from core.config import load_config

ZONES_SMALL = 8
ZONES_MEDIUM = 16


@pytest.fixture(scope="module")
def cfg():
    return load_config(scenario="sbsl_argon_water_fast")


@pytest.fixture(scope="module")
def deuterium_cfg():
    return load_config(scenario="sonofusion_d_acetone")


@pytest.mark.parametrize("stretch", [1.0 / 3.0, 1.0, 1.5])
@pytest.mark.parametrize("n_zones", [ZONES_SMALL, ZONES_MEDIUM])
def test_grid_partitions_the_gas_exactly(cfg, n_zones, stretch):
    nuclei = it.zone_nuclei(cfg, n_zones, stretch)
    mass = it.zone_mass(cfg, n_zones, stretch)
    assert float(jnp.sum(nuclei)) == pytest.approx(
        float(cfg.n_nuclei), rel=1e-14
    )
    assert float(jnp.sum(mass)) == pytest.approx(
        float(pc.total_gas_mass(cfg)), rel=1e-14
    )
    assert bool(jnp.all(nuclei > 0.0))


@pytest.mark.parametrize("stretch", [1.0 / 3.0, 1.0, 1.5])
def test_grid_fractions_are_monotone_and_span_the_bubble(stretch):
    frac = it.grid_fractions(ZONES_MEDIUM, stretch)
    assert len(frac) == ZONES_MEDIUM + 1
    assert float(frac[0]) == 0.0
    assert float(frac[-1]) == pytest.approx(1.0, rel=1e-15)
    assert bool(jnp.all(jnp.diff(frac) > 0.0))


def test_equal_mass_zoning_starves_the_centre():
    equal_mass = np.diff(np.asarray(it.grid_fractions(16, 1.0 / 3.0)))
    uniform = np.diff(np.asarray(it.grid_fractions(16, it.DEFAULT_STRETCH)))
    assert equal_mass[0] / equal_mass[-1] > 15.0
    assert uniform.max() / uniform.min() == pytest.approx(1.0, abs=1e-12)
    assert equal_mass[0] > 5.0 * uniform[0]


@pytest.mark.parametrize("stretch", [1.0 / 3.0, 1.0, 1.5])
@pytest.mark.parametrize("n_zones", [ZONES_SMALL, ZONES_MEDIUM])
def test_uniform_state_reproduces_the_single_zone_model(cfg, n_zones, stretch):
    radius = float(cfg.r0)
    temperature = float(cfg.liquid.ambient_temperature)

    single = jnp.asarray(
        [radius, 0.0, temperature, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    _, diag_single = km.rhs(jnp.asarray(0.0), single, cfg)

    resolved = it.initial_state_from_uniform(
        cfg, radius, 0.0, temperature, n_zones, stretch=stretch
    )
    _, diag_resolved = it.rhs(jnp.asarray(0.0), resolved, cfg, n_zones, stretch)

    assert float(diag_resolved.gas_pressure) == pytest.approx(
        float(diag_single.gas_pressure), rel=1e-12
    )
    assert float(diag_resolved.wall_acceleration) == pytest.approx(
        float(diag_single.radius_ddot), rel=1e-10
    )
    assert float(diag_resolved.amplification) == pytest.approx(1.0, rel=1e-14)


def test_uniform_state_has_no_artificial_viscosity(cfg):
    resolved = it.initial_state_from_uniform(
        cfg, float(cfg.r0), 0.0, float(cfg.liquid.ambient_temperature),
        ZONES_SMALL,
    )
    _, diag = it.rhs(jnp.asarray(0.0), resolved, cfg, ZONES_SMALL)
    assert float(jnp.max(jnp.abs(diag.zones.q_viscous))) == 0.0


def test_uniform_state_has_uniform_density(cfg):
    for stretch in (1.0 / 3.0, 1.0, 1.5):
        resolved = it.initial_state_from_uniform(
            cfg, float(cfg.r0), 0.0, 300.0, ZONES_SMALL, stretch=stretch
        )
        _, diag = it.rhs(jnp.asarray(0.0), resolved, cfg, ZONES_SMALL, stretch)
        rho = np.asarray(diag.zones.density)
        assert rho.max() / rho.min() == pytest.approx(1.0, rel=1e-12)


def test_uniform_state_is_in_internal_equilibrium(cfg):
    n_zones = ZONES_SMALL
    resolved = it.initial_state_from_uniform(
        cfg, float(cfg.r0), 0.0, float(cfg.liquid.ambient_temperature), n_zones
    )
    derivative, diag = it.rhs(jnp.asarray(0.0), resolved, cfg, n_zones)
    interior_accel = np.abs(np.asarray(derivative[n_zones : 2 * n_zones - 1]))
    assert interior_accel.max() < 1e-6 * abs(
        float(diag.wall_acceleration)
    )


def test_compression_switch_is_exact_outside_the_blend():
    scale = jnp.full((5,), 1000.0)
    eps = it.Q_SMOOTHING * 1000.0
    delta_u = jnp.asarray([-500.0, -2.0 * eps, 0.0, 2.0 * eps, 500.0])
    s = np.asarray(it._compression(delta_u, scale))
    assert s[0] == pytest.approx(500.0, rel=1e-15)
    assert s[1] == pytest.approx(2.0 * eps, rel=1e-15)
    assert s[2] == 0.0
    assert s[3] == 0.0
    assert s[4] == 0.0


def test_compression_switch_is_c1_at_both_joins():
    scale = jnp.asarray([1000.0])
    eps = it.Q_SMOOTHING * 1000.0
    h = eps * 1e-6

    def s(x):
        return float(it._compression(jnp.asarray([x]), scale)[0])

    for join in (-eps, 0.0):
        left, right = s(join - h), s(join + h)
        assert left == pytest.approx(right, abs=1e-3 * eps)
        slope_l = (s(join - h) - s(join - 2 * h)) / h
        slope_r = (s(join + 2 * h) - s(join + h)) / h
        assert slope_l == pytest.approx(slope_r, abs=1e-2)


def test_viscosity_activates_only_in_compression(cfg):
    for rdot, expect_viscosity in ((-900.0, True), (+900.0, False)):
        state = it.initial_state_from_uniform(
            cfg, 3.0e-6, rdot, 3000.0, ZONES_SMALL
        )
        _, diag = it.rhs(jnp.asarray(0.0), state, cfg, ZONES_SMALL)
        q_max = float(jnp.max(diag.zones.q_viscous))
        if expect_viscosity:
            assert q_max > 0.0
        else:
            assert q_max == 0.0


def test_time_is_integrated_in_scaled_units(cfg):
    n_zones = ZONES_SMALL
    y0 = it.initial_state_from_uniform(
        cfg, 3.0e-6, -400.0, 2000.0, n_zones
    )
    traj = it.integrate_interior(
        cfg, 0.0, 2.0e-11, y0, n_zones, n_save=8, max_steps=20_000,
        throw=False,
    )
    assert bool(traj.success)
    assert float(traj.times[-1]) == pytest.approx(2.0e-11, rel=1e-10)


def test_integration_conserves_the_grid_ordering(cfg):
    n_zones = ZONES_SMALL
    y0 = it.initial_state_from_uniform(cfg, 3.0e-6, -600.0, 3000.0, n_zones)
    traj = it.integrate_interior(
        cfg, 0.0, 1.0e-9, y0, n_zones, n_save=32, max_steps=200_000,
        throw=False,
    )
    assert bool(traj.success)
    radii = np.asarray(traj.states[:, 0:n_zones])
    assert np.all(np.diff(radii, axis=1) > 0.0)
    assert np.all(radii > 0.0)
    assert np.all(np.isfinite(np.asarray(traj.states)))


def test_resolved_collapse_develops_a_radial_profile(cfg):
    n_zones = ZONES_MEDIUM
    y0 = it.initial_state_from_uniform(cfg, 2.0e-6, -800.0, 6000.0, n_zones)
    traj = it.integrate_interior(
        cfg, 0.0, 2.0e-9, y0, n_zones, n_save=64, max_steps=400_000,
        throw=False,
    )
    assert bool(traj.success)
    temps = np.asarray(traj.temperatures)
    radii = np.asarray(traj.wall_radius)

    k = int(np.argmin(radii))
    assert 0 < k < len(radii) - 1, "window did not reach a turning point"
    profile = temps[k]
    assert np.all(np.diff(profile) <= 1e-9 * profile[0])
    assert profile[0] / profile[-1] > 1.5
    assert profile[0] > profile[n_zones // 2] > profile[-1]


def test_state_layout_round_trips(cfg):
    n_zones = ZONES_SMALL
    y0 = it.initial_state_from_uniform(
        cfg, 4.0e-6, -100.0, 1234.0, n_zones,
        distortion=3.0e-9, distortion_dot=2.0,
    )
    assert y0.shape == (it.state_dim(n_zones),)
    assert float(y0[n_zones - 1]) == pytest.approx(4.0e-6, rel=1e-15)
    assert float(y0[2 * n_zones - 1]) == pytest.approx(-100.0, rel=1e-15)
    assert float(y0[2 * n_zones]) == pytest.approx(1234.0, rel=1e-15)
    assert float(y0[3 * n_zones]) == pytest.approx(3.0e-9, rel=1e-15)
    assert float(y0[3 * n_zones + 1]) == pytest.approx(2.0, rel=1e-15)


def test_homologous_initial_velocity_is_exact(cfg):
    n_zones = ZONES_MEDIUM
    radius, radius_dot = 5.0e-6, -850.0
    y0 = np.asarray(
        it.initial_state_from_uniform(cfg, radius, radius_dot, 400.0, n_zones)
    )
    r = y0[0:n_zones]
    u = y0[n_zones : 2 * n_zones]
    assert np.allclose(u, radius_dot * r / radius, rtol=1e-14)


def test_gas_free_cavity_is_rejected_gracefully():
    cfg_empty = load_config(scenario="rayleigh_validation")
    n_zones = ZONES_SMALL
    y0 = it.initial_state_from_uniform(
        cfg_empty, float(cfg_empty.r0), 0.0, 300.0, n_zones
    )
    derivative, diag = it.rhs(jnp.asarray(0.0), y0, cfg_empty, n_zones)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(diag.gas_pressure) == 0.0


def test_tolerances_cover_every_component(cfg):
    n_zones = ZONES_MEDIUM
    atol = np.asarray(it.interior_tolerances(cfg, n_zones))
    assert atol.shape == (it.state_dim(n_zones),)
    assert np.all(atol > 0.0)
    assert atol[0] < float(cfg.integrator.atol_r)
