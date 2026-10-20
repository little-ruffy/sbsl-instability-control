from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from core import keller_miksis as km
from core import plasma_core as pc
from core import vapour as vap
from core.config import load_config
from core.integrator import integrate


@pytest.fixture(scope="module")
def cfg():
    return load_config(scenario="sbsl_argon_water_vapour")


@pytest.fixture(scope="module")
def dry_cfg():
    return load_config(scenario="sbsl_argon_water_fast")


@pytest.fixture(scope="module")
def cycle(cfg):
    period = 1.0 / float(cfg.acoustics.frequency)
    traj = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=8192, throw=True
    )
    valid = np.asarray(traj.valid)
    times = np.asarray(traj.times)[valid]
    states = np.asarray(traj.states)[valid]
    order = np.argsort(times)
    return times[order], states[order]


@pytest.fixture(scope="module")
def dry_cycle(dry_cfg):
    period = 1.0 / float(dry_cfg.acoustics.frequency)
    traj = integrate(
        dry_cfg, 0.0, period, save_at_steps=True, max_steps=8192, throw=True
    )
    valid = np.asarray(traj.valid)
    times = np.asarray(traj.times)[valid]
    states = np.asarray(traj.states)[valid]
    order = np.argsort(times)
    return times[order], states[order]


@pytest.mark.parametrize(
    "celsius, expected",
    [
        (0.0, 611.2),
        (10.0, 1228.0),
        (20.0, 2339.0),
        (30.0, 4246.0),
        (50.0, 12344.0),
        (100.0, 101325.0),
    ],
)
def test_saturation_pressure_matches_reference_values(celsius, expected):
    got = float(vap.saturation_pressure(jnp.asarray(celsius + 273.15)))
    assert got == pytest.approx(expected, rel=5e-3)


def test_saturation_pressure_is_monotone_and_finite():
    temperatures = jnp.asarray([1.0, 100.0, 273.15, 373.15, 600.0, 5000.0])
    values = np.asarray(vap.saturation_pressure(temperatures))
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    assert np.all(np.diff(values) > 0.0)


def test_diffusivity_scales_inversely_with_density(cfg):
    reference = float(cfg.vapour.density_ref)
    base = float(vap.diffusion_coefficient(jnp.asarray(reference), cfg))
    assert base == pytest.approx(float(cfg.vapour.diffusivity_ref), rel=1e-12)
    for factor in (10.0, 1e3, 1e5):
        scaled = float(
            vap.diffusion_coefficient(jnp.asarray(reference * factor), cfg)
        )
        assert scaled == pytest.approx(base / factor, rel=1e-12)


def test_diffusive_length_matches_its_two_branches(cfg):
    radius = 5.0e-6
    diffusivity = 2.0e-5
    cap = float(cfg.transport.boundary_layer_cap_factor)

    fast = float(
        vap.diffusive_length(
            jnp.asarray(radius), jnp.asarray(-5.0e3),
            jnp.asarray(diffusivity), cfg,
        )
    )
    assert fast == pytest.approx(
        math.sqrt(radius * diffusivity / 5.0e3), rel=2e-2
    )
    slow = float(
        vap.diffusive_length(
            jnp.asarray(radius), jnp.asarray(-1.0e-4),
            jnp.asarray(diffusivity), cfg,
        )
    )
    assert slow == pytest.approx(radius / cap, rel=2e-2)


def test_exchange_drives_towards_saturation(cfg):
    radius = 5.0e-6
    v_free = float(pc.free_volume(jnp.asarray(radius), cfg))
    saturated = float(vap.saturated_count(jnp.asarray(radius), cfg))

    dry_rate, _, _ = vap.exchange_rate(
        jnp.asarray(radius), jnp.asarray(-10.0), jnp.asarray(0.1 * saturated),
        jnp.asarray(v_free), jnp.asarray(1e10), cfg,
    )
    wet_rate, _, _ = vap.exchange_rate(
        jnp.asarray(radius), jnp.asarray(-10.0), jnp.asarray(10.0 * saturated),
        jnp.asarray(v_free), jnp.asarray(1e10), cfg,
    )
    assert float(dry_rate) > 0.0
    assert float(wet_rate) < 0.0


def test_exchange_is_exactly_zero_when_disabled(dry_cfg):
    rate, length, scale = vap.exchange_rate(
        jnp.asarray(5e-6), jnp.asarray(-100.0), jnp.asarray(1e9),
        jnp.asarray(1e-16), jnp.asarray(1e10), dry_cfg,
    )
    assert float(rate) == 0.0
    assert float(length) == 0.0
    assert float(scale) == 0.0
    assert float(
        vap.transfer_power(
            jnp.asarray(1.0e12), jnp.asarray(1.0e12), jnp.asarray(3.0e4),
            jnp.asarray(1.0e-19), dry_cfg,
        )
    ) == 0.0


def test_initial_bubble_is_saturated(cfg):
    state = km.initial_state(cfg)
    expected = float(vap.saturated_count(cfg.r0, cfg))
    assert float(state[km.IDX_NVAP]) == pytest.approx(expected, rel=1e-12)
    assert float(state[km.IDX_NVAP]) > 0.0


def test_initial_pressure_is_shared_with_the_vapour(cfg, dry_cfg):
    p_sat = float(vap.saturation_pressure(cfg.liquid.ambient_temperature))
    p_gas = float(
        cfg.liquid.ambient_pressure + 2.0 * cfg.liquid.surface_tension / cfg.r0
    )
    assert float(cfg.n_nuclei) / float(dry_cfg.n_nuclei) == pytest.approx(
        1.0 - p_sat / p_gas, rel=2e-3
    )


def test_vapour_species_carries_no_configured_mole_fraction(cfg):
    index = cfg.vapour.species_index
    assert cfg.gas.names[index] == "water_vapour"
    assert float(cfg.gas.mole_frac[index]) == 0.0
    assert float(jnp.sum(cfg.gas.mole_frac)) == pytest.approx(1.0, rel=1e-12)


def test_water_carries_rotational_and_vibrational_energy(cfg):
    index = cfg.vapour.species_index
    counts = jnp.zeros(cfg.gas.n_species).at[index].set(1.0e10)

    def energy_per_molecule(temperature):
        return float(
            pc.free_particle_internal_energy(
                jnp.asarray(temperature), counts, cfg.gas, cfg
            )
        ) / 1.0e10

    from core import constants as C

    cold = energy_per_molecule(300.0) / (C.K_B * 300.0)
    hot = energy_per_molecule(3000.0) / (C.K_B * 3000.0)
    assert cold == pytest.approx(1.5, rel=0.05)
    assert hot > cold


def test_adding_a_molecule_to_a_hot_core_costs_energy(cfg):
    from core import constants as C

    radius = 1.0e-6
    counts = pc.vapour_counts(cfg, jnp.asarray(1.0e9))
    v_free = pc.free_volume(jnp.asarray(radius), cfg, counts)
    incoming = float(vap.incoming_enthalpy(cfg))

    for temperature in (3.0e3, 1.0e4, 3.0e4):
        eos = pc.equation_of_state_and_derivatives(
            jnp.asarray(temperature), v_free, cfg, counts
        )
        cost = float(eos.du_dn)
        assert cost > 0.0
        assert cost > incoming
        assert cost > 1.4 * C.K_B * temperature
    assert incoming == pytest.approx(
        4.0 * C.K_B * float(cfg.liquid.ambient_temperature), rel=0.1
    )


def test_du_dn_is_zero_when_vapour_is_disabled(dry_cfg):
    v_free = pc.free_volume(jnp.asarray(1.0e-6), dry_cfg)
    eos = pc.equation_of_state_and_derivatives(
        jnp.asarray(1.0e4), v_free, dry_cfg
    )
    assert float(eos.du_dn) == 0.0
    assert float(eos.dp_dn) == 0.0


def test_bubble_is_mostly_vapour_at_maximum_radius(cfg, cycle):
    _, states = cycle
    radii = states[:, km.IDX_R]
    vapour_count = states[:, km.IDX_NVAP]
    k = int(np.argmax(radii))
    fraction = vapour_count[k] / (vapour_count[k] + float(cfg.n_nuclei))
    assert fraction > 0.8


def test_vapour_is_trapped_through_the_collapse(cfg, cycle):
    _, states = cycle
    radii = states[:, km.IDX_R]
    vapour_count = states[:, km.IDX_NVAP]
    k = int(np.argmin(radii))
    trapped = vapour_count[k]
    assert trapped > 0.1 * float(cfg.n_nuclei)
    assert trapped < 5.0 * float(cfg.n_nuclei)
    assert trapped < 0.2 * vapour_count.max()


def test_vapour_count_stays_positive(cycle):
    _, states = cycle
    assert np.all(states[:, km.IDX_NVAP] >= 0.0)
    assert np.all(np.isfinite(states))


def test_trapped_vapour_roughly_halves_the_peak_temperature(cycle, dry_cycle):
    _, wet = cycle
    _, dry = dry_cycle
    wet_peak = wet[:, km.IDX_T].max()
    dry_peak = dry[:, km.IDX_T].max()
    assert 1.5 < dry_peak / wet_peak < 4.0
    assert 5.0e3 < wet_peak < 2.5e4


def test_vapour_cools_the_core_during_the_collapse(cfg, cycle):
    times, states = cycle
    k = int(np.argmin(states[:, km.IDX_R]))
    probe = max(k - 10, 0)
    budget, _ = pc.energy_budget(
        jnp.asarray(states[probe, km.IDX_R]),
        jnp.asarray(states[probe, km.IDX_RDOT]),
        jnp.asarray(states[probe, km.IDX_T]),
        cfg,
        jnp.asarray(states[probe, km.IDX_NVAP]),
    )
    assert float(states[probe, km.IDX_RDOT]) < 0.0
    assert float(budget.vapour_power) < 0.0


def test_disabled_scenarios_are_untouched():
    for name in ("sbsl_argon_water", "sbsl_argon_water_fast",
                 "sonofusion_d_acetone", "rayleigh_validation"):
        config = load_config(scenario=name)
        assert not config.vapour.enabled
        assert float(km.initial_state(config)[km.IDX_NVAP]) == 0.0
