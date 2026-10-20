from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core import keller_miksis as km
from core.acoustics import (
    blake_threshold_pressure,
    drive_pressure,
    envelope,
    peak_negative_pressure,
)
from core.config import load_config
from core.integrator import diagnostics_along, integrate

STEPS_BUDGET_SBSL = 4096
STEPS_BUDGET_D2 = 8192


@pytest.fixture(scope="module")
def sbsl_cfg():
    return load_config(scenario="sbsl_argon_water_fast")


def test_lamb_frequency_bubble_and_drop_branches():
    radius, sigma, rho = 1e-5, 0.07275, 998.21

    bubble = float(
        km.lamb_shape_frequency(
            jnp.asarray(radius), 2, jnp.asarray(sigma), jnp.asarray(rho), 0.0
        )
    )
    assert bubble == pytest.approx(
        math.sqrt(12.0 * sigma / (rho * radius**3)), rel=1e-12
    )

    drop = float(
        km.lamb_shape_frequency(
            jnp.asarray(radius), 2, jnp.asarray(sigma), jnp.asarray(0.0),
            jnp.asarray(rho),
        )
    )
    assert drop == pytest.approx(
        math.sqrt(8.0 * sigma / (rho * radius**3)), rel=1e-12
    )
    assert bubble / drop == pytest.approx(math.sqrt(1.5), rel=1e-12)


@pytest.mark.parametrize("mode_l", [2, 3, 4, 5])
def test_shape_mode_oscillates_at_the_lamb_frequency(mode_l):
    cfg = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={
            "shape.mode_l": mode_l,
            "shape.viscous_damping_enabled": False,
        },
    )
    radius = float(cfg.r0)
    omega = float(
        km.lamb_shape_frequency(
            jnp.asarray(radius),
            mode_l,
            cfg.liquid.surface_tension,
            cfg.liquid.density,
            0.0,
        )
    )
    a_coeff, b_coeff = km.shape_mode_coefficients(
        jnp.asarray(radius), jnp.asarray(0.0), jnp.asarray(0.0), cfg
    )
    assert float(b_coeff) == 0.0
    assert float(-a_coeff) == pytest.approx(omega**2, rel=1e-12)

    def oscillator(t, y, _):
        del t
        return jnp.array([y[1], float(a_coeff) * y[0]])

    import diffrax as dfx

    period = 2.0 * math.pi / omega
    sol = dfx.diffeqsolve(
        dfx.ODETerm(oscillator),
        dfx.Tsit5(),
        t0=0.0,
        t1=4.0 * period,
        dt0=period / 1e4,
        y0=jnp.array([1.0, 0.0]),
        saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 4.0 * period, 4001)),
        stepsize_controller=dfx.PIDController(rtol=1e-11, atol=1e-13),
        max_steps=100000,
    )
    amplitude = np.asarray(sol.ys)[:, 0]
    times = np.asarray(sol.ts)
    sign_change = np.nonzero(np.diff(np.signbit(amplitude)))[0]
    assert sign_change.size >= 7
    crossings = times[sign_change] + (
        -amplitude[sign_change]
        * np.diff(times)[sign_change]
        / (amplitude[sign_change + 1] - amplitude[sign_change])
    )
    measured_half = float(np.mean(np.diff(crossings)))
    assert measured_half == pytest.approx(period / 2.0, rel=1e-6)


def test_shape_mode_coefficients_match_the_published_form(sbsl_cfg):
    radius, rdot, rddot = 1.2e-6, -430.0, 7.5e9
    cfg = sbsl_cfg
    sigma = float(cfg.liquid.surface_tension)
    rho = float(cfg.liquid.density)
    nu = float(cfg.liquid.kinematic_viscosity)

    a_coeff, b_coeff = km.shape_mode_coefficients(
        jnp.asarray(radius), jnp.asarray(rdot), jnp.asarray(rddot), cfg
    )
    expected_a = (
        rddot / radius
        - 12.0 * sigma / (rho * radius**3)
        - 8.0 * nu * rdot / radius**3
    )
    expected_b = 3.0 * rdot / radius + 40.0 * nu / radius**2
    assert float(a_coeff) == pytest.approx(expected_a, rel=1e-12)
    assert float(b_coeff) == pytest.approx(expected_b, rel=1e-12)


def test_rayleigh_taylor_term_is_destabilising_only_on_deceleration(sbsl_cfg):
    cfg = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={
            "liquid.surface_tension": 0.0,
            "shape.viscous_damping_enabled": False,
        },
    )
    radius = jnp.asarray(1e-6)
    for rddot, expect_positive in [(1e10, True), (-1e10, False)]:
        a_coeff, _ = km.shape_mode_coefficients(
            radius, jnp.asarray(0.0), jnp.asarray(rddot), cfg
        )
        assert (float(a_coeff) > 0.0) is expect_positive
    del sbsl_cfg


def test_geometric_term_amplifies_during_collapse(sbsl_cfg):
    cfg = load_config(
        scenario="sbsl_argon_water_fast", overrides={"shape.mode_l": 2}
    )
    _, b_collapse = km.shape_mode_coefficients(
        jnp.asarray(1e-6), jnp.asarray(-800.0), jnp.asarray(0.0), cfg
    )
    _, b_expand = km.shape_mode_coefficients(
        jnp.asarray(1e-6), jnp.asarray(+800.0), jnp.asarray(0.0), cfg
    )
    assert float(b_collapse) < 0.0, "collapse must anti-damp the shape mode"
    assert float(b_expand) > 0.0, "expansion must damp it"


def test_viscosity_damps_the_shape_mode(sbsl_cfg):
    common = {"shape.a_l_init_rel": 1e-2}
    cfg_visc = load_config(scenario="sbsl_argon_water_fast", overrides=common)
    cfg_inv = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={**common, "shape.viscous_damping_enabled": False},
    )
    period = 1.0 / float(cfg_visc.acoustics.frequency)
    d_visc = float(
        integrate(
            cfg_visc, 0.0, period, save_at_steps=True,
            max_steps=STEPS_BUDGET_SBSL, throw=True,
        ).max_relative_distortion
    )
    d_inv = float(
        integrate(
            cfg_inv, 0.0, period, save_at_steps=True,
            max_steps=STEPS_BUDGET_SBSL, throw=True,
        ).max_relative_distortion
    )
    assert d_visc < d_inv, (
        f"viscous damping did not reduce the distortion: "
        f"{d_visc:.4e} (viscous) vs {d_inv:.4e} (inviscid)"
    )
    del sbsl_cfg


def test_zero_seed_stays_exactly_zero(sbsl_cfg):
    cfg = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={"shape.a_l_init_rel": 0.0, "shape.adot_l_init": 0.0},
    )
    period = 1.0 / float(cfg.acoustics.frequency)
    traj = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_SBSL, throw=True
    )
    distortion = np.asarray(traj.distortion)[np.asarray(traj.valid)]
    assert np.all(distortion == 0.0)
    del sbsl_cfg


def test_radius_ddot_satisfies_the_implicit_keller_miksis_equation(sbsl_cfg):
    cfg = sbsl_cfg
    c = float(cfg.liquid.sound_speed)
    rho = float(cfg.liquid.density)
    mu = float(cfg.liquid.dynamic_viscosity)
    sigma = float(cfg.liquid.surface_tension)

    for radius, rdot, temperature in [
        (4.5e-6, 0.0, 293.15),
        (2.0e-6, -300.0, 5.0e3),
        (9.0e-7, -1200.0, 2.5e4),
        (3.0e-5, 50.0, 300.0),
    ]:
        state = km.initial_state(cfg)
        state = state.at[km.IDX_R].set(radius)
        state = state.at[km.IDX_RDOT].set(rdot)
        state = state.at[km.IDX_T].set(temperature)
        _deriv, diag = km.rhs(jnp.asarray(1.0e-6), state, cfg)

        rddot = float(diag.radius_ddot)
        p_wall = float(diag.wall_pressure)
        p_inf = float(diag.drive_pressure)
        p_inf_dot = float(diag.drive_pressure_dot)
        p_gas_dot = float(diag.gas_pressure_dot)
        mach = rdot / c

        dp_dt_full = (
            p_gas_dot
            + 2.0 * sigma * rdot / radius**2
            - 4.0 * mu * (rddot / radius - rdot**2 / radius**2)
            - p_inf_dot
        )
        residual = (
            (1.0 - mach) * radius * rddot
            + 1.5 * (1.0 - mach / 3.0) * rdot**2
            - (1.0 + mach) * (p_wall - p_inf) / rho
            - radius * dp_dt_full / (rho * c)
        )
        scale = abs(1.5 * rdot**2) + abs((p_wall - p_inf) / rho) + 1.0
        assert abs(residual) / scale < 1e-10, (
            f"R={radius:.2e}, Rdot={rdot:.1f}: implicit KM residual "
            f"{residual:.3e} (scale {scale:.3e})"
        )


def test_static_bubble_is_a_fixed_point(sbsl_cfg):
    cfg = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={"acoustics.amplitudes_init": [0.0, 0.0, 0.0, 0.0]},
    )
    state = km.initial_state(cfg)
    deriv, diag = km.rhs(jnp.asarray(0.0), state, cfg)
    assert float(deriv[km.IDX_R]) == 0.0
    assert abs(float(deriv[km.IDX_RDOT])) < 1e-6, (
        f"static bubble has Rddot = {float(deriv[km.IDX_RDOT]):.3e} m/s^2"
    )
    assert float(diag.drive_pressure) == pytest.approx(
        float(cfg.liquid.ambient_pressure), rel=1e-14
    )
    del sbsl_cfg


def test_rhs_is_differentiable_with_respect_to_the_drive(sbsl_cfg):
    import equinox as eqx

    cfg = sbsl_cfg
    state = km.initial_state(cfg).at[km.IDX_RDOT].set(-200.0)

    def rddot_of_amplitude(a1):
        amps = cfg.acoustics.amplitudes.at[0].set(a1)
        c = eqx.tree_at(lambda z: z.acoustics.amplitudes, cfg, amps)
        _, diag = km.rhs(jnp.asarray(1.0e-5), state, c)
        return diag.radius_ddot

    a0 = cfg.acoustics.amplitudes[0]
    grad = float(jax.grad(rddot_of_amplitude)(a0))
    h = float(a0) * 1e-6
    fd = (
        float(rddot_of_amplitude(a0 + h)) - float(rddot_of_amplitude(a0 - h))
    ) / (2.0 * h)
    assert math.isfinite(grad)
    assert abs(grad) > 0.0
    assert grad == pytest.approx(fd, rel=1e-5)


def test_drive_pressure_derivative_matches_finite_differences(sbsl_cfg):
    import equinox as eqx

    cfg = eqx.tree_at(
        lambda c: (c.acoustics.amplitudes, c.acoustics.phases),
        sbsl_cfg,
        (
            jnp.array([1.3e5, 4.0e4, 2.0e4, 1.0e4]),
            jnp.array([0.3, -1.1, 2.0, 0.7]),
        ),
    )
    p0 = cfg.liquid.ambient_pressure
    period = 1.0 / float(cfg.acoustics.frequency)
    for frac in (0.0, 0.13, 0.37, 0.5, 0.81, 0.99):
        t = frac * period
        _, dot = drive_pressure(jnp.asarray(t), cfg.acoustics, p0)
        h = period * 1e-7
        up, _ = drive_pressure(jnp.asarray(t + h), cfg.acoustics, p0)
        dn, _ = drive_pressure(jnp.asarray(t - h), cfg.acoustics, p0)
        assert float(dot) == pytest.approx(
            (float(up) - float(dn)) / (2.0 * h), rel=1e-6
        )


def test_drive_has_the_expected_harmonic_content(sbsl_cfg):
    import equinox as eqx

    for k in (1, 2, 3, 4):
        amps = jnp.zeros(4).at[k - 1].set(2.0e5)
        cfg = eqx.tree_at(lambda c: c.acoustics.amplitudes, sbsl_cfg, amps)
        omega = float(cfg.acoustics.omega)
        p0 = float(cfg.liquid.ambient_pressure)
        for t in (1e-6, 5e-6, 2e-5):
            p, _ = drive_pressure(
                jnp.asarray(t), cfg.acoustics, cfg.liquid.ambient_pressure
            )
            assert float(p) == pytest.approx(
                p0 - 2.0e5 * math.sin(k * omega * t), rel=1e-12
            )


def test_envelope_is_smooth_and_saturates(sbsl_cfg):
    import equinox as eqx

    cfg = eqx.tree_at(
        lambda c: c.acoustics.ramp_cycles, sbsl_cfg, jnp.asarray(3.0)
    )
    period = 1.0 / float(cfg.acoustics.frequency)
    e0, d0 = envelope(jnp.asarray(0.0), cfg.acoustics)
    assert float(e0) == pytest.approx(0.0, abs=1e-14)
    assert float(d0) == pytest.approx(0.0, abs=1e-9)
    e_end, d_end = envelope(jnp.asarray(3.0 * period), cfg.acoustics)
    assert float(e_end) == pytest.approx(1.0, rel=1e-12)
    assert float(d_end) == pytest.approx(0.0, abs=1e-6)
    e_mid, _ = envelope(jnp.asarray(1.5 * period), cfg.acoustics)
    assert float(e_mid) == pytest.approx(0.5, rel=1e-12)
    e_far, d_far = envelope(jnp.asarray(10.0 * period), cfg.acoustics)
    assert float(e_far) == 1.0
    assert float(d_far) == 0.0


def test_no_envelope_by_default(sbsl_cfg):
    for t in (0.0, 1e-6, 1.0):
        value, derivative = envelope(jnp.asarray(t), sbsl_cfg.acoustics)
        assert float(value) == 1.0
        assert float(derivative) == 0.0


def test_peak_negative_pressure_and_blake_threshold(sbsl_cfg):
    cfg = sbsl_cfg
    p0 = float(cfg.liquid.ambient_pressure)
    p_min = float(peak_negative_pressure(cfg.acoustics, cfg.liquid.ambient_pressure))
    a1 = float(cfg.acoustics.amplitudes[0])
    assert p_min == pytest.approx(p0 - a1, rel=1e-4)

    blake = float(
        blake_threshold_pressure(
            cfg.r0,
            cfg.liquid.surface_tension,
            cfg.liquid.ambient_pressure,
            cfg.liquid.vapour_pressure,
        )
    )
    assert 0.9 * p0 < blake < 1.3 * p0
    assert a1 > blake - p0


def test_guards_are_inactive_on_a_converged_trajectory(sbsl_cfg):
    cfg = sbsl_cfg
    period = 1.0 / float(cfg.acoustics.frequency)
    traj = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_SBSL, throw=True
    )
    valid = np.asarray(traj.valid)
    radius = np.asarray(traj.radius)[valid]
    temperature = np.asarray(traj.temperature)[valid]
    rdot = np.asarray(traj.radius_dot)[valid]

    r_floor = float(cfg.r_floor)
    assert radius.min() > r_floor * 1.001, (
        f"radius clamp active: min R = {radius.min():.4e} m vs floor "
        f"{r_floor:.4e} m"
    )
    assert temperature.min() > float(cfg.guards.t_min) * 1.001
    assert temperature.max() < float(cfg.guards.t_max) * 0.999
    rdot_cap = float(cfg.guards.rdot_max_mach) * float(cfg.liquid.sound_speed)
    assert np.abs(rdot).max() < rdot_cap * 0.999

    diag = diagnostics_along(cfg, traj)
    clamped = np.asarray(diag.radius_clamped)[valid]
    np.testing.assert_allclose(clamped, radius, rtol=0, atol=0)


def test_rhs_stays_finite_on_pathological_states(sbsl_cfg):
    cfg = sbsl_cfg
    h = float(cfg.hardcore_radius)
    pathological = [
        (0.5 * h, -5000.0, 1e7),
        (1e-12, 0.0, 300.0),
        (1e-3, 1e4, 1.0),
        (h, -1e5, 1e9),
        (4.5e-6, 0.0, 1e-3),
    ]
    for radius, rdot, temperature in pathological:
        state = km.initial_state(cfg)
        state = state.at[km.IDX_R].set(radius)
        state = state.at[km.IDX_RDOT].set(rdot)
        state = state.at[km.IDX_T].set(temperature)
        deriv, _diag = km.rhs(jnp.asarray(1e-6), state, cfg)
        assert np.all(np.isfinite(np.asarray(deriv))), (
            f"non-finite RHS at R={radius:.2e}, Rdot={rdot:.2e}, "
            f"T={temperature:.2e}: {np.asarray(deriv)}"
        )


def test_state_layout_is_self_consistent():
    assert km.STATE_DIM == len(km.STATE_NAMES) == len(km.STATE_UNITS)
    indices = [
        km.IDX_R,
        km.IDX_RDOT,
        km.IDX_T,
        km.IDX_A,
        km.IDX_ADOT,
        km.IDX_YIELD_N,
        km.IDX_YIELD_P,
        km.IDX_ERAD,
        km.IDX_NVAP,
    ]
    assert indices == list(range(km.STATE_DIM))
    assert km.STATE_UNITS[km.IDX_R] == "m"
    assert km.STATE_UNITS[km.IDX_T] == "K"
    assert km.STATE_UNITS[km.IDX_ERAD] == "J"
    assert km.STATE_UNITS[km.IDX_NVAP] == "1"


def test_initial_state_matches_the_configuration(sbsl_cfg):
    cfg = sbsl_cfg
    state = km.initial_state(cfg)
    assert float(state[km.IDX_R]) == float(cfg.r0)
    assert float(state[km.IDX_RDOT]) == 0.0
    assert float(state[km.IDX_T]) == float(cfg.liquid.ambient_temperature)
    assert float(state[km.IDX_A]) == pytest.approx(
        float(cfg.shape.a_l_init_rel) * float(cfg.r0), rel=1e-14
    )
    assert float(state[km.IDX_YIELD_N]) == 0.0
    assert float(state[km.IDX_YIELD_P]) == 0.0
    assert float(state[km.IDX_ERAD]) == 0.0


def test_cumulative_quantities_are_non_decreasing(sbsl_cfg):
    cfg = load_config(scenario="sonofusion_d_acetone")
    period = 1.0 / float(cfg.acoustics.frequency)
    traj = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_D2,
        throw=True,
    )
    valid = np.asarray(traj.valid)
    for index in (km.IDX_YIELD_N, km.IDX_YIELD_P, km.IDX_ERAD):
        series = np.asarray(traj.states)[valid, index]
        assert series[0] == 0.0
        assert np.all(series >= 0.0)
        assert np.all(np.diff(series) >= -1e-30 - 1e-12 * np.abs(series[:-1]))
    del sbsl_cfg


def test_mode_l_below_two_is_rejected():
    for mode_l in (0, 1):
        with pytest.raises(ValueError, match="mode_l must be >= 2"):
            load_config(
                scenario="sbsl_argon_water_fast",
                overrides={"shape.mode_l": mode_l},
            )
