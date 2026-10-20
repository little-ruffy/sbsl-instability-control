from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from core import keller_miksis as km
from core.config import load_config
from core.integrator import integrate

N_SAVE: int = 4001

R0_LARGE: float = 45.0e-6


def _zero_crossing_period(t: np.ndarray, x: np.ndarray) -> float:
    sign = np.signbit(x)
    upward = np.nonzero((sign[:-1]) & (~sign[1:]))[0]
    assert upward.size >= 3, "signal does not complete enough oscillations"
    crossings = (
        t[upward]
        - x[upward] * np.diff(t)[upward] / (x[upward + 1] - x[upward])
    )
    return float(np.mean(np.diff(crossings)))


def _envelope_decay_rate(t: np.ndarray, envelope: np.ndarray) -> float:
    good = envelope > 0.0
    slope = np.polyfit(t[good], np.log(envelope[good]), 1)[0]
    return -float(slope)


def _radius_ddot(cfg, radius: float, radius_dot: float) -> float:
    y = km.initial_state(cfg)
    y = y.at[km.IDX_R].set(radius).at[km.IDX_RDOT].set(radius_dot)
    _deriv, diagnostics = km.rhs(jnp.asarray(0.0), y, cfg)
    return float(diagnostics.radius_ddot)


def _free_bubble_cfg(c_sound: float, r0: float = R0_LARGE):
    return load_config(
        scenario="sbsl_argon_water_fast",
        overrides={
            "bubble.r0": r0,
            "acoustics.amplitudes_init": [0.0, 0.0, 0.0, 0.0],
            "acoustics.frequency": 0.0,
            "liquid.sound_speed": c_sound,
            "liquid.dynamic_viscosity": 0.0,
            "transport.conduction_enabled": False,
            "transport.bremsstrahlung_enabled": False,
            "shape.enabled": False,
        },
    )


@pytest.fixture(scope="module")
def capillary():
    cfg = load_config(scenario="sbsl_argon_water_fast")
    r0 = float(cfg.r0)
    sigma = float(cfg.liquid.surface_tension)
    rho = float(cfg.liquid.density)
    nu = float(cfg.liquid.kinematic_viscosity)
    omega0 = math.sqrt(12.0 * sigma / (rho * r0**3))
    beta = 20.0 * nu / r0**2
    return {
        "r0": r0,
        "omega0": omega0,
        "beta": beta,
        "zeta": beta / omega0,
    }


def _capillary_run(viscous: bool, n_periods: float, omega: float):
    cfg = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={
            "acoustics.amplitudes_init": [0.0, 0.0, 0.0, 0.0],
            "acoustics.frequency": 0.0,
            "shape.a_l_init_rel": 1.0e-4,
            "shape.viscous_damping_enabled": viscous,
            "transport.conduction_enabled": False,
        },
    )
    t_end = n_periods * 2.0 * math.pi / omega
    return cfg, integrate(cfg, 0.0, t_end, n_save=N_SAVE, throw=True)


def test_inviscid_shape_mode_rings_at_the_rayleigh_frequency(capillary):
    omega0 = capillary["omega0"]
    cfg, traj = _capillary_run(False, 6.0, omega0)
    t = np.asarray(traj.times)
    a_l = np.asarray(traj.distortion)
    radius = np.asarray(traj.radius)

    measured = _zero_crossing_period(t, a_l)
    expected = 2.0 * math.pi / omega0
    assert measured == pytest.approx(expected, rel=1e-5), (
        f"period {measured:.9e} s vs Lamb {expected:.9e} s"
    )

    assert np.max(np.abs(radius - capillary["r0"])) / capillary["r0"] < 1e-12

    envelope = np.abs(a_l)
    peaks = envelope[
        np.nonzero(
            (envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] > envelope[2:])
        )[0]
        + 1
    ]
    assert peaks.size >= 8
    assert peaks.max() / peaks.min() < 1.001, (
        "an inviscid shape mode must not decay"
    )


def test_viscous_shape_mode_matches_prosperetti_damping(capillary):
    omega0 = capillary["omega0"]
    beta = capillary["beta"]
    zeta = capillary["zeta"]
    assert 0.1 < zeta < 0.6, f"zeta = {zeta:.3f} is outside the useful range"

    omega_d = omega0 * math.sqrt(1.0 - zeta**2)
    cfg, traj = _capillary_run(True, 6.0, omega_d)
    t = np.asarray(traj.times)
    a_l = np.asarray(traj.distortion)

    measured = _zero_crossing_period(t, a_l)
    expected = 2.0 * math.pi / omega_d
    assert measured == pytest.approx(expected, rel=1e-3), (
        f"damped period {measured:.6e} s vs predicted {expected:.6e} s "
        f"(undamped would be {2 * math.pi / omega0:.6e} s)"
    )

    envelope = np.abs(a_l)
    idx = (
        np.nonzero(
            (envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] > envelope[2:])
        )[0]
        + 1
    )
    assert idx.size >= 8
    rate = _envelope_decay_rate(t[idx], envelope[idx])
    assert rate == pytest.approx(beta, rel=0.02), (
        f"decay rate {rate:.5e} s^-1 vs Prosperetti {beta:.5e} s^-1"
    )


def test_capillary_frequency_scales_with_radius():
    periods = {}
    for r0 in (4.5e-6, 14.2e-6, 45.0e-6):
        cfg = load_config(
            scenario="sbsl_argon_water_fast",
            overrides={
                "bubble.r0": r0,
                "acoustics.amplitudes_init": [0.0, 0.0, 0.0, 0.0],
                "acoustics.frequency": 0.0,
                "shape.a_l_init_rel": 1.0e-4,
                "shape.viscous_damping_enabled": False,
                "transport.conduction_enabled": False,
            },
        )
        omega = math.sqrt(
            12.0
            * float(cfg.liquid.surface_tension)
            / (float(cfg.liquid.density) * r0**3)
        )
        traj = integrate(
            cfg, 0.0, 5.0 * 2.0 * math.pi / omega, n_save=N_SAVE, throw=True
        )
        periods[r0] = _zero_crossing_period(
            np.asarray(traj.times), np.asarray(traj.distortion)
        )

    radii = sorted(periods)
    for small, large in zip(radii, radii[1:], strict=False):
        ratio = periods[large] / periods[small]
        expected = (large / small) ** 1.5
        assert ratio == pytest.approx(expected, rel=2e-4), (
            f"T({large:.1e})/T({small:.1e}) = {ratio:.5f}, expected "
            f"{expected:.5f} from the R^-3/2 scaling"
        )


@pytest.fixture(scope="module")
def parametric():
    mu = 1.0e-4
    cfg = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={"bubble.r0": R0_LARGE, "liquid.dynamic_viscosity": mu},
    )
    rho = float(cfg.liquid.density)
    sigma = float(cfg.liquid.surface_tension)
    nu = mu / rho
    omega2 = math.sqrt(12.0 * sigma / (rho * R0_LARGE**3))
    beta2 = 20.0 * nu / R0_LARGE**2
    omega_minnaert = math.sqrt(
        3.0
        * (5.0 / 3.0)
        * float(cfg.liquid.ambient_pressure)
        / (rho * R0_LARGE**2)
    )
    assert omega_minnaert / (2.0 * omega2) > 2.0, (
        "the Minnaert resonance is too close to 2*omega_2"
    )
    return {
        "mu": mu,
        "omega2": omega2,
        "f2": omega2 / (2.0 * math.pi),
        "beta2": beta2,
        "zeta2": beta2 / omega2,
    }


def _parametric_growth(
    parametric: dict, frequency_ratio: float, amplitude_atm: float = 0.35,
    n_cycles: int = 60,
) -> float:
    frequency = frequency_ratio * parametric["f2"]
    cfg = load_config(
        scenario="sbsl_argon_water_fast",
        overrides={
            "bubble.r0": R0_LARGE,
            "liquid.dynamic_viscosity": parametric["mu"],
            "transport.conduction_enabled": False,
            "shape.a_l_init_rel": 1.0e-6,
            "acoustics.frequency": frequency,
            "acoustics.amplitudes_init": [amplitude_atm, 0.0, 0.0, 0.0],
        },
    )
    traj = integrate(
        cfg, 0.0, n_cycles / frequency, n_save=N_SAVE, throw=True
    )
    a_l = np.asarray(traj.distortion)
    a_dot = np.asarray(traj.states)[:, km.IDX_ADOT]
    envelope = np.sqrt(a_l**2 + (a_dot / parametric["omega2"]) ** 2)
    head = envelope[: envelope.size // 10].max()
    tail = envelope[-envelope.size // 10 :].max()
    return float(tail / head)


def test_parametric_resonance_at_twice_the_capillary_frequency(parametric):
    on_resonance = _parametric_growth(parametric, 2.0)
    below = _parametric_growth(parametric, 1.4)
    above = _parametric_growth(parametric, 2.6)

    assert on_resonance > 1.0e3, (
        f"no parametric growth at Omega = 2 omega_2 (factor {on_resonance:.3e})"
    )
    assert below < 1.0, (
        f"the mode should decay at Omega = 1.4 omega_2, grew {below:.3e}"
    )
    assert above < 1.0, (
        f"the mode should decay at Omega = 2.6 omega_2, grew {above:.3e}"
    )
    assert on_resonance / max(below, above) > 1.0e4, (
        "the resonance is not sharply distinguished from its surroundings"
    )


def test_off_resonance_decay_matches_the_viscous_prediction(parametric):
    frequency_ratio = 1.4
    n_cycles = 60
    growth = _parametric_growth(parametric, frequency_ratio, n_cycles=n_cycles)
    duration = n_cycles / (frequency_ratio * parametric["f2"])
    measured = -math.log(growth) / duration
    assert measured == pytest.approx(parametric["beta2"], rel=0.15), (
        f"off-resonance decay {measured:.4e} s^-1 vs Prosperetti "
        f"{parametric['beta2']:.4e} s^-1"
    )


def test_parametric_growth_needs_a_threshold_drive(parametric):
    weak = _parametric_growth(parametric, 2.0, amplitude_atm=0.02)
    strong = _parametric_growth(parametric, 2.0, amplitude_atm=0.35)
    assert weak < 1.0, (
        f"a sub-threshold drive produced growth ({weak:.3e}); there is no "
        "threshold behaviour"
    )
    assert strong > 1.0e3


def _linearised_coefficients(cfg, r0: float) -> tuple[float, float]:
    eps = 1.0e-6
    a_plus = _radius_ddot(cfg, r0 * (1.0 + eps), 0.0)
    a_minus = _radius_ddot(cfg, r0 * (1.0 - eps), 0.0)
    omega_iso_sq = -(a_plus - a_minus) / (2.0 * eps * r0)

    v = 1.0e-4
    beta = -(_radius_ddot(cfg, r0, v) - _radius_ddot(cfg, r0, -v)) / (4.0 * v)
    return omega_iso_sq, beta


def test_radiation_damping_is_exactly_inverse_in_the_sound_speed():
    speeds = (5929.2, 2964.6, 1482.3, 741.15)
    products = []
    for c_sound in speeds:
        cfg = _free_bubble_cfg(c_sound)
        _omega_sq, beta = _linearised_coefficients(cfg, R0_LARGE)
        assert beta > 0.0, f"c = {c_sound}: damping is not dissipative"
        products.append(beta * c_sound)

    spread = (max(products) - min(products)) / np.mean(products)
    assert spread < 1e-9, (
        f"beta * c is not constant across sound speeds: {products}"
    )


def test_rayleigh_plesset_has_no_radiation_damping():
    cfg = _free_bubble_cfg(1.0e12)
    omega_sq, beta = _linearised_coefficients(cfg, R0_LARGE)
    omega = math.sqrt(omega_sq)
    assert beta / omega < 1e-10, (
        f"the incompressible limit damps at zeta = {beta / omega:.3e}"
    )


def test_radiation_damping_matches_monopole_theory():
    gamma = 5.0 / 3.0
    for c_sound in (5929.2, 2964.6, 1482.3):
        cfg = _free_bubble_cfg(c_sound)
        omega_iso_sq, beta = _linearised_coefficients(cfg, R0_LARGE)
        predicted_isothermal = R0_LARGE * omega_iso_sq / (2.0 * c_sound)
        ratio = beta / predicted_isothermal
        assert ratio == pytest.approx(gamma, rel=0.02), (
            f"c = {c_sound}: beta / [R0 w_iso^2 / 2c] = {ratio:.4f}, "
            f"expected the adiabatic index {gamma:.4f}"
        )


def test_adiabatic_resonance_matches_minnaert():
    cfg = _free_bubble_cfg(1.0e12)
    y0 = km.initial_state(cfg).at[km.IDX_R].set(1.001 * R0_LARGE)
    omega_minnaert = math.sqrt(
        3.0
        * (5.0 / 3.0)
        * float(cfg.liquid.ambient_pressure)
        / (float(cfg.liquid.density) * R0_LARGE**2)
    )
    traj = integrate(
        cfg,
        0.0,
        12.0 * 2.0 * math.pi / omega_minnaert,
        y0=y0,
        n_save=N_SAVE,
        throw=True,
    )
    period = _zero_crossing_period(
        np.asarray(traj.times), np.asarray(traj.radius) - R0_LARGE
    )
    omega = 2.0 * math.pi / period
    assert omega == pytest.approx(omega_minnaert, rel=0.02), (
        f"free oscillation at {omega / 2 / math.pi / 1e3:.3f} kHz vs Minnaert "
        f"{omega_minnaert / 2 / math.pi / 1e3:.3f} kHz"
    )


def test_keller_miksis_decays_while_rayleigh_plesset_does_not():
    omega_minnaert = math.sqrt(
        3.0 * (5.0 / 3.0) * 101325.0 / (998.21 * R0_LARGE**2)
    )
    t_end = 25.0 * 2.0 * math.pi / omega_minnaert
    amplitudes = {}
    for c_sound in (1.0e12, 1482.3):
        cfg = _free_bubble_cfg(c_sound)
        y0 = km.initial_state(cfg).at[km.IDX_R].set(1.01 * R0_LARGE)
        traj = integrate(
            cfg, 0.0, t_end, y0=y0, n_save=N_SAVE, throw=True
        )
        x = np.asarray(traj.radius) - R0_LARGE
        head = float(np.abs(x[: x.size // 5]).max())
        tail = float(np.abs(x[-x.size // 5 :]).max())
        amplitudes[c_sound] = tail / head

    assert amplitudes[1.0e12] > 0.99, (
        f"the incompressible bubble lost amplitude ({amplitudes[1.0e12]:.4f})"
    )
    assert amplitudes[1482.3] < 0.75, (
        f"the compressible bubble barely decayed ({amplitudes[1482.3]:.4f})"
    )
