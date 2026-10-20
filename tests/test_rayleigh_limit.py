from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate as sp_integrate
from scipy import optimize as sp_optimize
from scipy import special as sp_special

from core import constants as C
from core import keller_miksis as km
from core.config import load_config
from core.integrator import integrate


def rayleigh_collapse_time(r0: float, density: float, delta_p: float) -> float:
    beta = sp_special.beta(5.0 / 6.0, 0.5)
    return math.sqrt(1.5) * beta / 3.0 * r0 * math.sqrt(density / delta_p)


def rayleigh_partial_time(
    r0: float, density: float, delta_p: float, epsilon: float
) -> float:
    beta = sp_special.beta(5.0 / 6.0, 0.5)
    frac = 1.0 - sp_special.betainc(5.0 / 6.0, 0.5, epsilon**3)
    return math.sqrt(1.5) * beta * frac / 3.0 * r0 * math.sqrt(density / delta_p)


def rayleigh_wall_velocity(
    r_over_r0: float, density: float, delta_p: float
) -> float:
    return -math.sqrt(2.0 * delta_p / (3.0 * density) * (r_over_r0**-3 - 1.0))


EPSILON: float = 0.05

T_END_EPSILON: float = 0.03

WINDOW_RADIUS_FRAC: float = 0.30

WINDOW_RADIUS_MAX: float = 0.95

WINDOW_POINTS: int = 97

TIME_TOL: float = 1.0e-6


@pytest.fixture(scope="module")
def rayleigh_cfg():
    cfg = load_config(scenario="rayleigh_validation")
    assert not cfg.gas_enabled, "analytic solution requires an EMPTY cavity"
    assert float(cfg.n_nuclei) == 0.0
    assert float(cfg.liquid.surface_tension) == 0.0
    assert float(cfg.liquid.dynamic_viscosity) == 0.0
    assert float(cfg.liquid.vapour_pressure) == 0.0
    assert not cfg.shape.enabled
    assert float(cfg.acoustics.frequency) == 0.0
    assert float(jnp.max(jnp.abs(cfg.acoustics.amplitudes))) == 0.0
    return cfg


@pytest.fixture(scope="module")
def reference(rayleigh_cfg):
    r0 = float(rayleigh_cfg.r0)
    rho = float(rayleigh_cfg.liquid.density)
    dp = float(rayleigh_cfg.liquid.ambient_pressure)
    return {
        "r0": r0,
        "rho": rho,
        "dp": dp,
        "t_c": rayleigh_collapse_time(r0, rho, dp),
        "t_eps": rayleigh_partial_time(r0, rho, dp, EPSILON),
        "t_end": rayleigh_partial_time(r0, rho, dp, T_END_EPSILON),
    }


def _crossing_time(times: np.ndarray, radii: np.ndarray, target: float) -> float:
    below = np.nonzero(radii <= target)[0]
    assert below.size > 0, (
        f"radius never reached {target:.6e} m (min was {radii.min():.6e} m)"
    )
    k = int(below[0])
    assert k > 0, "radius started at or below the target"
    lo = max(0, k - 2)
    hi = min(len(times), k + 2)
    coeffs = np.polyfit(times[lo:hi], radii[lo:hi] - target, min(3, hi - lo - 1))
    return float(
        sp_optimize.brentq(
            np.poly1d(coeffs), times[k - 1], times[k], xtol=1e-20, rtol=1e-15
        )
    )


def _radius_window(
    r0: float, rho: float, dp: float, epsilon: float, n_points: int = WINDOW_POINTS
) -> tuple[jnp.ndarray, float]:
    eps_hi = min(epsilon * (1.0 + WINDOW_RADIUS_FRAC), WINDOW_RADIUS_MAX)
    eps_lo = epsilon * (1.0 - WINDOW_RADIUS_FRAC)
    t_start = rayleigh_partial_time(r0, rho, dp, eps_hi)
    t_stop = rayleigh_partial_time(r0, rho, dp, eps_lo)
    assert 0.0 < t_start < t_stop
    return jnp.linspace(t_start, t_stop, n_points), t_stop


def _measure_crossing(
    cfg, r0: float, rho: float, dp: float, epsilon: float, **kw
) -> float:
    save_ts, t_end = _radius_window(r0, rho, dp, epsilon)
    traj = integrate(cfg, 0.0, t_end, save_ts=save_ts, throw=True, **kw)
    assert bool(traj.success)
    return _crossing_time(
        np.asarray(traj.times), np.asarray(traj.radius), epsilon * r0
    )


def test_collapse_constant_matches_published_value():
    from_beta = rayleigh_collapse_time(1.0, 1.0, 1.0)
    assert from_beta == pytest.approx(C.RAYLEIGH_COLLAPSE_CONST, rel=1e-13)
    assert from_beta == pytest.approx(0.91468, rel=1e-5)

    quad, err = sp_integrate.quad(
        lambda x: x**1.5 / math.sqrt(1.0 - x**3), 0.0, 1.0, epsabs=1e-13
    )
    assert err < 1e-8
    assert math.sqrt(1.5) * quad == pytest.approx(from_beta, rel=1e-9)


def test_partial_time_limits_to_total_time():
    total = rayleigh_collapse_time(1.0, 1.0, 1.0)
    previous = 0.0
    for eps in (0.5, 0.2, 0.05, 1e-3, 1e-6, 1e-10):
        t = rayleigh_partial_time(1.0, 1.0, 1.0, eps)
        assert t >= previous
        assert t <= total
        previous = t
    assert rayleigh_partial_time(1.0, 1.0, 1.0, 0.05) < total
    assert rayleigh_partial_time(1.0, 1.0, 1.0, 1e-10) == pytest.approx(
        total, rel=1e-13
    )


def test_collapse_is_strongly_non_uniform_in_time():
    total = rayleigh_collapse_time(1.0, 1.0, 1.0)
    fractions = {
        eps: rayleigh_partial_time(1.0, 1.0, 1.0, eps) / total
        for eps in (0.5, 0.05, 0.03)
    }
    assert fractions[0.5] == pytest.approx(0.90245, abs=1e-4)
    assert fractions[0.05] == pytest.approx(0.99970, abs=1e-5)
    assert fractions[0.03] == pytest.approx(0.99992, abs=1e-5)
    assert fractions[T_END_EPSILON] < 1.0


@pytest.mark.parametrize("nondim", [True, False])
def test_collapse_time_matches_rayleigh(rayleigh_cfg, reference, nondim):
    cfg = rayleigh_cfg.with_integrator(nondimensionalise=nondim)
    t_sim = _measure_crossing(
        cfg,
        reference["r0"],
        reference["rho"],
        reference["dp"],
        EPSILON,
        rhs_fn=km.rayleigh_plesset_rhs,
    )
    rel = abs(t_sim - reference["t_eps"]) / reference["t_eps"]
    assert rel < TIME_TOL, (
        f"nondim={nondim}: collapse time to R = {EPSILON} R0 simulated "
        f"{t_sim:.12e} s vs analytic {reference['t_eps']:.12e} s, "
        f"relative error {rel:.3e}"
    )


def test_nondimensionalisation_is_an_exact_identity(rayleigh_cfg, reference):
    args = (reference["r0"], reference["rho"], reference["dp"], EPSILON)
    t_nd = _measure_crossing(
        rayleigh_cfg.with_integrator(nondimensionalise=True),
        *args,
        rhs_fn=km.rayleigh_plesset_rhs,
    )
    t_si = _measure_crossing(
        rayleigh_cfg.with_integrator(nondimensionalise=False),
        *args,
        rhs_fn=km.rayleigh_plesset_rhs,
    )
    assert abs(t_nd - t_si) / t_si < 1e-9


def test_collapse_time_converges_with_tolerance(rayleigh_cfg, reference):
    errors = []
    for rtol in (1e-5, 1e-7, 1e-9, 1e-11):
        cfg = rayleigh_cfg.with_integrator(rtol=rtol)
        t_sim = _measure_crossing(
            cfg,
            reference["r0"],
            reference["rho"],
            reference["dp"],
            EPSILON,
            rhs_fn=km.rayleigh_plesset_rhs,
        )
        errors.append(abs(t_sim - reference["t_eps"]) / reference["t_eps"])

    assert errors[-1] < errors[0], f"no convergence with tolerance: {errors}"
    assert errors[-1] < 1e-8, f"tightest-tolerance error too large: {errors}"


@pytest.mark.parametrize("epsilon", [0.8, 0.5, 0.3, 0.1, 0.05])
def test_collapse_time_matches_at_several_radii(rayleigh_cfg, reference, epsilon):
    t_analytic = rayleigh_partial_time(
        reference["r0"], reference["rho"], reference["dp"], epsilon
    )
    t_sim = _measure_crossing(
        rayleigh_cfg,
        reference["r0"],
        reference["rho"],
        reference["dp"],
        epsilon,
        rhs_fn=km.rayleigh_plesset_rhs,
    )
    rel = abs(t_sim - t_analytic) / t_analytic
    assert rel < TIME_TOL, f"eps={epsilon}: relative error {rel:.3e}"


def test_wall_velocity_matches_first_integral(rayleigh_cfg, reference):
    cfg = rayleigh_cfg.with_integrator(rtol=1e-10)
    traj = integrate(
        cfg,
        0.0,
        reference["t_end"],
        n_save=4000,
        rhs_fn=km.rayleigh_plesset_rhs,
        throw=True,
    )
    radii = np.asarray(traj.radius)
    rdots = np.asarray(traj.radius_dot)

    mask = (radii > 1.05 * T_END_EPSILON * reference["r0"]) & (
        radii < 0.999 * reference["r0"]
    )
    assert mask.sum() > 100

    expected = np.array(
        [
            rayleigh_wall_velocity(
                r / reference["r0"], reference["rho"], reference["dp"]
            )
            for r in radii[mask]
        ]
    )
    worst = float(np.max(np.abs(rdots[mask] - expected) / np.abs(expected)))
    assert worst < 1e-7, f"worst relative error in Rdot(R): {worst:.3e}"


def test_energy_first_integral_is_conserved(rayleigh_cfg, reference):
    cfg = rayleigh_cfg.with_integrator(rtol=1e-10)
    traj = integrate(
        cfg,
        0.0,
        reference["t_end"],
        n_save=4000,
        rhs_fn=km.rayleigh_plesset_rhs,
        throw=True,
    )
    radii = np.asarray(traj.radius)
    rdots = np.asarray(traj.radius_dot)
    mask = radii < 0.999 * reference["r0"]

    kinetic = 3.0 * reference["rho"] * rdots[mask] ** 2 / (2.0 * reference["dp"])
    potential = (reference["r0"] / radii[mask]) ** 3 - 1.0
    drift = np.abs(kinetic - potential) / np.abs(potential)
    assert drift.max() < 1e-7, f"first-integral drift {drift.max():.3e}"


@pytest.mark.parametrize("r0", [1e-5, 5e-5, 2e-4])
@pytest.mark.parametrize("dp_atm", [0.5, 1.0, 4.0])
def test_collapse_time_scaling_law(r0, dp_atm):
    dp = dp_atm * C.ATM_TO_PA
    cfg = load_config(
        scenario="rayleigh_validation",
        overrides={"bubble.r0": r0, "liquid.ambient_pressure": dp},
    )
    rho = float(cfg.liquid.density)
    t_eps = rayleigh_partial_time(r0, rho, dp, EPSILON)
    t_sim = _measure_crossing(
        cfg, r0, rho, dp, EPSILON, rhs_fn=km.rayleigh_plesset_rhs
    )
    rel = abs(t_sim - t_eps) / t_eps
    assert rel < TIME_TOL, (
        f"R0={r0:.1e} m, dp={dp_atm} atm: relative error {rel:.3e}"
    )


def test_keller_miksis_reduces_to_rayleigh_at_low_mach(rayleigh_cfg, reference):
    t_sim = _measure_crossing(
        rayleigh_cfg,
        reference["r0"],
        reference["rho"],
        reference["dp"],
        EPSILON,
    )
    rel = abs(t_sim - reference["t_eps"]) / reference["t_eps"]
    assert rel < 1e-5, f"full Keller-Miksis vs Rayleigh: relative error {rel:.3e}"


def test_compressibility_slows_the_collapse():
    r0 = 5e-5
    dp = 10.0 * C.ATM_TO_PA
    eps_probe = 0.3
    rho = 998.21
    save_ts = jnp.linspace(
        rayleigh_partial_time(r0, rho, dp, 0.60),
        rayleigh_partial_time(r0, rho, dp, 0.12),
        1024,
    )
    t_end = float(save_ts[-1])

    results = {}
    for c_sound in (1.0e12, 1482.3):
        cfg = load_config(
            scenario="rayleigh_validation",
            overrides={
                "bubble.r0": r0,
                "liquid.ambient_pressure": dp,
                "liquid.sound_speed": c_sound,
            },
        )
        traj = integrate(cfg, 0.0, t_end, save_ts=save_ts, throw=True)
        results[c_sound] = _crossing_time(
            np.asarray(traj.times), np.asarray(traj.radius), eps_probe * r0
        )

    t_incompressible = results[1.0e12]
    t_compressible = results[1482.3]
    assert t_compressible > t_incompressible, (
        "a finite sound speed must delay the collapse, got "
        f"{t_compressible:.6e} s vs {t_incompressible:.6e} s"
    )
    mach = abs(rayleigh_wall_velocity(eps_probe, rho, dp)) / 1482.3
    delay = (t_compressible - t_incompressible) / t_incompressible
    assert 0.0 < delay < 2.0 * mach, (
        f"fractional delay {delay:.5f} is not O(Mach = {mach:.4f})"
    )
