from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from core import gilmore as gm
from core import keller_miksis as km
from core.config import load_config
from core.integrator import integrate


@pytest.fixture(scope="module")
def cfg():
    return load_config(scenario="sbsl_argon_water_fast")


@pytest.fixture(scope="module")
def hard_cfg():
    return load_config(scenario="sonofusion_d_acetone")


def test_tait_constant_reproduces_the_configured_sound_speed(cfg):
    liquid = cfg.liquid
    b = float(gm.tait_reference_pressure(liquid))
    recovered = math.sqrt(
        gm.TAIT_EXPONENT * (float(liquid.ambient_pressure) + b)
        / float(liquid.density)
    )
    assert recovered == pytest.approx(float(liquid.sound_speed), rel=1e-12)
    assert 2.5e8 < b < 3.5e8


def test_density_is_ambient_at_ambient_pressure(cfg):
    rho = float(
        gm.liquid_density(cfg.liquid.ambient_pressure, cfg.liquid)
    )
    assert rho == pytest.approx(float(cfg.liquid.density), rel=1e-12)


def test_liquid_compresses_measurably_at_collapse_pressures(cfg):
    liquid = cfg.liquid
    ratios = {
        p: float(gm.liquid_density(jnp.asarray(p), liquid))
        / float(liquid.density)
        for p in (1e8, 1e9, 1e10)
    }
    assert ratios[1e8] == pytest.approx(1.04, rel=0.05)
    assert ratios[1e9] == pytest.approx(1.22, rel=0.05)
    assert ratios[1e10] > 1.5
    assert ratios[1e8] < ratios[1e9] < ratios[1e10]


def test_enthalpy_reduces_to_the_acoustic_form(cfg):
    liquid = cfg.liquid
    p0 = float(liquid.ambient_pressure)
    for delta in (1e2, 1e4, 1e6):
        h = float(
            gm.enthalpy(jnp.asarray(p0 + delta), liquid.ambient_pressure, liquid)
        )
        acoustic = delta / float(liquid.density)
        assert h == pytest.approx(acoustic, rel=max(1e-6, delta / 3.0e8))


def test_wall_sound_speed_rises_with_the_enthalpy(cfg):
    liquid = cfg.liquid
    assert float(gm.wall_sound_speed(jnp.asarray(0.0), liquid)) == pytest.approx(
        float(liquid.sound_speed), rel=1e-12
    )
    for enthalpy_value in (1e4, 1e5, 1e6):
        c = float(gm.wall_sound_speed(jnp.asarray(enthalpy_value), liquid))
        assert c == pytest.approx(
            math.sqrt(
                float(liquid.sound_speed) ** 2
                + (gm.TAIT_EXPONENT - 1.0) * enthalpy_value
            ),
            rel=1e-12,
        )
        assert c > float(liquid.sound_speed)


def test_reduces_to_rayleigh_in_the_incompressible_limit():
    import equinox as eqx

    from tests.test_rayleigh_limit import (
        _crossing_time,
        _radius_window,
        rayleigh_partial_time,
    )

    cfg = load_config(scenario="rayleigh_validation")
    fast = eqx.tree_at(
        lambda c: c.liquid.sound_speed, cfg, jnp.asarray(1.0e12)
    )
    r0 = float(cfg.r0)
    rho = float(cfg.liquid.density)
    dp = float(cfg.liquid.ambient_pressure)
    epsilon = 0.05

    save_ts, t_end = _radius_window(r0, rho, dp, epsilon)
    traj = integrate(
        fast, 0.0, t_end, save_ts=save_ts, rhs_fn=gm.rhs_only, throw=True,
    )
    assert bool(traj.success)
    measured = _crossing_time(
        np.asarray(traj.times), np.asarray(traj.radius), epsilon * r0
    )
    assert measured == pytest.approx(
        rayleigh_partial_time(r0, rho, dp, epsilon), rel=2e-4
    )


def test_agrees_with_keller_miksis_at_low_mach(cfg):
    period = 1.0 / float(cfg.acoustics.frequency)
    results = {}
    for label, fn in (("km", km.rhs_only), ("gilmore", gm.rhs_only)):
        traj = integrate(
            cfg, 0.0, period, save_at_steps=True, max_steps=20_000,
            rhs_fn=fn, throw=True,
        )
        valid = np.asarray(traj.valid)
        states = np.asarray(traj.states)[valid]
        results[label] = (
            states[:, km.IDX_R].max(),
            states[:, km.IDX_R].min(),
            states[:, km.IDX_T].max(),
        )
    r_max_km, r_min_km, t_km = results["km"]
    r_max_g, r_min_g, t_g = results["gilmore"]
    assert r_max_g == pytest.approx(r_max_km, rel=2e-3)
    assert r_min_g == pytest.approx(r_min_km, rel=0.08)
    assert t_g == pytest.approx(t_km, rel=0.08)


def test_local_mach_is_far_below_the_acoustic_one(hard_cfg):
    period = 1.0 / float(hard_cfg.acoustics.frequency)
    traj = integrate(
        hard_cfg, 0.0, period, save_at_steps=True, max_steps=40_000,
        rhs_fn=gm.rhs_only, throw=True,
    )
    valid = np.asarray(traj.valid)
    times = np.asarray(traj.times)[valid]
    states = np.asarray(traj.states)[valid]
    diag = gm.diagnostics_along(
        hard_cfg, jnp.asarray(times), jnp.asarray(states)
    )
    acoustic = np.abs(np.asarray(diag.mach_acoustic)).max()
    local = np.abs(np.asarray(diag.mach_local)).max()
    assert acoustic > 5.0
    assert local < acoustic
    assert float(np.asarray(diag.density_ratio).max()) > 1.5


def test_denominator_never_changes_sign_on_a_collapse(hard_cfg):
    period = 1.0 / float(hard_cfg.acoustics.frequency)
    traj = integrate(
        hard_cfg, 0.0, period, save_at_steps=True, max_steps=40_000,
        rhs_fn=gm.rhs_only, throw=True,
    )
    valid = np.asarray(traj.valid)
    times = np.asarray(traj.times)[valid]
    states = np.asarray(traj.states)[valid]
    diag = gm.diagnostics_along(
        hard_cfg, jnp.asarray(times), jnp.asarray(states)
    )
    denominator = 1.0 - np.asarray(diag.mach_local)
    assert np.all(denominator > 0.0)
    assert denominator.min() > 0.1
    rdot = states[:, km.IDX_RDOT]
    supersonic = np.abs(np.asarray(diag.mach_local)) > 1.0
    assert np.all(rdot[supersonic] < 0.0)


def test_fusion_yield_is_not_a_reportable_number(hard_cfg):
    period = 1.0 / float(hard_cfg.acoustics.frequency)
    peaks = {}
    yields = {}
    for label, fn in (("km", km.rhs_only), ("gilmore", gm.rhs_only)):
        traj = integrate(
            hard_cfg, 0.0, period, save_at_steps=True, max_steps=40_000,
            rhs_fn=fn, throw=True,
        )
        valid = np.asarray(traj.valid)
        states = np.asarray(traj.states)[valid]
        peaks[label] = states[:, km.IDX_T].max()
        yields[label] = states[:, km.IDX_YIELD_N].max()

    assert 0.4 < peaks["km"] / peaks["gilmore"] < 2.5
    for peak in peaks.values():
        assert 5.0e3 < peak < 1.0e5

    ratio = yields["gilmore"] / max(yields["km"], 1.0e-300)
    assert ratio > 1.0e6, (
        "if the two models ever agree on the yield, this test's premise -- "
        "that it is exponentially sensitive and must not be quoted -- needs "
        "revisiting"
    )
    for value in yields.values():
        assert value < 1.0e-20
