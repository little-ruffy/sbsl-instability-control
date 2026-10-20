from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from core import gilmore as gil
from core.config import load_config


def _with_n(cfg, n: float):
    liquid = dataclasses.replace(cfg.liquid, tait_n=jnp.asarray(float(n)))
    return cfg.with_liquid(liquid) if hasattr(cfg, "with_liquid") else None


def test_exponent_comes_from_the_liquid():
    water = load_config(scenario="sbsl_argon_water").liquid
    assert float(gil.tait_exponent(water)) == pytest.approx(7.15)

    metal = load_config(scenario="sbsl_argon_galinstan").liquid
    assert float(gil.tait_exponent(metal)) == pytest.approx(4.5)
    assert float(gil.tait_exponent(metal)) != float(gil.tait_exponent(water))


def test_water_still_gets_the_classical_tait_constant():
    water = load_config(scenario="sbsl_argon_water").liquid
    b = float(gil.tait_reference_pressure(water))
    assert b == pytest.approx(3.067e8, rel=2.0e-3), (
        "water's Tait constant must stay the classical 3.05-3.07e8 Pa"
    )


def test_the_module_constant_is_no_longer_load_bearing():
    water = load_config(scenario="sbsl_argon_water").liquid
    before = float(gil.tait_reference_pressure(water))
    saved = gil.TAIT_EXPONENT
    try:
        gil.TAIT_EXPONENT = 3.0
        after = float(gil.tait_reference_pressure(water))
    finally:
        gil.TAIT_EXPONENT = saved
    assert after == pytest.approx(before, rel=1.0e-15)


def test_a_stiffer_liquid_has_a_larger_tait_constant_at_fixed_sound_speed():
    metal = load_config(scenario="sbsl_argon_galinstan").liquid
    b_soft = float(gil.tait_reference_pressure(metal))

    stiff = dataclasses.replace(metal, tait_n=jnp.asarray(9.0))
    b_stiff = float(gil.tait_reference_pressure(stiff))
    assert b_stiff < b_soft

    rho = float(metal.density)
    c = float(metal.sound_speed)
    p0 = float(metal.ambient_pressure)
    assert b_stiff == pytest.approx(rho * c * c / 9.0 - p0, rel=1.0e-12)


def test_the_exponent_moves_the_enthalpy():
    metal = load_config(scenario="sbsl_argon_galinstan").liquid
    p_wall = jnp.asarray(1.0e9)
    p_inf = jnp.asarray(float(metal.ambient_pressure))

    h = {}
    for n in (3.0, 4.5, 7.15):
        liq = dataclasses.replace(metal, tait_n=jnp.asarray(n))
        h[n] = float(gil.enthalpy(p_wall, p_inf, liq))
        assert np.isfinite(h[n])

    assert h[3.0] != pytest.approx(h[4.5], rel=1.0e-6)
    assert h[4.5] != pytest.approx(h[7.15], rel=1.0e-6)


def test_enthalpy_still_reduces_to_the_acoustic_limit_for_any_exponent():
    metal = load_config(scenario="sbsl_argon_galinstan").liquid
    for n in (3.0, 4.5, 6.0, 9.0):
        liq = dataclasses.replace(metal, tait_n=jnp.asarray(n))
        p_inf = jnp.asarray(float(liq.ambient_pressure))
        dp = 1.0e3
        h = float(gil.enthalpy(p_inf + dp, p_inf, liq))
        assert h == pytest.approx(dp / float(liq.density), rel=1.0e-6), (
            f"acoustic limit broken at n = {n}"
        )
