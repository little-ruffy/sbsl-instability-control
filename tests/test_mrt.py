from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from core import mrt


def _params(k=1.0e4, atwood=0.9, density_sum=1.0e4, enabled=True):
    return mrt.MrtParams(
        wavenumber=k, atwood=atwood, density_sum=density_sum, enabled=enabled
    )


def test_a_static_interface_does_not_move():
    p = _params()
    dxi, dxi_dot = mrt.rhs(
        jnp.asarray(1.0e-6), jnp.asarray(0.0), p,
        radius=jnp.asarray(3.0e-3),
        radius_dot=jnp.asarray(0.0),
        gravity=jnp.asarray(0.0),
        b_axial=jnp.asarray(0.0),
    )
    assert float(dxi) == 0.0
    assert float(dxi_dot) == 0.0


def test_the_unmagnetised_growth_rate_is_exact():
    k, atwood = 1.0e4, 0.9
    p = _params(k=k, atwood=atwood)
    for g in (1.0e9, 1.0e11, 1.0e13):
        d = mrt.diagnostics(
            jnp.asarray(1.0e-6), p,
            radius=jnp.asarray(3.0e-3),
            gravity=jnp.asarray(g),
            b_axial=jnp.asarray(0.0),
        )
        expected = np.sqrt(k * atwood * g)
        assert float(d["growth_rate"]) == pytest.approx(expected, rel=1e-14)
        assert bool(d["growing"])


def test_the_azimuthal_drive_field_does_not_stabilise_m0():
    p = _params()
    g = 1.0e12
    bracket = float(
        mrt.growth_bracket(p, jnp.asarray(g), jnp.asarray(0.0))
    )
    assert bracket > 0.0
    assert bracket == pytest.approx(p.wavenumber * p.atwood * g, rel=1e-14)


def test_the_axial_field_stabilises_above_the_cutoff():
    p = _params()
    g = 1.0e12
    b_crit = float(mrt.field_for_cutoff(p, jnp.asarray(g)))
    assert b_crit > 0.0

    just_under = float(
        mrt.growth_bracket(p, jnp.asarray(g), jnp.asarray(0.99 * b_crit))
    )
    just_over = float(
        mrt.growth_bracket(p, jnp.asarray(g), jnp.asarray(1.01 * b_crit))
    )
    at_cutoff = float(
        mrt.growth_bracket(p, jnp.asarray(g), jnp.asarray(b_crit))
    )
    assert just_under > 0.0, "below the critical field the mode must grow"
    assert just_over < 0.0, "above it the field must hold the mode"
    assert abs(at_cutoff) < 1.0e-6 * abs(just_under)


def test_cutoff_wavenumber_and_critical_field_are_mutual_inverses():
    p = _params()
    g = 1.0e12
    b = 50.0
    k_c = float(mrt.cutoff_wavenumber(p, jnp.asarray(g), jnp.asarray(b)))
    at_kc = mrt.MrtParams(
        wavenumber=k_c, atwood=p.atwood, density_sum=p.density_sum
    )
    assert float(mrt.field_for_cutoff(at_kc, jnp.asarray(g))) == (
        pytest.approx(b, rel=1.0e-12)
    )


def test_bell_plesset_amplifies_without_any_instability():
    p = _params()
    r0, xi_dot0 = 3.0e-3, 1.0
    v = -1.0e4
    y = np.array([1.0e-9, xi_dot0])
    t, dt = 0.0, 1.0e-10
    for _ in range(2000):
        r = r0 + v * t
        dxi, dxi_dot = mrt.rhs(
            jnp.asarray(y[0]), jnp.asarray(y[1]), p,
            radius=jnp.asarray(r),
            radius_dot=jnp.asarray(v),
            gravity=jnp.asarray(0.0),
            b_axial=jnp.asarray(0.0),
        )
        y = y + dt * np.array([float(dxi), float(dxi_dot)])
        t += dt

    r_end = r0 + v * t
    assert r_end < r0, "the test must actually converge"
    expected = xi_dot0 * r0 / r_end
    assert y[1] == pytest.approx(expected, rel=2.0e-3), (
        f"Bell-Plesset scaling broken: got {y[1]:.6f}, expected {expected:.6f}"
    )


def test_disabling_the_mode_freezes_it():
    p = _params(enabled=False)
    dxi, dxi_dot = mrt.rhs(
        jnp.asarray(1.0e-3), jnp.asarray(1.0e3), p,
        radius=jnp.asarray(1.0e-3),
        radius_dot=jnp.asarray(-1.0e4),
        gravity=jnp.asarray(1.0e13),
        b_axial=jnp.asarray(0.0),
    )
    assert float(dxi) == 0.0 and float(dxi_dot) == 0.0


def test_the_relative_amplitude_is_reported_for_the_linearisation_bound():
    p = _params()
    d = mrt.diagnostics(
        jnp.asarray(3.0e-4), p,
        radius=jnp.asarray(3.0e-3),
        gravity=jnp.asarray(1.0e12),
        b_axial=jnp.asarray(10.0),
    )
    assert float(d["relative"]) == pytest.approx(0.1, rel=1e-12)


def test_the_destabilising_direction_depends_on_which_side_is_heavy():
    p = _params()
    decelerating = jnp.asarray(+1.0e12)
    accelerating = jnp.asarray(-1.0e12)

    g_liner_stag = mrt.effective_gravity(decelerating, heavy_outside=True)
    g_liner_drive = mrt.effective_gravity(accelerating, heavy_outside=True)
    assert float(g_liner_stag) > 0.0, "a decelerating liner must be unstable"
    assert float(g_liner_drive) < 0.0

    g_bubble = mrt.effective_gravity(accelerating, heavy_outside=False)
    assert float(g_bubble) > 0.0, (
        "an inward-accelerating bubble must be unstable, as in section 2.7"
    )

    grow = mrt.growth_bracket(p, g_liner_stag, jnp.asarray(0.0))
    hold = mrt.growth_bracket(p, g_liner_drive, jnp.asarray(0.0))
    assert float(grow) > 0.0 and float(hold) < 0.0
