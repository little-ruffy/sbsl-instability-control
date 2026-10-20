from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core import liner

R_SPECIFIC = 8.314462618 / 2.014e-3
GAMMA = 5.0 / 3.0


def _params(n_zones=24, sigma=0.0, liner_mass=1.0):
    return liner.LinerParams(
        gamma=GAMMA,
        n_zones=n_zones,
        mass_per_length=1.0e-3,
        conductivity=sigma,
        liner_mass_per_length=liner_mass,
    )


def test_a_uniform_column_at_rest_has_zero_acceleration():
    p = _params()
    st = liner.initial_state(
        p, radius=3.0e-3, temperature=250.0 * 11604.5,
        b_axial=30.0, gas_constant=R_SPECIFIC,
    )
    f = liner.zone_fields(st, p, R_SPECIFIC)
    p_total = float(f["pressure"][0]) + float(f["b_field"][0]) ** 2 / (
        2.0 * liner.MU0
    )

    d = liner.rhs(st, p, jnp.asarray(p_total), R_SPECIFIC)
    n = p.n_zones
    du = np.asarray(d[n : 2 * n])
    assert np.max(np.abs(du)) < 1.0e-6 * max(abs(p_total), 1.0), (
        f"uniform state is not a fixed point: max |du/dt| = "
        f"{np.max(np.abs(du)):.3e}"
    )
    de = np.asarray(d[2 * n : 3 * n])
    assert np.max(np.abs(de)) < 1.0e-20


def test_flux_is_conserved_exactly_in_the_ideal_limit():
    p = _params(sigma=0.0)
    st = liner.initial_state(
        p, radius=3.0e-3, temperature=250.0 * 11604.5,
        b_axial=30.0, gas_constant=R_SPECIFIC,
    )
    n = p.n_zones
    st = st.at[n : 2 * n].set(-1.0e4 * jnp.linspace(0.1, 1.0, n))
    d = liner.rhs(st, p, jnp.asarray(1.0e9), R_SPECIFIC)
    dflux = np.asarray(d[3 * n : 4 * n])
    assert np.all(dflux == 0.0), "ideal limit must conserve zone flux exactly"


def test_b_scales_as_one_over_area_under_compression():
    p = _params(sigma=0.0)
    r0 = 3.0e-3
    st = liner.initial_state(
        p, radius=r0, temperature=250.0 * 11604.5,
        b_axial=30.0, gas_constant=R_SPECIFIC,
    )
    f0 = liner.zone_fields(st, p, R_SPECIFIC)
    n = p.n_zones

    for conv in (2.0, 10.0, 100.0):
        shrunk = st.at[0:n].set(st[0:n] / conv)
        f = liner.zone_fields(shrunk, p, R_SPECIFIC)
        ratio = np.asarray(f["b_field"]) / np.asarray(f0["b_field"])
        assert np.allclose(ratio, conv**2, rtol=1.0e-12), (
            f"B should scale as conv^2 = {conv**2}, got {ratio[:3]}"
        )


def test_the_pdv_term_is_the_adiabatic_law_exactly():
    r0 = 3.0e-3
    rho0 = 1.0
    mass = float(rho0 * jnp.pi * r0**2)
    p = liner.LinerParams(
        gamma=GAMMA, n_zones=20, mass_per_length=mass,
        conductivity=0.0, liner_mass_per_length=1.0e-2,
    )
    st = liner.initial_state(
        p, radius=r0, temperature=250.0 * 11604.5, b_axial=0.0,
        gas_constant=R_SPECIFIC,
    )
    n = p.n_zones
    st = st.at[n : 2 * n].set(1.0e4 * st[0:n] / r0)

    f = liner.zone_fields(st, p, R_SPECIFIC)
    d = liner.rhs(st, p, jnp.asarray(0.0), R_SPECIFIC)
    de_dt = np.asarray(d[2 * n : 3 * n])

    r = np.asarray(f["r"])
    u = np.asarray(f["u"])
    area = np.asarray(f["area"])
    e = np.asarray(f["energy"])
    area_dot = 2.0 * np.pi * np.diff(r * u)

    lhs = area ** (GAMMA - 1.0) * de_dt + (GAMMA - 1.0) * e * area ** (
        GAMMA - 2.0
    ) * area_dot
    scale = np.abs(area ** (GAMMA - 1.0) * de_dt)
    rel = np.abs(lhs) / np.maximum(scale, 1.0e-300)
    assert np.max(rel) < 1.0e-12, (
        f"the PdV term is not the adiabatic law: worst relative residual "
        f"{np.max(rel):.3e}"
    )


def test_a_quasi_static_compression_stays_near_its_adiabat():
    r0 = 3.0e-3
    mass = float(1.0 * jnp.pi * r0**2)
    p = liner.LinerParams(
        gamma=GAMMA, n_zones=16, mass_per_length=mass,
        conductivity=0.0, liner_mass_per_length=1.0e-2,
    )
    st = liner.initial_state(
        p, radius=r0, temperature=250.0 * 11604.5, b_axial=0.0,
        gas_constant=R_SPECIFIC,
    )
    p0 = float(liner.zone_fields(st, p, R_SPECIFIC)["pressure"][0])

    step = jax.jit(
        lambda y, dt: _rk4(
            lambda z, t: liner.rhs(z, p, jnp.asarray(1.3 * p0), R_SPECIFIC),
            y, dt,
        )
    )
    y = st
    for _ in range(8000):
        y = step(y, 2.0e-10)

    a = liner.zone_fields(st, p, R_SPECIFIC)
    b = liner.zone_fields(y, p, R_SPECIFIC)
    conv = float(a["r"][-1]) / float(b["r"][-1])
    assert conv > 1.005, f"no compression at all (conv = {conv:.4f})"

    inv_a = np.asarray(a["temperature"]) * np.asarray(a["area"]) ** (GAMMA - 1)
    inv_b = np.asarray(b["temperature"]) * np.asarray(b["area"]) ** (GAMMA - 1)
    drift = np.max(np.abs(inv_b / inv_a - 1.0))
    assert drift < 0.05, (
        f"invariant drifted {drift:.1%} over a compression of {conv:.4f}"
    )


def _rk4(f, y, dt):
    k1 = f(y, 0.0)
    k2 = f(y + 0.5 * dt * k1, 0.0)
    k3 = f(y + 0.5 * dt * k2, 0.0)
    k4 = f(y + dt * k3, 0.0)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def test_magnetic_pressure_can_stop_an_implosion():
    p = _params(n_zones=12, sigma=0.0, liner_mass=1.0e-6)
    st = liner.initial_state(
        p, radius=3.0e-3, temperature=250.0 * 11604.5,
        b_axial=3.0e3, gas_constant=R_SPECIFIC,
    )
    f = liner.zone_fields(st, p, R_SPECIFIC)
    p_mag = float(f["b_field"][-1]) ** 2 / (2.0 * liner.MU0)

    n = p.n_zones
    d_weak = liner.rhs(st, p, jnp.asarray(0.1 * p_mag), R_SPECIFIC)
    assert float(d_weak[2 * n - 1]) > 0.0

    d_strong = liner.rhs(st, p, jnp.asarray(10.0 * p_mag), R_SPECIFIC)
    assert float(d_strong[2 * n - 1]) < 0.0


def test_resistive_transport_conserves_total_flux():
    p = _params(n_zones=20, sigma=1.0e4)
    st = liner.initial_state(
        p, radius=3.0e-3, temperature=250.0 * 11604.5,
        b_axial=30.0, gas_constant=R_SPECIFIC,
    )
    n = p.n_zones
    f = liner.zone_fields(st, p, R_SPECIFIC)
    bumped = st.at[3 * n : 4 * n].set(
        st[3 * n : 4 * n] * jnp.linspace(0.5, 1.5, n)
    )
    d = liner.rhs(bumped, p, jnp.asarray(1.0e9), R_SPECIFIC)
    dflux = np.asarray(d[3 * n : 4 * n])
    assert np.any(dflux != 0.0), "resistivity should move flux"
    scale = np.max(np.abs(dflux))
    assert abs(dflux.sum()) < 1.0e-10 * scale, (
        f"total flux not conserved: sum = {dflux.sum():.3e} against a "
        f"per-zone scale of {scale:.3e}"
    )


def test_resistive_transport_flattens_the_field():
    p = _params(n_zones=20, sigma=1.0e4)
    st = liner.initial_state(
        p, radius=3.0e-3, temperature=250.0 * 11604.5,
        b_axial=30.0, gas_constant=R_SPECIFIC,
    )
    n = p.n_zones
    st = st.at[3 * n : 4 * n].set(
        st[3 * n : 4 * n] * jnp.linspace(0.5, 1.5, n)
    )
    y = st
    for _ in range(400):
        d = liner.rhs(y, p, jnp.asarray(0.0), R_SPECIFIC)
        y = y.at[3 * n : 4 * n].add(1.0e-10 * d[3 * n : 4 * n])
    b0 = np.asarray(liner.zone_fields(st, p, R_SPECIFIC)["b_field"])
    b1 = np.asarray(liner.zone_fields(y, p, R_SPECIFIC)["b_field"])
    assert np.all(np.isfinite(b1)), "flux diffusion produced non-finite field"
    assert np.std(b1) < np.std(b0), (
        f"diffusion increased the field spread: {np.std(b0):.4e} -> "
        f"{np.std(b1):.4e}"
    )


def test_diagnostics_report_the_groups_that_bound_validity():
    p = _params(sigma=1.0e4)
    st = liner.initial_state(
        p, radius=3.0e-3, temperature=250.0 * 11604.5,
        b_axial=30.0, gas_constant=R_SPECIFIC,
    )
    d = liner.diagnostics(st, p, R_SPECIFIC, velocity=1.0e5)
    assert np.isfinite(float(d["beta"])) and float(d["beta"]) > 0.0
    assert np.isfinite(float(d["magnetic_reynolds"]))
    ideal = liner.diagnostics(st, _params(sigma=0.0), R_SPECIFIC, velocity=1e5)
    assert float(ideal["magnetic_reynolds"]) == np.inf


def _two_region(n_fuel=12, n_liner=8, sigma=0.0):
    r0, rho0 = 3.0e-3, 1.0
    return liner.LinerParams(
        gamma=GAMMA,
        n_zones=n_fuel + n_liner,
        mass_per_length=float(rho0 * jnp.pi * r0**2),
        conductivity=sigma,
        liner_mass_per_length=1.0e-2,
        n_fuel=n_fuel,
        liner_density=9000.0,
    ), r0


def test_the_default_still_lumps_the_liner():
    p = _params()
    assert not liner.liner_is_resolved(p)
    assert liner.n_fuel_zones(p) == p.n_zones


def test_the_two_regions_carry_their_own_masses():
    p, r0 = _two_region()
    frac, outer = liner.two_region_geometry(p, r0)
    m = np.asarray(liner.zone_masses(p, frac))
    nf = liner.n_fuel_zones(p)
    assert float(m[:nf].sum()) == pytest.approx(p.mass_per_length, rel=1e-12)
    assert float(m[nf:].sum()) == pytest.approx(
        p.liner_mass_per_length, rel=1e-12
    )
    st = liner.initial_state(
        p, radius=outer, temperature=250.0 * 11604.5, b_axial=30.0,
        gas_constant=R_SPECIFIC, fractions=frac,
    )
    f = liner.zone_fields(st, p, R_SPECIFIC, fractions=frac)
    rho = np.asarray(f["density"])
    assert rho[nf:].min() > 100.0 * rho[:nf].max()


def test_the_shell_thickness_follows_from_its_mass():
    p, r0 = _two_region()
    frac, outer = liner.two_region_geometry(p, r0)
    outer = r0 / float(frac[liner.n_fuel_zones(p)])
    predicted = np.sqrt(
        r0**2 + p.liner_mass_per_length / (np.pi * p.liner_density)
    )
    assert outer == pytest.approx(predicted, rel=1.0e-12)


def test_the_axial_field_is_excluded_from_the_shell():
    p, r0 = _two_region()
    frac, outer = liner.two_region_geometry(p, r0)
    st = liner.initial_state(
        p, radius=outer, temperature=250.0 * 11604.5, b_axial=30.0,
        gas_constant=R_SPECIFIC, fractions=frac,
    )
    n, nf = p.n_zones, liner.n_fuel_zones(p)
    flux = np.asarray(st[3 * n : 4 * n])
    assert np.all(flux[:nf] > 0.0)
    assert np.all(flux[nf:] == 0.0)


def test_resistive_transport_does_not_leak_flux_into_the_shell():
    p, r0 = _two_region(sigma=1.0e4)
    frac, outer = liner.two_region_geometry(p, r0)
    st = liner.initial_state(
        p, radius=outer, temperature=250.0 * 11604.5, b_axial=30.0,
        gas_constant=R_SPECIFIC, fractions=frac,
    )
    n, nf = p.n_zones, liner.n_fuel_zones(p)
    d = liner.rhs(st, p, jnp.asarray(1.0e12), R_SPECIFIC, fractions=frac)
    dflux = np.asarray(d[3 * n : 4 * n])
    assert np.all(dflux[nf:] == 0.0), "flux must not enter the conductor"


def test_the_resolved_shell_is_not_double_counted():
    p, r0 = _two_region()
    frac, outer = liner.two_region_geometry(p, r0)
    st = liner.initial_state(
        p, radius=outer, temperature=250.0 * 11604.5, b_axial=0.0,
        gas_constant=R_SPECIFIC, fractions=frac,
    )
    n = p.n_zones
    m = np.asarray(liner.zone_masses(p, frac))
    f = liner.zone_fields(st, p, R_SPECIFIC, fractions=frac)
    drive = 1.0e12
    d = liner.rhs(st, p, jnp.asarray(drive), R_SPECIFIC, fractions=frac)

    r_out = float(f["r"][-1])
    p_out = float(f["pressure"][-1])
    expected = 2.0 * np.pi * r_out * (p_out - drive) / (0.5 * m[-1])
    assert float(d[2 * n - 1]) == pytest.approx(expected, rel=1.0e-10)


def test_a_resolved_shell_starts_cold_and_is_pressure_negligible():
    p, r0 = _two_region()
    frac, outer = liner.two_region_geometry(p, r0)
    st = liner.initial_state(
        p, radius=outer, temperature=250.0 * 11604.5, b_axial=30.0,
        gas_constant=R_SPECIFIC, fractions=frac,
    )
    f = liner.zone_fields(st, p, R_SPECIFIC, fractions=frac)
    nf = liner.n_fuel_zones(p)
    shell_p = float(np.max(np.asarray(f["pressure"])[nf:]))
    fuel_p = float(np.max(np.asarray(f["pressure"])[:nf]))
    assert shell_p < 1.0e-3 * fuel_p, (
        f"the shell is not cold: {shell_p:.3e} Pa against a fuel "
        f"{fuel_p:.3e} Pa"
    )
    assert shell_p < 1.0e-4 * 5.0e12

    d = liner.diagnostics(st, p, R_SPECIFIC, velocity=1.0e5, fractions=frac)
    assert float(d["shell_pressure"]) == pytest.approx(shell_p, rel=1e-10)

    n = p.n_zones
    acc = float(
        liner.rhs(st, p, jnp.asarray(5.0e12), R_SPECIFIC, fractions=frac)[
            2 * n - 1
        ]
    )
    assert acc < 0.0, "a 5e12 Pa drive must accelerate the shell inward"
