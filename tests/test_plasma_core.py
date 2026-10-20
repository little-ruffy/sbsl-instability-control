from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize as sp_optimize

from core import constants as C
from core import plasma_core as pc
from core.config import load_config


@pytest.fixture(scope="module")
def argon_cfg():
    return load_config(scenario="sbsl_argon_water")


@pytest.fixture(scope="module")
def deuterium_cfg():
    return load_config(scenario="sonofusion_d_acetone")


def _v_free(cfg, radius: float):
    return pc.free_volume(jnp.asarray(radius), cfg)


def test_derived_constants_are_self_consistent():
    assert pytest.approx(8.31446261815324, rel=1e-12) == C.R_GAS
    assert pytest.approx(11604.518, rel=1e-6) == C.KELVIN_PER_EV
    assert pytest.approx(1.1604518e7, rel=1e-6) == C.KELVIN_PER_KEV
    assert pytest.approx(1.69e-38, rel=1e-12) == C.BREMSSTRAHLUNG_COEFF_SI
    assert pytest.approx(
        C.E_DD_NEUTRON_MEV + C.E_CHARGED_DD_N3HE_MEV, rel=1e-12
    ) == C.Q_DD_N3HE_MEV


def test_pressure_matches_ideal_gas_when_hard_core_is_negligible(argon_cfg):
    cfg = load_config(
        scenario="sbsl_argon_water",
        overrides={"transport.ionisation_enabled": False},
    )
    radius = 20.0 * float(cfg.r0)
    v_free = _v_free(cfg, radius)
    comp = pc.composition(jnp.asarray(300.0), v_free, cfg)
    p = float(pc.pressure_from_composition(comp, cfg))
    v_total = (4.0 / 3.0) * math.pi * radius**3
    p_ideal = float(cfg.n_nuclei) * C.K_B * 300.0 / v_total
    assert p == pytest.approx(p_ideal, rel=2e-5)


def test_hard_core_makes_pressure_diverge(argon_cfg):
    h = float(argon_cfg.hardcore_radius)
    assert h > 0.0
    pressures = []
    for factor in (3.0, 2.0, 1.5, 1.2, 1.05):
        v_free = _v_free(argon_cfg, factor * h)
        comp = pc.composition(jnp.asarray(5000.0), v_free, argon_cfg)
        pressures.append(float(pc.pressure_from_composition(comp, argon_cfg)))
    assert all(
        b > a for a, b in zip(pressures, pressures[1:], strict=False)
    ), f"pressure not monotone approaching the hard core: {pressures}"
    assert pressures[-1] / pressures[0] > 100.0


def test_initial_state_is_a_static_equilibrium(argon_cfg):
    v_free = _v_free(argon_cfg, float(argon_cfg.r0))
    comp = pc.composition(argon_cfg.liquid.ambient_temperature, v_free, argon_cfg)
    p_gas = float(pc.pressure_from_composition(comp, argon_cfg))
    expected = (
        float(argon_cfg.liquid.ambient_pressure)
        - float(argon_cfg.liquid.vapour_pressure)
        + 2.0 * float(argon_cfg.liquid.surface_tension) / float(argon_cfg.r0)
    )
    assert p_gas == pytest.approx(expected, rel=1e-10)


def test_hardcore_radius_matches_the_conventional_alpha(argon_cfg):
    ratio = float(argon_cfg.r0 / argon_cfg.hardcore_radius)
    assert 7.5 < ratio < 10.0, f"R_0/h = {ratio:.3f}, expected ~8.9"


def test_heat_capacity_is_exactly_ideal_without_chemistry(argon_cfg):
    cfg = load_config(
        scenario="sbsl_argon_water",
        overrides={
            "transport.ionisation_enabled": False,
            "transport.dissociation_enabled": False,
        },
    )
    v_free = _v_free(cfg, float(cfg.r0))
    _, c_v, du_dv = pc.caloric_coefficients(
        jnp.asarray(293.15), v_free, cfg
    )
    assert float(c_v) == pytest.approx(
        1.5 * float(cfg.n_nuclei) * C.K_B, rel=1e-12
    )
    assert float(du_dv) == 0.0


@pytest.mark.parametrize("temperature", [3e3, 2e4, 1e5, 1e6, 1e7])
@pytest.mark.parametrize("scenario", ["sbsl_argon_water", "sonofusion_d_acetone"])
def test_caloric_derivatives_match_finite_differences(scenario, temperature):
    cfg = load_config(scenario=scenario)
    v_free = float(_v_free(cfg, 1e-6))
    _, c_v, du_dv = pc.caloric_coefficients(
        jnp.asarray(temperature), jnp.asarray(v_free), cfg
    )

    def u(t: float, v: float) -> float:
        return float(pc.internal_energy(jnp.asarray(t), jnp.asarray(v), cfg))

    h_t = temperature * 1e-6
    c_v_fd = (u(temperature + h_t, v_free) - u(temperature - h_t, v_free)) / (
        2.0 * h_t
    )
    assert float(c_v) == pytest.approx(c_v_fd, rel=1e-6)

    h_v = v_free * 1e-6
    du_dv_fd = (u(temperature, v_free + h_v) - u(temperature, v_free - h_v)) / (
        2.0 * h_v
    )
    comp = pc.composition(jnp.asarray(temperature), jnp.asarray(v_free), cfg)
    p_scale = float(pc.pressure_from_composition(comp, cfg))
    assert float(du_dv) == pytest.approx(
        du_dv_fd, rel=1e-5, abs=1e-8 * p_scale
    )


@pytest.mark.parametrize("scenario", ["sbsl_argon_water", "sonofusion_d_acetone"])
def test_pressure_derivatives_match_finite_differences(scenario):
    cfg = load_config(scenario=scenario)
    temperature, v_free = 3.0e4, float(_v_free(cfg, 1e-6))
    eos = pc.equation_of_state_and_derivatives(
        jnp.asarray(temperature), jnp.asarray(v_free), cfg
    )

    def p(t: float, v: float) -> float:
        comp = pc.composition(jnp.asarray(t), jnp.asarray(v), cfg)
        return float(pc.pressure_from_composition(comp, cfg))

    h_t = temperature * 1e-6
    h_v = v_free * 1e-6
    assert float(eos.dp_dt) == pytest.approx(
        (p(temperature + h_t, v_free) - p(temperature - h_t, v_free))
        / (2.0 * h_t),
        rel=1e-6,
    )
    assert float(eos.dp_dv) == pytest.approx(
        (p(temperature, v_free + h_v) - p(temperature, v_free - h_v))
        / (2.0 * h_v),
        rel=1e-6,
    )
    assert float(eos.pressure) == pytest.approx(p(temperature, v_free), rel=1e-12)


def test_ionisation_raises_the_effective_heat_capacity(argon_cfg):
    v_free = _v_free(argon_cfg, 1e-6)
    ratios = []
    for t in (3e3, 1e4, 3e4, 1e5):
        _, c_v, _ = pc.caloric_coefficients(jnp.asarray(t), v_free, argon_cfg)
        comp = pc.composition(jnp.asarray(t), v_free, argon_cfg)
        ideal = 1.5 * float(comp.n_translational) * C.K_B
        ratios.append(float(c_v) / ideal)
    assert ratios[0] == pytest.approx(1.0, abs=1e-3), (
        "no chemistry should be active at 3000 K"
    )
    assert max(ratios) > 2.0, f"no ionisation latent heat seen: {ratios}"


@pytest.mark.parametrize("scenario", ["sbsl_argon_water", "sonofusion_d_acetone"])
@pytest.mark.parametrize(
    "temperature", [1.0, 100.0, 300.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
)
@pytest.mark.parametrize("radius", [1e-7, 1e-6, 5e-5])
def test_caloric_quantities_stay_finite_everywhere(
    scenario, temperature, radius
):
    cfg = load_config(scenario=scenario)
    v_free = _v_free(cfg, radius)
    energy, c_v, du_dv = pc.caloric_coefficients(
        jnp.asarray(temperature), v_free, cfg
    )
    comp = pc.composition(jnp.asarray(temperature), v_free, cfg)
    values = [
        float(energy),
        float(c_v),
        float(du_dv),
        float(comp.n_e_density),
        float(comp.z_bar_overall),
        float(comp.alpha[0]),
        float(pc.pressure_from_composition(comp, cfg)),
    ]
    assert all(math.isfinite(v) for v in values), values
    assert float(c_v) > 0.0
    assert float(comp.n_e_density) >= 0.0


def test_single_stage_saha_matches_an_independent_solve(deuterium_cfg):
    cfg = deuterium_cfg
    for temperature in (1e4, 3e4, 1e5, 1e6):
        v_free = _v_free(cfg, 1e-6)
        comp = pc.composition(jnp.asarray(temperature), v_free, cfg)
        n_e = float(comp.n_e_density)
        n_free = float(jnp.sum(comp.n_free_atoms) / v_free)
        if n_free <= 0.0 or n_e <= 0.0:
            continue

        lam_e = C.H_PLANCK / math.sqrt(
            2.0 * math.pi * C.M_ELECTRON * C.K_B * temperature
        )
        chi = float(cfg.gas.ion_pot[0, 0])
        s1 = (
            float(cfg.gas.saha_g_ratio[0])
            * lam_e**-3
            * math.exp(-chi / (C.K_B * temperature))
        )
        residual = n_e**2 - s1 * (n_free - n_e)
        scale = max(n_e**2, s1 * n_free)
        assert abs(residual) / scale < 1e-10, (
            f"T={temperature:.1e}: Saha residual {residual:.3e}"
        )

        reference = sp_optimize.brentq(
            lambda x, _s=s1, _n=n_free: x**2 - _s * (_n - x),
            0.0,
            n_free,
            xtol=1e-30,
        )
        assert n_e == pytest.approx(reference, rel=1e-9)


def test_multi_stage_saha_satisfies_charge_neutrality(argon_cfg):
    v_free = _v_free(argon_cfg, 1e-6)
    for temperature in (2e4, 1e5, 1e6, 1e7):
        comp = pc.composition(jnp.asarray(temperature), v_free, argon_cfg)
        n_e = float(comp.n_e_density)
        predicted = float(jnp.sum(comp.n_free_atoms * comp.z_mean) / v_free)
        assert n_e == pytest.approx(predicted, rel=1e-8), (
            f"T={temperature:.1e}: n_e={n_e:.6e} vs n_free <Z>={predicted:.6e}"
        )


def test_mean_ionisation_is_monotone_and_saturates(argon_cfg):
    v_free = _v_free(argon_cfg, 1e-6)
    temps = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7, 1e8]
    z = [
        float(pc.composition(jnp.asarray(t), v_free, argon_cfg).z_bar_overall)
        for t in temps
    ]
    assert all(b >= a for a, b in zip(z, z[1:], strict=False)), f"Zbar not monotone: {z}"
    z_max = float(argon_cfg.gas.ion_pot.shape[1])
    assert z[-1] == pytest.approx(z_max, rel=1e-3), (
        f"argon should be fully stripped at 1e8 K, got Zbar={z[-1]:.4f}"
    )
    assert z[0] < 1e-10, "argon must be essentially neutral at 1000 K"


def test_truncated_ladder_matches_full_ladder_in_its_validity_range():
    full = load_config(scenario="sbsl_argon_water")
    trunc = load_config(
        scenario="sbsl_argon_water",
        overrides={"gas_species.argon.max_ionisation_stages": 1},
    )
    assert trunc.gas.ion_pot.shape == (1, 1), "truncation did not take effect"

    v_full = _v_free(full, 1e-6)
    v_trunc = _v_free(trunc, 1e-6)

    budget = {
        1.0e4: 1.0e-9,
        2.0e4: 1.0e-4,
        3.0e4: 2.0e-3,
        4.0e4: 1.0e-2,
        5.0e4: 4.0e-2,
    }
    for temperature, tol in budget.items():
        a = float(
            pc.composition(jnp.asarray(temperature), v_full, full).n_e_density
        )
        b = float(
            pc.composition(jnp.asarray(temperature), v_trunc, trunc).n_e_density
        )
        rel = abs(b - a) / a
        assert rel < tol, (
            f"T={temperature:.1e}: truncated n_e differs by {rel:.3e}, "
            f"above the documented budget {tol:.1e}"
        )

    a = float(pc.composition(jnp.asarray(1e6), v_full, full).n_e_density)
    b = float(pc.composition(jnp.asarray(1e6), v_trunc, trunc).n_e_density)
    assert b < 0.5 * a, (
        "a single-stage ladder must badly under-predict n_e at 1e6 K; "
        f"got {b:.3e} vs {a:.3e}"
    )


def test_saha_solution_is_differentiable(deuterium_cfg):
    cfg = deuterium_cfg
    v_free = _v_free(cfg, 1e-6)

    def n_e(t):
        return pc.composition(t, v_free, cfg).n_e_density

    t0 = jnp.asarray(5.0e4)
    grad = float(jax.grad(n_e)(t0))
    h = 5.0
    fd = (float(n_e(t0 + h)) - float(n_e(t0 - h))) / (2.0 * h)
    assert math.isfinite(grad)
    assert grad > 0.0, "ionisation must increase with temperature"
    assert grad == pytest.approx(fd, rel=1e-5)


def test_dissociation_fraction_has_the_right_limits(deuterium_cfg):
    cfg = deuterium_cfg
    v_free = _v_free(cfg, 1e-6)
    temps = [100.0, 1e3, 3e3, 5e3, 1e4, 3e4, 1e5, 1e6, 1e7]
    alphas = [
        float(
            pc.dissociation_fraction(
                jnp.asarray(t), v_free, cfg.gas, cfg.n_nuclei, True
            )[0]
        )
        for t in temps
    ]
    assert all(0.0 <= a <= 1.0 for a in alphas), alphas
    assert all(b >= a for a, b in zip(alphas, alphas[1:], strict=False)), (
        f"alpha not monotone in T: {dict(zip(temps, alphas, strict=False))}"
    )
    assert alphas[0] < 1e-6, "D2 must be molecular at 100 K"
    assert alphas[-1] > 0.9, (
        f"D2 must be essentially atomic at 1e7 K, got {alphas[-1]:.4f}"
    )


def test_dissociation_satisfies_the_law_of_mass_action(deuterium_cfg):
    cfg = deuterium_cfg
    v_free = float(_v_free(cfg, 1e-6))
    temperature = 6.0e3
    comp = pc.composition(jnp.asarray(temperature), jnp.asarray(v_free), cfg)
    n_atoms = float(comp.n_free_atoms[0]) / v_free
    n_mol = float(comp.n_molecules[0]) / v_free
    assert n_mol > 0.0 and n_atoms > 0.0

    m_atom = float(cfg.gas.molar_mass_atomic[0]) / C.N_AVOGADRO
    lam = C.H_PLANCK / math.sqrt(
        2.0 * math.pi * m_atom * C.K_B * temperature
    )
    theta_rot = float(cfg.gas.theta_rot[0])
    theta_vib = float(cfg.gas.theta_vib[0])
    e_diss = float(cfg.gas.e_diss[0])
    n_levels = math.floor(e_diss / (C.K_B * theta_vib)) + 1.0
    x = theta_vib / temperature
    z_vib = (1.0 - math.exp(-n_levels * x)) / (1.0 - math.exp(-x))
    k_d = (
        float(cfg.gas.diss_g_ratio[0])
        / (2.0**1.5 * lam**3)
        * (2.0 * theta_rot / temperature)
        / z_vib
        * math.exp(-e_diss / (C.K_B * temperature))
    )
    assert n_levels == 12.0, "D2 should have 12 bound vibrational levels"
    assert n_atoms**2 / n_mol == pytest.approx(k_d, rel=1e-8)


def test_monatomic_species_are_always_fully_atomic(argon_cfg):
    v_free = _v_free(argon_cfg, 1e-6)
    for t in (1.0, 300.0, 1e5):
        comp = pc.composition(jnp.asarray(t), v_free, argon_cfg)
        assert float(comp.alpha[0]) == 1.0
        assert float(comp.n_molecules[0]) == 0.0
        assert float(comp.e_dissociation) == 0.0


def test_bremsstrahlung_matches_the_nrl_formula(deuterium_cfg):
    cfg = deuterium_cfg
    v_free = _v_free(cfg, 1e-6)
    temperature = 1.0e6
    comp = pc.composition(jnp.asarray(temperature), v_free, cfg)
    got = float(pc.bremsstrahlung_power_density(comp, cfg))
    expected = (
        1.69e-38
        * float(comp.n_e_density)
        * float(comp.n_zsq_density)
        * math.sqrt(temperature / C.KELVIN_PER_EV)
    )
    assert got == pytest.approx(expected, rel=1e-10)
    assert got > 0.0


def test_bremsstrahlung_scales_as_n_squared_sqrt_t(deuterium_cfg):
    cfg = deuterium_cfg
    temperature = 1.0e7
    radii = [1e-6, 2e-6]
    densities, powers = [], []
    for r in radii:
        v_free = _v_free(cfg, r)
        comp = pc.composition(jnp.asarray(temperature), v_free, cfg)
        densities.append(float(comp.n_e_density))
        powers.append(float(pc.bremsstrahlung_power_density(comp, cfg)))
    expected_ratio = (densities[1] / densities[0]) ** 2
    assert powers[1] / powers[0] == pytest.approx(expected_ratio, rel=1e-3)

    v_free = _v_free(cfg, 1e-6)
    reduced = {}
    for t in (1e7, 4e7):
        comp = pc.composition(jnp.asarray(t), v_free, cfg)
        power = float(pc.bremsstrahlung_power_density(comp, cfg))
        reduced[t] = power / (
            float(comp.n_e_density) * float(comp.n_zsq_density)
        )
    assert reduced[4e7] / reduced[1e7] == pytest.approx(2.0, rel=1e-6)


def test_blackbody_cap_bounds_the_radiative_loss(deuterium_cfg):
    cfg = deuterium_cfg
    radius = 1.0e-6
    for temperature in (1e5, 1e6, 1e7, 1e8):
        v_free = _v_free(cfg, radius)
        comp = pc.composition(jnp.asarray(temperature), v_free, cfg)
        thin = float(pc.bremsstrahlung_power_density(comp, cfg)) * float(v_free)
        capped = float(pc.radiative_loss(jnp.asarray(radius), comp, cfg))
        black = (
            4.0 * math.pi * radius**2 * C.SIGMA_SB * temperature**4
        )
        assert capped <= min(thin, black) * 1.001, (
            f"T={temperature:.1e}: capped loss {capped:.3e} exceeds "
            f"min(thin={thin:.3e}, blackbody={black:.3e})"
        )
        assert capped > 0.0


def test_soft_min_approximates_the_hard_minimum():
    sharpness = jnp.asarray(8.0)
    for a, b in [(1.0, 1.0), (1.0, 2.0), (1.0, 100.0), (1e-9, 1e3)]:
        got = float(pc.soft_min(jnp.asarray(a), jnp.asarray(b), sharpness))
        hard = min(a, b)
        assert got <= hard * (1.0 + 1e-12)
        assert got >= hard * 2.0 ** (-1.0 / 8.0) * (1.0 - 1e-12)
    d = jax.grad(lambda x: pc.soft_min(x, jnp.asarray(1.0), sharpness))(
        jnp.asarray(1.0)
    )
    assert math.isfinite(float(d))


def test_conduction_carries_heat_down_the_gradient(argon_cfg):
    radius, rdot = jnp.asarray(1e-6), jnp.asarray(-100.0)
    v_free = _v_free(argon_cfg, 1e-6)

    hot = pc.composition(jnp.asarray(5.0e4), v_free, argon_cfg)
    cold = pc.composition(jnp.asarray(50.0), v_free, argon_cfg)
    q_hot = float(pc.conduction_loss(radius, rdot, hot, argon_cfg))
    q_cold = float(pc.conduction_loss(radius, rdot, cold, argon_cfg))
    assert q_hot > 0.0, "a hot core must lose heat"
    assert q_cold < 0.0, "a core below T_inf must gain heat"


def test_boundary_layer_is_capped_by_the_bubble_radius(argon_cfg):
    v_free = _v_free(argon_cfg, 1e-6)
    comp = pc.composition(jnp.asarray(1e4), v_free, argon_cfg)
    cap = 1e-6 / float(argon_cfg.transport.boundary_layer_cap_factor)
    for rdot in (1e-12, 1e-6, 1.0, 100.0, 1000.0):
        ell = float(
            pc.boundary_layer_thickness(
                jnp.asarray(1e-6), jnp.asarray(-rdot), comp, argon_cfg
            )
        )
        assert 0.0 < ell <= cap * 1.001, f"Rdot={rdot}: ell={ell:.3e} > {cap:.3e}"


def test_flux_limiter_bounds_the_heat_flux(argon_cfg):
    radius, rdot = jnp.asarray(6e-7), jnp.asarray(-1000.0)
    v_free = _v_free(argon_cfg, 6e-7)
    unlimited_cfg = load_config(
        scenario="sbsl_argon_water",
        overrides={"transport.flux_limiter_enabled": False},
    )

    def ratio(temperature: float) -> float:
        comp = pc.composition(jnp.asarray(temperature), v_free, argon_cfg)
        limited = float(pc.conduction_loss(radius, rdot, comp, argon_cfg))
        raw = float(pc.conduction_loss(radius, rdot, comp, unlimited_cfg))
        assert 0.0 < limited <= raw, (
            f"T={temperature:.1e}: limiter increased the flux "
            f"({limited:.3e} vs {raw:.3e})"
        )
        return limited / raw

    assert ratio(1.0e5) > 0.9, "limiter should barely act at 1e5 K"
    assert ratio(1.0e7) < 0.85, "limiter should be clearly active at 1e7 K"
    assert ratio(3.0e7) < 0.5, (
        "at 3e7 K the free-streaming bound must cut the Spitzer flux by "
        "more than half"
    )


_BH_REFERENCE_CM3S = {
    "DD_pT": {1.0: 1.0166e-22, 10.0: 5.7813e-19, 100.0: 2.2439e-17},
    "DD_n3He": {1.0: 9.9330e-23, 10.0: 6.0227e-19, 100.0: 2.6817e-17},
}


@pytest.mark.parametrize("reaction", ["DD_pT", "DD_n3He"])
def test_bosch_hale_reproduces_reference_values(reaction):
    for t_keV, expected in _BH_REFERENCE_CM3S[reaction].items():
        got = float(pc.bosch_hale_reactivity(jnp.asarray(t_keV), reaction))
        assert got * 1e6 == pytest.approx(expected, rel=1e-4), (
            f"{reaction} at {t_keV} keV: {got * 1e6:.5e} vs {expected:.5e} cm^3/s"
        )


def test_dd_total_reactivity_matches_the_textbook_value():
    total = sum(
        float(pc.bosch_hale_reactivity(jnp.asarray(10.0), r)) * 1e6
        for r in ("DD_pT", "DD_n3He")
    )
    assert total == pytest.approx(1.18e-18, rel=0.03)


@pytest.mark.parametrize("reaction", ["DD_pT", "DD_n3He"])
def test_reactivity_is_monotone_over_the_validity_window(reaction):
    temps = np.geomspace(0.5, 190.0, 60)
    values = np.array(
        [float(pc.bosch_hale_reactivity(jnp.asarray(t), reaction)) for t in temps]
    )
    assert np.all(values > 0.0)
    assert np.all(np.diff(values) > 0.0), "reactivity is not monotone in T"


@pytest.mark.parametrize("reaction", ["DD_pT", "DD_n3He"])
def test_reactivity_follows_the_gamow_form_at_low_temperature(reaction):
    bg2 = 31.3970**2
    for t_keV in (0.5, 1.0, 2.0):
        h = 1e-4 * t_keV
        up = float(pc.bosch_hale_reactivity(jnp.asarray(t_keV + h), reaction))
        dn = float(pc.bosch_hale_reactivity(jnp.asarray(t_keV - h), reaction))
        slope = (math.log(up) - math.log(dn)) / (
            math.log(t_keV + h) - math.log(t_keV - h)
        )
        gamow = (bg2 / (4.0 * t_keV)) ** (1.0 / 3.0)
        assert slope == pytest.approx(gamow - 2.0 / 3.0, rel=0.05), (
            f"{reaction} at {t_keV} keV: log-slope {slope:.3f} vs "
            f"Gamow {gamow - 2.0 / 3.0:.3f}"
        )


def test_unknown_reaction_raises():
    with pytest.raises(KeyError, match="unknown reaction"):
        pc.bosch_hale_reactivity(jnp.asarray(10.0), "DT")


def test_fusion_rate_has_the_identical_particle_half(deuterium_cfg):
    cfg = deuterium_cfg
    v_free = _v_free(cfg, 1e-6)
    temperature = 1.0e8
    comp = pc.composition(jnp.asarray(temperature), v_free, cfg)
    rates, neutron_rate, deposited = pc.fusion_rates(comp, cfg)

    n_d = float(comp.n_deuteron_density)
    for name, rate in rates.items():
        sv = float(
            pc.bosch_hale_reactivity(
                jnp.asarray(temperature / C.KELVIN_PER_KEV), name
            )
        )
        expected = 0.5 * n_d**2 * sv * float(v_free)
        assert float(rate) == pytest.approx(expected, rel=1e-10), name

    assert float(neutron_rate) == pytest.approx(
        float(rates["DD_n3He"]), rel=1e-12
    )
    assert float(deposited) == 0.0


def test_no_fusion_without_deuterium(argon_cfg):
    v_free = _v_free(argon_cfg, 1e-6)
    comp = pc.composition(jnp.asarray(1.0e8), v_free, argon_cfg)
    assert float(comp.n_deuteron_density) == 0.0
    _rates, neutron_rate, deposited = pc.fusion_rates(comp, argon_cfg)
    assert float(neutron_rate) == 0.0
    assert float(deposited) == 0.0


def test_charged_product_deposition_is_opt_in(deuterium_cfg):
    cfg = load_config(
        scenario="sonofusion_d_acetone",
        overrides={"fusion.charged_product_deposition_fraction": 1.0},
    )
    v_free = _v_free(cfg, 1e-6)
    comp = pc.composition(jnp.asarray(1.0e8), v_free, cfg)
    rates, _neutron, deposited = pc.fusion_rates(comp, cfg)
    expected = (
        float(rates["DD_pT"]) * C.E_CHARGED_DD_PT_MEV
        + float(rates["DD_n3He"]) * C.E_CHARGED_DD_N3HE_MEV
    ) * C.MEV_TO_J
    assert float(deposited) == pytest.approx(expected, rel=1e-10)
    assert float(deposited) > 0.0


def test_bosch_hale_validity_flag(deuterium_cfg):
    lo = float(deuterium_cfg.fusion.t_valid_min_keV) * C.KELVIN_PER_KEV
    hi = float(deuterium_cfg.fusion.t_valid_max_keV) * C.KELVIN_PER_KEV
    assert not bool(
        pc.bosch_hale_in_range(jnp.asarray(0.5 * lo), deuterium_cfg)
    )
    assert bool(pc.bosch_hale_in_range(jnp.asarray(2.0 * lo), deuterium_cfg))
    assert bool(pc.bosch_hale_in_range(jnp.asarray(0.5 * hi), deuterium_cfg))
    assert not bool(
        pc.bosch_hale_in_range(jnp.asarray(2.0 * hi), deuterium_cfg)
    )


def test_energy_budget_signs_and_consistency(argon_cfg):
    radius, temperature = jnp.asarray(1e-6), jnp.asarray(1e4)
    compressing, _ = pc.energy_budget(
        radius, jnp.asarray(-500.0), temperature, argon_cfg
    )
    expanding, _ = pc.energy_budget(
        radius, jnp.asarray(+500.0), temperature, argon_cfg
    )
    assert float(compressing.temperature_dot) > 0.0
    assert float(expanding.temperature_dot) < 0.0
    assert float(compressing.pressure) > 0.0
    assert float(compressing.heat_capacity) > 0.0


def test_energy_budget_is_zero_for_an_empty_cavity():
    cfg = load_config(scenario="rayleigh_validation")
    budget, comp = pc.energy_budget(
        jnp.asarray(1e-6), jnp.asarray(-500.0), jnp.asarray(1e4), cfg
    )
    assert float(budget.pressure) == 0.0
    assert float(budget.temperature_dot) == 0.0
    assert float(budget.conduction) == 0.0
    assert float(budget.radiation) == 0.0
    assert float(budget.neutron_rate) == 0.0
    assert float(comp.n_e_density) == 0.0


def test_energy_equation_reproduces_the_first_law(argon_cfg):
    radius, rdot, temperature = (
        jnp.asarray(8e-7),
        jnp.asarray(-800.0),
        jnp.asarray(3e4),
    )
    budget, _comp = pc.energy_budget(radius, rdot, temperature, argon_cfg)
    v_dot = 4.0 * math.pi * float(radius) ** 2 * float(rdot)
    reconstructed = (
        -(float(budget.pressure) + float(budget.internal_pressure)) * v_dot
        - float(budget.conduction)
        - float(budget.radiation)
        + float(budget.fusion_deposition)
    ) / float(budget.heat_capacity)
    assert float(budget.temperature_dot) == pytest.approx(
        reconstructed, rel=1e-10
    )


def test_total_gas_mass_matches_species_bookkeeping(argon_cfg):
    expected = float(
        np.sum(
            np.asarray(argon_cfg.gas.mole_frac)
            * float(argon_cfg.n_nuclei)
            * np.asarray(argon_cfg.gas.mass_per_nucleus)
        )
    )
    assert float(pc.total_gas_mass(argon_cfg)) == pytest.approx(expected, rel=1e-14)
    per_nucleus = float(pc.total_gas_mass(argon_cfg)) / float(argon_cfg.n_nuclei)
    assert per_nucleus / C.ATOMIC_MASS_UNIT == pytest.approx(39.948, rel=1e-4)


@pytest.mark.parametrize("radius_um", [10.0, 5.0, 2.0, 1.2, 1.0])
def test_sound_speed_matches_the_exact_hardcore_result(argon_cfg, radius_um):
    volume = (4.0 / 3.0) * math.pi * (radius_um * 1e-6) ** 3
    v_hc = float(argon_cfg.n_nuclei * argon_cfg.gas.hardcore_volume_per_nucleus)
    v_free = jnp.asarray(volume - v_hc)
    temperature = jnp.asarray(300.0)

    comp = pc.composition(temperature, v_free, argon_cfg)
    eos = pc.equation_of_state_and_derivatives(temperature, v_free, argon_cfg)
    c_model = float(pc.sound_speed_gas(comp, eos, argon_cfg))

    mass_per = float(pc.total_gas_mass(argon_cfg)) / float(argon_cfg.n_nuclei)
    c_exact = math.sqrt((5.0 / 3.0) * C.K_B * 300.0 / mass_per) / (
        1.0 - v_hc / volume
    )
    assert c_model == pytest.approx(c_exact, rel=1e-12)


def test_sound_speed_is_lowered_by_the_ionisation_latent_heat(argon_cfg):
    v_free = pc.free_volume(jnp.asarray(2.0e-6), argon_cfg)
    hot = jnp.asarray(5.0e4)
    comp = pc.composition(hot, v_free, argon_cfg)
    eos = pc.equation_of_state_and_derivatives(hot, v_free, argon_cfg)

    n_translational = float(comp.n_translational)
    cv_ideal = 1.5 * n_translational * C.K_B
    assert float(eos.du_dt) > 2.0 * cv_ideal, "ionisation latent heat missing"

    c_model = float(pc.sound_speed_gas(comp, eos, argon_cfg))
    frozen = dataclasses.replace(eos, du_dt=jnp.asarray(cv_ideal))
    c_frozen = float(pc.sound_speed_gas(comp, frozen, argon_cfg))
    assert c_model < c_frozen


def test_sound_speed_is_zero_without_gas():
    cfg = load_config(scenario="rayleigh_validation")
    comp = pc.composition(jnp.asarray(1e4), jnp.asarray(1e-18), cfg)
    eos = pc.equation_of_state_and_derivatives(
        jnp.asarray(1e4), jnp.asarray(1e-18), cfg
    )
    assert float(pc.sound_speed_gas(comp, eos, cfg)) == 0.0
    assert (
        float(pc.internal_mach_number(jnp.asarray(-500.0), comp, eos, cfg)) == 0.0
    )


def test_internal_mach_is_the_wall_speed_over_the_gas_sound_speed(argon_cfg):
    v_free = pc.free_volume(jnp.asarray(1.5e-6), argon_cfg)
    temperature = jnp.asarray(1.5e4)
    comp = pc.composition(temperature, v_free, argon_cfg)
    eos = pc.equation_of_state_and_derivatives(temperature, v_free, argon_cfg)
    c_gas = float(pc.sound_speed_gas(comp, eos, argon_cfg))

    for rdot in (-900.0, 900.0):
        mach = float(
            pc.internal_mach_number(jnp.asarray(rdot), comp, eos, argon_cfg)
        )
        assert mach == pytest.approx(abs(rdot) / c_gas, rel=1e-12)


def test_internal_mach_differs_from_the_liquid_mach(argon_cfg):
    budget, _comp = pc.energy_budget(
        jnp.asarray(4.0e-6), jnp.asarray(-900.0), jnp.asarray(1.0e3), argon_cfg
    )
    liquid_mach = 900.0 / float(argon_cfg.liquid.sound_speed)
    ratio = float(budget.internal_mach) / liquid_mach
    assert ratio == pytest.approx(
        float(argon_cfg.liquid.sound_speed) / float(budget.sound_speed_gas),
        rel=1e-12,
    )
    assert ratio > 2.0


def test_energy_budget_reports_a_finite_subsonic_core_at_rest(argon_cfg):
    budget, _comp = pc.energy_budget(
        jnp.asarray(4.5e-6), jnp.asarray(0.0), jnp.asarray(300.0), argon_cfg
    )
    assert float(budget.internal_mach) == 0.0
    assert math.isfinite(float(budget.sound_speed_gas))
    assert float(budget.sound_speed_gas) > 0.0


def test_escape_factor_is_exact_in_both_limits():
    thick = jnp.asarray(1.0)
    for ratio in (1e-12, 1e-8, 1e-4):
        thin = jnp.asarray(ratio)
        assert float(pc.escape_factor(thin, thick)) == pytest.approx(
            ratio * (1.0 - 0.5 * ratio), rel=1e-9
        )
    for ratio in (50.0, 500.0):
        assert float(
            pc.escape_factor(jnp.asarray(ratio), thick)
        ) == pytest.approx(1.0, rel=1e-12)


def test_escape_factor_never_exceeds_either_limit():
    ratios = jnp.asarray(np.logspace(-10, 10, 201))
    thick = jnp.asarray(3.7)
    emitted = np.asarray(pc.escape_factor(ratios * thick, thick))
    assert np.all(emitted <= np.asarray(ratios * thick) * (1.0 + 1e-12))
    assert np.all(emitted <= float(thick) * (1.0 + 1e-12))
    assert np.all(np.diff(emitted) >= 0.0)
    assert emitted[-1] == pytest.approx(float(thick), rel=1e-12)


def test_planck_mean_absorption_is_consistent_with_the_emissivity(argon_cfg):
    v_free = _v_free(argon_cfg, 1.0e-6)
    for temperature in (3.0e4, 1.0e5):
        comp = pc.composition(jnp.asarray(temperature), v_free, argon_cfg)
        kappa = float(pc.planck_mean_absorption(comp, argon_cfg))
        emissivity = float(
            pc.bremsstrahlung_power_density(comp, argon_cfg)
        )
        assert emissivity == pytest.approx(
            4.0 * C.SIGMA_SB * temperature**4 * kappa, rel=1e-12
        )
        assert kappa > 0.0


def test_sbsl_core_is_optically_thin(argon_cfg):
    v_free = _v_free(argon_cfg, 1.0e-6)
    comp = pc.composition(jnp.asarray(2.7e4), v_free, argon_cfg)
    thin = float(pc.bremsstrahlung_power_density(comp, argon_cfg) * comp.v_free)
    actual = float(pc.radiative_loss(jnp.asarray(1.0e-6), comp, argon_cfg))
    assert actual / thin > 0.98
