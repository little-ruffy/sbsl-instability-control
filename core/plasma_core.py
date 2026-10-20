from __future__ import annotations

from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float

from core import constants as C
from core.config import GasMixture, SimConfig, TransportParams


# Bosch-Hale (1992) D-D reactivity fit parameters
_BOSCH_HALE: Final[dict[str, dict[str, float]]] = {
    "DD_pT": {
        "bg": 31.3970,
        "mrc2": 937814.0,
        "c1": 5.65718e-12,
        "c2": 3.41267e-3,
        "c3": 1.99167e-3,
        "c4": 0.0,
        "c5": 1.05060e-5,
        "c6": 0.0,
        "c7": 0.0,
        "e_charged_mev": C.E_CHARGED_DD_PT_MEV,
        "neutron_yield": 0.0,
    },
    "DD_n3He": {
        "bg": 31.3970,
        "mrc2": 937814.0,
        "c1": 5.43360e-12,
        "c2": 5.85778e-3,
        "c3": 7.68222e-3,
        "c4": 0.0,
        "c5": -2.96400e-6,
        "c6": 0.0,
        "c7": 0.0,
        "e_charged_mev": C.E_CHARGED_DD_N3HE_MEV,
        "neutron_yield": 1.0,
    },
}

BOSCH_HALE_REACTIONS: Final[tuple[str, ...]] = tuple(_BOSCH_HALE)
_LN_ZERO: Final[float] = -1.0e30
_LOG_SAHA_FLOOR: Final[float] = -200.0


def _electron_density_floor(
    n_free_atoms_total: Float[Array, ""]
) -> Float[Array, ""]:
    return jnp.maximum(n_free_atoms_total, 1.0e-30) * jnp.exp(_LOG_SAHA_FLOOR)


def thermal_wavelength(
    mass: Float[Array, "..."], temperature: Float[Array, "..."]
) -> Float[Array, "..."]:
    return C.H_PLANCK / jnp.sqrt(2.0 * jnp.pi * mass * C.K_B * temperature)


def soft_min(
    a: Float[Array, ""], b: Float[Array, ""], sharpness: Float[Array, ""]
) -> Float[Array, ""]:
    la = jnp.log(jnp.maximum(a, 1.0e-300))
    lb = jnp.log(jnp.maximum(b, 1.0e-300))
    stacked = jnp.stack([-sharpness * la, -sharpness * lb])
    return jnp.exp(-logsumexp(stacked) / sharpness)


def vibrational_levels(
    e_diss: Float[Array, " S"], theta_vib: Float[Array, " S"]
) -> Float[Array, " S"]:
    quanta = e_diss / (C.K_B * theta_vib)
    return jnp.maximum(jnp.floor(quanta), 0.0) + 1.0


def log_z_vibrational(
    temperature: Float[Array, ""],
    theta_vib: Float[Array, " S"],
    n_levels: Float[Array, " S"],
) -> Float[Array, " S"]:
    x = jnp.clip(theta_vib / temperature, 1.0e-300, 700.0)
    nx = jnp.clip(n_levels * x, 1.0e-300, 700.0)
    return jnp.log(-jnp.expm1(-nx)) - jnp.log(-jnp.expm1(-x))


def vibrational_energy_per_molecule(
    temperature: Float[Array, ""],
    theta_vib: Float[Array, " S"],
    n_levels: Float[Array, " S"],
) -> Float[Array, " S"]:
    x = jnp.clip(theta_vib / temperature, 1.0e-300, 700.0)
    nx = jnp.clip(n_levels * x, 1.0e-300, 700.0)
    return C.K_B * theta_vib * (
        1.0 / jnp.expm1(x) - n_levels / jnp.expm1(nx)
    )


def species_counts(
    cfg: SimConfig, n_per_species: Float[Array, " S"] | None
) -> Float[Array, " S"]:
    if n_per_species is None:
        return cfg.gas.mole_frac * cfg.n_nuclei
    return n_per_species


def hardcore_volume(
    n_per_species: Float[Array, " S"], cfg: SimConfig
) -> Float[Array, ""]:
    return jnp.sum(n_per_species * cfg.gas.hardcore_volume_per_nucleus_species)


def free_volume(
    radius: Float[Array, ""],
    cfg: SimConfig,
    n_per_species: Float[Array, " S"] | None = None,
) -> Float[Array, ""]:
    v_total = (4.0 / 3.0) * jnp.pi * radius**3
    counts = species_counts(cfg, n_per_species)
    v_hc = hardcore_volume(counts, cfg)
    v_hc_min = (4.0 / 3.0) * jnp.pi * cfg.r_floor**3 - v_hc
    return jnp.maximum(v_total - v_hc, jnp.maximum(v_hc_min, 1.0e-300))


# law of mass action in log-space (s-variable quadratic root)
def dissociation_fraction(
    temperature: Float[Array, ""],
    v_free: Float[Array, ""],
    gas: GasMixture,
    n_per_species: Float[Array, " S"],
    enabled: bool,
) -> Float[Array, " S"]:
    is_diatomic = gas.n_atoms > 1.5
    if not enabled:
        return jnp.where(is_diatomic, 0.0, 1.0)

    lam_a = thermal_wavelength(gas.mass_per_nucleus, temperature)
    log_z_rot_inv = jnp.log(2.0 * gas.theta_rot / temperature)
    n_levels = vibrational_levels(gas.e_diss, gas.theta_vib)
    log_z_vib_inv = -log_z_vibrational(temperature, gas.theta_vib, n_levels)
    log_k_d = (
        jnp.log(gas.diss_g_ratio)
        - 1.5 * jnp.log(2.0)
        - 3.0 * jnp.log(lam_a)
        + log_z_rot_inv
        + log_z_vib_inv
        - gas.e_diss / (C.K_B * temperature)
    )
    n_species = jnp.maximum(n_per_species, 1.0e-30)
    log_lambda = log_k_d + jnp.log(v_free) - jnp.log(2.0) - jnp.log(n_species)

    s_param = jnp.exp(jnp.clip(0.5 * log_lambda, -700.0, 700.0))
    alpha = 2.0 * s_param / (s_param + jnp.sqrt(s_param**2 + 4.0))
    return jnp.where(is_diatomic, jnp.clip(alpha, 0.0, 1.0), 1.0)


def _saha_weights(
    log_ne: Float[Array, ""],
    temperature: Float[Array, ""],
    gas: GasMixture,
) -> tuple[Float[Array, " S"], Float[Array, " S"], Float[Array, " S"]]:
    lam_e = thermal_wavelength(C.M_ELECTRON, temperature)
    log_s = (
        jnp.log(gas.saha_g_ratio)[:, None]
        - 3.0 * jnp.log(lam_e)
        - gas.ion_pot / (C.K_B * temperature)
    )
    log_s = jnp.where(gas.ion_mask > 0.5, log_s, _LN_ZERO)

    log_w = jnp.concatenate(
        [
            jnp.zeros((gas.ion_pot.shape[0], 1)),
            jnp.cumsum(log_s - log_ne, axis=1),
        ],
        axis=1,
    )
    chi_cum = jnp.concatenate(
        [
            jnp.zeros((gas.ion_pot.shape[0], 1)),
            jnp.cumsum(gas.ion_pot * gas.ion_mask, axis=1),
        ],
        axis=1,
    )
    j_idx = jnp.arange(log_w.shape[1], dtype=log_w.dtype)
    log_j = jnp.where(j_idx > 0, jnp.log(jnp.maximum(j_idx, 1.0)), _LN_ZERO)

    log_den = logsumexp(log_w, axis=1)
    log_z1 = logsumexp(log_w + log_j, axis=1) - log_den
    log_z2 = logsumexp(log_w + 2.0 * log_j, axis=1) - log_den
    chi_mean = jnp.sum(chi_cum * jax.nn.softmax(log_w, axis=1), axis=1)
    return log_z1, log_z2, chi_mean


def _saha_residual(
    log_ne: Float[Array, ""],
    temperature: Float[Array, ""],
    n_free_atoms_density: Float[Array, " S"],
    gas: GasMixture,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    log_z1, log_z2, _ = _saha_weights(log_ne, temperature, gas)
    log_nf = jnp.where(
        n_free_atoms_density > 0.0,
        jnp.log(jnp.maximum(n_free_atoms_density, 1.0e-300)),
        _LN_ZERO,
    )
    log_contrib = log_nf + log_z1
    residual = log_ne - logsumexp(log_contrib)
    var_over_mean = jnp.exp(log_z2 - log_z1) - jnp.exp(log_z1)
    derivative = 1.0 + jnp.sum(jax.nn.softmax(log_contrib) * var_over_mean)
    return residual, derivative


# Saha ladder solver: bisection + analytic-Jacobian Newton + 1-step implicit AD
def solve_electron_density(
    temperature: Float[Array, ""],
    n_free_atoms_density: Float[Array, " S"],
    gas: GasMixture,
    transport: TransportParams,
) -> Float[Array, ""]:
    n_f_total = jnp.sum(n_free_atoms_density)

    if gas.ion_pot.shape == (1, 1):
        lam_e = thermal_wavelength(C.M_ELECTRON, temperature)
        log_s1 = (
            jnp.log(gas.saha_g_ratio[0])
            - 3.0 * jnp.log(lam_e)
            - gas.ion_pot[0, 0] / (C.K_B * temperature)
        )
        u = jnp.exp(jnp.clip(0.5 * log_s1, -700.0, 700.0))
        nf = jnp.maximum(n_f_total, 0.0)
        n_e = 2.0 * u * nf / (u + jnp.sqrt(u**2 + 4.0 * nf))
        return jnp.maximum(n_e, _electron_density_floor(n_f_total))

    z_max = float(gas.ion_pot.shape[1])
    log_nf = jax.lax.stop_gradient(
        jnp.log(jnp.maximum(n_f_total, 1.0e-30))
    )
    x_lo = log_nf + _LOG_SAHA_FLOOR
    x_hi = log_nf + jnp.log(z_max) + 1.0e-12

    t_sg = jax.lax.stop_gradient(temperature)
    nf_sg = jax.lax.stop_gradient(n_free_atoms_density)
    gas_sg = jax.lax.stop_gradient(gas)

    def bisect_body(
        _: int, bounds: tuple[Float[Array, ""], Float[Array, ""]]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        g, _unused = _saha_residual(mid, t_sg, nf_sg, gas_sg)
        return jnp.where(g < 0.0, mid, lo), jnp.where(g < 0.0, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0, transport.saha_bisection_iters, bisect_body, (x_lo, x_hi)
    )

    def newton_body(_: int, xi: Float[Array, ""]) -> Float[Array, ""]:
        g, dg = _saha_residual(xi, t_sg, nf_sg, gas_sg)
        return jnp.clip(xi - g / dg, x_lo, x_hi)

    x = jax.lax.fori_loop(
        0, transport.saha_newton_iters, newton_body, 0.5 * (lo + hi)
    )

    x = jax.lax.stop_gradient(x)
    g, dg = _saha_residual(x, temperature, n_free_atoms_density, gas)
    x = jnp.clip(x - g / dg, x_lo, x_hi)
    return jnp.exp(x)


class CoreComposition(eqx.Module):
    temperature: Float[Array, ""]
    v_free: Float[Array, ""]
    n_per_species: Float[Array, " S"]
    alpha: Float[Array, " S"]
    n_molecules: Float[Array, " S"]
    n_free_atoms: Float[Array, " S"]
    n_electrons: Float[Array, ""]
    n_translational: Float[Array, ""]
    z_mean: Float[Array, " S"]
    z2_mean: Float[Array, " S"]
    n_e_density: Float[Array, ""]
    n_zsq_density: Float[Array, ""]
    n_heavy_density: Float[Array, ""]
    n_deuteron_density: Float[Array, ""]
    e_ionisation: Float[Array, ""]
    e_dissociation: Float[Array, ""]
    z_bar_overall: Float[Array, ""]

    @property
    def heavy_fraction(self) -> Float[Array, " S"]:
        total = jnp.maximum(jnp.sum(self.n_per_species), 1.0e-300)
        return self.n_per_species / total


def composition(
    temperature: Float[Array, ""],
    v_free: Float[Array, ""],
    cfg: SimConfig,
    n_per_species: Float[Array, " S"] | None = None,
) -> CoreComposition:
    gas = cfg.gas
    n_per_species = species_counts(cfg, n_per_species)
    t_safe = jnp.clip(temperature, cfg.guards.t_min, cfg.guards.t_max)

    alpha = dissociation_fraction(
        t_safe, v_free, gas, n_per_species, cfg.transport.dissociation_enabled
    )
    n_free_atoms = alpha * n_per_species
    n_molecules = (1.0 - alpha) * n_per_species / gas.n_atoms

    if cfg.transport.ionisation_enabled:
        nf_density = n_free_atoms / v_free
        n_e_density = solve_electron_density(
            t_safe, nf_density, gas, cfg.transport
        )
        log_z1, log_z2, chi_mean = _saha_weights(
            jnp.log(n_e_density), t_safe, gas
        )
        z_mean = jnp.exp(log_z1)
        z2_mean = jnp.exp(log_z2)
    else:
        n_e_density = jnp.zeros(())
        z_mean = jnp.zeros_like(gas.mole_frac)
        z2_mean = jnp.zeros_like(gas.mole_frac)
        chi_mean = jnp.zeros_like(gas.mole_frac)

    n_electrons = n_e_density * v_free
    n_translational = jnp.sum(n_molecules) + jnp.sum(n_free_atoms) + n_electrons

    n_zsq_density = jnp.sum(n_free_atoms * z2_mean) / v_free
    n_heavy_density = (jnp.sum(n_molecules) + jnp.sum(n_free_atoms)) / v_free
    n_deuteron_density = jnp.sum(gas.is_deuteron * n_per_species) / v_free

    e_ion = jnp.sum(n_free_atoms * chi_mean)
    e_diss = jnp.sum(alpha * n_per_species * gas.e_diss / gas.n_atoms)

    free_atom_total = jnp.maximum(jnp.sum(n_free_atoms), 1.0e-30)
    z_bar_overall = jnp.sum(n_free_atoms * z_mean) / free_atom_total

    return CoreComposition(
        temperature=t_safe,
        v_free=v_free,
        n_per_species=n_per_species,
        alpha=alpha,
        n_molecules=n_molecules,
        n_free_atoms=n_free_atoms,
        n_electrons=n_electrons,
        n_translational=n_translational,
        z_mean=z_mean,
        z2_mean=z2_mean,
        n_e_density=n_e_density,
        n_zsq_density=n_zsq_density,
        n_heavy_density=n_heavy_density,
        n_deuteron_density=n_deuteron_density,
        e_ionisation=e_ion,
        e_dissociation=e_diss,
        z_bar_overall=z_bar_overall,
    )


def pressure_from_composition(
    comp: CoreComposition, cfg: SimConfig
) -> Float[Array, ""]:
    if not cfg.gas_enabled:
        return jnp.zeros(())
    return comp.n_translational * C.K_B * comp.temperature / comp.v_free


def free_particle_internal_energy(
    temperature: Float[Array, ""],
    n_free: Float[Array, " S"],
    gas: GasMixture,
    cfg: SimConfig,
) -> Float[Array, ""]:
    has_structure = gas.free_rot_dof > 0.0
    u = temperature / gas.free_theta_rot
    g_rot = u**2 / (1.0 + u**2)
    e_rot = 0.5 * gas.free_rot_dof * C.K_B * temperature * g_rot
    energy = jnp.sum(jnp.where(has_structure, n_free * e_rot, 0.0))

    if cfg.transport.vibrational_dof_enabled:
        levels = vibrational_levels(
            gas.free_e_bind[:, None], gas.free_theta_vib
        )
        e_vib = vibrational_energy_per_molecule(
            temperature, gas.free_theta_vib, levels
        )
        per_species = jnp.sum(gas.free_vib_mask * e_vib, axis=1)
        energy = energy + jnp.sum(
            jnp.where(has_structure, n_free * per_species, 0.0)
        )
    return energy


def internal_energy_from_composition(
    comp: CoreComposition, cfg: SimConfig
) -> Float[Array, ""]:
    if not cfg.gas_enabled:
        return jnp.zeros(())

    gas = cfg.gas
    t = comp.temperature
    u_trans = 1.5 * comp.n_translational * C.K_B * t

    is_diatomic = gas.n_atoms > 1.5
    u = t / gas.theta_rot
    g_rot = u**2 / (1.0 + u**2)
    u_rot = jnp.sum(
        jnp.where(is_diatomic, comp.n_molecules * C.K_B * t * g_rot, 0.0)
    )
    u_free = free_particle_internal_energy(t, comp.n_free_atoms, gas, cfg)

    if cfg.transport.vibrational_dof_enabled:
        n_levels = vibrational_levels(gas.e_diss, gas.theta_vib)
        e_vib = vibrational_energy_per_molecule(t, gas.theta_vib, n_levels)
        u_vib = jnp.sum(
            jnp.where(is_diatomic, comp.n_molecules * e_vib, 0.0)
        )
    else:
        u_vib = jnp.zeros(())

    return (
        u_trans + u_rot + u_vib + u_free
        + comp.e_dissociation + comp.e_ionisation
    )


def internal_energy(
    temperature: Float[Array, ""],
    v_free: Float[Array, ""],
    cfg: SimConfig,
) -> Float[Array, ""]:
    return internal_energy_from_composition(
        composition(temperature, v_free, cfg), cfg
    )


class EosDerivatives(eqx.Module):
    energy: Float[Array, ""]
    pressure: Float[Array, ""]
    du_dt: Float[Array, ""]
    du_dv: Float[Array, ""]
    dp_dt: Float[Array, ""]
    dp_dv: Float[Array, ""]
    du_dn: Float[Array, ""]
    dp_dn: Float[Array, ""]


# single-pass forward JVP for (U, p) partials wrt (T, V, N_vap)
def equation_of_state_and_derivatives(
    temperature: Float[Array, ""],
    v_free: Float[Array, ""],
    cfg: SimConfig,
    n_per_species: Float[Array, " S"] | None = None,
) -> EosDerivatives:
    if not cfg.gas_enabled:
        zero = jnp.zeros(())
        return EosDerivatives(
            energy=zero,
            pressure=zero,
            du_dt=jnp.ones(()),
            du_dv=zero,
            dp_dt=zero,
            dp_dv=zero,
            du_dn=zero,
            dp_dn=zero,
        )

    counts = species_counts(cfg, n_per_species)
    index = cfg.vapour.species_index

    if cfg.vapour.enabled:
        def _eos(x: Float[Array, " 3"]) -> Float[Array, " 2"]:
            mixture = counts.at[index].set(x[2])
            comp = composition(x[0], x[1], cfg, mixture)
            return jnp.stack(
                [
                    internal_energy_from_composition(comp, cfg),
                    pressure_from_composition(comp, cfg),
                ]
            )

        x0 = jnp.stack([temperature, v_free, counts[index]])
        primal, jvp = jax.linearize(_eos, x0)
        d_dt = jvp(jnp.array([1.0, 0.0, 0.0]))
        d_dv = jvp(jnp.array([0.0, 1.0, 0.0]))
        d_dn = jvp(jnp.array([0.0, 0.0, 1.0]))
        du_dn, dp_dn = d_dn[0], d_dn[1]
    else:
        def _eos(x: Float[Array, " 2"]) -> Float[Array, " 2"]:
            comp = composition(x[0], x[1], cfg, counts)
            return jnp.stack(
                [
                    internal_energy_from_composition(comp, cfg),
                    pressure_from_composition(comp, cfg),
                ]
            )

        x0 = jnp.stack([temperature, v_free])
        primal, jvp = jax.linearize(_eos, x0)
        d_dt = jvp(jnp.array([1.0, 0.0]))
        d_dv = jvp(jnp.array([0.0, 1.0]))
        du_dn = jnp.zeros(())
        dp_dn = jnp.zeros(())

    return EosDerivatives(
        energy=primal[0],
        pressure=primal[1],
        du_dt=d_dt[0],
        du_dv=d_dv[0],
        dp_dt=d_dt[1],
        dp_dv=d_dv[1],
        du_dn=du_dn,
        dp_dn=dp_dn,
    )


def caloric_coefficients(
    temperature: Float[Array, ""],
    v_free: Float[Array, ""],
    cfg: SimConfig,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    eos = equation_of_state_and_derivatives(temperature, v_free, cfg)
    return eos.energy, eos.du_dt, eos.du_dv


def total_gas_mass(
    cfg: SimConfig, n_per_species: Float[Array, " S"] | None = None
) -> Float[Array, ""]:
    counts = species_counts(cfg, n_per_species)
    return jnp.sum(counts * cfg.gas.mass_per_nucleus)


def sound_speed_gas(
    comp: CoreComposition, eos: "EosDerivatives", cfg: SimConfig
) -> Float[Array, ""]:
    if not cfg.gas_enabled:
        return jnp.zeros(())
    v_total = comp.v_free + hardcore_volume(comp.n_per_species, cfg)
    mass = jnp.maximum(total_gas_mass(cfg, comp.n_per_species), 1.0e-300)
    c_v = jnp.maximum(eos.du_dt, 1.0e-300)
    dp_ds = eos.dp_dv - comp.temperature * eos.dp_dt**2 / c_v
    c_squared = -(v_total**2 / mass) * dp_ds
    return jnp.sqrt(jnp.maximum(c_squared, 0.0))


def internal_mach_number(
    radius_dot: Float[Array, ""],
    comp: CoreComposition,
    eos: "EosDerivatives",
    cfg: SimConfig,
) -> Float[Array, ""]:
    if not cfg.gas_enabled:
        return jnp.zeros(())
    c_gas = sound_speed_gas(comp, eos, cfg)
    return jnp.abs(radius_dot) / jnp.maximum(c_gas, 1.0e-300)


def coulomb_logarithm(
    comp: CoreComposition, transport: TransportParams
) -> Float[Array, ""]:
    n_e_cgs = jnp.maximum(comp.n_e_density * 1.0e-6, 1.0)
    t_ev = jnp.maximum(comp.temperature / C.KELVIN_PER_EV, 1.0e-3)
    z = jnp.maximum(comp.z_bar_overall, 1.0e-3)
    raw = 23.0 - jnp.log(jnp.sqrt(n_e_cgs) * z * t_ev**-1.5)
    return jnp.clip(raw, transport.coulomb_log_min, transport.coulomb_log_max)


def thermal_conductivity(
    comp: CoreComposition, cfg: SimConfig
) -> Float[Array, ""]:
    transport = cfg.transport
    if transport.conduction_model == "none":
        return jnp.zeros(())

    t = comp.temperature
    kappa_n = jnp.sum(
        comp.heavy_fraction
        * cfg.gas.kappa_ref
        * (t / cfg.gas.kappa_T_ref) ** cfg.gas.kappa_exponent
    )
    if transport.conduction_model == "neutral_power_law":
        return kappa_n

    t_ev = jnp.maximum(t / C.KELVIN_PER_EV, 1.0e-6)
    z = jnp.maximum(comp.z_bar_overall, 1.0e-3)
    ln_lambda = coulomb_logarithm(comp, transport)
    kappa_e = C.SPITZER_KAPPA_COEFF * t_ev**2.5 / (z * ln_lambda)

    denom = jnp.maximum(comp.n_e_density + comp.n_heavy_density, 1.0e-30)
    f_e = comp.n_e_density / denom
    return (1.0 - f_e) * kappa_n + f_e * kappa_e


def boundary_layer_thickness(
    radius: Float[Array, ""],
    radius_dot: Float[Array, ""],
    comp: CoreComposition,
    cfg: SimConfig,
) -> Float[Array, ""]:
    kappa = thermal_conductivity(comp, cfg)
    c_p_per_particle = 2.5 * C.K_B
    n_density = jnp.maximum(comp.n_translational / comp.v_free, 1.0e-30)
    chi = kappa / (n_density * c_p_per_particle)

    rdot_abs = jnp.abs(radius_dot) + 1.0e-12
    diffusive = jnp.sqrt(radius * chi / rdot_abs)
    geometric = radius / cfg.transport.boundary_layer_cap_factor
    return soft_min(diffusive, geometric, cfg.guards.softmin_sharpness)


# Toegel-Lohse boundary layer + flux-limited Spitzer conduction
def conduction_loss(
    radius: Float[Array, ""],
    radius_dot: Float[Array, ""],
    comp: CoreComposition,
    cfg: SimConfig,
) -> Float[Array, ""]:
    if not cfg.transport.conduction_enabled or not cfg.gas_enabled:
        return jnp.zeros(())

    kappa = thermal_conductivity(comp, cfg)
    ell = boundary_layer_thickness(radius, radius_dot, comp, cfg)
    t = comp.temperature
    q = kappa * (t - cfg.liquid.ambient_temperature) / jnp.maximum(ell, 1.0e-15)

    if cfg.transport.flux_limiter_enabled:
        v_th_e = jnp.sqrt(C.K_B * t / C.M_ELECTRON)
        m_heavy = jnp.sum(comp.heavy_fraction * cfg.gas.mass_per_nucleus)
        v_th_h = jnp.sqrt(C.K_B * t / jnp.maximum(m_heavy, 1.0e-30))
        q_fs = cfg.transport.flux_limiter_factor * C.K_B * t * (
            comp.n_e_density * v_th_e + comp.n_heavy_density * v_th_h
        )
        q = q / (1.0 + jnp.abs(q) / jnp.maximum(q_fs, 1.0e-30))

    return 4.0 * jnp.pi * radius**2 * q


def bremsstrahlung_power_density(
    comp: CoreComposition, cfg: SimConfig
) -> Float[Array, ""]:
    if not cfg.transport.bremsstrahlung_enabled or not cfg.gas_enabled:
        return jnp.zeros(())
    t_ev = jnp.maximum(comp.temperature / C.KELVIN_PER_EV, 0.0)
    return (
        C.BREMSSTRAHLUNG_COEFF_SI
        * cfg.transport.bremsstrahlung_gaunt
        * comp.n_e_density
        * comp.n_zsq_density
        * jnp.sqrt(t_ev)
    )


def planck_mean_absorption(
    comp: CoreComposition, cfg: SimConfig
) -> Float[Array, ""]:
    if not cfg.gas_enabled:
        return jnp.zeros(())
    emissivity = bremsstrahlung_power_density(comp, cfg)
    t4 = jnp.maximum(comp.temperature, 1.0e-30) ** 4
    return emissivity / (4.0 * C.SIGMA_SB * t4)


def escape_factor(
    thin: Float[Array, ""], thick: Float[Array, ""]
) -> Float[Array, ""]:
    safe = jnp.maximum(thick, 1.0e-300)
    return safe * -jnp.expm1(thin / -safe)


# free-free bremsstrahlung with grey-body escape factor
def radiative_loss(
    radius: Float[Array, ""], comp: CoreComposition, cfg: SimConfig
) -> Float[Array, ""]:
    if not cfg.gas_enabled:
        return jnp.zeros(())
    thin = bremsstrahlung_power_density(comp, cfg) * comp.v_free
    if not cfg.transport.blackbody_cap_enabled:
        return thin
    thick = (
        cfg.transport.blackbody_emissivity
        * 4.0
        * jnp.pi
        * radius**2
        * C.SIGMA_SB
        * comp.temperature**4
    )
    return escape_factor(thin, thick)


def bosch_hale_reactivity(
    t_keV: Float[Array, "..."],
    reaction: str,
    theta_denominator_floor: float | Float[Array, ""] = 1.0e-3,
    t_max_keV: float | Float[Array, ""] = 190.0,
) -> Float[Array, "..."]:
    if reaction not in _BOSCH_HALE:
        raise KeyError(
            f"unknown reaction {reaction!r}; implemented: "
            f"{sorted(_BOSCH_HALE)}"
        )
    p = _BOSCH_HALE[reaction]
    t = jnp.clip(t_keV, 1.0e-6, t_max_keV)

    num = t * (p["c2"] + t * (p["c4"] + t * p["c6"]))
    den = 1.0 + t * (p["c3"] + t * (p["c5"] + t * p["c7"]))
    bracket = jnp.maximum(1.0 - num / den, theta_denominator_floor)
    theta = t / bracket

    xi = (p["bg"] ** 2 / (4.0 * theta)) ** (1.0 / 3.0)
    sv_cgs = (
        p["c1"]
        * theta
        * jnp.sqrt(xi / (p["mrc2"] * t**3))
        * jnp.exp(jnp.clip(-3.0 * xi, -700.0, 0.0))
    )
    return sv_cgs * 1.0e-6


def fusion_rates(
    comp: CoreComposition, cfg: SimConfig
) -> tuple[dict[str, Float[Array, ""]], Float[Array, ""], Float[Array, ""]]:
    if not cfg.fusion.enabled or not cfg.gas_enabled:
        zero = jnp.zeros(())
        return {r: zero for r in cfg.fusion.reactions}, zero, zero

    t_keV = comp.temperature / C.KELVIN_PER_KEV
    n_d = comp.n_deuteron_density
    prefactor = 0.5 * n_d**2 * comp.v_free

    rates: dict[str, Float[Array, ""]] = {}
    neutron_rate = jnp.zeros(())
    deposited = jnp.zeros(())
    for name in cfg.fusion.reactions:
        sv = bosch_hale_reactivity(
            t_keV,
            name,
            cfg.fusion.theta_denominator_floor,
            cfg.fusion.t_valid_max_keV,
        )
        rate = prefactor * sv
        rates[name] = rate
        neutron_rate = neutron_rate + rate * _BOSCH_HALE[name]["neutron_yield"]
        deposited = deposited + rate * _BOSCH_HALE[name][
            "e_charged_mev"
        ] * C.MEV_TO_J
    deposited = deposited * cfg.fusion.charged_product_deposition_fraction
    return rates, neutron_rate, deposited


def bosch_hale_in_range(
    temperature: Float[Array, ""], cfg: SimConfig
) -> Bool[Array, ""]:
    t_keV = temperature / C.KELVIN_PER_KEV
    return (t_keV >= cfg.fusion.t_valid_min_keV) & (
        t_keV <= cfg.fusion.t_valid_max_keV
    )


class EnergyBudget(eqx.Module):
    pressure: Float[Array, ""]
    heat_capacity: Float[Array, ""]
    internal_pressure: Float[Array, ""]
    dp_dt: Float[Array, ""]
    dp_dv: Float[Array, ""]
    dp_dn: Float[Array, ""]
    sound_speed_gas: Float[Array, ""]
    internal_mach: Float[Array, ""]
    conduction: Float[Array, ""]
    radiation: Float[Array, ""]
    fusion_deposition: Float[Array, ""]
    neutron_rate: Float[Array, ""]
    reaction_rates: dict[str, Float[Array, ""]]
    temperature_dot: Float[Array, ""]
    vapour_rate: Float[Array, ""]
    vapour_length: Float[Array, ""]
    vapour_power: Float[Array, ""]


def vapour_counts(
    cfg: SimConfig, n_vapour: Float[Array, ""] | None
) -> Float[Array, " S"]:
    counts = cfg.gas.mole_frac * cfg.n_nuclei
    if not cfg.vapour.enabled or n_vapour is None:
        return counts
    return counts.at[cfg.vapour.species_index].set(jnp.maximum(n_vapour, 0.0))


def energy_budget(
    radius: Float[Array, ""],
    radius_dot: Float[Array, ""],
    temperature: Float[Array, ""],
    cfg: SimConfig,
    n_vapour: Float[Array, ""] | None = None,
) -> tuple[EnergyBudget, CoreComposition]:
    counts = vapour_counts(cfg, n_vapour)
    v_free = free_volume(radius, cfg, counts)
    comp = composition(temperature, v_free, cfg, counts)

    if not cfg.gas_enabled:
        zero = jnp.zeros(())
        return (
            EnergyBudget(
                pressure=zero,
                heat_capacity=jnp.ones(()),
                internal_pressure=zero,
                dp_dt=zero,
                dp_dv=zero,
                dp_dn=zero,
                sound_speed_gas=zero,
                internal_mach=zero,
                vapour_rate=zero,
                vapour_length=zero,
                vapour_power=zero,
                conduction=zero,
                radiation=zero,
                fusion_deposition=zero,
                neutron_rate=zero,
                reaction_rates={r: zero for r in cfg.fusion.reactions},
                temperature_dot=zero,
            ),
            comp,
        )

    eos = equation_of_state_and_derivatives(temperature, v_free, cfg, counts)
    c_v = eos.du_dt
    internal_p = eos.du_dv
    p_gas = eos.pressure
    q_cond = conduction_loss(radius, radius_dot, comp, cfg)
    q_rad = radiative_loss(radius, comp, cfg)
    rates, neutron_rate, q_fus = fusion_rates(comp, cfg)

    v_dot = 4.0 * jnp.pi * radius**2 * radius_dot

    from core import vapour as vap

    vapour_rate, vapour_length, vapour_scale = vap.exchange_rate(
        radius, radius_dot, counts[cfg.vapour.species_index]
        if cfg.vapour.enabled else jnp.zeros(()),
        v_free, comp.n_translational, cfg,
    )
    q_vapour = vap.transfer_power(
        vapour_rate, vapour_scale, comp.temperature, eos.du_dn, cfg
    )

    c_v_safe = jnp.maximum(c_v, 1.0e-30)
    t_dot = (
        -(p_gas + internal_p) * v_dot - q_cond - q_rad + q_fus + q_vapour
    ) / c_v_safe

    return (
        EnergyBudget(
            pressure=p_gas,
            heat_capacity=c_v,
            internal_pressure=internal_p,
            dp_dt=eos.dp_dt,
            dp_dv=eos.dp_dv,
            dp_dn=eos.dp_dn,
            sound_speed_gas=sound_speed_gas(comp, eos, cfg),
            internal_mach=internal_mach_number(radius_dot, comp, eos, cfg),
            conduction=q_cond,
            radiation=q_rad,
            fusion_deposition=q_fus,
            neutron_rate=neutron_rate,
            reaction_rates=rates,
            temperature_dot=t_dot,
            vapour_rate=vapour_rate,
            vapour_length=vapour_length,
            vapour_power=q_vapour,
        ),
        comp,
    )


__all__ = [
    "BOSCH_HALE_REACTIONS",
    "CoreComposition",
    "EnergyBudget",
    "EosDerivatives",
    "boundary_layer_thickness",
    "bosch_hale_in_range",
    "bosch_hale_reactivity",
    "bremsstrahlung_power_density",
    "caloric_coefficients",
    "composition",
    "conduction_loss",
    "coulomb_logarithm",
    "dissociation_fraction",
    "energy_budget",
    "equation_of_state_and_derivatives",
    "free_particle_internal_energy",
    "free_volume",
    "fusion_rates",
    "internal_energy",
    "internal_energy_from_composition",
    "internal_mach_number",
    "log_z_vibrational",
    "escape_factor",
    "planck_mean_absorption",
    "pressure_from_composition",
    "radiative_loss",
    "hardcore_volume",
    "soft_min",
    "species_counts",
    "solve_electron_density",
    "sound_speed_gas",
    "thermal_conductivity",
    "thermal_wavelength",
    "total_gas_mass",
    "vibrational_energy_per_molecule",
    "vapour_counts",
    "vibrational_levels",
]
