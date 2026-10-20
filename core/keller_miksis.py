from __future__ import annotations

from typing import Final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from core import plasma_core as pc
from core import vapour
from core.acoustics import drive_pressure
from core.config import SimConfig


# 9-state ODE vector indices (SI units)
IDX_R: Final[int] = 0
IDX_RDOT: Final[int] = 1
IDX_T: Final[int] = 2
IDX_A: Final[int] = 3
IDX_ADOT: Final[int] = 4
IDX_YIELD_N: Final[int] = 5
IDX_YIELD_P: Final[int] = 6
IDX_ERAD: Final[int] = 7
IDX_NVAP: Final[int] = 8
STATE_DIM: Final[int] = 9
STATE_NAMES: Final[tuple[str, ...]] = (
    "R",
    "Rdot",
    "T",
    "a_l",
    "adot_l",
    "yield_DD_n3He",
    "yield_DD_pT",
    "E_radiated",
    "N_vapour",
)

STATE_UNITS: Final[tuple[str, ...]] = (
    "m",
    "m/s",
    "K",
    "m",
    "m/s",
    "1",
    "1",
    "J",
    "1",
)


def initial_state(cfg: SimConfig) -> Float[Array, " 9"]:
    n_vapour = (
        vapour.saturated_count(cfg.r0, cfg) if cfg.vapour.enabled else 0.0
    )
    return jnp.array(
        [
            cfg.r0,
            0.0,
            cfg.liquid.ambient_temperature,
            cfg.shape.a_l_init_rel * cfg.r0 if cfg.shape.enabled else 0.0,
            cfg.shape.adot_l_init if cfg.shape.enabled else 0.0,
            0.0,
            0.0,
            0.0,
            n_vapour,
        ],
        dtype=jnp.float64,
    )


# Plesset-Prosperetti linear spherical harmonic mode l
def shape_mode_coefficients(
    radius: Float[Array, ""],
    radius_dot: Float[Array, ""],
    radius_ddot: Float[Array, ""],
    cfg: SimConfig,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    el = float(cfg.shape.mode_l)
    nu = cfg.liquid.kinematic_viscosity
    sigma = cfg.liquid.surface_tension
    rho = cfg.liquid.density

    capillary = (el - 1.0) * (el + 1.0) * (el + 2.0) * sigma / (rho * radius**3)
    a_coeff = (el - 1.0) * radius_ddot / radius - capillary
    b_coeff = 3.0 * radius_dot / radius

    if cfg.shape.viscous_damping_enabled:
        a_coeff = a_coeff - cfg.shape.damping_coeff_c2 * (
            2.0 * (el - 1.0) * (el + 2.0) * nu * radius_dot / radius**3
        )
        b_coeff = b_coeff + cfg.shape.damping_coeff_c1 * (
            2.0 * (el + 2.0) * (2.0 * el + 1.0) * nu / radius**2
        )
    return a_coeff, b_coeff


def lamb_shape_frequency(
    radius: Float[Array, ""],
    mode_l: int,
    surface_tension: Float[Array, ""],
    density_outer: Float[Array, ""],
    density_inner: Float[Array, ""] | float = 0.0,
) -> Float[Array, ""]:
    el = float(mode_l)
    numerator = el * (el + 1.0) * (el - 1.0) * (el + 2.0) * surface_tension
    denominator = radius**3 * (
        (el + 1.0) * jnp.asarray(density_inner) + el * density_outer
    )
    return jnp.sqrt(numerator / denominator)


class Diagnostics(eqx.Module):
    radius_clamped: Float[Array, ""]
    mach: Float[Array, ""]
    internal_mach: Float[Array, ""]
    sound_speed_gas: Float[Array, ""]
    gas_pressure: Float[Array, ""]
    gas_pressure_dot: Float[Array, ""]
    wall_pressure: Float[Array, ""]
    drive_pressure: Float[Array, ""]
    drive_pressure_dot: Float[Array, ""]
    radius_ddot: Float[Array, ""]
    shape_a_coeff: Float[Array, ""]
    shape_b_coeff: Float[Array, ""]
    relative_distortion: Float[Array, ""]
    heat_capacity: Float[Array, ""]
    conduction_loss: Float[Array, ""]
    radiative_loss: Float[Array, ""]
    neutron_rate: Float[Array, ""]
    electron_density: Float[Array, ""]
    mean_ionisation: Float[Array, ""]
    dissociation_fraction: Float[Array, ""]
    bosch_hale_in_range: Float[Array, ""]


# Keller-Miksis radial equation with implicit viscous Rddot term on LHS
def rhs(
    time: Float[Array, ""],
    state: Float[Array, " 9"],
    cfg: SimConfig,
    drive: tuple[Float[Array, ""], Float[Array, ""]] | None = None,
) -> tuple[Float[Array, " 9"], Diagnostics]:
    guards = cfg.guards
    liquid = cfg.liquid

    radius = jnp.maximum(state[IDX_R], cfg.r_floor)
    temperature = jnp.clip(state[IDX_T], guards.t_min, guards.t_max)
    rdot_cap = guards.rdot_max_mach * liquid.sound_speed
    radius_dot = jnp.clip(state[IDX_RDOT], -rdot_cap, rdot_cap)
    a_l = state[IDX_A]
    adot_l = state[IDX_ADOT]

    c = liquid.sound_speed
    rho = liquid.density
    mu = liquid.dynamic_viscosity
    sigma = liquid.surface_tension
    mach = radius_dot / c

    n_vapour = jnp.maximum(state[IDX_NVAP], 0.0)
    budget, comp = pc.energy_budget(
        radius, radius_dot, temperature, cfg, n_vapour
    )
    p_gas = budget.pressure

    v_dot = 4.0 * jnp.pi * radius**2 * radius_dot
    t_dot = budget.temperature_dot
    p_gas_dot = (
        budget.dp_dt * t_dot
        + budget.dp_dv * v_dot
        + budget.dp_dn * budget.vapour_rate
    )

    p_wall = (
        p_gas
        + liquid.vapour_pressure
        - 2.0 * sigma / radius
        - 4.0 * mu * radius_dot / radius
    )
    if drive is None:
        p_inf, p_inf_dot = drive_pressure(
            time, cfg.acoustics, liquid.ambient_pressure
        )
    else:
        p_inf, p_inf_dot = drive

    dp_dt_explicit = (
        p_gas_dot
        + 2.0 * sigma * radius_dot / radius**2
        + 4.0 * mu * radius_dot**2 / radius**2
        - p_inf_dot
    )

    delta_p = p_wall - p_inf
    rhs_terms = (
        (1.0 + mach) * delta_p / rho
        + radius * dp_dt_explicit / (rho * c)
        - 1.5 * (1.0 - mach / 3.0) * radius_dot**2
    )
    lhs_coeff = (1.0 - mach) * radius + 4.0 * mu / (rho * c)
    lhs_coeff = jnp.where(
        jnp.abs(lhs_coeff) < guards.km_denominator_floor,
        guards.km_denominator_floor,
        lhs_coeff,
    )
    radius_ddot = rhs_terms / lhs_coeff

    if cfg.shape.enabled:
        a_coeff, b_coeff = shape_mode_coefficients(
            radius, radius_dot, radius_ddot, cfg
        )
        addot_l = a_coeff * a_l - b_coeff * adot_l
    else:
        a_coeff = jnp.zeros(())
        b_coeff = jnp.zeros(())
        adot_l = jnp.zeros(())
        addot_l = jnp.zeros(())

    rates = budget.reaction_rates
    yield_n_dot = rates.get("DD_n3He", jnp.zeros(()))
    yield_p_dot = rates.get("DD_pT", jnp.zeros(()))

    derivative = jnp.stack(
        [
            radius_dot,
            radius_ddot,
            t_dot,
            adot_l,
            addot_l,
            yield_n_dot,
            yield_p_dot,
            budget.radiation,
            budget.vapour_rate,
        ]
    )

    diagnostics = Diagnostics(
        radius_clamped=radius,
        mach=mach,
        internal_mach=budget.internal_mach,
        sound_speed_gas=budget.sound_speed_gas,
        gas_pressure=p_gas,
        gas_pressure_dot=p_gas_dot,
        wall_pressure=p_wall,
        drive_pressure=p_inf,
        drive_pressure_dot=p_inf_dot,
        radius_ddot=radius_ddot,
        shape_a_coeff=a_coeff,
        shape_b_coeff=b_coeff,
        relative_distortion=jnp.abs(a_l) / radius,
        heat_capacity=budget.heat_capacity,
        conduction_loss=budget.conduction,
        radiative_loss=budget.radiation,
        neutron_rate=budget.neutron_rate,
        electron_density=comp.n_e_density,
        mean_ionisation=comp.z_bar_overall,
        dissociation_fraction=jnp.sum(cfg.gas.mole_frac * comp.alpha),
        bosch_hale_in_range=pc.bosch_hale_in_range(
            temperature, cfg
        ).astype(jnp.float64),
    )
    return derivative, diagnostics


def rhs_only(
    time: Float[Array, ""], state: Float[Array, " 9"], cfg: SimConfig
) -> Float[Array, " 9"]:
    derivative, _ = rhs(time, state, cfg)
    return derivative


def rayleigh_plesset_rhs(
    time: Float[Array, ""], state: Float[Array, " 9"], cfg: SimConfig
) -> Float[Array, " 9"]:
    incompressible = eqx.tree_at(
        lambda c: c.liquid.sound_speed, cfg, jnp.asarray(1.0e18)
    )
    derivative, _ = rhs(time, state, incompressible)
    return derivative


__all__ = [
    "Diagnostics",
    "IDX_NVAP",
    "IDX_A",
    "IDX_ADOT",
    "IDX_ERAD",
    "IDX_R",
    "IDX_RDOT",
    "IDX_T",
    "IDX_YIELD_N",
    "IDX_YIELD_P",
    "STATE_DIM",
    "STATE_NAMES",
    "STATE_UNITS",
    "initial_state",
    "lamb_shape_frequency",
    "rayleigh_plesset_rhs",
    "rhs",
    "rhs_only",
    "shape_mode_coefficients",
]
