from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from core import keller_miksis as km
from core import plasma_core as pc
from core.acoustics import drive_pressure
from core.config import LiquidParams, SimConfig

TAIT_EXPONENT: float = 7.15


def tait_exponent(liquid: LiquidParams) -> Float[Array, ""]:
    return liquid.tait_n


# Tait EOS liquid sound speed and enthalpy (n=7.15, B=304.6 MPa)
def tait_reference_pressure(liquid: LiquidParams) -> Float[Array, ""]:
    return (
        liquid.density * liquid.sound_speed**2 / tait_exponent(liquid)
        - liquid.ambient_pressure
    )


def liquid_density(
    pressure: Float[Array, ""], liquid: LiquidParams
) -> Float[Array, ""]:
    b = tait_reference_pressure(liquid)
    a = liquid.ambient_pressure + b
    x = jnp.maximum((pressure - liquid.ambient_pressure) / a, -1.0 + 1.0e-9)
    return liquid.density * jnp.exp(jnp.log1p(x) / tait_exponent(liquid))


def enthalpy(
    wall_pressure: Float[Array, ""],
    far_pressure: Float[Array, ""],
    liquid: LiquidParams,
) -> Float[Array, ""]:
    b = tait_reference_pressure(liquid)
    a = liquid.ambient_pressure + b
    n_tait = tait_exponent(liquid)
    exponent = (n_tait - 1.0) / n_tait

    far_over_a = jnp.log1p(
        (far_pressure - liquid.ambient_pressure) / a
    )
    ratio = jnp.maximum(
        (wall_pressure - far_pressure) / jnp.maximum(far_pressure + b, 1.0e-30),
        -1.0 + 1.0e-12,
    )
    return (
        n_tait
        / (n_tait - 1.0)
        * (a / liquid.density)
        * jnp.exp(exponent * far_over_a)
        * jnp.expm1(exponent * jnp.log1p(ratio))
    )


def wall_sound_speed(
    enthalpy_value: Float[Array, ""], liquid: LiquidParams
) -> Float[Array, ""]:
    return jnp.sqrt(
        jnp.maximum(
            liquid.sound_speed**2
            + (tait_exponent(liquid) - 1.0) * enthalpy_value,
            1.0e-6 * liquid.sound_speed**2,
        )
    )


class GilmoreDiagnostics(eqx.Module):
    radius_clamped: Float[Array, ""]
    mach_acoustic: Float[Array, ""]
    mach_local: Float[Array, ""]
    wall_sound_speed: Float[Array, ""]
    enthalpy: Float[Array, ""]
    wall_density: Float[Array, ""]
    density_ratio: Float[Array, ""]
    gas_pressure: Float[Array, ""]
    wall_pressure: Float[Array, ""]
    drive_pressure: Float[Array, ""]
    radius_ddot: Float[Array, ""]
    relative_distortion: Float[Array, ""]


# Gilmore radial ODE + Plesset l-mode shape coupling
def rhs(
    time: Float[Array, ""],
    state: Float[Array, " 9"],
    cfg: SimConfig,
    drive: tuple[Float[Array, ""], Float[Array, ""]] | None = None,
) -> tuple[Float[Array, " 9"], GilmoreDiagnostics]:
    guards = cfg.guards
    liquid = cfg.liquid

    radius = jnp.maximum(state[km.IDX_R], cfg.r_floor)
    temperature = jnp.clip(state[km.IDX_T], guards.t_min, guards.t_max)
    rdot_cap = guards.rdot_max_mach * liquid.sound_speed
    radius_dot = jnp.clip(state[km.IDX_RDOT], -rdot_cap, rdot_cap)
    a_l = state[km.IDX_A]
    adot_l = state[km.IDX_ADOT]

    mu = liquid.dynamic_viscosity
    sigma = liquid.surface_tension

    n_vapour = jnp.maximum(state[km.IDX_NVAP], 0.0)
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

    h = enthalpy(p_wall, p_inf, liquid)
    c_wall = wall_sound_speed(h, liquid)
    rho_wall = liquid_density(p_wall, liquid)
    rho_inf = liquid_density(p_inf, liquid)

    p_wall_dot_explicit = (
        p_gas_dot
        + 2.0 * sigma * radius_dot / radius**2
        + 4.0 * mu * radius_dot**2 / radius**2
    )
    h_dot_explicit = p_wall_dot_explicit / rho_wall - p_inf_dot / rho_inf

    mach_local = radius_dot / c_wall
    one_minus = 1.0 - mach_local

    numerator = (
        h * (1.0 + mach_local)
        + (radius / c_wall) * one_minus * h_dot_explicit
        - 1.5 * radius_dot**2 * (1.0 - mach_local / 3.0)
    )
    denominator = one_minus * (radius + 4.0 * mu / (c_wall * rho_wall))
    denominator = jnp.where(
        jnp.abs(denominator) < guards.km_denominator_floor,
        jnp.sign(denominator + 1.0e-300) * guards.km_denominator_floor,
        denominator,
    )
    radius_ddot = numerator / denominator

    if cfg.shape.enabled:
        a_coeff, b_coeff = km.shape_mode_coefficients(
            radius, radius_dot, radius_ddot, cfg
        )
        addot_l = a_coeff * a_l - b_coeff * adot_l
    else:
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
    diagnostics = GilmoreDiagnostics(
        radius_clamped=radius,
        mach_acoustic=radius_dot / liquid.sound_speed,
        mach_local=mach_local,
        wall_sound_speed=c_wall,
        enthalpy=h,
        wall_density=rho_wall,
        density_ratio=rho_wall / liquid.density,
        gas_pressure=p_gas,
        wall_pressure=p_wall,
        drive_pressure=p_inf,
        radius_ddot=radius_ddot,
        relative_distortion=jnp.abs(a_l) / radius,
    )
    return derivative, diagnostics


def rhs_only(
    time: Float[Array, ""], state: Float[Array, " 9"], cfg: SimConfig
) -> Float[Array, " 9"]:
    return rhs(time, state, cfg)[0]


def diagnostics_along(
    cfg: SimConfig,
    times: Float[Array, " S"],
    states: Float[Array, "S 9"],
) -> GilmoreDiagnostics:
    import jax

    return jax.vmap(lambda t, y: rhs(t, y, cfg)[1])(times, states)


__all__ = [
    "GilmoreDiagnostics",
    "TAIT_EXPONENT",
    "tait_exponent",
    "diagnostics_along",
    "enthalpy",
    "liquid_density",
    "rhs",
    "rhs_only",
    "tait_reference_pressure",
    "wall_sound_speed",
]
