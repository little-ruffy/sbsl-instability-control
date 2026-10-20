from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from core import constants as C
from core.config import SimConfig

BUCK_P0: float = 611.21
BUCK_A: float = 18.678
BUCK_B: float = 234.5
BUCK_C: float = 257.14

UPWIND_BLEND: float = 1.0e-2


def saturation_pressure(temperature: Float[Array, ""]) -> Float[Array, ""]:
    t_celsius = jnp.clip(temperature - 273.15, -200.0, 400.0)
    exponent = (BUCK_A - t_celsius / BUCK_B) * (
        t_celsius / (BUCK_C + t_celsius)
    )
    return BUCK_P0 * jnp.exp(jnp.clip(exponent, -200.0, 200.0))


def saturated_count(
    radius: Float[Array, ""], cfg: SimConfig
) -> Float[Array, ""]:
    from core import plasma_core as pc

    v_free = pc.free_volume(radius, cfg)
    t_liquid = cfg.liquid.ambient_temperature
    return (
        saturation_pressure(t_liquid)
        * v_free
        / (C.K_B * jnp.maximum(t_liquid, 1.0e-30))
    )


def diffusion_coefficient(
    n_total_density: Float[Array, ""], cfg: SimConfig
) -> Float[Array, ""]:
    reference = cfg.vapour.diffusivity_ref * cfg.vapour.density_ref
    return reference / jnp.maximum(n_total_density, 1.0e-30)


def diffusive_length(
    radius: Float[Array, ""],
    radius_dot: Float[Array, ""],
    diffusivity: Float[Array, ""],
    cfg: SimConfig,
) -> Float[Array, ""]:
    from core import plasma_core as pc

    rdot_abs = jnp.abs(radius_dot) + 1.0e-12
    diffusive = jnp.sqrt(radius * diffusivity / rdot_abs)
    geometric = radius / cfg.transport.boundary_layer_cap_factor
    return pc.soft_min(diffusive, geometric, cfg.guards.softmin_sharpness)


# diffusive water-vapour mass transfer across the bubble wall
def exchange_rate(
    radius: Float[Array, ""],
    radius_dot: Float[Array, ""],
    n_vapour: Float[Array, ""],
    v_free: Float[Array, ""],
    n_total: Float[Array, ""],
    cfg: SimConfig,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    if not cfg.vapour.enabled:
        return jnp.zeros(()), jnp.zeros(()), jnp.zeros(())

    density = n_total / jnp.maximum(v_free, 1.0e-300)
    diffusivity = diffusion_coefficient(density, cfg)
    length = diffusive_length(radius, radius_dot, diffusivity, cfg)

    t_liquid = cfg.liquid.ambient_temperature
    n_sat_density = saturation_pressure(t_liquid) / (
        C.K_B * jnp.maximum(t_liquid, 1.0e-30)
    )
    n_vapour_density = n_vapour / jnp.maximum(v_free, 1.0e-300)

    area = 4.0 * jnp.pi * radius**2
    conductance = area * diffusivity / jnp.maximum(length, 1.0e-15)
    rate = conductance * (n_sat_density - n_vapour_density)
    rate = jnp.where(n_vapour <= 0.0, jnp.maximum(rate, 0.0), rate)
    scale = conductance * n_sat_density
    return rate, length, scale


def transfer_power(
    rate: Float[Array, ""],
    rate_scale: Float[Array, ""],
    temperature: Float[Array, ""],
    du_dn: Float[Array, ""],
    cfg: SimConfig,
) -> Float[Array, ""]:
    if not cfg.vapour.enabled:
        return jnp.zeros(())
    reduced = rate / jnp.maximum(rate_scale, 1.0e-300)
    weight = 0.5 * (1.0 + jnp.tanh(reduced / UPWIND_BLEND))
    stream = (
        weight * incoming_enthalpy(cfg)
        + (1.0 - weight) * (du_dn + C.K_B * temperature)
    )
    return rate * (stream - du_dn)


def incoming_enthalpy(cfg: SimConfig) -> Float[Array, ""]:
    t_liquid = cfg.liquid.ambient_temperature
    index = cfg.vapour.species_index
    if index < 0:
        return jnp.zeros(())
    gas = cfg.gas
    rot = 0.5 * gas.free_rot_dof[index] * C.K_B * t_liquid
    u = t_liquid / gas.free_theta_rot[index]
    rot = rot * u**2 / (1.0 + u**2)
    theta = gas.free_theta_vib[index]
    mask = gas.free_vib_mask[index]
    ratio = jnp.clip(theta / jnp.maximum(t_liquid, 1.0e-30), 0.0, 700.0)
    vib = jnp.sum(
        mask * C.K_B * theta / jnp.maximum(jnp.expm1(ratio), 1.0e-300)
    )
    return 1.5 * C.K_B * t_liquid + rot + vib + C.K_B * t_liquid


__all__ = [
    "BUCK_A",
    "BUCK_B",
    "BUCK_C",
    "BUCK_P0",
    "diffusion_coefficient",
    "diffusive_length",
    "exchange_rate",
    "incoming_enthalpy",
    "saturated_count",
    "UPWIND_BLEND",
    "saturation_pressure",
    "transfer_power",
]
