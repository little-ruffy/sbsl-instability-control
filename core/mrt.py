from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

MU0: float = 4.0e-7 * 3.141592653589793


class MrtParams(eqx.Module):
    wavenumber: float
    atwood: float
    density_sum: float
    enabled: bool = eqx.field(static=True, default=True)


def effective_gravity(
    radius_ddot: Float[Array, ""], heavy_outside: bool
) -> Float[Array, ""]:
    return radius_ddot if heavy_outside else -radius_ddot


def growth_bracket(
    params: MrtParams,
    gravity: Float[Array, ""],
    b_axial: Float[Array, ""],
) -> Float[Array, ""]:
    k = params.wavenumber
    drive = k * params.atwood * gravity
    tension = k * k * b_axial**2 / (
        MU0 * jnp.maximum(params.density_sum, 1.0e-300)
    )
    return drive - tension


def cutoff_wavenumber(
    params: MrtParams,
    gravity: Float[Array, ""],
    b_axial: Float[Array, ""],
) -> Float[Array, ""]:
    b2 = jnp.maximum(b_axial**2, 1.0e-300)
    return MU0 * params.density_sum * params.atwood * gravity / b2


def field_for_cutoff(
    params: MrtParams, gravity: Float[Array, ""]
) -> Float[Array, ""]:
    k = jnp.maximum(params.wavenumber, 1.0e-300)
    return jnp.sqrt(
        jnp.maximum(
            MU0 * params.density_sum * params.atwood * gravity / k,
            0.0,
        )
    )


def rhs(
    xi: Float[Array, ""],
    xi_dot: Float[Array, ""],
    params: MrtParams,
    radius: Float[Array, ""],
    radius_dot: Float[Array, ""],
    gravity: Float[Array, ""],
    b_axial: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    if not params.enabled:
        return jnp.zeros_like(xi), jnp.zeros_like(xi_dot)

    convergence = radius_dot / jnp.maximum(radius, 1.0e-300)
    bracket = growth_bracket(params, gravity, b_axial)
    return xi_dot, bracket * xi - convergence * xi_dot


def diagnostics(
    xi: Float[Array, ""],
    params: MrtParams,
    radius: Float[Array, ""],
    gravity: Float[Array, ""],
    b_axial: Float[Array, ""],
) -> dict:
    bracket = growth_bracket(params, gravity, b_axial)
    return dict(
        relative=jnp.abs(xi) / jnp.maximum(radius, 1.0e-300),
        bracket=bracket,
        growing=bracket > 0.0,
        growth_rate=jnp.sqrt(jnp.maximum(bracket, 0.0)),
        b_critical=field_for_cutoff(params, gravity),
    )


__all__ = [
    "MU0",
    "MrtParams",
    "cutoff_wavenumber",
    "diagnostics",
    "effective_gravity",
    "field_for_cutoff",
    "growth_bracket",
    "rhs",
]
