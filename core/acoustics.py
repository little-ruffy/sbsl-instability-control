from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from core.config import AcousticParams


def envelope(
    time: Float[Array, ""], acoustics: AcousticParams
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    ramp_time = acoustics.ramp_cycles / jnp.maximum(acoustics.frequency, 1.0e-300)
    active = (acoustics.ramp_cycles > 0.0) & (time < ramp_time)
    safe_ramp = jnp.where(ramp_time > 0.0, ramp_time, 1.0)
    phase = jnp.pi * jnp.clip(time / safe_ramp, 0.0, 1.0)
    value = jnp.where(active, 0.5 * (1.0 - jnp.cos(phase)), 1.0)
    derivative = jnp.where(
        active, 0.5 * jnp.pi / safe_ramp * jnp.sin(phase), 0.0
    )
    return value, derivative


# multi-harmonic acoustic drive p_inf(t) and dp_inf/dt
def drive_pressure(
    time: Float[Array, ""],
    acoustics: AcousticParams,
    ambient_pressure: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    k = jnp.arange(1, acoustics.n_harmonics + 1, dtype=jnp.float64)
    theta = k * acoustics.omega * time + acoustics.phases
    env, env_dot = envelope(time, acoustics)

    sin_sum = jnp.sum(acoustics.amplitudes * jnp.sin(theta))
    cos_sum = jnp.sum(acoustics.amplitudes * k * acoustics.omega * jnp.cos(theta))

    pressure = ambient_pressure - env * sin_sum
    pressure_dot = -env_dot * sin_sum - env * cos_sum
    return pressure, pressure_dot


def peak_negative_pressure(
    acoustics: AcousticParams, ambient_pressure: Float[Array, ""], n_samples: int = 512
) -> Float[Array, ""]:
    period = jnp.where(
        acoustics.frequency > 0.0,
        1.0 / jnp.maximum(acoustics.frequency, 1.0e-300),
        1.0,
    )
    t = jnp.linspace(0.0, period, n_samples, endpoint=False)
    k = jnp.arange(1, acoustics.n_harmonics + 1, dtype=jnp.float64)
    theta = k[None, :] * acoustics.omega * t[:, None] + acoustics.phases[None, :]
    series = jnp.sum(acoustics.amplitudes[None, :] * jnp.sin(theta), axis=1)
    return ambient_pressure - jnp.max(series)


def blake_threshold_pressure(
    r0: Float[Array, ""],
    surface_tension: Float[Array, ""],
    ambient_pressure: Float[Array, ""],
    vapour_pressure: Float[Array, ""],
) -> Float[Array, ""]:
    p_net = ambient_pressure - vapour_pressure + 2.0 * surface_tension / r0
    return ambient_pressure - vapour_pressure + (
        4.0 * surface_tension / 3.0
    ) * jnp.sqrt(
        2.0 * surface_tension / (3.0 * r0**3 * jnp.maximum(p_net, 1.0e-30))
    )


__all__ = [
    "blake_threshold_pressure",
    "drive_pressure",
    "envelope",
    "peak_negative_pressure",
]
