from __future__ import annotations

from collections.abc import Callable
from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from core import gilmore as gil
from core import keller_miksis as km
from core import plasma_core as pc
from core.config import SimConfig
from core.integrator import _solve_impl

DEFAULT_STEP_BUDGET: Final[int] = 4096


# per-lane parameters for vmapped multi-bubble integration
class BatchSpec(eqx.Module):
    r0: Float[Array, " B"]
    n_nuclei: Float[Array, " B"]
    ambient_temperature: Float[Array, " B"]
    amplitudes: Float[Array, "B K"]
    phases: Float[Array, "B K"]

    @property
    def batch_size(self) -> int:
        return int(self.r0.shape[0])

    def at(self, index: int) -> tuple:
        return (
            self.r0[index],
            self.n_nuclei[index],
            self.ambient_temperature[index],
            self.amplitudes[index],
            self.phases[index],
        )


class CycleSummary(eqx.Module):
    final_state: Float[Array, "B 9"]
    peak_temperature: Float[Array, " B"]
    min_radius: Float[Array, " B"]
    max_radius: Float[Array, " B"]
    peak_wall_speed: Float[Array, " B"]
    peak_mach_local: Float[Array, " B"]
    max_relative_distortion: Float[Array, " B"]
    neutron_yield_delta: Float[Array, " B"]
    proton_yield_delta: Float[Array, " B"]
    radiated_energy_delta: Float[Array, " B"]
    mean_ionisation_at_peak: Float[Array, " B"]
    success: Bool[Array, " B"]
    num_steps: Float[Array, " B"]
    steps_used: Float[Array, " B"]


def apply_spec(cfg: SimConfig, index_spec: tuple) -> SimConfig:
    r0, n_nuclei, t_inf, amplitudes, phases = index_spec
    return eqx.tree_at(
        lambda c: (
            c.r0,
            c.n_nuclei,
            c.liquid.ambient_temperature,
            c.acoustics.amplitudes,
            c.acoustics.phases,
        ),
        cfg,
        (r0, n_nuclei, t_inf, amplitudes, phases),
    )


# collapse summary reduction (ambient and local Tait wall Mach)
def _reduce_one(
    cfg: SimConfig,
    times: Float[Array, " N"],
    states: Float[Array, "N 9"],
    success: Bool[Array, ""],
    num_steps: Float[Array, ""],
) -> CycleSummary:
    valid = jnp.isfinite(times) & jnp.all(jnp.isfinite(states), axis=1)
    any_valid = jnp.any(valid)

    radius = jnp.where(valid, states[:, km.IDX_R], jnp.inf)
    temperature = jnp.where(valid, states[:, km.IDX_T], -jnp.inf)
    speed = jnp.where(valid, jnp.abs(states[:, km.IDX_RDOT]), -jnp.inf)
    distortion = jnp.where(
        valid,
        jnp.abs(states[:, km.IDX_A]) / jnp.maximum(states[:, km.IDX_R], 1e-300),
        -jnp.inf,
    )

    index = jnp.arange(valid.shape[0])
    first = jnp.argmax(valid)
    last = jnp.argmax(jnp.where(valid, index, -1))
    hottest = jnp.argmax(temperature)

    final_state = states[last]
    start_state = states[first]

    peak_radius = jnp.maximum(states[hottest, km.IDX_R], cfg.r_floor)
    n_vap_hot = jnp.maximum(states[hottest, km.IDX_NVAP], 0.0)
    counts_hot = pc.vapour_counts(cfg, n_vap_hot)
    comp = pc.composition(
        jnp.clip(
            states[hottest, km.IDX_T], cfg.guards.t_min, cfg.guards.t_max
        ),
        pc.free_volume(peak_radius, cfg, n_per_species=counts_hot),
        cfg,
        n_per_species=counts_hot,
    )

    fastest = jnp.argmax(speed)
    r_fast = jnp.maximum(states[fastest, km.IDX_R], cfg.r_floor)
    rdot_fast = states[fastest, km.IDX_RDOT]
    t_fast = jnp.where(valid[fastest], times[fastest], 0.0)
    n_vap_fast = jnp.maximum(states[fastest, km.IDX_NVAP], 0.0)
    counts_fast = pc.vapour_counts(cfg, n_vap_fast)
    comp_fast = pc.composition(
        jnp.clip(
            states[fastest, km.IDX_T], cfg.guards.t_min, cfg.guards.t_max
        ),
        pc.free_volume(r_fast, cfg, n_per_species=counts_fast),
        cfg,
        n_per_species=counts_fast,
    )
    p_wall = (
        pc.pressure_from_composition(comp_fast, cfg)
        + cfg.liquid.vapour_pressure
        - 2.0 * cfg.liquid.surface_tension / r_fast
        - 4.0 * cfg.liquid.dynamic_viscosity * rdot_fast / r_fast
    )
    p_inf_fast, _ = km.drive_pressure(
        t_fast, cfg.acoustics, cfg.liquid.ambient_pressure
    )
    c_wall = gil.wall_sound_speed(
        gil.enthalpy(p_wall, p_inf_fast, cfg.liquid),
        cfg.liquid,
    )
    peak_mach_local = jnp.abs(rdot_fast) / jnp.maximum(c_wall, 1.0)

    ok = success & any_valid
    return CycleSummary(
        final_state=final_state,
        peak_temperature=jnp.max(temperature),
        min_radius=jnp.min(radius),
        max_radius=jnp.max(jnp.where(valid, states[:, km.IDX_R], -jnp.inf)),
        peak_wall_speed=jnp.max(speed),
        peak_mach_local=peak_mach_local,
        max_relative_distortion=jnp.max(distortion),
        neutron_yield_delta=jnp.maximum(
            final_state[km.IDX_YIELD_N] - start_state[km.IDX_YIELD_N], 0.0
        ),
        proton_yield_delta=jnp.maximum(
            final_state[km.IDX_YIELD_P] - start_state[km.IDX_YIELD_P], 0.0
        ),
        radiated_energy_delta=jnp.maximum(
            final_state[km.IDX_ERAD] - start_state[km.IDX_ERAD], 0.0
        ),
        mean_ionisation_at_peak=comp.z_bar_overall,
        success=ok,
        num_steps=num_steps,
        steps_used=jnp.sum(valid.astype(jnp.float64)),
    )


@eqx.filter_jit
def batched_cycle_summary(
    cfg: SimConfig,
    spec: BatchSpec,
    y0: Float[Array, "B 9"],
    t0: Float[Array, " B"],
    t1: Float[Array, " B"],
    max_steps: int = DEFAULT_STEP_BUDGET,
    rhs_fn: Callable[
        [Float[Array, ""], Float[Array, " 9"], SimConfig], Float[Array, " 9"]
    ] = km.rhs_only,
) -> CycleSummary:

    def one(
        r0: Float[Array, ""],
        n_nuclei: Float[Array, ""],
        t_inf: Float[Array, ""],
        amplitudes: Float[Array, " K"],
        phases: Float[Array, " K"],
        y0_i: Float[Array, " 9"],
        t0_i: Float[Array, ""],
        t1_i: Float[Array, ""],
    ) -> CycleSummary:
        local = apply_spec(cfg, (r0, n_nuclei, t_inf, amplitudes, phases))
        times, states, success, num_steps = _solve_impl(
            local,
            t0_i,
            t1_i,
            y0_i,
            None,
            rhs_fn,
            max_steps,
            False,
            True,
        )
        return _reduce_one(local, times, states, success, num_steps)

    return jax.vmap(one)(
        spec.r0,
        spec.n_nuclei,
        spec.ambient_temperature,
        spec.amplitudes,
        spec.phases,
        y0,
        t0,
        t1,
    )


@eqx.filter_jit
def batched_initial_state(
    cfg: SimConfig,
    spec: BatchSpec,
    a_l_init_rel: Float[Array, " B"],
    adot_l_init: Float[Array, " B"],
) -> Float[Array, "B 9"]:
    enabled = 1.0 if cfg.shape.enabled else 0.0
    zeros = jnp.zeros_like(spec.r0)
    return jnp.stack(
        [
            spec.r0,
            zeros,
            spec.ambient_temperature,
            enabled * a_l_init_rel * spec.r0,
            enabled * adot_l_init,
            zeros,
            zeros,
            zeros,
            _batched_vapour(cfg, spec.r0, spec.ambient_temperature),
        ],
        axis=1,
    )


def _batched_vapour(
    cfg: SimConfig,
    r0: Float[Array, " B"],
    ambient_temperature: Float[Array, " B"] | None = None,
) -> Float[Array, " B"]:
    if not cfg.vapour.enabled:
        return jnp.zeros_like(r0)
    from core import vapour as vap

    if ambient_temperature is None:
        return jax.vmap(
            lambda r: vap.saturated_count(
                r, eqx.tree_at(lambda c: c.r0, cfg, r)
            )
        )(r0)
    return jax.vmap(
        lambda r, t_inf: vap.saturated_count(
            r,
            eqx.tree_at(
                lambda c: (c.r0, c.liquid.ambient_temperature),
                cfg,
                (r, t_inf),
            ),
        )
    )(r0, ambient_temperature)


def estimate_device_memory(batch_size: int, max_steps: int) -> float:
    return float(batch_size) * float(max_steps) * 9.0 * 8.0


__all__ = [
    "DEFAULT_STEP_BUDGET",
    "BatchSpec",
    "CycleSummary",
    "apply_spec",
    "batched_cycle_summary",
    "batched_initial_state",
    "estimate_device_memory",
]
