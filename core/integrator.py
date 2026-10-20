from __future__ import annotations

from collections.abc import Callable
from typing import Final

import diffrax as dfx
import optimistix as optx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from core import keller_miksis as km
from core.config import SimConfig

_SOLVERS: Final[dict[str, Callable[[], dfx.AbstractSolver]]] = {
    "Tsit5": dfx.Tsit5,
    "Dopri8": dfx.Dopri8,
    "Kvaerno3": dfx.Kvaerno3,
    "Kvaerno5": dfx.Kvaerno5,
}


class Scaling(eqx.Module):
    state: Float[Array, " 9"]
    time: Float[Array, ""]

    def to_nondim(self, y: Float[Array, " 9"]) -> Float[Array, " 9"]:
        return y / self.state

    def from_nondim(self, y_hat: Float[Array, " 9"]) -> Float[Array, " 9"]:
        return y_hat * self.state


def error_norm(cfg: SimConfig):
    if cfg.vapour.enabled:
        return optx.rms_norm

    def _norm(y):
        leaves = jax.tree_util.tree_leaves(y)
        total = sum(jnp.sum(leaf[..., :-1] ** 2) for leaf in leaves)
        count = sum(leaf[..., :-1].size for leaf in leaves)
        return jnp.sqrt(total / max(count, 1))

    return _norm


# characteristic scales (R0, t_R, T0) for diffrax step control
def nondim_scaling(cfg: SimConfig) -> Scaling:
    t_s = cfg.time_scale
    r_s = cfg.r0
    v_s = r_s / t_s
    e_s = cfg.liquid.ambient_pressure * cfg.v0
    state = jnp.stack(
        [
            r_s,
            v_s,
            cfg.liquid.ambient_temperature,
            r_s,
            v_s,
            jnp.ones(()),
            jnp.ones(()),
            e_s,
            jnp.maximum(cfg.n_nuclei, 1.0),
        ]
    )
    return Scaling(state=state, time=t_s)


def absolute_tolerances(
    cfg: SimConfig, scaling: Scaling | None
) -> Float[Array, " 9"]:
    it = cfg.integrator
    atol = jnp.stack(
        [
            it.atol_r,
            it.atol_rdot,
            it.atol_T,
            it.atol_a,
            it.atol_adot,
            it.atol_yield,
            it.atol_yield,
            it.atol_yield,
            1.0e-6 * jnp.maximum(cfg.n_nuclei, 1.0),
        ]
    )
    if scaling is None:
        return atol
    return atol / scaling.state


class Trajectory(eqx.Module):
    times: Float[Array, " N"]
    states: Float[Array, "N 9"]
    success: Bool[Array, ""]
    num_steps: Float[Array, ""]
    valid: Bool[Array, " N"]


    def _masked(
        self, column: int, fill: float
    ) -> Float[Array, " N"]:
        return jnp.where(self.valid, self.states[:, column], fill)

    @property
    def radius(self) -> Float[Array, " N"]:
        return self.states[:, km.IDX_R]

    @property
    def radius_dot(self) -> Float[Array, " N"]:
        return self.states[:, km.IDX_RDOT]

    @property
    def temperature(self) -> Float[Array, " N"]:
        return self.states[:, km.IDX_T]

    @property
    def distortion(self) -> Float[Array, " N"]:
        return self.states[:, km.IDX_A]

    @property
    def relative_distortion(self) -> Float[Array, " N"]:
        return jnp.abs(self.distortion) / jnp.maximum(self.radius, 1.0e-300)

    @property
    def n_valid(self) -> Float[Array, ""]:
        return jnp.sum(self.valid.astype(jnp.float64))

    @property
    def final_state(self) -> Float[Array, " 9"]:
        last = jnp.argmax(
            jnp.where(self.valid, jnp.arange(self.valid.shape[0]), -1)
        )
        return self.states[last]

    @property
    def neutron_yield(self) -> Float[Array, ""]:
        return self.final_state[km.IDX_YIELD_N]

    @property
    def peak_temperature(self) -> Float[Array, ""]:
        return jnp.max(self._masked(km.IDX_T, -jnp.inf))

    @property
    def min_radius(self) -> Float[Array, ""]:
        return jnp.min(self._masked(km.IDX_R, jnp.inf))

    @property
    def max_radius(self) -> Float[Array, ""]:
        return jnp.max(self._masked(km.IDX_R, -jnp.inf))

    @property
    def peak_wall_speed(self) -> Float[Array, ""]:
        return jnp.max(jnp.abs(self._masked(km.IDX_RDOT, 0.0)))

    @property
    def max_relative_distortion(self) -> Float[Array, ""]:
        return jnp.max(jnp.where(self.valid, self.relative_distortion, -jnp.inf))

    @property
    def expansion_ratio(self) -> Float[Array, ""]:
        return self.max_radius / jnp.maximum(self.min_radius, 1.0e-300)


def make_term(
    cfg: SimConfig,
    scaling: Scaling | None,
    rhs_fn: Callable[
        [Float[Array, ""], Float[Array, " 9"], SimConfig], Float[Array, " 9"]
    ] = km.rhs_only,
) -> dfx.ODETerm:
    if scaling is None:

        def _f_si(
            t: Float[Array, ""], y: Float[Array, " 9"], _: None
        ) -> Float[Array, " 9"]:
            return rhs_fn(t, y, cfg)

        return dfx.ODETerm(_f_si)

    def _f_hat(
        t_hat: Float[Array, ""], y_hat: Float[Array, " 9"], _: None
    ) -> Float[Array, " 9"]:
        y = scaling.from_nondim(y_hat)
        dydt = rhs_fn(t_hat * scaling.time, y, cfg)
        return dydt * scaling.time / scaling.state

    return dfx.ODETerm(_f_hat)


def _solve_impl(
    cfg: SimConfig,
    t0: Float[Array, ""],
    t1: Float[Array, ""],
    y0: Float[Array, " 9"],
    save_ts: Float[Array, " N"] | None,
    rhs_fn: Callable[
        [Float[Array, ""], Float[Array, " 9"], SimConfig], Float[Array, " 9"]
    ],
    max_steps: int,
    throw: bool,
    save_at_steps: bool,
) -> tuple[
    Float[Array, " N"], Float[Array, "N 9"], Bool[Array, ""], Float[Array, ""]
]:
    it = cfg.integrator
    scaling = nondim_scaling(cfg) if it.nondimensionalise else None

    if scaling is None:
        t0_s, t1_s, y_s, dt0 = t0, t1, y0, it.dt0
        ts_s = save_ts
    else:
        t0_s = t0 / scaling.time
        t1_s = t1 / scaling.time
        ts_s = None if save_ts is None else save_ts / scaling.time
        y_s = scaling.to_nondim(y0)
        dt0 = it.dt0 / scaling.time

    controller = dfx.PIDController(
        rtol=it.rtol,
        atol=absolute_tolerances(cfg, scaling),
        pcoeff=it.pcoeff,
        icoeff=it.icoeff,
        dcoeff=it.dcoeff,
        norm=error_norm(cfg),
    )
    solution = dfx.diffeqsolve(
        make_term(cfg, scaling, rhs_fn),
        _SOLVERS[it.solver](),
        t0=t0_s,
        t1=t1_s,
        dt0=dt0,
        y0=y_s,
        saveat=(
            dfx.SaveAt(t0=True, steps=True)
            if save_at_steps
            else dfx.SaveAt(ts=ts_s)
        ),
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=throw,
    )
    states = solution.ys
    times = solution.ts
    assert states is not None and times is not None
    if scaling is not None:
        states = states * scaling.state
        times = times * scaling.time
    return (
        times,
        states,
        solution.result == dfx.RESULTS.successful,
        jnp.asarray(solution.stats["num_steps"], dtype=jnp.float64),
    )


_solve = eqx.filter_jit(_solve_impl)


# adaptive diffrax ODE solve over [t0, t1]
def integrate(
    cfg: SimConfig,
    t0: float | Float[Array, ""],
    t1: float | Float[Array, ""],
    y0: Float[Array, " 9"] | None = None,
    n_save: int = 2048,
    rhs_fn: Callable[
        [Float[Array, ""], Float[Array, " 9"], SimConfig], Float[Array, " 9"]
    ] = km.rhs_only,
    max_steps: int | None = None,
    throw: bool = False,
    save_ts: Float[Array, " N"] | None = None,
    save_at_steps: bool = False,
) -> Trajectory:
    t0_a = jnp.asarray(t0, dtype=jnp.float64)
    t1_a = jnp.asarray(t1, dtype=jnp.float64)
    if save_at_steps:
        grid = None
    elif save_ts is None:
        grid = jnp.linspace(t0_a, t1_a, n_save)
    else:
        grid = jnp.asarray(save_ts, dtype=jnp.float64)
    y_init = km.initial_state(cfg) if y0 is None else y0

    times, states, success, num_steps = _solve(
        cfg,
        t0_a,
        t1_a,
        y_init,
        grid,
        rhs_fn,
        cfg.integrator.max_steps if max_steps is None else max_steps,
        throw,
        save_at_steps,
    )
    valid = jnp.isfinite(times) & jnp.all(jnp.isfinite(states), axis=1)
    return Trajectory(
        times=times,
        states=states,
        success=success,
        num_steps=num_steps,
        valid=valid,
    )


def diagnostics_along(
    cfg: SimConfig, trajectory: Trajectory
) -> km.Diagnostics:

    def _diag(
        t: Float[Array, ""], y: Float[Array, " 9"], ok: Bool[Array, ""]
    ) -> km.Diagnostics:
        t_safe = jnp.where(ok, t, 0.0)
        y_safe = jnp.where(ok, y, km.initial_state(cfg))
        _, diagnostics = km.rhs(t_safe, y_safe, cfg)
        return diagnostics

    return jax.vmap(_diag)(
        trajectory.times, trajectory.states, trajectory.valid
    )


__all__ = [
    "Scaling",
    "Trajectory",
    "absolute_tolerances",
    "diagnostics_along",
    "integrate",
    "make_term",
    "error_norm",
    "nondim_scaling",
]
