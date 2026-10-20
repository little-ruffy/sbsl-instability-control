from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from core import constants as C
from core import keller_miksis as km
from core import plasma_core as pc
from core.config import SimConfig
from core.integrator import _SOLVERS, nondim_scaling
from core.keller_miksis import drive_pressure

# Von Neumann-Richtmyer artificial viscosity parameters
Q_QUADRATIC: float = 2.0

Q_LINEAR: float = 0.5

Q_SMOOTHING: float = 1.0e-3


DEFAULT_STRETCH: float = 1.0


def state_dim(n_zones: int) -> int:
    return 3 * n_zones + 4


class ZoneState(eqx.Module):
    radius_inner: Float[Array, " N"]
    radius_outer: Float[Array, " N"]
    volume: Float[Array, " N"]
    v_free: Float[Array, " N"]
    density: Float[Array, " N"]
    temperature: Float[Array, " N"]
    pressure: Float[Array, " N"]
    q_viscous: Float[Array, " N"]
    sound_speed: Float[Array, " N"]
    mach: Float[Array, " N"]


class InteriorDiagnostics(eqx.Module):
    zones: ZoneState
    wall_radius: Float[Array, ""]
    wall_velocity: Float[Array, ""]
    wall_acceleration: Float[Array, ""]
    wall_mach: Float[Array, ""]
    gas_pressure: Float[Array, ""]
    central_temperature: Float[Array, ""]
    mean_temperature: Float[Array, ""]
    peak_temperature: Float[Array, ""]
    amplification: Float[Array, ""]
    conduction_flux: Float[Array, " M"]
    radiated_power: Float[Array, ""]
    neutron_rate: Float[Array, ""]
    shock_index: Float[Array, ""]


def grid_fractions(n_zones: int, stretch: float = DEFAULT_STRETCH):
    j = jnp.arange(0, n_zones + 1, dtype=jnp.float64) / n_zones
    return j**stretch


def zone_nuclei(
    cfg: SimConfig, n_zones: int, stretch: float = DEFAULT_STRETCH
) -> Float[Array, " N"]:
    frac = grid_fractions(n_zones, stretch)
    return cfg.n_nuclei * jnp.diff(frac**3)


def zone_mass(
    cfg: SimConfig, n_zones: int, stretch: float = DEFAULT_STRETCH
) -> Float[Array, " N"]:
    frac = grid_fractions(n_zones, stretch)
    return pc.total_gas_mass(cfg) * jnp.diff(frac**3)


def uniform_grid(
    radius: Float[Array, ""],
    n_zones: int,
    stretch: float = DEFAULT_STRETCH,
) -> Float[Array, " N"]:
    return radius * grid_fractions(n_zones, stretch)[1:]


def initial_state_from_uniform(
    cfg: SimConfig,
    radius: float | Float[Array, ""],
    radius_dot: float | Float[Array, ""],
    temperature: float | Float[Array, ""],
    n_zones: int,
    distortion: float | Float[Array, ""] = 0.0,
    distortion_dot: float | Float[Array, ""] = 0.0,
    stretch: float = DEFAULT_STRETCH,
) -> Float[Array, " D"]:
    radius = jnp.asarray(radius, dtype=jnp.float64)
    radius_dot = jnp.asarray(radius_dot, dtype=jnp.float64)
    temperature = jnp.asarray(temperature, dtype=jnp.float64)

    r = uniform_grid(radius, n_zones, stretch)
    u = radius_dot * r / radius
    t = jnp.full((n_zones,), temperature)
    return jnp.concatenate(
        [
            r,
            u,
            t,
            jnp.asarray([distortion, distortion_dot, 0.0, 0.0], dtype=jnp.float64),
        ]
    )


def _unpack(
    state: Float[Array, " D"], n_zones: int
) -> tuple[
    Float[Array, " M"],
    Float[Array, " M"],
    Float[Array, " N"],
    Float[Array, ""],
    Float[Array, ""],
]:
    zero = jnp.zeros((1,))
    r = jnp.concatenate([zero, state[0:n_zones]])
    u = jnp.concatenate([zero, state[n_zones : 2 * n_zones]])
    t = state[2 * n_zones : 3 * n_zones]
    return r, u, t, state[3 * n_zones], state[3 * n_zones + 1]


# C1 cubic Hermite compression switch (exact zero in expansion)
def _compression(delta_u: Float[Array, " N"], scale: Float[Array, " N"]):
    eps = jnp.maximum(Q_SMOOTHING * scale, 1.0e-30)
    u = jnp.clip(-delta_u / eps, 0.0, 1.0)
    return jnp.where(delta_u <= -eps, -delta_u, eps * u**2 * (2.0 - u))


def _flux_limited(
    q: Float[Array, " M"],
    q_free_stream: Float[Array, " M"],
) -> Float[Array, " M"]:
    return q / (1.0 + jnp.abs(q) / jnp.maximum(q_free_stream, 1.0e-30))


def zone_thermo(
    temperature: Float[Array, " N"],
    v_free: Float[Array, " N"],
    nuclei: Float[Array, " N"],
    cfg: SimConfig,
):
    counts = nuclei[:, None] * cfg.gas.mole_frac[None, :]
    comp = jax.vmap(lambda t, v, n: pc.composition(t, v, cfg, n))(
        temperature, v_free, counts
    )
    eos = jax.vmap(
        lambda t, v, n: pc.equation_of_state_and_derivatives(t, v, cfg, n)
    )(temperature, v_free, counts)
    return comp, eos


# 1D spherical Lagrangian hydro + KM wall boundary
def rhs(
    time: Float[Array, ""],
    state: Float[Array, " D"],
    cfg: SimConfig,
    n_zones: int,
    stretch: float = DEFAULT_STRETCH,
) -> tuple[Float[Array, " D"], InteriorDiagnostics]:
    guards = cfg.guards
    liquid = cfg.liquid

    r_raw, u, temperature_raw, a_l, adot_l = _unpack(state, n_zones)

    increments = jnp.maximum(jnp.diff(r_raw), 0.0)
    r = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(increments)])
    radius = jnp.maximum(r[-1], cfg.r_floor)

    temperature = jnp.clip(temperature_raw, guards.t_min, guards.t_max)

    nuclei = zone_nuclei(cfg, n_zones, stretch)
    masses = zone_mass(cfg, n_zones, stretch)

    r_in, r_out = r[:-1], r[1:]
    u_in, u_out = u[:-1], u[1:]
    volume = (4.0 / 3.0) * jnp.pi * (r_out**3 - r_in**3)
    v_hc = nuclei * cfg.gas.hardcore_volume_per_nucleus
    v_free = jnp.maximum(volume - v_hc, 0.02 * jnp.maximum(v_hc, 1.0e-30))
    volume_dot = 4.0 * jnp.pi * (r_out**2 * u_out - r_in**2 * u_in)
    density = masses / jnp.maximum(volume, 1.0e-300)

    comp, eos = zone_thermo(temperature, v_free, nuclei, cfg)
    pressure = eos.pressure
    sound_speed = jax.vmap(lambda c, e: pc.sound_speed_gas(c, e, cfg))(comp, eos)

    delta_u = u_out - u_in
    compression = _compression(delta_u, jnp.maximum(sound_speed, 1.0))
    q_visc = density * (
        Q_QUADRATIC**2 * compression**2 + Q_LINEAR * sound_speed * compression
    )
    total_pressure = pressure + q_visc

    m_interface = jnp.maximum(0.5 * (masses[:-1] + masses[1:]), 1.0e-300)
    r_mid = r[1:-1]
    du_dt_interior = (
        4.0
        * jnp.pi
        * r_mid**2
        * (total_pressure[:-1] - total_pressure[1:])
        / m_interface
    )

    kappa = jax.vmap(lambda c: pc.thermal_conductivity(c, cfg))(comp)
    r_centre = 0.5 * (r_in + r_out)

    if cfg.transport.conduction_enabled and cfg.gas_enabled:
        kappa_face = 0.5 * (kappa[:-1] + kappa[1:])
        distance = jnp.maximum(r_centre[1:] - r_centre[:-1], 1.0e-15)
        q_face = kappa_face * (temperature[:-1] - temperature[1:]) / distance

        d_wall = jnp.maximum(radius - r_centre[-1], 1.0e-15)
        q_wall = (
            kappa[-1] * (temperature[-1] - liquid.ambient_temperature) / d_wall
        )
        q_all = jnp.concatenate([q_face, q_wall[None]])

        if cfg.transport.flux_limiter_enabled:
            v_th_e = jnp.sqrt(C.K_B * temperature / C.M_ELECTRON)
            m_heavy = jnp.sum(cfg.gas.mole_frac * cfg.gas.mass_per_nucleus)
            v_th_h = jnp.sqrt(
                C.K_B * temperature / jnp.maximum(m_heavy, 1.0e-30)
            )
            q_fs = (
                cfg.transport.flux_limiter_factor
                * C.K_B
                * temperature
                * (comp.n_e_density * v_th_e + comp.n_heavy_density * v_th_h)
            )
            q_fs_face = jnp.concatenate(
                [jnp.minimum(q_fs[:-1], q_fs[1:]), q_fs[-1][None]]
            )
            q_all = _flux_limited(q_all, q_fs_face)

        power_out = jnp.concatenate(
            [jnp.zeros((1,)), 4.0 * jnp.pi * r[1:] ** 2 * q_all]
        )
    else:
        power_out = jnp.zeros((n_zones + 1,))

    conduction_net = power_out[1:] - power_out[:-1]

    if cfg.transport.bremsstrahlung_enabled and cfg.gas_enabled:
        thin = (
            jax.vmap(lambda c: pc.bremsstrahlung_power_density(c, cfg))(comp)
            * v_free
        )
        if cfg.transport.blackbody_cap_enabled:
            kappa = jax.vmap(lambda c: pc.planck_mean_absorption(c, cfg))(comp)
            thickness = jnp.maximum(r_out - r_in, 0.0)
            tau_self = kappa * thickness
            tau_outside = jnp.concatenate(
                [jnp.cumsum(tau_self[::-1])[::-1][1:], jnp.zeros((1,))]
            )
            self_absorption = jnp.where(
                tau_self > 1.0e-12,
                -jnp.expm1(-tau_self) / jnp.maximum(tau_self, 1.0e-300),
                1.0,
            )
            radiated = thin * self_absorption * jnp.exp(-tau_outside)
        else:
            radiated = thin
    else:
        radiated = jnp.zeros((n_zones,))

    rates, neutron_rate, deposited = jax.vmap(
        lambda c: pc.fusion_rates(c, cfg)
    )(comp)
    yield_n_dot = jnp.sum(neutron_rate)
    yield_p_dot = jnp.sum(rates.get("DD_pT", jnp.zeros((n_zones,))))

    heat_capacity = jnp.maximum(eos.du_dt, 1.0e-300)
    t_dot = (
        -(pressure + eos.du_dv + q_visc) * volume_dot
        - conduction_net
        - radiated
        + deposited
    ) / heat_capacity

    c = liquid.sound_speed
    rho = liquid.density
    mu = liquid.dynamic_viscosity
    sigma = liquid.surface_tension
    rdot_cap = guards.rdot_max_mach * c
    radius_dot = jnp.clip(u[-1], -rdot_cap, rdot_cap)
    mach = radius_dot / c

    p_gas = total_pressure[-1]
    p_gas_dot = eos.dp_dt[-1] * t_dot[-1] + eos.dp_dv[-1] * volume_dot[-1]

    p_wall = (
        p_gas
        + liquid.vapour_pressure
        - 2.0 * sigma / radius
        - 4.0 * mu * radius_dot / radius
    )
    p_inf, p_inf_dot = drive_pressure(
        time, cfg.acoustics, liquid.ambient_pressure
    )
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
        a_coeff, b_coeff = km.shape_mode_coefficients(
            radius, radius_dot, radius_ddot, cfg
        )
        addot_l = a_coeff * a_l - b_coeff * adot_l
    else:
        adot_l = jnp.zeros(())
        addot_l = jnp.zeros(())

    du_dt = jnp.concatenate([du_dt_interior, radius_ddot[None]])
    derivative = jnp.concatenate(
        [
            u[1:],
            du_dt,
            t_dot,
            jnp.stack([adot_l, addot_l, yield_n_dot, yield_p_dot]),
        ]
    )

    mean_temperature = jnp.sum(temperature * volume) / jnp.sum(volume)
    diagnostics = InteriorDiagnostics(
        zones=ZoneState(
            radius_inner=r_in,
            radius_outer=r_out,
            volume=volume,
            v_free=v_free,
            density=density,
            temperature=temperature,
            pressure=pressure,
            q_viscous=q_visc,
            sound_speed=sound_speed,
            mach=jnp.abs(delta_u) / jnp.maximum(sound_speed, 1.0e-300),
        ),
        wall_radius=radius,
        wall_velocity=radius_dot,
        wall_acceleration=radius_ddot,
        wall_mach=mach,
        gas_pressure=p_gas,
        central_temperature=temperature[0],
        mean_temperature=mean_temperature,
        peak_temperature=jnp.max(temperature),
        amplification=temperature[0] / jnp.maximum(mean_temperature, 1.0e-300),
        conduction_flux=power_out,
        radiated_power=jnp.sum(radiated),
        neutron_rate=yield_n_dot,
        shock_index=jnp.argmax(compression).astype(jnp.float64),
    )
    return derivative, diagnostics


def rhs_only(
    time: Float[Array, ""],
    state: Float[Array, " D"],
    args: tuple[SimConfig, int, float],
) -> Float[Array, " D"]:
    cfg, n_zones, stretch = args
    return rhs(time, state, cfg, n_zones, stretch)[0]


def diagnostics_along(
    cfg: SimConfig,
    times: Float[Array, " S"],
    states: Float[Array, "S D"],
    n_zones: int,
    stretch: float = DEFAULT_STRETCH,
) -> InteriorDiagnostics:
    return jax.vmap(lambda t, y: rhs(t, y, cfg, n_zones, stretch)[1])(
        times, states
    )


class InteriorTrajectory(eqx.Module):
    times: Float[Array, " S"]
    states: Float[Array, "S D"]
    n_zones: int = eqx.field(static=True)
    success: Bool[Array, ""]
    num_steps: Float[Array, ""]
    valid: Bool[Array, " S"]

    @property
    def wall_radius(self) -> Float[Array, " S"]:
        return self.states[:, self.n_zones - 1]

    @property
    def temperatures(self) -> Float[Array, "S N"]:
        return self.states[:, 2 * self.n_zones : 3 * self.n_zones]

    @property
    def central_temperature(self) -> Float[Array, " S"]:
        return self.states[:, 2 * self.n_zones]


def interior_tolerances(
    cfg: SimConfig, n_zones: int
) -> Float[Array, " D"]:
    it = cfg.integrator
    radius_scale = float(n_zones) ** (1.0 / 3.0)
    return jnp.concatenate(
        [
            jnp.full((n_zones,), it.atol_r / radius_scale),
            jnp.full((n_zones,), it.atol_rdot),
            jnp.full((n_zones,), it.atol_T),
            jnp.stack([it.atol_a, it.atol_adot, it.atol_yield, it.atol_yield]),
        ]
    )


# rescale time by t_s to avoid diffrax 1e-10 s clip_to_end lockup
def _integrate_impl(
    cfg: SimConfig,
    t0: Float[Array, ""],
    t1: Float[Array, ""],
    y0: Float[Array, " D"],
    n_zones: int,
    stretch: float,
    save_ts: Float[Array, " S"] | None,
    max_steps: int,
    throw: bool,
    save_at_steps: bool,
):
    it_cfg = cfg.integrator
    t_scale = nondim_scaling(cfg).time

    def _f(t_hat, y, _):
        return t_scale * rhs_only(t_hat * t_scale, y, (cfg, n_zones, stretch))

    controller = dfx.PIDController(
        rtol=it_cfg.rtol,
        atol=interior_tolerances(cfg, n_zones),
        pcoeff=it_cfg.pcoeff,
        icoeff=it_cfg.icoeff,
        dcoeff=it_cfg.dcoeff,
    )
    solution = dfx.diffeqsolve(
        dfx.ODETerm(_f),
        _SOLVERS[it_cfg.solver](),
        t0=t0 / t_scale,
        t1=t1 / t_scale,
        dt0=it_cfg.dt0 / t_scale,
        y0=y0,
        saveat=(
            dfx.SaveAt(t0=True, steps=True)
            if save_at_steps
            else dfx.SaveAt(ts=save_ts / t_scale)
        ),
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=throw,
    )
    return (
        solution.ts * t_scale,
        solution.ys,
        solution.result == dfx.RESULTS.successful,
        jnp.asarray(solution.stats["num_steps"], dtype=jnp.float64),
    )


_integrate_jit = eqx.filter_jit(_integrate_impl)


def integrate_interior(
    cfg: SimConfig,
    t0: float | Float[Array, ""],
    t1: float | Float[Array, ""],
    y0: Float[Array, " D"],
    n_zones: int,
    stretch: float = DEFAULT_STRETCH,
    n_save: int = 1024,
    max_steps: int = 1_000_000,
    throw: bool = False,
    save_ts: Float[Array, " S"] | None = None,
    save_at_steps: bool = False,
) -> InteriorTrajectory:
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    t1 = jnp.asarray(t1, dtype=jnp.float64)
    if save_ts is None and not save_at_steps:
        save_ts = jnp.linspace(t0, t1, n_save)

    times, states, success, num_steps = _integrate_jit(
        cfg, t0, t1, y0, n_zones, stretch, save_ts, max_steps, throw,
        save_at_steps,
    )
    valid = jnp.isfinite(times)
    return InteriorTrajectory(
        times=jnp.where(valid, times, t1),
        states=states,
        n_zones=n_zones,
        success=success,
        num_steps=num_steps,
        valid=valid,
    )


__all__ = [
    "InteriorDiagnostics",
    "InteriorTrajectory",
    "Q_LINEAR",
    "Q_QUADRATIC",
    "Q_SMOOTHING",
    "ZoneState",
    "DEFAULT_STRETCH",
    "diagnostics_along",
    "grid_fractions",
    "integrate_interior",
    "interior_tolerances",
    "initial_state_from_uniform",
    "rhs",
    "rhs_only",
    "state_dim",
    "uniform_grid",
    "zone_mass",
    "zone_nuclei",
    "zone_thermo",
]
