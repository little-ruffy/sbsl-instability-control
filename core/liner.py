from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

MU0: float = 4.0e-7 * 3.141592653589793
Q_QUADRATIC: float = 2.0
Q_LINEAR: float = 0.5
Q_SMOOTHING: float = 1.0e-6


class LinerParams(eqx.Module):
    gamma: float = eqx.field(static=True)
    n_zones: int = eqx.field(static=True)
    mass_per_length: float
    conductivity: float
    liner_mass_per_length: float
    n_fuel: int = eqx.field(static=True, default=-1)
    liner_density: float = 9000.0
    liner_sound_speed: float = 3300.0


def shell_reference_pressure(params: "LinerParams") -> Float[Array, ""]:
    return (
        params.liner_density * params.liner_sound_speed**2 / params.gamma
    )


def zone_reference_pressure(params: "LinerParams") -> Float[Array, " N"]:
    nf = n_fuel_zones(params)
    inside = jnp.arange(params.n_zones) < nf
    return jnp.where(inside, 0.0, shell_reference_pressure(params))


def n_fuel_zones(params: "LinerParams") -> int:
    return params.n_zones if params.n_fuel < 0 else params.n_fuel


def liner_is_resolved(params: "LinerParams") -> bool:
    return n_fuel_zones(params) < params.n_zones


def state_dim(n_zones: int) -> int:
    return 4 * n_zones


def zone_masses(params: LinerParams, fractions: Float[Array, " N1"]):
    nf = n_fuel_zones(params)
    areas = jnp.diff(fractions**2)
    if not liner_is_resolved(params):
        return params.mass_per_length * areas

    fuel_area = jnp.sum(areas[:nf])
    liner_area = jnp.sum(areas[nf:])
    fuel = params.mass_per_length * areas[:nf] / jnp.maximum(
        fuel_area, 1.0e-300
    )
    shell = params.liner_mass_per_length * areas[nf:] / jnp.maximum(
        liner_area, 1.0e-300
    )
    return jnp.concatenate([fuel, shell])


def uniform_fractions(n_zones: int, stretch: float = 1.0):
    j = jnp.arange(n_zones + 1, dtype=jnp.float64)
    return (j / n_zones) ** stretch


def initial_state(
    params: LinerParams,
    radius: float,
    temperature: float,
    b_axial: float,
    gas_constant: float,
    fractions: Float[Array, " N1"] | None = None,
    liner_energy: float | None = None,
) -> Float[Array, " D"]:
    n = params.n_zones
    nf = n_fuel_zones(params)
    if fractions is None:
        if liner_is_resolved(params):
            fractions = two_region_fractions(params, radius)
        else:
            fractions = uniform_fractions(n)
    frac = fractions
    r = radius * frac
    e = gas_constant * temperature / (params.gamma - 1.0)
    areas = jnp.pi * jnp.diff(r**2)

    if liner_is_resolved(params):
        p_inf = shell_reference_pressure(params)
        rho_shell = params.liner_density
        e_cold = (
            params.gamma * p_inf / ((params.gamma - 1.0) * rho_shell)
        )
        e_shell = e_cold if liner_energy is None else liner_energy
        energy = jnp.where(jnp.arange(n) < nf, e, e_shell)
    else:
        energy = jnp.full((n,), e)
    flux = b_axial * areas * (jnp.arange(n) < nf)
    return jnp.concatenate(
        [
            r[1:],
            jnp.zeros((n,)),
            energy,
            flux,
        ]
    )


def two_region_geometry(params: LinerParams, fuel_radius: float):
    frac = two_region_fractions(params, fuel_radius)
    nf = n_fuel_zones(params)
    return frac, fuel_radius / frac[nf]


def two_region_fractions(params: LinerParams, fuel_radius: float):
    nf = n_fuel_zones(params)
    nl = params.n_zones - nf
    outer_sq = fuel_radius**2 + params.liner_mass_per_length / (
        jnp.pi * params.liner_density
    )
    outer = jnp.sqrt(outer_sq)
    fuel = fuel_radius * jnp.arange(nf + 1, dtype=jnp.float64) / nf
    shell = fuel_radius + (outer - fuel_radius) * (
        jnp.arange(1, nl + 1, dtype=jnp.float64) / nl
    )
    return jnp.concatenate([fuel, shell]) / outer


def _unpack(state: Float[Array, " D"], n_zones: int):
    n = n_zones
    r_out = state[0:n]
    u_out = state[n : 2 * n]
    energy = state[2 * n : 3 * n]
    flux = state[3 * n : 4 * n]
    r = jnp.concatenate([jnp.zeros((1,)), r_out])
    u = jnp.concatenate([jnp.zeros((1,)), u_out])
    return r, u, energy, flux


def _compression(delta_u: Float[Array, " N"], scale: Float[Array, " N"]):
    eps = jnp.maximum(Q_SMOOTHING * scale, 1.0e-30)
    x = jnp.clip(-delta_u / eps, 0.0, 1.0)
    return jnp.where(delta_u <= -eps, -delta_u, eps * x**2 * (2.0 - x))


def zone_fields(
    state: Float[Array, " D"],
    params: LinerParams,
    gas_constant: float,
    fractions: Float[Array, " N1"] | None = None,
):
    n = params.n_zones
    frac = uniform_fractions(n) if fractions is None else fractions
    masses = zone_masses(params, frac)

    r, u, energy, flux = _unpack(state, n)
    area = jnp.maximum(jnp.pi * jnp.diff(r**2), 1.0e-300)
    density = masses / area
    p_inf = zone_reference_pressure(params)
    pressure = (params.gamma - 1.0) * density * energy - params.gamma * p_inf
    b_field = flux / area
    temperature = (params.gamma - 1.0) * energy / gas_constant
    return dict(
        r=r, u=u, area=area, density=density, pressure=pressure,
        b_field=b_field, temperature=temperature, masses=masses,
        energy=energy, flux=flux, p_inf=p_inf,
    )


# 1D cylindrical magnetically imploded conducting liner + D2 fuel ODE
def rhs(
    state: Float[Array, " D"],
    params: LinerParams,
    drive_pressure: Float[Array, ""],
    gas_constant: float,
    fractions: Float[Array, " N1"] | None = None,
) -> Float[Array, " D"]:
    n = params.n_zones
    f = zone_fields(state, params, gas_constant, fractions)
    r, u = f["r"], f["u"]
    masses = f["masses"]

    sound_speed = jnp.sqrt(
        jnp.maximum(
            params.gamma * (f["pressure"] + f["p_inf"]) / f["density"],
            1.0e-30,
        )
    )

    delta_u = jnp.diff(u)
    comp = _compression(delta_u, sound_speed)
    q_visc = f["density"] * comp * (
        Q_QUADRATIC**2 * comp + Q_LINEAR * sound_speed
    )

    p_mag = f["b_field"] ** 2 / (2.0 * MU0)
    total = f["pressure"] + p_mag + q_visc

    m_int = jnp.maximum(0.5 * (masses[:-1] + masses[1:]), 1.0e-300)
    du_interior = (
        2.0 * jnp.pi * r[1:-1] * (total[:-1] - total[1:]) / m_int
    )
    extra = 0.0 if liner_is_resolved(params) else params.liner_mass_per_length
    m_outer = jnp.maximum(0.5 * masses[-1] + extra, 1.0e-300)
    du_outer = 2.0 * jnp.pi * r[-1] * (total[-1] - drive_pressure) / m_outer
    du_dt = jnp.concatenate([du_interior, du_outer[None]])

    area_dot = 2.0 * jnp.pi * jnp.diff(r * u)
    de_dt = -(f["pressure"] + q_visc) * area_dot / masses

    if params.conductivity > 0.0:
        eta = 1.0 / (MU0 * params.conductivity)
        r_centre = 0.5 * (r[:-1] + r[1:])
        db_dr = jnp.diff(f["b_field"]) / jnp.maximum(
            jnp.diff(r_centre), 1.0e-30
        )
        transport = 2.0 * jnp.pi * r[1:-1] * eta * db_dr
        inflow = jnp.concatenate([jnp.zeros((1,)), transport])
        outflow = jnp.concatenate([transport, jnp.zeros((1,))])
        dflux_dt = outflow - inflow
        dflux_dt = dflux_dt * (jnp.arange(n) < n_fuel_zones(params))
    else:
        dflux_dt = jnp.zeros((n,))

    return jnp.concatenate([u[1:], du_dt, de_dt, dflux_dt])


def diagnostics(
    state: Float[Array, " D"],
    params: LinerParams,
    gas_constant: float,
    velocity: float = 0.0,
    fractions: Float[Array, " N1"] | None = None,
) -> dict:
    f = zone_fields(state, params, gas_constant, fractions)
    radius = f["r"][-1]
    b_max = jnp.max(jnp.abs(f["b_field"]))
    p_gas_max = jnp.max(f["pressure"])
    p_mag_max = b_max**2 / (2.0 * MU0)

    rm = (
        MU0 * params.conductivity * jnp.abs(velocity) * radius
        if params.conductivity > 0.0
        else jnp.inf
    )
    total_flux = jnp.sum(f["flux"])
    nf = n_fuel_zones(params)
    shell_pressure = (
        jnp.max(f["pressure"][nf:]) if liner_is_resolved(params)
        else jnp.asarray(0.0)
    )
    return dict(
        radius=radius,
        shell_pressure=shell_pressure,
        b_max=b_max,
        beta=p_gas_max / jnp.maximum(p_mag_max, 1.0e-300),
        magnetic_reynolds=rm,
        total_flux=total_flux,
        temperature_max=jnp.max(f["temperature"]),
        density_max=jnp.max(f["density"]),
    )


__all__ = [
    "MU0",
    "LinerParams",
    "diagnostics",
    "liner_is_resolved",
    "n_fuel_zones",
    "shell_reference_pressure",
    "two_region_fractions",
    "two_region_geometry",
    "zone_reference_pressure",
    "initial_state",
    "rhs",
    "state_dim",
    "uniform_fractions",
    "zone_fields",
    "zone_masses",
]
