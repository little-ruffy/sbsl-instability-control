from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

RHO0_DEFAULT: float = 1.0
PINF_DEFAULT: float = 500.0
GAMMA_L_DEFAULT: float = 7.14
PI1_DEFAULT: float = 30000.0
C_INF_DEFAULT: float = math.sqrt(
    GAMMA_L_DEFAULT * (PINF_DEFAULT + PI1_DEFAULT) / RHO0_DEFAULT
)


@dataclass(frozen=True)
class SphericalEulerResult:
    r_min: float
    t_min: float
    compression: float
    peak_mach_ambient: float
    peak_mach_local: float
    steps: int


# analytical gas cushion compliance S(gamma, C)
def thermodynamic_compliance(gamma: float, compression: float) -> float:
    c = max(float(compression), 1.0)
    dg = float(gamma) - 1.0
    if abs(dg) < 1.0e-8:
        return math.log(c)
    return (1.0 - c ** (-3.0 * dg)) / (3.0 * dg)


def quantized_volume_radius(
    r_exact: float, decimals: int = 6
) -> tuple[float, float, float]:
    r = max(float(r_exact), 0.0)
    v_code = (r ** 3) / 3.0
    v_quant = round(v_code, decimals)
    r_quant = (3.0 * max(v_quant, 0.0)) ** (1.0 / 3.0)
    return v_code, v_quant, r_quant


def _minmod(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(
        a * b <= 0.0,
        0.0,
        jnp.where(jnp.abs(a) < jnp.abs(b), a, b),
    )


@functools.partial(jax.jit, static_argnames=("n_cells", "init_laplace", "max_steps"))
def _solve_spherical_euler_jit(
    pg0: float,
    gam_g: float,
    sig: float,
    mu_l: float,
    t_end: float,
    r_max: float,
    pinf: float,
    gamma_l: float,
    pi1: float,
    rho0: float,
    n_cells: int = 350,
    init_laplace: bool = True,
    max_steps: int = 250_000,
):
    c_inf = jnp.sqrt(gamma_l * (pinf + pi1) / rho0)
    eta = jnp.linspace(0.0, 1.0, n_cells + 1, dtype=jnp.float64)
    r_w0 = jnp.asarray(1.0, dtype=jnp.float64)
    u_w0 = jnp.asarray(0.0, dtype=jnp.float64)

    r_f0 = r_w0 * (r_max / r_w0) ** eta
    r_c0 = 0.5 * (r_f0[:-1] + r_f0[1:])

    p_w0 = pg0 - 2.0 * sig / r_w0
    if init_laplace:
        p0 = pinf * (1.0 - r_w0 / r_c0) + p_w0 * (r_w0 / r_c0)
    else:
        p0 = jnp.full((n_cells,), pinf, dtype=jnp.float64)
    rho_init = rho0 * ((p0 + pi1) / (pinf + pi1)) ** (1.0 / gamma_l)
    u0 = jnp.zeros((n_cells,), dtype=jnp.float64)

    def eval_rhs(r_w, u_w, rho, u, p):
        ratio = r_max / r_w
        r_f = r_w * (ratio ** eta)
        w_f = (1.0 - eta) * (u_w / r_w) * r_f
        r_c = 0.5 * (r_f[:-1] + r_f[1:])
        dr = r_f[1:] - r_f[:-1]
        w_c = 0.5 * (w_f[:-1] + w_f[1:])

        c = jnp.sqrt(
            jnp.maximum(gamma_l * (p + pi1) / jnp.maximum(rho, 1.0e-12), 1.0e-6)
        )
        dt_c = 0.45 * dr / (c + jnp.abs(u - w_c) + 1.0e-12)
        dt_min = jnp.minimum(jnp.min(dt_c), 1.0e-4)

        p_gas = pg0 * (1.0 / r_w) ** (3.0 * gam_g)
        p_wall = p_gas - 2.0 * sig / r_w - 4.0 * mu_l * u_w / r_w
        rho_w = rho0 * (
            (jnp.maximum(p_wall, -pi1 + 1.0) + pi1) / (pinf + pi1)
        ) ** (1.0 / gamma_l)

        q = (r_c * r_c) * u
        q_w = (r_w * r_w) * u_w

        r_ext = jnp.concatenate([jnp.array([r_w]), r_c])
        rho_ext = jnp.concatenate([jnp.array([rho_w]), rho])
        q_ext = jnp.concatenate([jnp.array([q_w]), q])
        p_ext = jnp.concatenate([jnp.array([p_wall]), p])

        dr_l = r_c[:-1] - r_ext[:-2]
        dr_r = r_c[1:] - r_c[:-1]

        sl_rho = jnp.concatenate(
            [
                _minmod(
                    (rho[:-1] - rho_ext[:-2]) / dr_l,
                    (rho[1:] - rho[:-1]) / dr_r,
                ),
                jnp.zeros((1,), dtype=jnp.float64),
            ]
        )
        sl_q = jnp.concatenate(
            [
                _minmod(
                    (q[:-1] - q_ext[:-2]) / dr_l,
                    (q[1:] - q[:-1]) / dr_r,
                ),
                jnp.zeros((1,), dtype=jnp.float64),
            ]
        )
        sl_p = jnp.concatenate(
            [
                _minmod(
                    (p[:-1] - p_ext[:-2]) / dr_l,
                    (p[1:] - p[:-1]) / dr_r,
                ),
                jnp.zeros((1,), dtype=jnp.float64),
            ]
        )

        rf = r_f[1:-1]
        drL = rf - r_c[:-1]
        drR = r_c[1:] - rf
        pL = p[:-1] + sl_p[:-1] * drL
        pR = p[1:] - sl_p[1:] * drR
        qL = q[:-1] + sl_q[:-1] * drL
        qR = q[1:] - sl_q[1:] * drR
        uL = qL / (rf * rf)
        uR = qR / (rf * rf)
        zL = rho[:-1] * c[:-1]
        zR = rho[1:] * c[1:]
        inv_z = 1.0 / (zL + zR)

        u_int = (zL * uL + zR * uR + (pL - pR)) * inv_z
        p_int = (zR * pL + zL * pR + zL * zR * (uL - uR)) * inv_z

        u_outer = q[-1] / (r_f[-1] * r_f[-1])
        p_outer = pinf + rho0 * c_inf * u_outer

        u_face = jnp.concatenate(
            [jnp.array([u_w]), u_int, jnp.array([u_outer])]
        )
        p_face = jnp.concatenate(
            [jnp.array([p_wall]), p_int, jnp.array([p_outer])]
        )

        h0 = r_c[0] - r_w
        h1 = r_c[1] - r_w
        dp_dr_w = (
            p_wall * (-(h0 + h1) / (h0 * h1))
            + p[0] * (h1 / (h0 * (h1 - h0)))
            - p[1] * (h0 / (h1 * (h1 - h0)))
        )
        du_w_dt = -dp_dr_w / rho_w

        r0 = r_f[:-1]
        r1 = r_f[1:]
        vol = (r1 * r1 * r1 - r0 * r0 * r0) / 3.0
        v_rel = u - w_c

        div_u = (r1 * r1 * u_face[1:] - r0 * r0 * u_face[:-1]) / vol
        dp_dr = (p_face[1:] - p_face[:-1]) / dr
        du_dr_adv = jnp.where(
            v_rel <= 0.0,
            (u_face[1:] - u) / (0.5 * dr),
            (u - u_face[:-1]) / (0.5 * dr),
        )

        drho_dt = -v_rel * sl_rho - rho * div_u
        du_dt = -v_rel * du_dr_adv - dp_dr / rho
        dp_dt = -v_rel * sl_p - rho * (c * c) * div_u
        return dt_min, du_w_dt, p_wall, drho_dt, du_dt, dp_dt

    init_carry = (
        r_w0,
        u_w0,
        rho_init,
        u0,
        p0,
        jnp.asarray(0.0, dtype=jnp.float64),
        r_w0,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def cond_fn(carry):
        r_w, u_w, _, _, _, t, r_min, _, _, _, steps = carry
        not_rebounded = jnp.logical_not(
            jnp.logical_and(u_w > 0.0, jnp.logical_and(r_w > 1.05 * r_min, steps > 50))
        )
        return jnp.logical_and(
            t < t_end - 1.0e-15,
            jnp.logical_and(steps < max_steps, not_rebounded),
        )

    def body_fn(carry):
        r_w, u_w, rho, u, p, t, r_min, t_min, max_m_amb, max_m_loc, steps = carry
        dt_min, du_w1, p_wall, drho1, du1, dp1 = eval_rhs(r_w, u_w, rho, u, p)
        dt = jnp.minimum(dt_min, t_end - t)

        m_amb = jnp.abs(u_w) / c_inf
        max_m_amb = jnp.maximum(max_m_amb, m_amb)
        c_w = jnp.sqrt(jnp.maximum(gamma_l * (p_wall + pi1) / rho0, 1.0e-6))
        m_loc = jnp.abs(u_w) / c_w
        max_m_loc = jnp.maximum(max_m_loc, m_loc)

        rw1 = r_w + dt * u_w
        uw1 = u_w + dt * du_w1
        rho1 = rho + dt * drho1
        u1 = u + dt * du1
        p1 = p + dt * dp1

        _, du_w2, _, drho2, du2, dp2 = eval_rhs(rw1, uw1, rho1, u1, p1)

        r_w_new = r_w + 0.5 * dt * (u_w + uw1)
        u_w_new = u_w + 0.5 * dt * (du_w1 + du_w2)
        rho_new = rho + 0.5 * dt * (drho1 + drho2)
        u_new = u + 0.5 * dt * (du1 + du2)
        p_new = p + 0.5 * dt * (dp1 + dp2)

        t_new = t + dt
        is_smaller = r_w_new < r_min
        r_min_new = jnp.where(is_smaller, r_w_new, r_min)
        t_min_new = jnp.where(is_smaller, t_new, t_min)

        return (
            r_w_new,
            u_w_new,
            rho_new,
            u_new,
            p_new,
            t_new,
            r_min_new,
            t_min_new,
            max_m_amb,
            max_m_loc,
            steps + 1,
        )

    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    _, _, _, _, _, _, r_min, t_min, max_m_amb, max_m_loc, steps = final_carry
    return r_min, t_min, max_m_amb, max_m_loc, steps


# 1D co-collapsing ALE spherical compressible Euler solver
def solve_spherical_euler(
    pg0: float,
    gam_g: float,
    sig: float = 0.0,
    mu_l: float = 0.0,
    t_end: float | None = None,
    *,
    n_cells: int = 350,
    r_max: float = 50.0,
    pinf: float = PINF_DEFAULT,
    gamma_l: float = GAMMA_L_DEFAULT,
    pi1: float = PI1_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    init_laplace: bool = True,
) -> SphericalEulerResult:
    if t_end is None:
        t_end = 2.0 * 0.915 * math.sqrt(rho0 / max(pinf - pg0, 1.0))
    r_min, t_min, m_amb, m_loc, steps = _solve_spherical_euler_jit(
        float(pg0),
        float(gam_g),
        float(sig),
        float(mu_l),
        float(t_end),
        float(r_max),
        float(pinf),
        float(gamma_l),
        float(pi1),
        float(rho0),
        n_cells=int(n_cells),
        init_laplace=bool(init_laplace),
    )
    r_min_f = float(r_min)
    return SphericalEulerResult(
        r_min=r_min_f,
        t_min=float(t_min),
        compression=1.0 / max(r_min_f, 1.0e-30),
        peak_mach_ambient=float(m_amb),
        peak_mach_local=float(m_loc),
        steps=int(steps),
    )


def solve_gilmore_polytrope(
    pg0: float,
    gam_g: float,
    sig: float = 0.0,
    mu_l: float = 0.0,
    t_end: float | None = None,
    *,
    pinf: float = PINF_DEFAULT,
    gamma_l: float = GAMMA_L_DEFAULT,
    pi1: float = PI1_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    dt: float = 1.5e-7,
) -> SphericalEulerResult:
    if t_end is None:
        t_end = 2.0 * 0.915 * math.sqrt(rho0 / max(pinf - pg0, 1.0))
    c_inf = math.sqrt(gamma_l * (pinf + pi1) / rho0)
    b_tait = pinf + pi1

    def gil_rhs(r: float, rd: float) -> tuple[float, float, float, float]:
        r_s = max(r, 1.0e-12)
        p_gas = pg0 * (1.0 / r_s) ** (3.0 * gam_g)
        dp_gas_dr = -3.0 * gam_g * p_gas / r_s
        p_w = p_gas - 2.0 * sig / r_s - 4.0 * mu_l * rd / r_s
        dp_w_dr = (
            dp_gas_dr
            + 2.0 * sig / (r_s * r_s)
            + 4.0 * mu_l * rd / (r_s * r_s)
        )
        ratio = max((p_w + pi1) / b_tait, 1.0e-12)
        h = (
            (gamma_l / (gamma_l - 1.0))
            * (b_tait / rho0)
            * (ratio ** ((gamma_l - 1.0) / gamma_l) - 1.0)
        )
        c_w = c_inf * (ratio ** ((gamma_l - 1.0) / (2.0 * gamma_l)))
        dh_dp = (1.0 / rho0) * (ratio ** (-1.0 / gamma_l))
        dh_dr = dh_dp * dp_w_dr
        mach_loc = rd / c_w
        denom = r_s * (1.0 - mach_loc)
        num = (
            (1.0 + mach_loc) * h
            + (r_s / c_w) * (1.0 - mach_loc) * dh_dr * rd
            - 1.5 * (1.0 - mach_loc / 3.0) * rd * rd
        )
        return rd, num / denom, abs(rd) / c_inf, abs(mach_loc)

    r, rd = 1.0, 0.0
    tc = 0.0
    rmin, tmin = 1.0, 0.0
    max_m_amb, max_m_loc = 0.0, 0.0
    steps = 0
    while tc < t_end and r > 1.0e-4:
        k1_r, k1_v, m_a, m_l = gil_rhs(r, rd)
        if m_a > max_m_amb:
            max_m_amb = m_a
        if m_l > max_m_loc:
            max_m_loc = m_l
        k2_r, k2_v, _, _ = gil_rhs(r + 0.5 * dt * k1_r, rd + 0.5 * dt * k1_v)
        k3_r, k3_v, _, _ = gil_rhs(r + 0.5 * dt * k2_r, rd + 0.5 * dt * k2_v)
        k4_r, k4_v, _, _ = gil_rhs(r + dt * k3_r, rd + dt * k3_v)
        r += (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        rd += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        tc += dt
        steps += 1
        if r < rmin:
            rmin = r
            tmin = tc
        if rd > 0.0 and r > 1.02 * rmin:
            break

    return SphericalEulerResult(
        r_min=rmin,
        t_min=tmin,
        compression=1.0 / max(rmin, 1.0e-30),
        peak_mach_ambient=max_m_amb,
        peak_mach_local=max_m_loc,
        steps=steps,
    )


def solve_km_polytrope(
    pg0: float,
    gam_g: float,
    sig: float = 0.0,
    mu_l: float = 0.0,
    t_end: float | None = None,
    *,
    pinf: float = PINF_DEFAULT,
    gamma_l: float = GAMMA_L_DEFAULT,
    pi1: float = PI1_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    dt: float = 1.5e-7,
) -> SphericalEulerResult:
    if t_end is None:
        t_end = 2.0 * 0.915 * math.sqrt(rho0 / max(pinf - pg0, 1.0))
    c_inf = math.sqrt(gamma_l * (pinf + pi1) / rho0)
    b_tait = pinf + pi1

    def km_rhs(r: float, rd: float) -> tuple[float, float, float, float]:
        r_s = max(r, 1.0e-12)
        p_gas = pg0 * (1.0 / r_s) ** (3.0 * gam_g)
        dp_gas_dr = -3.0 * gam_g * p_gas / r_s
        p_w = p_gas - 2.0 * sig / r_s - 4.0 * mu_l * rd / r_s
        dp_w_dr = (
            dp_gas_dr
            + 2.0 * sig / (r_s * r_s)
            + 4.0 * mu_l * rd / (r_s * r_s)
        )
        mach = rd / c_inf
        denom = r_s * (1.0 - mach) + 4.0 * mu_l / (rho0 * c_inf)
        num = (
            (1.0 + mach) * (p_w - pinf) / rho0
            + (r_s / (rho0 * c_inf)) * dp_w_dr * rd
            - 1.5 * (1.0 - mach / 3.0) * rd * rd
        )
        ratio = max((p_w + pi1) / b_tait, 1.0e-12)
        c_w = c_inf * (ratio ** ((gamma_l - 1.0) / (2.0 * gamma_l)))
        return rd, num / denom, abs(mach), abs(rd) / c_w

    r, rd = 1.0, 0.0
    tc = 0.0
    rmin, tmin = 1.0, 0.0
    max_m_amb, max_m_loc = 0.0, 0.0
    steps = 0
    while tc < t_end and r > 1.0e-4:
        k1_r, k1_v, m_a, m_l = km_rhs(r, rd)
        if m_a > max_m_amb:
            max_m_amb = m_a
        if m_l > max_m_loc:
            max_m_loc = m_l
        k2_r, k2_v, _, _ = km_rhs(r + 0.5 * dt * k1_r, rd + 0.5 * dt * k1_v)
        k3_r, k3_v, _, _ = km_rhs(r + 0.5 * dt * k2_r, rd + 0.5 * dt * k2_v)
        k4_r, k4_v, _, _ = km_rhs(r + dt * k3_r, rd + dt * k3_v)
        r += (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        rd += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        tc += dt
        steps += 1
        if r < rmin:
            rmin = r
            tmin = tc
        if rd > 0.0 and r > 1.02 * rmin:
            break

    return SphericalEulerResult(
        r_min=rmin,
        t_min=tmin,
        compression=1.0 / max(rmin, 1.0e-30),
        peak_mach_ambient=max_m_amb,
        peak_mach_local=max_m_loc,
        steps=steps,
    )


__all__ = [
    "C_INF_DEFAULT",
    "GAMMA_L_DEFAULT",
    "PI1_DEFAULT",
    "PINF_DEFAULT",
    "RHO0_DEFAULT",
    "SphericalEulerResult",
    "quantized_volume_radius",
    "solve_gilmore_polytrope",
    "solve_km_polytrope",
    "solve_spherical_euler",
    "thermodynamic_compliance",
]
