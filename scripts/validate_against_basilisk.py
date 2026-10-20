from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp

from core import gilmore as gil
from core.config import load_config

RHO_L = 1.0
PG0 = 100.0
PINF = 5.0 * PG0
GAMMA_L = 7.14
PI1 = 300.0 * PG0
GAMMA_G = 1.4
R0 = 1.0
RE = 10.0
MU_L = np.sqrt(PINF - PG0) / RE
C_INF = float(np.sqrt(GAMMA_L * (PINF + PI1) / RHO_L))


# match nondimensional Basilisk VOF liquid parameters (gamma_l=7.14, Pi_1=300*pg0)
def _liquid():
    water = load_config(scenario="sbsl_argon_water").liquid
    return dataclasses.replace(
        water,
        density=jnp.asarray(RHO_L),
        sound_speed=jnp.asarray(C_INF),
        ambient_pressure=jnp.asarray(PINF),
        surface_tension=jnp.asarray(0.0),
        dynamic_viscosity=jnp.asarray(MU_L),
        tait_n=jnp.asarray(GAMMA_L),
    )


# Gilmore + Tait enthalpy radial RHS for direct CFD cross-check
def _rhs(y, liquid):
    r, rdot = y

    def wall_pressure(rr, rd):
        p_gas = PG0 * (R0 / rr) ** (3.0 * GAMMA_G)
        return p_gas - 4.0 * MU_L * rd / rr

    p_l = wall_pressure(r, rdot)
    h = float(
        gil.enthalpy(jnp.asarray(p_l), jnp.asarray(PINF), liquid)
    )
    c = float(gil.wall_sound_speed(jnp.asarray(h), liquid))

    eps = 1.0e-7 * r
    h_p = float(
        gil.enthalpy(
            jnp.asarray(wall_pressure(r + eps, rdot)),
            jnp.asarray(PINF), liquid,
        )
    )
    h_m = float(
        gil.enthalpy(
            jnp.asarray(wall_pressure(r - eps, rdot)),
            jnp.asarray(PINF), liquid,
        )
    )
    hdot = (h_p - h_m) / (2.0 * eps) * rdot

    mach = rdot / c
    num = (
        h * (1.0 + mach)
        + r * hdot / c * (1.0 - mach)
        - 1.5 * rdot**2 * (1.0 - mach / 3.0)
    )
    rddot = num / (r * (1.0 - mach))
    return np.array([rdot, rddot])


def integrate(liquid, t_end, dt):
    y = np.array([R0, 0.0])
    ts, rs = [0.0], [R0]
    t = 0.0
    while t < t_end:
        k1 = _rhs(y, liquid)
        k2 = _rhs(y + 0.5 * dt * k1, liquid)
        k3 = _rhs(y + 0.5 * dt * k2, liquid)
        k4 = _rhs(y + dt * k3, liquid)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if not np.isfinite(y).all() or y[0] <= 0.0:
            break
        ts.append(t)
        rs.append(y[0])
    return np.array(ts), np.array(rs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basilisk", required=True,
                    help="Basilisk `log` file: t volume KE p_gas")
    ap.add_argument("--dt", type=float, default=2.0e-6)
    args = ap.parse_args()

    d = np.loadtxt(args.basilisk, skiprows=1)
    t_b, vol = d[:, 0], d[:, 1]
    r_b = (3.0 * vol) ** (1.0 / 3.0)

    liquid = _liquid()
    b = float(gil.tait_reference_pressure(liquid))
    print("liquid, as the two codes see it")
    print(f"  Basilisk  gamma_1 = {GAMMA_L}, Pi_1 = {PI1:.0f}, "
          f"c = {C_INF:.3f}")
    print(f"  this code tait_n  = {float(liquid.tait_n)}, "
          f"B = {b:.0f}, c = {float(liquid.sound_speed):.3f}")
    assert abs(b - PI1) < 1.0, "the liquids do not match; stop here"
    print(f"  -> B matches Pi_1 to {abs(b - PI1):.3e}\n")

    t_g, r_g = integrate(liquid, float(t_b[-1]), args.dt)

    kb = int(np.argmin(r_b))
    kg = int(np.argmin(r_g))
    t_r = 0.915 * np.sqrt(1.0 / (PINF - PG0))

    print(f"{'quantity':>26} {'Basilisk':>12} {'this code':>12} {'rel':>10}")
    print("-" * 64)
    for name, vb, vg in (
        ("R_min / R_0", r_b[kb], r_g[kg]),
        ("t(R_min) / t_R", t_b[kb] / t_r, t_g[kg] / t_r),
        ("R_0 / R_min", 1.0 / r_b[kb], 1.0 / r_g[kg]),
    ):
        rel = abs(vg - vb) / max(abs(vb), 1e-30)
        print(f"{name:>26} {vb:>12.5f} {vg:>12.5f} {rel:>9.2%}")

    r_i = np.interp(t_b, t_g, r_g)
    mask = t_b <= t_g[-1]
    err = np.abs(r_i[mask] - r_b[mask])
    print()
    print(f"pointwise |dR|/R_0 over {mask.sum()} samples: "
          f"mean {err.mean():.4f}, max {err.max():.4f}")
    print()
    print("A disagreement here is in the LIQUID model -- the Tait enthalpy,")
    print("the wall sound speed and the Gilmore reduction -- because the gas")
    print("is the same polytropic law on both sides by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
