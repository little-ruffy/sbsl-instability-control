from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp

from core import gilmore as gil
from core.config import load_config

RHO, PINF, GAMMA_L, PI1 = 1.0, 500.0, 7.14, 30000.0
C_INF = float(np.sqrt(GAMMA_L * (PINF + PI1) / RHO))
ROW = re.compile(r"^\s*(\d+\.\d{6})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
MACH_ROW = re.compile(r"^\s*(\d+\.\d{8})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")

CASES = [
    ("fast101", 1.01, 0.0, 0.0),
    ("fast102", 1.02, 0.0, 0.0),
    ("mu105", 1.05, 0.0, 0.2064),
    ("sig2", 1.05, 2.3100, 0.0),
    ("real2", 1.05, 2.3100, 0.2064),
]


def liquid_for(sig, mu):
    return dataclasses.replace(
        load_config(scenario="sbsl_argon_water").liquid,
        density=jnp.asarray(RHO), sound_speed=jnp.asarray(C_INF),
        ambient_pressure=jnp.asarray(PINF),
        surface_tension=jnp.asarray(sig), dynamic_viscosity=jnp.asarray(mu),
        tait_n=jnp.asarray(GAMMA_L),
    )


def read(path):
    rows, mach_rows = [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = ROW.match(line)
        if m:
            try:
                v = [float(x) for x in m.groups()]
            except ValueError:
                continue
            if not any(x != x for x in v):
                rows.append(v)
            continue
        m8 = MACH_ROW.match(line)
        if m8:
            try:
                v = [float(x) for x in m8.groups()]
            except ValueError:
                continue
            if not any(x != x for x in v):
                mach_rows.append(v)
    a = np.array(rows)
    b = np.array(mach_rows) if mach_rows else None
    if b is not None and len(b) > 0 and b[0, 3] > 10.0:
        scale = (4.0 * np.pi) ** (1.0 / 3.0)
        r2, rdot2 = b[:, 1], b[:, 2]
        r_prev = np.concatenate([[r2[0]], r2[:-1]])
        dt_ex = (r2 - r_prev / scale) / np.where(
            np.abs(rdot2) > 1.0e-30, rdot2, 1.0e-30
        )
        b[:, 2] = (r2 - r_prev) / np.where(
            np.abs(dt_ex) > 1.0e-30, dt_ex, 1.0e-30
        )
        b[:, 3] = np.abs(b[:, 2]) / C_INF
    return a[:, 0], (3.0 * a[:, 1]) ** (1.0 / 3.0), a[:, 3], b


def gas_from(r_b, p_b, gam):
    o = np.argsort(r_b)
    rr, pp = r_b[o], p_b[o]
    u, i = np.unique(np.round(rr, 9), return_index=True)
    pu = pp[i]

    def p_wall(r):
        r = float(r)
        if r <= u[0]:
            return float(pu[0]) * (u[0] / max(r, 1e-12)) ** (3.0 * gam)
        if r >= u[-1]:
            return float(pu[-1])
        return float(np.interp(r, u, pu))

    return p_wall


def _tait_hc(p: float) -> tuple[float, float]:
    denom = PINF + PI1
    alpha = (GAMMA_L - 1.0) / GAMMA_L
    arg = max((p - PINF) / denom, -1.0 + 1.0e-12)
    h = (GAMMA_L / (GAMMA_L - 1.0)) * (denom / RHO) * float(
        np.expm1(alpha * np.log1p(arg))
    )
    c = float(np.sqrt(max(C_INF**2 + (GAMMA_L - 1.0) * h, 1.0e-6 * C_INF**2)))
    return h, c


def rhs(y, liq, p_gas, sig, mu):
    del liq
    r, rd = y

    def wall(rr, rdot):
        return p_gas(rr) - 2.0 * sig / rr - 4.0 * mu * rdot / rr

    h, c = _tait_hc(wall(r, rd))
    e = 1.0e-7 * r
    hp, _ = _tait_hc(wall(r + e, rd))
    hm, _ = _tait_hc(wall(r - e, rd))
    m = rd / c
    num = (h * (1 + m) + r * ((hp - hm) / (2 * e) * rd) / c * (1 - m)
           - 1.5 * rd * rd * (1 - m / 3))
    return np.array([rd, num / (r * (1 - m))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    print(f"{'case':>9} {'gam':>5} {'sigma':>8} {'mu':>7} {'comp':>7} "
          f"{'M(loc)':>7} {'M(amb)':>7} {'R_min NS':>9} {'R_min mdl':>10} "
          f"{'error':>8}")
    print("-" * 88)
    out = []
    for tag, gam, sig, mu in CASES:
        f = root / f"rad5_run_{tag}" / "log.txt"
        if not f.exists():
            print(f"{tag:>9}  missing")
            continue
        try:
            t_b, r_b, p_b, mach_rows = read(f)
        except (ValueError, IndexError) as exc:
            print(f"{tag:>9}  unreadable: {exc}")
            continue
        if len(t_b) < 50:
            print(f"{tag:>9}  only {len(t_b)} rows")
            continue

        liq = liquid_for(sig, mu)
        p_gas = gas_from(r_b, p_b, gam)
        if mach_rows is not None and len(mach_rows) > 0:
            t_eval = mach_rows[:, 0]
            r_eval = mach_rows[:, 1]
            rdot = mach_rows[:, 2]
            k_pk = int(np.argmax(np.abs(rdot)))
            mach_amb = float(abs(rdot[k_pk]) / C_INF)
            r_pk = max(float(r_eval[k_pk]), 1.0e-12)
            pw = (
                float(np.interp(t_eval[k_pk], t_b, p_b))
                - 2.0 * sig / r_pk
                - 4.0 * mu * float(rdot[k_pk]) / r_pk
            )
        else:
            rdot = np.gradient(r_b, t_b)
            k_pk = int(np.argmax(np.abs(rdot)))
            mach_amb = float(abs(rdot[k_pk]) / C_INF)
            r_pk = max(float(r_b[k_pk]), 1.0e-12)
            pw = (
                float(p_b[k_pk])
                - 2.0 * sig / r_pk
                - 4.0 * mu * float(rdot[k_pk]) / r_pk
            )
        h = float(gil.enthalpy(jnp.asarray(pw), jnp.asarray(PINF), liq))
        cw = float(gil.wall_sound_speed(jnp.asarray(h), liq))
        mach = float(abs(rdot[k_pk]) / max(cw, 1e-30))

        y, t, dt = np.array([1.0, 0.0]), 0.0, 5.0e-7
        rs = [1.0]
        steps = 0
        while t < t_b[-1] and y[0] > 1.0e-4 and steps < 4_000_000:
            k1 = rhs(y, liq, p_gas, sig, mu)
            k2 = rhs(y + 0.5 * dt * k1, liq, p_gas, sig, mu)
            k3 = rhs(y + 0.5 * dt * k2, liq, p_gas, sig, mu)
            k4 = rhs(y + dt * k3, liq, p_gas, sig, mu)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            steps += 1
            if not np.isfinite(y).all():
                break
            rs.append(y[0])
            if y[1] > 0.0 and len(rs) > 10 and y[0] > 1.02 * min(rs):
                break
        rs = np.array(rs)
        kb, kg = int(np.argmin(r_b)), int(np.argmin(rs))
        if r_b[kb] <= 1.0e-6:
            print(f"{tag:>9}  bubble collapsed to zero radius -- unusable")
            continue
        err = abs(rs[kg] - r_b[kb]) / max(r_b[kb], 1e-30)
        print(f"{tag:>9} {gam:>5.2f} {sig:>8.2f} {mu:>7.4f} "
              f"{1.0 / r_b[kb]:>7.1f} {mach:>7.3f} {mach_amb:>7.3f} "
              f"{r_b[kb]:>9.5f} {rs[kg]:>10.5f} {err:>7.1%}")
        out.append((tag, sig, mu, mach, mach_amb, err))

    print()
    print("Which term, if any, removes the disagreement:")
    for tag, sig, mu, mach, mach_amb, err in out:
        what = ("neither" if sig == 0 and mu == 0
                else "sigma only" if mu == 0
                else "mu only" if sig == 0 else "both")
        print(f"   {tag:>9}  {what:>11}  M(loc) {mach:5.3f}  "
              f"M(amb) {mach_amb:5.3f}  error {err:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
