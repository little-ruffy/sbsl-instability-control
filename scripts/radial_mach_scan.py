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


def _liquid():
    return dataclasses.replace(
        load_config(scenario="sbsl_argon_water").liquid,
        density=jnp.asarray(RHO), sound_speed=jnp.asarray(C_INF),
        ambient_pressure=jnp.asarray(PINF), surface_tension=jnp.asarray(0.0),
        dynamic_viscosity=jnp.asarray(0.0), tait_n=jnp.asarray(GAMMA_L),
    )


def _tait_hc(p: float) -> tuple[float, float]:
    denom = PINF + PI1
    alpha = (GAMMA_L - 1.0) / GAMMA_L
    arg = max((p - PINF) / denom, -1.0 + 1.0e-12)
    h = (GAMMA_L / (GAMMA_L - 1.0)) * (denom / RHO) * float(
        np.expm1(alpha * np.log1p(arg))
    )
    c = float(np.sqrt(max(C_INF**2 + (GAMMA_L - 1.0) * h, 1.0e-6 * C_INF**2)))
    return h, c


def _rhs(y, liquid, p_wall):
    del liquid
    r, rd = y
    h, c = _tait_hc(p_wall(r))
    eps = 1.0e-7 * r
    hp, _ = _tait_hc(p_wall(r + eps))
    hm, _ = _tait_hc(p_wall(r - eps))
    mach = rd / c
    num = (h * (1.0 + mach) + r * ((hp - hm) / (2.0 * eps) * rd) / c
           * (1.0 - mach) - 1.5 * rd * rd * (1.0 - mach / 3.0))
    return np.array([rd, num / (r * (1.0 - mach))])


def read_log(path: Path):
    rows, mach_rows = [], []
    for line in path.read_text(errors="ignore").splitlines():
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


def measured_gas(r_b, p_b, pg0, gam):
    order = np.argsort(r_b)
    rr, pp = r_b[order], p_b[order]
    uniq, idx = np.unique(np.round(rr, 9), return_index=True)
    pu = pp[idx]

    def p_wall(r):
        r = float(r)
        if r <= uniq[0]:
            return float(pu[0]) * (uniq[0] / max(r, 1e-12)) ** (3.0 * gam)
        if r >= uniq[-1]:
            return float(pu[-1])
        return float(np.interp(r, uniq, pu))

    return p_wall


def integrate(liquid, p_wall, t_end, dt=1.0e-6):
    y, t = np.array([1.0, 0.0]), 0.0
    ts, rs = [0.0], [1.0]
    while t < t_end and y[0] > 1.0e-4:
        k1 = _rhs(y, liquid, p_wall)
        k2 = _rhs(y + 0.5 * dt * k1, liquid, p_wall)
        k3 = _rhs(y + 0.5 * dt * k2, liquid, p_wall)
        k4 = _rhs(y + dt * k3, liquid, p_wall)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if not np.isfinite(y).all():
            break
        ts.append(t)
        rs.append(y[0])
    return np.array(ts), np.array(rs)


CASES = [
    ("rad4_run_p30g11", 30.0, 1.1),
    ("rad4_run_p10g12", 10.0, 1.2),
    ("rad4_run_p10g11", 10.0, 1.1),
    ("rad4_run_p10g105", 10.0, 1.05),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    liquid = _liquid()
    print("core/gilmore.py vs 1-D spherical Navier-Stokes, rising violence\n")
    print(f"{'p_g0':>6} {'gam':>5} {'compression':>12} {'Mach(loc)':>10} "
          f"{'Mach(amb)':>10} {'R_min NS':>10} {'R_min lin':>10} "
          f"{'error':>8} {'sign':>12}")
    print("-" * 92)

    out = []
    for name, pg0, gam in CASES:
        d = Path(args.root) / name / "log.txt"
        if not d.exists():
            continue
        t_b, r_b, p_b, mach_rows = read_log(d)
        if len(t_b) < 20:
            continue
        p_wall = measured_gas(r_b, p_b, pg0, gam)
        if mach_rows is not None and len(mach_rows) > 0:
            t_eval = mach_rows[:, 0]
            rdot = mach_rows[:, 2]
            k_pk = int(np.argmax(np.abs(rdot)))
            mach_amb = float(abs(rdot[k_pk]) / C_INF)
            p_w = float(np.interp(t_eval[k_pk], t_b, p_b))
        else:
            rdot = np.gradient(r_b, t_b)
            k_pk = int(np.argmax(np.abs(rdot)))
            mach_amb = float(abs(rdot[k_pk]) / C_INF)
            p_w = float(p_b[k_pk])
        h = float(gil.enthalpy(jnp.asarray(p_w), jnp.asarray(PINF), liquid))
        c_w = float(gil.wall_sound_speed(jnp.asarray(h), liquid))
        mach_loc = float(abs(rdot[k_pk]) / max(c_w, 1e-30))

        t_g, r_g = integrate(liquid, p_wall, float(t_b[-1]))
        kb, kg = int(np.argmin(r_b)), int(np.argmin(r_g))
        err = (r_g[kg] - r_b[kb]) / max(r_b[kb], 1e-30)
        sign = "optimistic" if err < 0 else "conservative"
        print(f"{pg0:>6.0f} {gam:>5.2f} {r_b[0]/r_b[kb]:>12.1f} "
              f"{mach_loc:>10.3f} {mach_amb:>10.3f} {r_b[kb]:>10.5f} "
              f"{r_g[kg]:>10.5f} {abs(err):>7.1%} {sign:>12}")
        out.append((mach_amb, abs(err)))

    if len(out) >= 3:
        arr = np.array(out)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        print()
        print("Error against peak ambient wall Mach:")
        for m, e in arr:
            print(f"   Mach {m:6.3f} -> {e:6.1%}")
        ok = (arr[:, 0] > 0) & (arr[:, 1] > 0)
        if ok.sum() >= 2:
            s, b = np.polyfit(np.log(arr[ok, 0]), np.log(arr[ok, 1]), 1)
            print(f"\n   power law: error ~ Mach^{s:.2f}")
            for target in (0.824, 1.051):
                print(f"   extrapolated to Mach {target:.3f}: "
                      f"{np.exp(b) * target ** s:.1%}")
            print("\n   An extrapolation over few points, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
