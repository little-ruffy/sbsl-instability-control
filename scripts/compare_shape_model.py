from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp

from core.config import load_config
from core.keller_miksis import shape_mode_coefficients

RHO_L, PG0, GAMMA_L = 1.0, 100.0, 7.14
PINF, PI1 = 5.0 * PG0, 300.0 * PG0
MU_L = float(np.sqrt(PINF - PG0) / 10.0)
C_INF = float(np.sqrt(GAMMA_L * (PINF + PI1) / RHO_L))


def basilisk_cfg(inviscid: bool = False):
    cfg = load_config(scenario="sbsl_argon_unstable")
    liquid = dataclasses.replace(
        cfg.liquid,
        density=jnp.asarray(RHO_L),
        sound_speed=jnp.asarray(C_INF),
        ambient_pressure=jnp.asarray(PINF),
        surface_tension=jnp.asarray(0.0),
        dynamic_viscosity=jnp.asarray(0.0 if inviscid else MU_L),
        tait_n=jnp.asarray(GAMMA_L),
    )
    return dataclasses.replace(cfg, liquid=liquid, r0=jnp.asarray(1.0))


# project axisymmetric VOF facets onto P_l(cos theta) over [0, pi/2]
def legendre_a2(x, y, lmax=4):
    r = np.hypot(x, np.abs(y))
    mu = np.cos(np.arctan2(np.abs(y), x))
    key = np.round(mu, 9)
    uniq, inv = np.unique(key, return_inverse=True)
    rr = np.bincount(inv, weights=r) / np.bincount(inv)
    out = []
    for l in range(lmax + 1):
        p = np.polynomial.legendre.Legendre.basis(l)(uniq)
        out.append((2 * l + 1) * np.trapezoid(rr * p, uniq))
    return out[0], out[2]


def facet_volume(x, y):
    r = np.hypot(x, np.abs(y))
    mu = np.cos(np.arctan2(np.abs(y), x))
    uniq, inv = np.unique(np.round(mu, 9), return_inverse=True)
    rr = np.bincount(inv, weights=r) / np.bincount(inv)
    return 4.0 * np.pi / 3.0 * np.trapezoid(rr**3, uniq)


def read_case(d: Path):
    log = np.loadtxt(d / "log.txt", skiprows=1)
    t_log, vol = log[:, 0], log[:, 1]
    r_vol = (3.0 * vol) ** (1.0 / 3.0)
    idx = np.atleast_2d(np.loadtxt(d / "frames.dat"))
    rows = []
    for fr, tt in idx:
        f = d / f"facets-{int(fr):04d}.dat"
        if not f.exists():
            continue
        a = np.loadtxt(f)
        if a.ndim != 2 or len(a) < 8:
            continue
        r0, a2 = legendre_a2(a[:, 0], a[:, 1])
        rows.append((float(tt), r0, a2))
    return t_log, r_vol, np.array(rows)


def startup_index(t_log, rddot, factor=20.0):
    med = np.median(np.abs(rddot))
    ok = np.abs(rddot) < factor * max(med, 1e-30)
    idx = int(np.argmax(ok)) if ok.any() else 0
    while idx + 3 < len(ok) and not ok[idx:idx + 3].all():
        idx += 1
    return idx


# integrate linear Plesset l=2 ODE driven by Basilisk radial history R(t)
def predict(t_log, r_vol, t_out, a_ns_t, a_ns, cfg):
    rdot = np.gradient(r_vol, t_log)
    rddot = np.gradient(rdot, t_log)
    rddot = np.convolve(rddot, np.ones(5) / 5.0, mode="same")

    i0 = startup_index(t_log, rddot)
    t0 = t_log[i0]
    a = float(np.interp(t0, a_ns_t, a_ns))
    j = int(np.searchsorted(a_ns_t, t0))
    j = min(max(j, 0), len(a_ns) - 2)
    adot = float((a_ns[j + 1] - a_ns[j]) / (a_ns_t[j + 1] - a_ns_t[j]))

    ts, amps = [t0], [a]
    for i in range(i0 + 1, len(t_log)):
        dt = t_log[i] - t_log[i - 1]
        A, B = shape_mode_coefficients(
            jnp.asarray(max(r_vol[i], 1e-12)), jnp.asarray(rdot[i]),
            jnp.asarray(rddot[i]), cfg,
        )
        addot = float(A) * a - float(B) * adot
        adot += dt * addot
        a += dt * adot
        ts.append(t_log[i])
        amps.append(a)
    return np.interp(t_out, np.array(ts), np.array(amps)), t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="directory holding scan_<tag> subdirectories")
    ap.add_argument("--inviscid", action="store_true",
                    help="the runs were made with mu = 0; set the shape "
                         "equation's viscosity to zero to match")
    args = ap.parse_args()

    cfg = basilisk_cfg(inviscid=args.inviscid)
    root = Path(args.root)
    cases = sorted(root.glob("scan_*"))
    if not cases:
        print(f"no scan_* directories under {root}")
        return 1

    print("§2.7 linear shape equation vs Basilisk Navier-Stokes")
    print("radial history taken from Basilisk on both sides, so any")
    print("disagreement below belongs to the SHAPE model.\n")
    print(f"{'seed a2/R':>10} {'peak |a2|/R':>12} {'peak (lin)':>11} "
          f"{'ratio':>8} {'final NS':>10} {'final lin':>10} {'rel err':>9}")
    print("-" * 78)

    summary = []
    for d in cases:
        tag = d.name.replace("scan_", "")
        seed = float(tag[0] + "." + tag[1:]) if len(tag) > 1 else float(tag)
        try:
            t_log, r_vol, rows = read_case(d)
        except (OSError, ValueError) as exc:
            print(f"{seed:>10.3f}  unreadable: {exc}")
            continue
        if len(rows) < 3:
            print(f"{seed:>10.3f}  too few frames ({len(rows)})")
            continue
        t_out, r_ns, a2_ns = rows[:, 0], rows[:, 1], rows[:, 2]
        a2_lin, t0 = predict(t_log, r_vol, t_out, t_out, a2_ns, cfg)

        rel_ns = np.abs(a2_ns) / np.maximum(r_ns, 1e-30)
        rel_lin = np.abs(a2_lin) / np.maximum(r_ns, 1e-30)
        keep = t_out >= t0
        peak_ns = float(np.median(
            np.sort(rel_ns[keep])[-3:])) if keep.sum() >= 3 else rel_ns.max()
        peak_lin = float(np.median(
            np.sort(rel_lin[keep])[-3:])) if keep.sum() >= 3 else rel_lin.max()
        err = abs(rel_lin[-1] - rel_ns[-1]) / max(rel_ns[-1], 1e-30)
        ratio = peak_lin / max(peak_ns, 1e-30)
        print(f"{seed:>10.3f} {peak_ns:>12.4f} {peak_lin:>11.4f} "
              f"{ratio:>8.3f} {rel_ns[-1]:>10.4f} {rel_lin[-1]:>10.4f} "
              f"{err:>8.1%}")
        summary.append((seed, peak_ns, peak_lin, ratio, err))

    if not summary:
        return 1
    print()
    print("ratio < 1 means the linear model UNDER-predicts the distortion,")
    print("i.e. the real interface is worse behaved than §2.7 reports and the")
    print("baselines in §7.6.4 were judged too leniently. ratio > 1 means the")
    print("opposite, which is the direction §2.7 assumed.")
    print()
    print("Interpretation is pre-committed in docs/VALIDATION_CFD.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
