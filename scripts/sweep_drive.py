from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import constants as C
from core.acoustics import blake_threshold_pressure
from core.config import load_config
from core.integrator import integrate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="sbsl_argon_water_fast")
    ap.add_argument("--amin", type=float, default=0.6, help="min A_1 [atm]")
    ap.add_argument("--amax", type=float, default=2.2, help="max A_1 [atm]")
    ap.add_argument("--n", type=int, default=17, help="number of amplitudes")
    ap.add_argument(
        "--cycles", type=int, default=1, help="acoustic cycles to integrate"
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=65536,
        help="solver step budget (also the save-buffer size)",
    )
    ap.add_argument("--json", type=Path, default=None, help="write results here")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = load_config(scenario=args.scenario)
    blake_pa = float(
        blake_threshold_pressure(
            base.r0,
            base.liquid.surface_tension,
            base.liquid.ambient_pressure,
            base.liquid.vapour_pressure,
        )
    )
    period = 1.0 / float(base.acoustics.frequency)
    threshold = float(base.shape.instability_threshold)

    print(f"scenario           : {args.scenario}")
    print(f"R_0                : {float(base.r0) * 1e6:.3f} um")
    print(f"hard core h        : {float(base.hardcore_radius) * 1e6:.4f} um")
    print(f"drive frequency    : {float(base.acoustics.frequency) / 1e3:.2f} kHz")
    print(f"Blake threshold    : {blake_pa / C.ATM_TO_PA:.3f} atm")
    print(f"instability limit  : |a_2|/R = {threshold}")
    print(
        f"\n{'A_1[atm]':>9} {'Rmax[um]':>9} {'Rmin[um]':>9} {'Rmin/h':>7} "
        f"{'Rmax/Rmin':>10} {'T_peak[K]':>11} {'Mach':>6} "
        f"{'|a2|/R':>10} {'Y_n':>10} {'steps':>7} {'stable':>7}"
    )
    rows = []
    t_start = time.perf_counter()
    for amp_atm in np.linspace(args.amin, args.amax, args.n):
        cfg = load_config(
            scenario=args.scenario,
            overrides={
                "acoustics.amplitudes_init": [float(amp_atm), 0.0, 0.0, 0.0]
            },
        )
        traj = integrate(
            cfg,
            0.0,
            args.cycles * period,
            save_at_steps=True,
            max_steps=args.max_steps,
            throw=False,
        )
        if not bool(traj.success):
            print(f"{amp_atm:9.3f}   solver failed (step budget exhausted)")
            rows.append({"amplitude_atm": float(amp_atm), "ok": False})
            continue
        r_min = float(traj.min_radius)
        r_max = float(traj.max_radius)
        t_peak = float(traj.peak_temperature)
        mach = float(traj.peak_wall_speed) / float(cfg.liquid.sound_speed)
        d_max = float(traj.max_relative_distortion)
        yield_n = float(traj.neutron_yield)
        stable = d_max < threshold
        print(
            f"{amp_atm:9.3f} {r_max * 1e6:9.3f} {r_min * 1e6:9.5f} "
            f"{r_min / float(cfg.hardcore_radius):7.3f} "
            f"{r_max / r_min:10.1f} {t_peak:11.4e} {mach:6.3f} "
            f"{d_max:10.3e} {yield_n:10.3e} {int(traj.num_steps):7d} "
            f"{'yes' if stable else 'NO':>7}"
        )
        rows.append(
            {
                "amplitude_atm": float(amp_atm),
                "ok": True,
                "r_max_m": r_max,
                "r_min_m": r_min,
                "r_min_over_hardcore": r_min / float(cfg.hardcore_radius),
                "compression_ratio": r_max / r_min,
                "t_peak_K": t_peak,
                "peak_mach": mach,
                "max_relative_distortion": d_max,
                "neutron_yield": yield_n,
                "solver_steps": int(traj.num_steps),
                "shape_stable": bool(stable),
            }
        )
    print(f"\nelapsed: {time.perf_counter() - t_start:.1f} s")

    ok = [r for r in rows if r.get("ok")]
    if ok:
        hottest = max(ok, key=lambda r: r["t_peak_K"])
        hottest_stable = max(
            (r for r in ok if r["shape_stable"]),
            key=lambda r: r["t_peak_K"],
            default=None,
        )
        print(
            f"hottest overall      : A_1 = {hottest['amplitude_atm']:.3f} atm, "
            f"T = {hottest['t_peak_K']:.4e} K, "
            f"|a2|/R = {hottest['max_relative_distortion']:.3e}"
        )
        if hottest_stable is not None:
            print(
                f"hottest SHAPE-STABLE : A_1 = "
                f"{hottest_stable['amplitude_atm']:.3f} atm, "
                f"T = {hottest_stable['t_peak_K']:.4e} K, "
                f"|a2|/R = "
                f"{hottest_stable['max_relative_distortion']:.3e}"
            )
            if hottest_stable is not hottest:
                print(
                    "  -> the hottest reachable point is UNSTABLE, so a "
                    "single-harmonic sweep cannot have both.  That gap is the "
                    "headroom a harmonic-shaping policy has to work with."
                )
            else:
                print(
                    "  -> the hottest point in this range is still stable: "
                    "widen --amax to expose the trade-off."
                )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"scenario": args.scenario, "rows": rows}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
