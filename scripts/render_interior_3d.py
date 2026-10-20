from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp

from core.config import load_config
from core.integrator import integrate
from core import interior as it


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="sbsl_argon_water_fast")
    ap.add_argument("--zones", type=int, default=24)
    ap.add_argument(
        "--stretch",
        type=float,
        default=it.DEFAULT_STRETCH,
        help="grid exponent; >1 refines towards the centre",
    )
    ap.add_argument(
        "--start-frac",
        type=float,
        default=1.0,
        help=(
            "start the resolved solve at this fraction of R_max on the inward "
            "leg; 1.0 starts at maximum radius, where the gas is uniform and "
            "at ambient temperature"
        ),
    )
    ap.add_argument("--max-steps", type=int, default=400_000)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--tau-ns", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument(
        "--still",
        action="store_true",
        help="render one PNG at peak central temperature instead of a movie",
    )
    ap.add_argument("--out", type=Path, default=Path("runs/interior3d"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(scenario=args.scenario)
    period = 1.0 / float(cfg.acoustics.frequency)

    print(f"single-zone cycle for {cfg.scenario} ...")
    traj = integrate(cfg, 0.0, period, save_at_steps=True, throw=False)
    valid = np.asarray(traj.valid)
    t = np.asarray(traj.times)[valid]
    y = np.asarray(traj.states)[valid]
    order = np.argsort(t)
    t, y = t[order], y[order]
    radius, radius_dot, temperature = y[:, 0], y[:, 1], y[:, 2]
    k_max = int(np.argmax(radius))
    k_min = int(np.argmin(radius))
    print(
        f"  R_max={radius[k_max] * 1e6:.3f} um  "
        f"R_min={radius[k_min] * 1e6:.4f} um  "
        f"T_peak={temperature.max():.4e} K"
    )

    leg = slice(k_max, k_min + 1)
    idx = k_max + int(
        np.argmin(np.abs(radius[leg] - args.start_frac * radius[k_max]))
    )
    print(
        f"  starting resolved solve at R={radius[idx] * 1e6:.4f} um, "
        f"T={temperature[idx]:.2f} K "
        f"(ambient {float(cfg.liquid.ambient_temperature):.2f} K)"
    )

    t_start, t_end = t[idx], t[k_min] + 1.0e-9
    y0 = it.initial_state_from_uniform(
        cfg, radius[idx], radius_dot[idx], temperature[idx], args.zones,
        distortion=y[idx, 3], distortion_dot=y[idx, 4], stretch=args.stretch,
    )

    tau = args.tau_ns * 1e-9
    u_lo = np.arcsinh((t_start - t[k_min]) / tau)
    u_hi = np.arcsinh((t_end - t[k_min]) / tau)
    grid = t[k_min] + tau * np.sinh(np.linspace(u_lo, u_hi, 4 * args.frames))
    save_ts = jnp.asarray(np.clip(np.sort(grid), t_start, t_end))

    print(
        f"  N={args.zones} stretch={args.stretch} over "
        f"{(t_end - t_start) * 1e9:.2f} ns, budget {args.max_steps} ..."
    )
    started = time.perf_counter()
    resolved = it.integrate_interior(
        cfg, t_start, t_end, y0, args.zones, stretch=args.stretch,
        save_ts=save_ts, max_steps=args.max_steps, throw=False,
    )
    resolved.states.block_until_ready()
    print(
        f"  success={bool(resolved.success)} "
        f"steps={float(resolved.num_steps):.0f} "
        f"({time.perf_counter() - started:.1f} s)"
    )

    ts = np.asarray(resolved.times)
    ys = np.asarray(resolved.states)
    usable = np.isfinite(ys).all(axis=1)
    if not usable.any():
        print("  no usable saves; raise --max-steps")
        return 1
    if not usable.all():
        print(
            f"  WARNING: only {usable.sum()}/{len(usable)} saves reached; "
            "the solver ran out of budget before the end of the window"
        )
    ts, ys = ts[usable], ys[usable]

    n = args.zones
    radii = ys[:, 0:n]
    temps = ys[:, 2 * n : 3 * n]
    wall = ys[:, n - 1]
    print(
        f"  T_centre peak = {temps[:, 0].max():.4e} K   "
        f"hottest zone = {temps.max():.4e} K at r/R = "
        f"{radii[np.unravel_index(np.argmax(temps), temps.shape)] / wall[np.argmax(temps) // n]:.3f}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    from viz.volume import animate_collapse_3d_profile, plot_cutaway_still

    if args.still:
        k = int(np.argmax(temps.max(axis=1)))
        out = args.out / f"interior_{cfg.scenario}_N{n}.png"
        written = plot_cutaway_still(
            cfg, float(wall[k]), 0.0, 0.0, float(temps[k, 0]), out,
            dpi=max(args.dpi, 160),
            zone_edges=radii[k] / wall[k],
            zone_temperature=temps[k],
        )
        print(f"  wrote {written}")
        return 0

    out = args.out / f"interior_{cfg.scenario}_N{n}.mp4"
    print(f"encoding {args.frames} frames ...")
    started = time.perf_counter()
    written = animate_collapse_3d_profile(
        cfg, ts, ys, args.zones, out,
        n_frames=args.frames, fps=args.fps, dpi=args.dpi,
        t_reference=float(t[k_min]),
        single_zone=(t, radius, temperature),
    )
    size_mb = written.stat().st_size / 1e6
    print(
        f"  wrote {written} ({size_mb:.2f} MB, "
        f"{time.perf_counter() - started:.1f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
