from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import constants as C
from core.config import load_config
from viz.animate import animate_collapse
from viz.render import collapse_window


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="sbsl_argon_water_fast")
    ap.add_argument(
        "--amplitude", type=float, default=None, help="A_1 override [atm]"
    )
    ap.add_argument(
        "--window-ns",
        type=float,
        default=60.0,
        help="half-width of the simulated window around the collapse [ns]",
    )
    ap.add_argument(
        "--tau-ns",
        type=float,
        default=2.0,
        help="clock-warp scale [ns]; smaller = more frames spent at the peak",
    )
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument(
        "--samples",
        type=int,
        default=8001,
        help="trajectory samples inside the window (must exceed --frames)",
    )
    ap.add_argument(
        "--gain",
        type=float,
        default=None,
        help="shape-mode exaggeration; default picks one that is visible",
    )
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("runs/movie"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    overrides: dict[str, object] = {}
    if args.amplitude is not None:
        base = load_config(scenario=args.scenario)
        amps = [
            float(a) / C.ATM_TO_PA
            for a in np.asarray(base.acoustics.amplitudes)
        ]
        amps[0] = float(args.amplitude)
        overrides["acoustics.amplitudes_init"] = amps
    cfg = load_config(scenario=args.scenario, overrides=overrides)

    period = 1.0 / float(cfg.acoustics.frequency)
    print(f"integrating one cycle of {cfg.scenario} ...")
    started = time.perf_counter()
    collapse, t_min = collapse_window(
        cfg,
        0.0,
        period,
        half_width=args.window_ns * 1e-9,
        n_points=args.samples,
    )
    print(
        f"  R_min   = {float(collapse.min_radius) * 1e6:.5f} um\n"
        f"  T_peak  = {float(collapse.peak_temperature):.4e} K\n"
        f"  |a2|/R  = {float(collapse.max_relative_distortion):.4e}\n"
        f"  collapse at t = {t_min * 1e6:.4f} us "
        f"({time.perf_counter() - started:.1f} s)"
    )

    gain = args.gain
    if gain is None:
        rel = float(collapse.max_relative_distortion)
        gain = float(np.clip(0.25 / max(rel, 1e-12), 1.0, 1e6))
    if gain > 1.0:
        print(f"  shape mode exaggerated {gain:.4g}x so it is visible")

    out = args.out / f"collapse_{cfg.scenario}.mp4"
    print(f"encoding {args.frames} frames at {args.fps} fps ...")
    started = time.perf_counter()
    written = animate_collapse(
        cfg,
        collapse,
        out,
        n_frames=args.frames,
        fps=args.fps,
        tau=args.tau_ns * 1e-9,
        distortion_gain=gain,
        dpi=args.dpi,
    )
    size_mb = written.stat().st_size / 1e6
    print(
        f"  wrote {written}  ({size_mb:.2f} MB, "
        f"{args.frames / args.fps:.1f} s runtime, "
        f"{time.perf_counter() - started:.1f} s to encode)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
