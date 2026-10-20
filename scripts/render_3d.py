from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import constants as C
from core.config import load_config
from viz.render import collapse_window
from viz.volume import animate_collapse_3d, plot_cutaway_still


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
    ap.add_argument("--frames", type=int, default=180)
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
    ap.add_argument(
        "--spin",
        type=float,
        default=0.0,
        help=(
            "total camera azimuth sweep [deg]; zero by default because the "
            "body is axisymmetric and azimuth reveals nothing"
        ),
    )
    ap.add_argument(
        "--elev", type=float, default=20.0, help="mean camera elevation [deg]"
    )
    ap.add_argument(
        "--elev-sweep",
        type=float,
        default=44.0,
        help="peak-to-peak elevation sweep [deg]; this is the rotation that "
             "actually shows the l=2 shape",
    )
    ap.add_argument("--theta", type=int, default=44, help="polar divisions")
    ap.add_argument("--phi", type=int, default=56, help="azimuthal divisions")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument(
        "--still",
        action="store_true",
        help="render one PNG at the collapse instead of a movie",
    )
    ap.add_argument("--out", type=Path, default=Path("runs/movie3d"))
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

    args.out.mkdir(parents=True, exist_ok=True)

    if args.still:
        valid = np.asarray(collapse.valid)
        radius = np.asarray(collapse.radius)[valid]
        rdot = np.asarray(collapse.radius_dot)[valid]
        temperature = np.asarray(collapse.temperature)[valid]
        a_l = np.asarray(collapse.distortion)[valid]
        k = int(np.argmin(radius))
        from viz.projection import _boundary_layer

        layer = _boundary_layer(
            cfg, float(radius[k]), float(rdot[k]), float(temperature[k])
        )
        out = args.out / f"cutaway_{cfg.scenario}.png"
        written = plot_cutaway_still(
            cfg,
            float(radius[k]),
            float(a_l[k]) * gain,
            layer,
            float(temperature[k]),
            out,
            dpi=max(args.dpi, 160),
        )
        print(f"  wrote {written}")
        return 0

    out = args.out / f"cutaway_{cfg.scenario}.mp4"
    print(f"encoding {args.frames} 3-D frames at {args.fps} fps ...")
    started = time.perf_counter()
    written = animate_collapse_3d(
        cfg,
        collapse,
        out,
        n_frames=args.frames,
        fps=args.fps,
        tau=args.tau_ns * 1e-9,
        distortion_gain=gain,
        n_theta=args.theta,
        n_phi=args.phi,
        spin_degrees=args.spin,
        elev_mid=args.elev,
        elev_sweep=args.elev_sweep,
        dpi=args.dpi,
    )
    elapsed = time.perf_counter() - started
    size_mb = written.stat().st_size / 1e6
    print(
        f"  wrote {written}  ({size_mb:.2f} MB, "
        f"{args.frames / args.fps:.1f} s runtime, "
        f"{elapsed:.1f} s to encode, {elapsed / args.frames:.2f} s/frame)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
