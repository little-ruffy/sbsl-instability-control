from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import constants as C
from core.config import load_config
from core.integrator import integrate
from viz.projection import (
    plot_action_spectrogram,
    plot_bubble_projection,
    plot_phase_portrait,
)
from viz.render import collapse_window, save_figures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="sbsl_argon_water_fast")
    ap.add_argument(
        "--amplitude", type=float, default=None, help="A_1 override [atm]"
    )
    ap.add_argument(
        "--only",
        choices=["projection", "phase", "spectrogram", "all"],
        default="all",
    )
    ap.add_argument(
        "--policy", type=Path, default=None, help="a saved PPO policy to drive"
    )
    ap.add_argument("--episode-steps", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("runs/showcase"))
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args(argv)


def _config(args: argparse.Namespace):
    overrides: dict[str, object] = {}
    if args.amplitude is not None:
        base = load_config(scenario=args.scenario)
        amps = [
            float(a) / C.ATM_TO_PA for a in np.asarray(base.acoustics.amplitudes)
        ]
        amps[0] = float(args.amplitude)
        overrides["acoustics.amplitudes_init"] = amps
    return load_config(scenario=args.scenario, overrides=overrides)


def _demo_schedule(step: int, n_steps: int, n_harmonics: int) -> np.ndarray:
    progress = step / max(n_steps - 1, 1)
    action = -np.ones(2 * n_harmonics)
    action[0] = -0.2 + 1.0 * progress
    for k in range(1, n_harmonics):
        start = k / n_harmonics
        local = np.clip((progress - start) * n_harmonics, 0.0, 1.0)
        action[k] = -1.0 + 1.6 * local
        action[n_harmonics + k] = np.sin(2.0 * np.pi * (progress + 0.25 * k))
    action[n_harmonics] = np.sin(2.0 * np.pi * progress)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_episode(args: argparse.Namespace) -> dict:
    from core import keller_miksis as km
    cfg = _config(args)
    n_harmonics = cfg.acoustics.n_harmonics
    period = 1.0 / float(cfg.acoustics.frequency)
    y = km.initial_state(cfg)
    amplitudes, phases, interval_times, collapse_times = [], [], [], []
    peak_temperatures, min_radii = [], []
    for step in range(args.episode_steps):
        act = _demo_schedule(step, args.episode_steps, n_harmonics)
        amps_pa = np.maximum(0.0, (act[:n_harmonics] + 1.0) * 0.75) * C.ATM_TO_PA
        phis_rad = act[n_harmonics:] * np.pi
        step_cfg = cfg.with_action(amps_pa, phis_rad)
        t0 = step * period
        traj = integrate(step_cfg, t0, t0 + period, y0=y, n_save=1024, throw=False)
        y = traj.final_state
        amplitudes.append(amps_pa)
        phases.append(phis_rad)
        interval_times.append(t0)
        peak_temperatures.append(float(traj.peak_temperature))
        min_radii.append(float(traj.min_radius))
        valid = np.asarray(traj.valid)
        if valid.any():
            ts = np.asarray(traj.times)[valid]
            rs = np.asarray(traj.radius)[valid]
            collapse_times.append(float(ts[int(np.argmin(rs))]))
        else:
            collapse_times.append(np.nan)
    return {
        "cfg": cfg,
        "amplitudes": np.asarray(amplitudes),
        "phases": np.asarray(phases),
        "interval_times": np.asarray(interval_times),
        "collapse_times": np.asarray(collapse_times),
        "peak_temperatures": np.asarray(peak_temperatures),
        "min_radii": np.asarray(min_radii),
        "scripted": True,
    }



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    figures = []

    want = args.only
    if want in ("projection", "phase", "all"):
        cfg = _config(args)
        period = 1.0 / float(cfg.acoustics.frequency)
        print(f"integrating one cycle of {cfg.scenario} ...")
        cycle = integrate(cfg, 0.0, period, n_save=4096, throw=True)
        collapse, t_min = collapse_window(
            cfg, 0.0, period, half_width=3.0e-8, n_points=4001
        )
        print(
            f"  R_min = {float(collapse.min_radius) * 1e6:.5f} um, "
            f"T_peak = {float(collapse.peak_temperature):.4e} K, "
            f"collapse at t = {t_min * 1e6:.4f} us"
        )

        if want in ("projection", "all"):
            figures.append(
                ("bubble_projection", plot_bubble_projection(cfg, collapse))
            )
            rel = float(collapse.max_relative_distortion)
            gain = float(np.clip(0.25 / max(rel, 1e-12), 1.0, 1e6))
            if gain > 2.0:
                figures.append(
                    (
                        "bubble_projection_exaggerated",
                        plot_bubble_projection(
                            cfg, collapse, distortion_gain=gain
                        ),
                    )
                )
        if want in ("phase", "all"):
            figures.append((
                "phase_portrait_cycle",
                plot_phase_portrait(cfg, cycle, note="  [cycle-scale grid]"),
            ))
            figures.append((
                "phase_portrait_collapse",
                plot_phase_portrait(
                    cfg, collapse, note="  [refined collapse window]"
                ),
            ))

    if want in ("spectrogram", "all"):
        print(f"running a {args.episode_steps}-step episode ...")
        episode = run_episode(args)
        figure = plot_action_spectrogram(
            episode["cfg"],
            episode["amplitudes"],
            episode["phases"],
            episode["collapse_times"],
            episode["interval_times"],
            peak_temperatures=episode["peak_temperatures"],
            min_radii=episode["min_radii"],
        )
        if episode["scripted"]:
            figure.suptitle(
                figure._suptitle.get_text()
                + "   [scripted demonstration schedule, not a trained policy]",
                fontsize=11,
            )
        figures.append(("action_spectrogram", figure))

    written = save_figures(figures, args.out, dpi=args.dpi)
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
