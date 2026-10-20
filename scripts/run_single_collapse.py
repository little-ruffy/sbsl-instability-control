from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import constants as C
from core import gilmore as gil
from core import keller_miksis as km
from core.acoustics import (
    blake_threshold_pressure,
    peak_negative_pressure,
)
from core.config import load_config
from core.integrator import integrate
from viz.render import (
    collapse_window,
    plot_cycle,
    save_figures,
    summarise,
)

SBSL_REFERENCE = {
    "r_max_um": (40.0, 50.0),
    "r_min_um": (0.5, 0.8),
    "t_peak_K": (1.5e4, 3.0e4),
    "peak_mach": (0.5, 1.5),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="sbsl_argon_water_fast")
    ap.add_argument(
        "--closure",
        choices=("km", "gilmore"),
        default="km",
        help="radial liquid closure: km (Keller-Miksis, default) or gilmore",
    )
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="override the fundamental amplitude A_1 [atm]",
    )
    ap.add_argument(
        "--harmonics",
        type=float,
        nargs=3,
        default=None,
        metavar=("A2", "A3", "A4"),
        help="amplitudes of harmonics 2-4 [atm]",
    )
    ap.add_argument(
        "--phases",
        type=float,
        nargs=4,
        default=None,
        metavar=("P1", "P2", "P3", "P4"),
        help="harmonic phases [rad]",
    )
    ap.add_argument("--n-save", type=int, default=4096, help="cycle-scale points")
    ap.add_argument(
        "--window-ns",
        type=float,
        default=30.0,
        help="half-width of the refined collapse window [ns]",
    )
    ap.add_argument("--max-steps", type=int, default=65536)
    ap.add_argument("--out", type=Path, default=Path("runs/single_collapse"))
    ap.add_argument(
        "--no-figure", action="store_true", help="print numbers only"
    )
    return ap.parse_args(argv)


def _build_config(args: argparse.Namespace):
    overrides: dict[str, object] = {}
    if args.amplitude is not None or args.harmonics is not None:
        base = load_config(scenario=args.scenario)
        amps = [
            float(a) / C.ATM_TO_PA for a in np.asarray(base.acoustics.amplitudes)
        ]
        if args.amplitude is not None:
            amps[0] = float(args.amplitude)
        if args.harmonics is not None:
            amps[1:4] = [float(v) for v in args.harmonics]
        overrides["acoustics.amplitudes_init"] = amps
    if args.phases is not None:
        overrides["acoustics.phases_init"] = [float(v) for v in args.phases]
    return load_config(scenario=args.scenario, overrides=overrides)


def _bracket(value: float, bounds: tuple[float, float]) -> str:
    lo, hi = bounds
    if lo <= value <= hi:
        return "within"
    return "BELOW" if value < lo else "ABOVE"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args)
    rhs_fn = gil.rhs_only if args.closure == "gilmore" else km.rhs_only
    period = 1.0 / float(cfg.acoustics.frequency)
    t_end = args.cycles * period

    blake = float(
        blake_threshold_pressure(
            cfg.r0,
            cfg.liquid.surface_tension,
            cfg.liquid.ambient_pressure,
            cfg.liquid.vapour_pressure,
        )
    )
    p_min = float(
        peak_negative_pressure(cfg.acoustics, cfg.liquid.ambient_pressure)
    )

    print("=" * 74)
    print(f"Sonofusion Sim -- single collapse: {cfg.scenario}")
    print("=" * 74)
    print(f"  closure                {args.closure}")
    print(f"  liquid                 {cfg.liquid.name}")
    print(f"  gas                    {', '.join(cfg.gas.names)}")
    print(f"  R_0                    {float(cfg.r0) * 1e6:.4f} um")
    print(f"  van der Waals core h   {float(cfg.hardcore_radius) * 1e6:.4f} um "
          f"(R_0/h = {float(cfg.r0 / cfg.hardcore_radius):.3f})")
    print(f"  gas nuclei N           {float(cfg.n_nuclei):.4e}")
    print(f"  drive frequency        {float(cfg.acoustics.frequency) / 1e3:.3f} kHz"
          f"   (period {period * 1e6:.3f} us)")
    amps_atm = np.asarray(cfg.acoustics.amplitudes) / C.ATM_TO_PA
    print(f"  harmonic amplitudes    "
          f"{np.array2string(amps_atm, precision=3)} atm")
    print(f"  harmonic phases        "
          f"{np.array2string(np.asarray(cfg.acoustics.phases), precision=3)} rad")
    print(f"  deepest tension        {p_min / C.ATM_TO_PA:+.3f} atm")
    print(f"  Blake threshold        {blake / C.ATM_TO_PA:.3f} atm "
          f"({'super' if (float(cfg.liquid.ambient_pressure) - p_min) > blake else 'SUB'}-threshold)")
    print(f"  solver                 {cfg.integrator.solver}, rtol="
          f"{float(cfg.integrator.rtol):.1e}, nondim="
          f"{cfg.integrator.nondimensionalise}")

    t0 = time.perf_counter()
    cycle = integrate(
        cfg, 0.0, t_end, rhs_fn=rhs_fn, n_save=args.n_save, throw=False
    )
    if not bool(cycle.success):
        print("\n  SOLVER FAILED on the cycle-scale pass "
              "(step budget exhausted).")
        return 1
    collapse, t_min = collapse_window(
        cfg,
        0.0,
        t_end,
        half_width=args.window_ns * 1e-9,
        n_points=2001,
        rhs_fn=rhs_fn,
    )
    elapsed = time.perf_counter() - t0

    stats = summarise(cfg, cycle, collapse)
    print("\n  --- collapse observables (peaks from the refined window) ---")
    print(f"  R_max                  {stats['r_max_m'] * 1e6:9.4f} um")
    print(f"  R_min                  {stats['r_min_m'] * 1e6:9.5f} um "
          f"({stats['r_min_over_hardcore']:.3f} h)")
    print(f"  compression R_max/R_min{stats['expansion_ratio']:9.1f}")
    print(f"  T_peak                 {stats['t_peak_K']:9.4e} K "
          f"({stats['t_peak_K'] / C.KELVIN_PER_EV:.4f} eV)")
    print(f"  peak wall Mach         {stats['peak_mach']:9.4f} "
          f"(vs c_inf)")
    print(f"  peak wall Mach         {stats['peak_mach_local']:9.4f} "
          f"(vs C at the wall)")
    print(f"  max |a_{cfg.shape.mode_l}|/R             "
          f"{stats['max_relative_distortion']:9.4e} "
          f"(threshold {float(cfg.shape.instability_threshold)})")
    print(f"  neutron yield          {stats['neutron_yield']:9.4e} per "
          f"{args.cycles} cycle(s)")
    print(f"  radiated energy        {stats['radiated_energy_J']:9.4e} J")
    print(f"  solver steps           {int(stats['solver_steps'])}")
    print(f"  wall time              {elapsed:9.2f} s")

    if args.scenario.startswith("sbsl_argon_water"):
        print("\n  --- against published SBSL values (BHL02) ---")
        checks = [
            ("R_max [um]", stats["r_max_m"] * 1e6, SBSL_REFERENCE["r_max_um"]),
            ("R_min [um]", stats["r_min_m"] * 1e6, SBSL_REFERENCE["r_min_um"]),
            ("T_peak [K]", stats["t_peak_K"], SBSL_REFERENCE["t_peak_K"]),
            ("peak Mach", stats["peak_mach"], SBSL_REFERENCE["peak_mach"]),
        ]
        for name, value, bounds in checks:
            print(f"  {name:14s} {value:12.4g}   reference "
                  f"{bounds[0]:g} - {bounds[1]:g}   -> "
                  f"{_bracket(value, bounds)}")

    if stats["max_relative_distortion"] > 0.3:
        print("\n  NOTE: |a_l|/R exceeds ~0.3, the limit of the LINEAR shape")
        print("        theory.  The value above shows the mode is unstable but")
        print("        its magnitude is not quantitative.  See docs/REALISM.md.")
    if stats["peak_mach_local"] > 1.0:
        print("\n  NOTE: peak wall Mach against the LOCAL sound speed")
        print("        exceeds 1.  Keller-Miksis and Gilmore are first")
        print("        order in that ratio, so this run is outside the")
        print("        model's trust region.  See docs/REALISM.md.")
    elif stats["peak_mach"] > 1.5:
        print("\n  NOTE: peak wall Mach against the AMBIENT sound")
        print(f"        speed exceeds 1.5, but the local ratio is "
              f"{stats['peak_mach_local']:.3f}, and that is the one")
        print("        the trust region is stated in.  The wall is")
        print("        compressed, so the liquid there is far stiffer")
        print("        than ambient.  Not a validity failure.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "scenario": cfg.scenario,
                "cycles": args.cycles,
                "amplitudes_atm": amps_atm.tolist(),
                "phases_rad": np.asarray(cfg.acoustics.phases).tolist(),
                "blake_threshold_atm": blake / C.ATM_TO_PA,
                "t_collapse_s": float(t_min),
                "wall_time_s": elapsed,
                **stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {args.out / 'summary.json'}")

    if not args.no_figure:
        fig = plot_cycle(
            cfg,
            cycle,
            collapse=collapse,
            t_min=t_min,
            title=(
                f"{cfg.scenario}: A_1 = {amps_atm[0]:.3f} atm, "
                f"T_peak = {stats['t_peak_K']:.3e} K, "
                f"R_min = {stats['r_min_m'] * 1e6:.4f} um"
            ),
        )
        written = save_figures([("cycle", fig)], args.out)
        for path in written:
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
