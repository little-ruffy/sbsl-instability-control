from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from core.config import load_config
from core.integrator import integrate


def _timed(fn: Callable[[], object], repeats: int = 3) -> tuple[float, float]:
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return first, float(np.median(samples))


def _solve(cfg, period: float, **kwargs):

    def run():
        traj = integrate(cfg, 0.0, period, throw=False, **kwargs)
        traj.states.block_until_ready()
        return traj

    return run


def bench_solvers(quick: bool) -> list[dict]:
    print("\n--- 1. solver x non-dimensionalisation (one SBSL cycle) ---")
    print(f"{'solver':10s} {'nondim':7s} {'ok':6s} {'steps':>7s} "
          f"{'compile[s]':>11s} {'run[s]':>9s}")
    cfg = load_config(scenario="sbsl_argon_water_fast")
    period = 1.0 / float(cfg.acoustics.frequency)
    solvers = ["Tsit5"] if quick else ["Tsit5", "Dopri8", "Kvaerno3", "Kvaerno5"]
    rows = []
    for solver in solvers:
        for nondim in (True, False):
            variant = cfg.with_integrator(
                solver=solver, nondimensionalise=nondim, max_steps=40000
            )
            try:
                compile_s, run_s = _timed(
                    _solve(variant, period, n_save=1024), repeats=2
                )
                traj = integrate(
                    variant, 0.0, period, n_save=1024, throw=False
                )
                ok, steps = bool(traj.success), int(traj.num_steps)
            except Exception as exc:
                print(f"{solver:10s} {nondim!s:7s} EXCEPTION "
                      f"{type(exc).__name__}")
                rows.append({"solver": solver, "nondim": nondim,
                             "error": type(exc).__name__})
                continue
            print(f"{solver:10s} {nondim!s:7s} {ok!s:6s} {steps:7d} "
                  f"{compile_s:11.2f} {run_s:9.4f}")
            rows.append({
                "solver": solver, "nondim": nondim, "ok": ok, "steps": steps,
                "compile_s": compile_s, "run_s": run_s,
            })
    return rows


def bench_saha() -> list[dict]:
    print("\n--- 2. Saha ladder depth (argon) ---")
    print(f"{'stages':>7s} {'ok':6s} {'steps':>7s} {'R_min[um]':>11s} "
          f"{'T_peak[K]':>11s} {'compile[s]':>11s} {'run[s]':>9s} "
          f"{'speedup':>8s}")
    rows = []
    baseline_run = None
    for stages in (None, 4, 2, 1):
        overrides = (
            {} if stages is None
            else {"gas_species.argon.max_ionisation_stages": stages}
        )
        cfg = load_config(scenario="sbsl_argon_water", overrides=overrides)
        period = 1.0 / float(cfg.acoustics.frequency)
        compile_s, run_s = _timed(_solve(cfg, period, n_save=1024), repeats=2)
        traj = integrate(
            cfg, 0.0, period, save_at_steps=True, max_steps=8192, throw=False
        )
        if baseline_run is None:
            baseline_run = run_s
        label = "all 18" if stages is None else str(stages)
        print(f"{label:>7s} {bool(traj.success)!s:6s} {int(traj.num_steps):7d} "
              f"{float(traj.min_radius) * 1e6:11.6f} "
              f"{float(traj.peak_temperature):11.5e} "
              f"{compile_s:11.2f} {run_s:9.4f} "
              f"{baseline_run / run_s:7.2f}x")
        rows.append({
            "stages": stages, "steps": int(traj.num_steps),
            "r_min_m": float(traj.min_radius),
            "t_peak_K": float(traj.peak_temperature),
            "compile_s": compile_s, "run_s": run_s,
            "speedup": baseline_run / run_s,
        })
    return rows


def bench_save_grid() -> list[dict]:
    print("\n--- 3. save-grid length (a compiled shape) ---")
    print(f"{'mode':>14s} {'points':>8s} {'compile[s]':>11s} {'run[s]':>9s} "
          f"{'T_peak[K]':>11s}")
    cfg = load_config(scenario="sbsl_argon_water_fast")
    period = 1.0 / float(cfg.acoustics.frequency)
    rows = []
    for n_save in (256, 1024, 4096, 16384):
        compile_s, run_s = _timed(_solve(cfg, period, n_save=n_save), repeats=2)
        traj = integrate(cfg, 0.0, period, n_save=n_save, throw=False)
        print(f"{'uniform grid':>14s} {n_save:8d} {compile_s:11.2f} "
              f"{run_s:9.4f} {float(traj.peak_temperature):11.5e}")
        rows.append({"mode": "grid", "points": n_save,
                     "compile_s": compile_s, "run_s": run_s,
                     "t_peak_K": float(traj.peak_temperature)})
    for budget in (4096, 16384):
        compile_s, run_s = _timed(
            _solve(cfg, period, save_at_steps=True, max_steps=budget),
            repeats=2,
        )
        traj = integrate(
            cfg, 0.0, period, save_at_steps=True, max_steps=budget, throw=False
        )
        print(f"{'steps':>14s} {budget:8d} {compile_s:11.2f} {run_s:9.4f} "
              f"{float(traj.peak_temperature):11.5e}")
        rows.append({"mode": "steps", "points": budget,
                     "compile_s": compile_s, "run_s": run_s,
                     "t_peak_K": float(traj.peak_temperature)})
    return rows


def bench_batched(quick: bool) -> list[dict]:
    import numpy as _np

    from core.batched import (
        BatchSpec,
        batched_cycle_summary,
        batched_initial_state,
        estimate_device_memory,
    )
    from core.constants import ATM_TO_PA

    print("\n--- 5. jax.vmap batch scaling (one SBSL cycle per bubble) ---")
    print(f"{'B':>6} {'compile[s]':>11} {'run[s]':>9} {'s/bubble':>10} "
          f"{'speedup':>8} {'buffer':>10}")
    cfg = load_config(scenario="sbsl_argon_water_fast")
    period = 1.0 / float(cfg.acoustics.frequency)
    rng = _np.random.default_rng(0)
    sizes = (1, 8, 32) if quick else (1, 4, 16, 64, 128, 256)
    rows, baseline = [], None

    for batch in sizes:
        amps = _np.zeros((batch, 4))
        amps[:, 0] = rng.uniform(1.0, 1.5, size=batch) * ATM_TO_PA
        spec = BatchSpec(
            r0=jnp.asarray(rng.uniform(0.9, 1.1, size=batch) * float(cfg.r0)),
            n_nuclei=jnp.full((batch,), float(cfg.n_nuclei)),
            ambient_temperature=jnp.full(
                (batch,), float(cfg.liquid.ambient_temperature)
            ),
            amplitudes=jnp.asarray(amps),
            phases=jnp.zeros((batch, 4)),
        )
        y0 = batched_initial_state(
            cfg, spec, jnp.full((batch,), 1e-3), jnp.zeros((batch,))
        )
        t0v = jnp.zeros((batch,))
        t1v = jnp.full((batch,), period)

        def run(_spec=spec, _y0=y0, _t0=t0v, _t1=t1v):
            out = batched_cycle_summary(cfg, _spec, _y0, _t0, _t1, 4096)
            out.peak_temperature.block_until_ready()
            return out

        compile_s, run_s = _timed(run, repeats=2)
        per_bubble = run_s / batch
        if baseline is None:
            baseline = per_bubble
        print(f"{batch:6d} {compile_s:11.2f} {run_s:9.3f} {per_bubble:10.4f} "
              f"{baseline / per_bubble:7.2f}x "
              f"{estimate_device_memory(batch, 4096) / 1e6:8.1f}MB")
        rows.append({
            "batch": batch, "compile_s": compile_s, "run_s": run_s,
            "s_per_bubble": per_bubble, "speedup": baseline / per_bubble,
            "buffer_bytes": estimate_device_memory(batch, 4096),
        })
    print(
        "  NOTE: an XLA CPU backend vectorises only across SIMD lanes,\n"
        "        so this curve flattens near 5x.  The design target is\n"
        "        an accelerator, which has NOT been measured here."
    )
    return rows


def bench_env(n_steps: int) -> dict:
    print(f"\n--- 4. environment throughput ({n_steps} steps) ---")
    try:
        from envs.sonofusion_env import SonofusionEnv
    except ImportError:
        print('  [skipped: envs/ not present in open-core preview]')
        return {}

    env = SonofusionEnv()
    action = np.array([0.8, -1, -1, -1, 0, 0, 0, 0], dtype=np.float32)

    start = time.perf_counter()
    env.reset(seed=0)
    env.step(action)
    first = time.perf_counter() - start

    start = time.perf_counter()
    done_count = 0
    for _ in range(n_steps):
        _obs, _r, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            env.reset()
            done_count += 1
    elapsed = time.perf_counter() - start
    per_step = elapsed / n_steps

    print(f"  first step (incl. compile) : {first:8.2f} s")
    print(f"  steady-state per step      : {per_step:8.4f} s")
    print(f"  throughput                 : {1.0 / per_step:8.2f} steps/s")
    print(f"  episodes completed         : {done_count}")
    print(f"  -> 1e5 env steps would take: {1e5 * per_step / 3600:8.2f} h "
          f"on this machine, single process")
    return {
        "first_step_s": first,
        "per_step_s": per_step,
        "steps_per_s": 1.0 / per_step,
        "hours_per_1e5_steps": 1e5 * per_step / 3600.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow solver comparison")
    ap.add_argument("--env-steps", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=" * 74)
    print("Sonofusion Sim -- performance benchmark")
    print("=" * 74)
    print(f"  platform  : {platform.platform()}")
    print(f"  python    : {platform.python_version()}")
    print(f"  jax       : {jax.__version__}")
    print(f"  devices   : {jax.devices()}")
    print(f"  x64       : {jax.config.jax_enable_x64}")

    results = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "solvers": bench_solvers(args.quick),
        "saha": bench_saha(),
        "save_grid": bench_save_grid(),
        "batched": bench_batched(args.quick),
        "environment": bench_env(args.env_steps),
    }

    print("\n" + "=" * 74)
    print("Read docs/OPTIMIZATION.md for what to do with these numbers.")
    print("=" * 74)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
