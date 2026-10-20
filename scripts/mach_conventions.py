from __future__ import annotations

import argparse
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.batched import BatchSpec, batched_cycle_summary, batched_initial_state
from core.config import load_config
from core.constants import ATM_TO_PA


# Compare ambient |Rdot|/c_inf vs local Tait |Rdot|/C(p_w) across the constant-drive grid
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="sbsl_argon_unstable")
    ap.add_argument("--constants", default="1.250,1.375,1.450,1.550,1.650")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config(scenario=args.scenario)
    period = 1.0 / float(cfg.acoustics.frequency)
    grid = [float(x) for x in args.constants.split(",")]

    print(f"scenario: {cfg.scenario}, c_inf = {float(cfg.liquid.sound_speed):.1f} m/s")
    print(f"{'A_1 [atm]':>10} {'|Rdot|/c_inf':>14} {'|Rdot|/C_wall':>15} {'ratio':>9}")
    print("-" * 52)

    for a1 in grid:
        amps = np.zeros((1, 4))
        amps[0, 0] = a1 * ATM_TO_PA
        spec = BatchSpec(
            r0=jnp.asarray([float(cfg.r0)]),
            n_nuclei=jnp.asarray([float(cfg.n_nuclei)]),
            ambient_temperature=jnp.asarray([float(cfg.liquid.ambient_temperature)]),
            amplitudes=jnp.asarray(amps),
            phases=jnp.zeros((1, 4)),
        )
        y = batched_initial_state(cfg, spec, jnp.asarray([1.0e-3]), jnp.zeros((1,)))
        m_amb = m_loc = 0.0
        for step in range(args.steps):
            t0 = jnp.asarray([step * period])
            t1 = jnp.asarray([(step + 1) * period])
            out = batched_cycle_summary(cfg, spec, y, t0, t1, 4096)
            m_amb = max(m_amb, float(out.peak_mach[0]))
            m_loc = max(m_loc, float(out.peak_mach_local[0]))
            y = out.final_state
            if float(out.max_relative_distortion[0]) >= 0.334 or not bool(out.success[0]):
                break
        ratio = m_amb / max(m_loc, 1.0e-12)
        print(f"{a1:>10.3f} {m_amb:>14.3f} {m_loc:>15.3f} {ratio:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
