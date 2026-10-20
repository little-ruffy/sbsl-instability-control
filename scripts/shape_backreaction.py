from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_shape_model import facet_volume, legendre_a2


def case(d: Path):
    tag = d.name.replace("scan_", "")
    eps = float(tag[0] + "." + tag[1:])
    log = np.loadtxt(d / "log.txt", skiprows=1)
    f0 = np.loadtxt(d / "facets-0000.dat")
    r0, a2 = legendre_a2(f0[:, 0], f0[:, 1])
    return {
        "eps": eps,
        "t": log[:, 0],
        "v": log[:, 1],
        "r_proj": r0,
        "a2_meas": a2,
        "v_facet0": facet_volume(f0[:, 0], f0[:, 1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    cases = [case(d) for d in sorted(Path(args.root).glob("scan_*"))
             if (d / "facets-0000.dat").exists()]
    if len(cases) < 3:
        print("need at least three cases")
        return 1
    cases.sort(key=lambda c: c["eps"])
    ref = cases[0]

    print("Step A -- is the seed what was commanded, and is the geometric")
    print("excess the analytic 0.6 eps^2?  Both must hold before the")
    print("dynamic part means anything.\n")
    print(f"{'eps':>6} {'R (l=0)':>9} {'a2/R meas':>10} {'seed err':>9} "
          f"{'V(facets)':>11} {'geom exc':>9} {'0.6 eps^2':>10} {'ratio':>7}")
    print("-" * 78)
    for c in cases:
        rel = abs(c["a2_meas"]) / c["r_proj"]
        geom = c["v_facet0"] / ref["v_facet0"] - 1.0
        pred = 0.6 * (c["eps"] ** 2 - ref["eps"] ** 2)
        ratio = geom / pred if abs(pred) > 1e-12 else float("nan")
        print(f"{c['eps']:>6.2f} {c['r_proj']:>9.5f} {rel:>10.4f} "
              f"{rel/c['eps']-1:>8.2%} {c['v_facet0']:>11.6f} "
              f"{geom:>8.3%} {pred:>9.3%} {ratio:>7.3f}")

    print()
    print("Step B -- the dynamic part: total volume excess at the end of the")
    print("run, minus the geometric excess that was there at t = 0.\n")
    print(f"{'eps':>6} {'V_final':>10} {'total exc':>10} {'geom exc':>9} "
          f"{'dynamic':>9} {'dyn/eps^2':>10} {'-> dR/R':>9} "
          f"{'(dR/R)/eps^2':>13}")
    print("-" * 82)
    rows = []
    for c in cases:
        v_fin = float(np.interp(ref["t"][-1], c["t"], c["v"]))
        v_fin_ref = ref["v"][-1]
        total = v_fin / v_fin_ref - 1.0
        geom = c["v_facet0"] / ref["v_facet0"] - 1.0
        dyn = total - geom
        dr = (1.0 + dyn) ** (1.0 / 3.0) - 1.0
        e2 = c["eps"] ** 2
        print(f"{c['eps']:>6.2f} {v_fin:>10.6f} {total:>9.3%} {geom:>8.3%} "
              f"{dyn:>8.3%} {dyn/e2:>10.4f} {dr:>8.3%} {dr/e2:>13.4f}")
        rows.append((c["eps"], dyn, dr))

    arr = np.array(rows)
    big = arr[arr[:, 0] >= 0.1]
    print()
    print("What §2.7 needs: the radial contamination dR/R must be no larger")
    print("than O((a/R)^2).  Comparing the two directly --\n")
    print(f"{'a/R':>6} {'(a/R)^2':>10} {'measured dR/R':>14} {'margin':>9}")
    print("-" * 44)
    for e, _dyn, dr in big:
        print(f"{e:>6.2f} {e**2:>10.4f} {dr:>13.3%} "
              f"{e**2/max(abs(dr),1e-30):>8.1f}x")
    print()
    print("A margin above 1 means the bound holds with room to spare; below 1")
    print("means the neglected term is bigger than §2.7 claims and the radial")
    print("solution is contaminated.  Reading is pre-committed in")
    print("docs/VALIDATION_CFD.md and is not to be adjusted to suit these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
