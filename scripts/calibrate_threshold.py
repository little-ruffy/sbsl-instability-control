from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_shape_model import (
    basilisk_cfg, legendre_a2, predict,
)

FOLD_BINS = 60
FOLD_SPREAD = 0.25
FOLD_ONSET = 0.02


def fold_fraction(path: Path) -> float:
    a = np.loadtxt(path)
    if a.ndim != 2 or len(a) < 8:
        return float("nan")
    r = np.hypot(a[:, 0], np.abs(a[:, 1]))
    mu = np.cos(np.arctan2(np.abs(a[:, 1]), a[:, 0]))
    idx = np.digitize(mu, np.linspace(0.0, 1.0, FOLD_BINS))
    folded = total = 0
    for b in np.unique(idx):
        rr = r[idx == b]
        if len(rr) < 2:
            continue
        total += 1
        if (rr.max() - rr.min()) / max(rr.mean(), 1e-30) > FOLD_SPREAD:
            folded += 1
    return folded / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--inviscid", action="store_true")
    args = ap.parse_args()

    cfg = basilisk_cfg(inviscid=args.inviscid)
    print("Where the interface first folds, against what §2.7 predicts there")
    print(f"(fold = more than {FOLD_ONSET:.0%} of rays crossing twice)\n")
    print(f"{'seed':>6} {'t_fold':>9} {'frame':>6} {'NS |a2|/R':>11} "
          f"{'§2.7 says':>11} {'NS at end':>10} {'fold end':>9}")
    print("-" * 70)

    rows = []
    for d in sorted(Path(args.root).glob("scan_*")):
        tag = d.name.replace("scan_", "")
        seed = float(tag[0] + "." + tag[1:])
        log = np.loadtxt(d / "log.txt", skiprows=1)
        t_log, r_vol = log[:, 0], (3.0 * log[:, 1]) ** (1.0 / 3.0)
        idx = np.atleast_2d(np.loadtxt(d / "frames.dat"))

        ts, r_ns, a2_ns, folds = [], [], [], []
        for fr, tt in idx:
            f = d / f"facets-{int(fr):04d}.dat"
            if not f.exists():
                continue
            a = np.loadtxt(f)
            if a.ndim != 2 or len(a) < 8:
                continue
            r0, a2 = legendre_a2(a[:, 0], a[:, 1])
            ts.append(float(tt))
            r_ns.append(r0)
            a2_ns.append(a2)
            folds.append(fold_fraction(f))
        if len(ts) < 5:
            continue
        ts = np.array(ts)
        r_ns = np.array(r_ns)
        a2_ns = np.array(a2_ns)
        folds = np.array(folds)

        a2_lin, _t0 = predict(t_log, r_vol, ts, ts, a2_ns, cfg)
        rel_lin = np.abs(a2_lin) / np.maximum(r_ns, 1e-30)
        rel_ns = np.abs(a2_ns) / np.maximum(r_ns, 1e-30)

        hit = np.nonzero(folds > FOLD_ONSET)[0]
        if len(hit) == 0:
            print(f"{seed:>6.2f} {'never':>9} {'-':>6} {'-':>11} {'-':>11} "
                  f"{rel_ns[-1]:>10.3f} {folds[-1]:>9.3f}")
            continue
        k = int(hit[0])
        print(f"{seed:>6.2f} {ts[k]:>9.5f} {k:>6d} {rel_ns[k]:>11.3f} "
              f"{rel_lin[k]:>11.3f} {rel_ns[-1]:>10.3f} {folds[-1]:>9.3f}")
        rows.append((seed, rel_lin[k], rel_ns[k]))

    if not rows:
        print("\nNo run folded: the single-mode description held throughout.")
        return 0

    arr = np.array(rows)
    print()
    print(f"§2.7's |a_2|/R at the onset of folding, over {len(arr)} seeds:")
    print(f"  mean {arr[:,1].mean():.3f}, sd {arr[:,1].std(ddof=1) if len(arr)>1 else 0.0:.3f}, "
          f"range {arr[:,1].min():.3f}..{arr[:,1].max():.3f}")
    print()
    print("This is the number §7.6's termination threshold should be set")
    print("against: the model's own reading at the moment the real interface")
    print("stops being describable by one amplitude. It is a calibration of")
    print("the criterion, not a correction to the amplitude -- the amplitude")
    print("has no corrected value once the surface has folded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
