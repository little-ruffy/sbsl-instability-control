from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

LOG_ROW = re.compile(r"^\s*(\d+\.\d{6})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
MACH_ROW = re.compile(r"^\s*(\d+\.\d{8})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def parse(path: Path, pattern):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.match(line)
        if not m:
            continue
        try:
            v = [float(x) for x in m.groups()]
        except ValueError:
            continue
        if not any(x != x for x in v):
            rows.append(v)
    return np.array(rows) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="a radial run's log.txt")
    ap.add_argument("--c-inf", type=float, default=466.658,
                    help="far-field sound speed, sqrt(gamma1 (pinf+PI1)/rho)")
    args = ap.parse_args()

    path = Path(args.log)
    a = parse(path, LOG_ROW)
    b = parse(path, MACH_ROW)
    if a is None:
        print("no bubble.h rows found")
        return 1
    if b is None:
        print("no instrumented rows found -- was the solver rebuilt?")
        return 1

    t1, r1 = a[:, 0], (3.0 * a[:, 1]) ** (1.0 / 3.0)
    mach1 = np.abs(np.gradient(r1, t1)) / args.c_inf

    t2, r2, rdot2, mach2 = b[:, 0], b[:, 1], b[:, 2].copy(), b[:, 3].copy()
    if len(r2) > 0 and mach2[0] > 10.0:
        scale = (4.0 * np.pi) ** (1.0 / 3.0)
        r_prev = np.concatenate([[r2[0]], r2[:-1]])
        dt_ex = (r2 - r_prev / scale) / np.where(
            np.abs(rdot2) > 1.0e-30, rdot2, 1.0e-30
        )
        rdot2 = (r2 - r_prev) / np.where(np.abs(dt_ex) > 1.0e-30, dt_ex, 1.0e-30)
        mach2 = np.abs(rdot2) / args.c_inf

    print(f"route 1 (volume log, differenced):  {len(t1)} samples")
    print(f"route 2 (solver instrumentation):   {len(t2)} samples\n")

    print(f"{'quantity':>22} {'route 1':>12} {'route 2':>12} {'rel':>9}")
    print("-" * 58)
    pairs = [
        ("R(0)", r1[0], r2[0]),
        ("R_min", r1.min(), r2.min()),
        ("peak Mach", mach1.max(), mach2.max()),
    ]
    worst = 0.0
    for name, v1, v2 in pairs:
        rel = abs(v2 - v1) / max(abs(v1), 1e-30)
        worst = max(worst, rel)
        print(f"{name:>22} {v1:>12.5f} {v2:>12.5f} {rel:>8.2%}")

    r2i = np.interp(t1, t2, r2)
    dr = np.abs(r2i - r1) / np.maximum(r1, 1e-30)
    print(f"\npointwise |dR|/R over {len(t1)} samples: "
          f"mean {dr.mean():.3%}, max {dr.max():.3%}")

    rdot1 = np.gradient(r1, t1)
    rdot2i = np.interp(t1, t2, rdot2)
    live = np.abs(rdot2i) > 0.05 * np.abs(rdot2i).max()
    series = np.median(
        np.abs(rdot1[live] - rdot2i[live])
        / np.maximum(np.abs(rdot2i[live]), 1e-12)
    )
    peak_gap = abs(mach2.max() - mach1.max()) / max(mach1.max(), 1e-30)

    print(f"velocity series, where |Rdot| > 5 % of peak: "
          f"median disagreement {series:.2%}")
    print(f"peak disagreement: {peak_gap:.1%}")
    print()
    if max(worst_r := abs(r1.min() - r2.min()) / max(r1.min(), 1e-30),
           0.0) > 0.05:
        print(f"The RADII disagree by {worst_r:.1%}. The two routes are not")
        print("measuring the same run, and nothing below is meaningful.")
    elif series < 0.05:
        print("The trajectories agree. Any peak disagreement is the estimator,")
        print("not the data --- but note that it makes the PEAK itself a poor")
        print("statistic, which is why the paper's independent variable is")
        print("compression and not Mach.")
    else:
        print("The velocity series DISAGREE beyond estimator noise. One route")
        print("is wrong and the published Mach values cannot be relied on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
