from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def legendre_coeffs(theta, r, lmax=6):
    mu = np.cos(theta)
    order = np.argsort(mu)
    mu, r = mu[order], r[order]
    uniq, inv = np.unique(np.round(mu, 10), return_inverse=True)
    if len(uniq) < len(mu):
        r = np.bincount(inv, weights=r) / np.bincount(inv)
        mu = uniq
    out = []
    for l in range(lmax + 1):
        p = np.polynomial.legendre.Legendre.basis(l)(mu)
        out.append((2 * l + 1) * np.trapezoid(r * p, mu))
    return np.array(out)


def frames(path):
    blocks, cur = [], []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            if cur:
                blocks.append(np.array(cur))
                cur = []
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                cur.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if cur:
        blocks.append(np.array(cur))
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interfaces", required=True, help="Basilisk `out`")
    ap.add_argument("--log", required=True, help="Basilisk `log`")
    ap.add_argument("--lmax", type=int, default=6)
    args = ap.parse_args()

    blocks = frames(args.interfaces)
    print(f"interface frames: {len(blocks)}")
    if not blocks:
        print("no frames parsed -- check the output format")
        return 1

    d = np.loadtxt(args.log, skiprows=1)
    t_log, vol = d[:, 0], d[:, 1]
    r_vol = (3.0 * vol) ** (1.0 / 3.0)

    print(f"\n{'frame':>6} {'pts':>6} {'R(l=0)':>10} {'a_2':>11} "
          f"{'|a2|/R':>9} {'a_4':>11} {'|a4|/R':>9}")
    print("-" * 70)
    rows = []
    for i, b in enumerate(blocks):
        x, y = b[:, 0], np.abs(b[:, 1])
        rr = np.hypot(x, y)
        th = np.arctan2(y, x)
        if len(rr) < 8:
            continue
        c = legendre_coeffs(th, rr, args.lmax)
        r0, a2, a4 = c[0], c[2], c[4] if len(c) > 4 else 0.0
        rows.append((i, r0, a2, a4))
        if i % max(1, len(blocks) // 20) == 0 or i == len(blocks) - 1:
            print(f"{i:>6} {len(rr):>6} {r0:>10.5f} {a2:>11.3e} "
                  f"{abs(a2)/max(r0,1e-30):>9.4f} {a4:>11.3e} "
                  f"{abs(a4)/max(r0,1e-30):>9.4f}")

    arr = np.array([(r[1], r[2], r[3]) for r in rows])
    rel2 = np.abs(arr[:, 1]) / np.maximum(arr[:, 0], 1e-30)
    print()
    print(f"R from l=0 projection: {arr[0,0]:.5f} -> {arr[-1,0]:.5f}")
    print(f"R from volume        : {r_vol[0]:.5f} -> {r_vol[-1]:.5f}")
    print(f"  (these must agree, or the projection geometry is wrong)")
    print()
    print(f"peak |a_2|/R over the run: {rel2.max():.4f} "
          f"at frame {int(np.argmax(rel2))}")
    print(f"final |a_2|/R            : {rel2[-1]:.4f}")
    print()
    print("Interpretation is pre-committed in docs/VALIDATION_CFD.md and is")
    print("NOT to be adjusted to suit these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
