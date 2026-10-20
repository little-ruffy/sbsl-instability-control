from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_shape_model import (
    MU_L, basilisk_cfg, legendre_a2, predict,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="renders")
    ap.add_argument("--inviscid", action="store_true",
                    help="zero viscosity on the linear side to match an "
                         "inviscid Basilisk scan")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = basilisk_cfg(mu=0.0 if args.inviscid else MU_L)
    cases = sorted(
        d for d in Path(args.root).glob("scan_*")
        if (d / "facets-0000.dat").exists() and (d / "log.txt").exists()
    )
    if not cases:
        print("nothing to render")
        return 1

    fig, axes = plt.subplots(2, len(cases), figsize=(3.1 * len(cases), 6.4))
    axes = np.atleast_2d(axes)

    for col, d in enumerate(cases):
        tag = d.name.replace("scan_", "")
        eps = float(tag[0] + "." + tag[1:]) if len(tag) > 1 else 0.0
        log = np.loadtxt(d / "log.txt", skiprows=1)
        t_log, r_vol = log[:, 0], (3.0 * log[:, 1]) ** (1.0 / 3.0)
        idx = np.atleast_2d(np.loadtxt(d / "frames.dat"))

        rows, shapes = [], []
        for fr, tt in idx:
            f = d / f"facets-{int(fr):04d}.dat"
            if not f.exists():
                continue
            a = np.loadtxt(f)
            if a.ndim != 2 or len(a) < 8:
                continue
            r0, a2 = legendre_a2(a[:, 0], a[:, 1])
            rows.append((float(tt), r0, a2))
            shapes.append((float(tt), a))
        if len(rows) < 3:
            continue
        rows = np.array(rows)

        ax = axes[0, col]
        picks = np.linspace(0, len(shapes) - 1, 5).astype(int)
        cmap = plt.get_cmap("viridis")
        for j, i in enumerate(picks):
            tt, a = shapes[i]
            x, y = a[:, 0], np.abs(a[:, 1])
            c = cmap(j / max(len(picks) - 1, 1))
            ax.scatter(x, y, s=0.8, color=c, linewidths=0)
            ax.scatter(x, -y, s=0.8, color=c, linewidths=0)
            ax.scatter(-x, y, s=0.8, color=c, linewidths=0)
            ax.scatter(-x, -y, s=0.8, color=c, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"seed $a_2/R$ = {eps:.2f}", fontsize=10)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        if col == 0:
            ax.set_ylabel("interface (dark = early)")

        ax = axes[1, col]
        t_out, r_ns, a2_ns = rows[:, 0], rows[:, 1], rows[:, 2]
        rel_ns = np.abs(a2_ns) / np.maximum(r_ns, 1e-30)
        ax.plot(t_out, rel_ns, "o-", ms=2.5, lw=1.2,
                label="Navier-Stokes", color="#1f77b4")
        if eps > 0:
            a2_lin, _ = predict(t_log, r_vol, t_out, t_out, a2_ns, cfg)
            ax.plot(t_out, np.abs(a2_lin) / np.maximum(r_ns, 1e-30),
                    "--", lw=1.2, label=u"§2.7 linear", color="#d62728")
        ax.axhline(0.3, color="k", ls=":", lw=0.9)
        ax.text(t_out[0], 0.32, "claimed validity limit", fontsize=6)
        ax.axhline(1.0, color="grey", ls=":", lw=0.9)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 20.0)
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel(r"$|a_2|/R$")
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Seeded $l=2$ collapse in Basilisk, against this project's linear "
        "shape equation\n(radial history taken from Basilisk on both sides)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "shape_validation.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
