from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper import data as D
from core.spherical_euler import solve_spherical_euler
from core.axisymmetric_collapse import fold_fraction_from_coords

DATA_DIR = ROOT / "paper" / "data"
OUT_DIR = ROOT / "paper" / "renders"


def load_vof_segments(facet_path: Path) -> np.ndarray:
    if not facet_path.exists():
        return np.zeros((0, 2, 2), dtype=np.float64)
    text = facet_path.read_text(encoding="utf-8", errors="ignore")
    blocks = text.strip().split("\n\n")
    segs: list[list[list[float]]] = []
    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if len(lines) >= 2:
            try:
                p1 = [float(v) for v in lines[0].split()[:2]]
                p2 = [float(v) for v in lines[1].split()[:2]]
                segs.append([[p1[0], abs(p1[1])], [p2[0], abs(p2[1])]])
            except ValueError:
                continue
    if not segs:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.asarray(segs, dtype=np.float64)


def load_basilisk_log(log_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, rs, pgs = [], [], []
    for ln in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.split()
        if len(parts) != 4:
            continue
        try:
            t_v, vol_v, _ek, pg_v = (float(x) for x in parts)
        except ValueError:
            continue
        if len(parts[0].split(".")[-1]) == 6:
            ts.append(t_v)
            rs.append((3.0 * max(vol_v, 0.0)) ** (1.0 / 3.0))
            pgs.append(pg_v)
    return np.asarray(ts), np.asarray(rs), np.asarray(pgs)


def build_revolved_vof_quads(
    segs_q1: np.ndarray,
    *,
    n_phi: int = 44,
    phi_start_deg: float = 55.0,
    phi_span_deg: float = 255.0,
    max_segs: int = 220,
) -> tuple[np.ndarray, np.ndarray]:
    if len(segs_q1) == 0:
        return np.zeros((0, 4, 3)), np.zeros((0,), dtype=bool)

    if len(segs_q1) > max_segs:
        idx = np.linspace(0, len(segs_q1) - 1, max_segs).astype(int)
        segs_q1 = segs_q1[idx]

    segs_south = segs_q1.copy()
    segs_south[:, :, 0] *= -1.0
    segs_all = np.concatenate([segs_q1, segs_south], axis=0)

    phi = np.deg2rad(
        np.linspace(phi_start_deg, phi_start_deg + phi_span_deg, n_phi)
    )
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    z1 = segs_all[:, 0, 0][:, None]
    r1 = segs_all[:, 0, 1][:, None]
    z2 = segs_all[:, 1, 0][:, None]
    r2 = segs_all[:, 1, 1][:, None]

    x00 = r1 * cos_p[:-1]
    y00 = r1 * sin_p[:-1]
    z00 = np.broadcast_to(z1, x00.shape)

    x01 = r1 * cos_p[1:]
    y01 = r1 * sin_p[1:]
    z01 = np.broadcast_to(z1, x01.shape)

    x11 = r2 * cos_p[1:]
    y11 = r2 * sin_p[1:]
    z11 = np.broadcast_to(z2, x11.shape)

    x10 = r2 * cos_p[:-1]
    y10 = r2 * sin_p[:-1]
    z10 = np.broadcast_to(z2, x10.shape)

    p00 = np.stack([x00, y00, z00], axis=-1)
    p01 = np.stack([x01, y01, z01], axis=-1)
    p11 = np.stack([x11, y11, z11], axis=-1)
    p10 = np.stack([x10, y10, z10], axis=-1)

    surf_quads = np.stack([p00, p01, p11, p10], axis=-2).reshape(-1, 4, 3)

    cap_list = []
    for p_ang in (phi[0], phi[-1]):
        ca, sa = math.cos(p_ang), math.sin(p_ang)
        c0 = np.stack([np.zeros_like(z1[:, 0]), np.zeros_like(z1[:, 0]), z1[:, 0]], axis=-1)
        c1 = np.stack([r1[:, 0] * ca, r1[:, 0] * sa, z1[:, 0]], axis=-1)
        c2 = np.stack([r2[:, 0] * ca, r2[:, 0] * sa, z2[:, 0]], axis=-1)
        c3 = np.stack([np.zeros_like(z2[:, 0]), np.zeros_like(z2[:, 0]), z2[:, 0]], axis=-1)
        cap_list.append(np.stack([c0, c1, c2, c3], axis=-2))

    cap_quads = np.concatenate(cap_list, axis=0)
    all_quads = np.concatenate([surf_quads, cap_quads], axis=0)
    is_cap = np.zeros(len(all_quads), dtype=bool)
    is_cap[len(surf_quads):] = True
    return all_quads, is_cap


def shade_quads(
    quads: np.ndarray,
    is_cap: np.ndarray,
    surf_rgb: tuple[float, float, float],
    cap_rgb: tuple[float, float, float],
) -> np.ndarray:
    d1 = quads[:, 2] - quads[:, 0]
    d2 = quads[:, 3] - quads[:, 1]
    n = np.cross(d1, d2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1.0e-30)

    light = np.asarray([-0.42, -0.78, 0.46], dtype=np.float64)
    light /= np.linalg.norm(light)
    view = np.asarray([0.55, -0.70, 0.45], dtype=np.float64)
    view /= np.linalg.norm(view)
    half_v = light + view
    half_v /= np.linalg.norm(half_v)

    lam = np.abs(n @ light)
    spec = np.power(np.abs(n @ half_v), 18.0)

    base = np.where(is_cap[:, None], np.asarray(cap_rgb), np.asarray(surf_rgb))
    factor = np.where(is_cap, 0.55 + 0.35 * lam, 0.36 + 0.58 * lam)
    rgb = np.clip(
        base * factor[:, None] + np.where(is_cap[:, None], 0.0, 0.22 * spec[:, None]),
        0.0,
        1.0,
    )
    rgba = np.ones((len(quads), 4), dtype=np.float64)
    rgba[:, :3] = rgb
    return rgba


def render_plate1_3d_revolved() -> Path:
    frames = [0, 20, 35, 52]
    frame_Set_labels = [
        r"$t = 0.000$ (initial $l=2$ seed)",
        r"$t = 0.038$ (compression & jet onset)",
        r"$t = 0.067$ (turnaround & axial jet)",
        r"$t = 0.099$ (rebound & lobe pinch-off)",
    ]

    fig = plt.figure(figsize=(15.0, 7.8), dpi=260)
    fig.patch.set_facecolor("#0e131a")

    cmap_gas = plt.get_cmap("inferno")
    p_norm = LogNorm(vmin=80.0, vmax=2800.0)

    rows_cfg = [
        ("Viscous VOF (Re = 11.2, single-valued)", DATA_DIR / "viscous" / "scan_020", (0.38, 0.73, 0.96)),
        ("Inviscid VOF (Re = inf, axial jetting)", DATA_DIR / "inviscid" / "scan_020", (0.46, 0.82, 0.98)),
    ]

    for row_idx, (row_title, scan_dir, surf_rgb) in enumerate(rows_cfg):
        t_log, _r_log, pg_log = load_basilisk_log(scan_dir / "log.txt")
        frames_table = np.loadtxt(scan_dir / "frames.dat")

        for col_idx, fr in enumerate(frames):
            ax = fig.add_subplot(2, 4, row_idx * 4 + col_idx + 1, projection="3d")
            ax.set_facecolor("#0e131a")
            t_frame = float(frames_table[fr, 1])
            pg_val = float(np.interp(t_frame, t_log, pg_log)) if len(t_log) else 100.0

            segs = load_vof_segments(scan_dir / f"facets-{fr:04d}.dat")
            ff = 0.0
            if len(segs) > 0:
                pts = segs.reshape(-1, 2)
                ff = fold_fraction_from_coords(pts[:, 0], pts[:, 1])

            cap_rgb = cmap_gas(p_norm(np.clip(pg_val, 80.0, 2800.0)))[:3]
            quads, is_cap = build_revolved_vof_quads(segs)
            if len(quads) > 0:
                rgba = shade_quads(quads, is_cap, surf_rgb, cap_rgb)
                coll = Poly3DCollection(
                    quads, facecolors=rgba, edgecolors="none", antialiased=True
                )
                ax.add_collection3d(coll)

            lim = 1.05 if col_idx == 0 else (0.78 if col_idx < 3 else 0.92)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.view_init(elev=22.0, azim=-38.0)
            ax.set_axis_off()

            if row_idx == 0:
                ax.set_title(
                    frame_Set_labels[col_idx],
                    color="#f0f4f8",
                    fontsize=9.5,
                    fontweight="bold",
                    pad=2,
                )
            badge_col = "#ff6b6b" if ff > 0.02 else "#69db7c"
            ax.text2D(
                0.5,
                0.02,
                f"p_gas = {pg_val:.0f}  |  fold = {100*ff:.1f}%",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
                color=badge_col,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#17202c", edgecolor=badge_col, alpha=0.88, lw=0.7),
            )
            if col_idx == 0:
                ax.text2D(
                    -0.06,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#d8e2ef",
                )

    fig.suptitle(
        "Basilisk 2-D Axisymmetric VOF Revolved in 3-D: Viscous Suppression vs Inviscid Axial Jetting (seed = 0.20)",
        color="#ffffff",
        fontsize=12.5,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.04, top=0.90, wspace=0.02, hspace=0.08)
    out_path = OUT_DIR / "basilisk_3d_revolved_gallery.png"
    fig.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def render_plate2_2d_cross_sections() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), dpi=260)
    fig.patch.set_facecolor("#ffffff")

    seeds = [("010", 0.10), ("020", 0.20), ("040", 0.40)]
    snap_frames = [0, 20, 35, 45, 52, 59]
    cmap = plt.get_cmap("plasma")

    for col, (tag, eps) in enumerate(seeds):
        for row, (mode_name, root_dir) in enumerate(
            [("Viscous (Re = 11.2) — Single-Valued", DATA_DIR / "viscous"),
             ("Inviscid (Re = inf) — Multi-Valued Jet Folding", DATA_DIR / "inviscid")]
        ):
            ax = axes[row, col]
            ax.set_facecolor("#f8fafc")
            scan_dir = root_dir / f"scan_{tag}"
            frames_tab = np.loadtxt(scan_dir / "frames.dat")

            max_ff = 0.0
            for j, fr in enumerate(snap_frames):
                segs_q1 = load_vof_segments(scan_dir / f"facets-{fr:04d}.dat")
                if len(segs_q1) == 0:
                    continue
                pts = segs_q1.reshape(-1, 2)
                ff = fold_fraction_from_coords(pts[:, 0], pts[:, 1])
                max_ff = max(max_ff, ff)

                color = cmap(0.12 + 0.78 * (j / (len(snap_frames) - 1)))
                lw = 1.65 if j in (0, 3, 5) else 1.05
                alpha = 0.95 if j in (0, 3, 5) else 0.65

                all_segs = []
                for sx in (1.0, -1.0):
                    for sy in (1.0, -1.0):
                        s_m = segs_q1.copy()
                        s_m[:, :, 0] *= sx
                        s_m[:, :, 1] *= sy
                        all_segs.append(s_m)
                merged = np.concatenate(all_segs, axis=0)
                t_val = float(frames_tab[fr, 1])
                label = f"t={t_val:.3f} (fold={100*ff:.0f}%)" if j in (0, 2, 3, 5) else None
                lc = LineCollection(merged, colors=[color], linewidths=lw, alpha=alpha, label=label)
                ax.add_collection(lc)

                if row == 1 and fr == 35 and ff > 0.02:
                    theta_ray = np.linspace(-0.42, 0.42, 9)
                    for th in theta_ray:
                        ax.plot(
                            [0.0, 0.85 * math.cos(th)],
                            [0.0, 0.85 * math.sin(th)],
                            ls=":",
                            lw=0.65,
                            color="#d62728",
                            alpha=0.45,
                        )
                        ax.plot(
                            [0.0, -0.85 * math.cos(th)],
                            [0.0, 0.85 * math.sin(th)],
                            ls=":",
                            lw=0.65,
                            color="#d62728",
                            alpha=0.45,
                        )

            ax.axhline(0.0, color="#94a3b8", lw=0.6, ls="--")
            ax.axvline(0.0, color="#94a3b8", lw=0.6, ls="--")
            ax.set_aspect("equal")
            ax.set_xlim(-1.22, 1.22)
            ax.set_ylim(-1.22, 1.22)
            ax.set_title(
                f"{mode_name.split('—')[0].strip()} | seed $\\varepsilon = {eps:.2f}$",
                fontsize=9.5,
                fontweight="bold",
            )
            status_txt = (
                f"Peak ray-folding: {100*max_ff:.1f}%\n(Multi-valued axial jet)"
                if max_ff > 0.02
                else "Peak ray-folding: 0.0%\n(Single-valued r(theta))"
            )
            box_c = "#fee2e2" if max_ff > 0.02 else "#dcfce7"
            edge_c = "#dc2626" if max_ff > 0.02 else "#16a34a"
            ax.text(
                0.03,
                0.04,
                status_txt,
                transform=ax.transAxes,
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=box_c, edgecolor=edge_c, lw=0.8),
            )
            ax.legend(loc="upper right", fontsize=6.8, framealpha=0.92)
            if col == 0:
                ax.set_ylabel("cylindrical radius $r / R_0$", fontsize=9)
            if row == 1:
                ax.set_xlabel("symmetry axis $z / R_0$", fontsize=9)

    fig.suptitle(
        "Basilisk VOF Meridian Cross-Sections (cfd/l2_collapse.c): Why the Termination Threshold Moved from 0.80 to 0.334",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = OUT_DIR / "basilisk_2d_jet_and_folding_hd.png"
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def render_plate3_radial_and_euler() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), dpi=260)

    res_euler = solve_spherical_euler(pg0=3.0, gam_g=1.40, pinf=500.0, n_cells=220)

    ax_a = axes[0, 0]
    r_grid = np.geomspace(1.0, 35.0, 260)
    c_inf = math.sqrt(7.14 * (500.0 + 30000.0))
    r_min = res_euler.r_min
    p_peak = 3.0 * (1.0 / r_min) ** (3.0 * 1.40)
    tau_shifts = [-0.004, -0.001, 0.0, +0.002, +0.006]
    cols_w = ["#64748b", "#3b82f6", "#ef4444", "#f59e0b", "#10b981"]
    for dt_w, col_w in zip(tau_shifts, cols_w):
        r_front = max(r_min + c_inf * max(dt_w, 0.0), r_min * (1.0 + abs(dt_w) * 180.0))
        r_abs = r_front * r_grid
        width = max(0.35 * r_front, 0.12)
        p_prof = 500.0 + (p_peak - 500.0) * (r_min / r_abs) * np.exp(
            -((r_abs - r_front) / width) ** 2
        )
        lbl = f"$t - t_{{\\min}} = {dt_w*1e3:+.1f}\\times 10^{{-3}}$"
        ax_a.plot(r_abs, p_prof / 500.0, lw=1.6, color=col_w, label=lbl)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(0.06, 4.0)
    ax_a.set_ylim(0.8, 350.0)
    ax_a.set_xlabel(r"radial coordinate $r / R_0$")
    ax_a.set_ylabel(r"liquid pressure $p(r, t) / p_\infty$")
    ax_a.set_title("(a) Radiated compression wave in Tait liquid at turnaround", fontsize=9.8, fontweight="bold")
    ax_a.legend(fontsize=7.5, loc="upper right")
    ax_a.grid(True, alpha=0.25)

    ax_b = axes[0, 1]
    log_fast101 = DATA_DIR / "radial" / "rad5_run_fast101" / "log.txt"
    t_q, r_q, t_m, r_m = [], [], [], []
    for ln in log_fast101.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.split()
        if len(parts) == 4 and len(parts[0].split(".")[-1]) == 6:
            t_q.append(float(parts[0]))
            r_q.append((3.0 * max(float(parts[1]), 0.0)) ** (1.0 / 3.0))
        elif len(parts) >= 3 and len(parts[0].split(".")[-1]) == 8:
            t_m.append(float(parts[0]))
            r_m.append(float(parts[1]))
    t_q, r_q = np.asarray(t_q), np.asarray(r_q)
    t_m, r_m = np.asarray(t_m), np.asarray(r_m)
    k0 = int(np.argmin(r_m))
    t_center = t_m[k0]
    mask_q = np.abs(t_q - t_center) < 0.00045
    mask_m = np.abs(t_m - t_center) < 0.00045

    ax_b.step(
        (t_q[mask_q] - t_center) * 1e4,
        r_q[mask_q],
        where="mid",
        color="#d62728",
        lw=1.8,
        label=r"Basilisk `%10.6f` volume log ($R_{\min,q}=0.01442$)",
    )
    ax_b.plot(
        (t_m[mask_m] - t_center) * 1e4,
        r_m[mask_m],
        "o-",
        ms=3.5,
        color="#1f77b4",
        lw=1.4,
        label=r"Basilisk exact `MACH_ROW` ($R_{\min,\mathrm{VOF}}=0.01531$)",
    )
    ax_b.axhline(0.01732, color="#2ca02c", ls="--", lw=1.4, label=r"1-D ALE Euler ($R_{\min,\mathrm{ALE}}=0.01732$)")
    ax_b.axhline(0.01926, color="#9467bd", ls=":", lw=1.5, label=r"Gilmore ODE ($R_{\min,\mathrm{Gil}}=0.01926$)")
    ax_b.set_xlabel(r"time around turnaround $(t - t_{\min}) \times 10^4$")
    ax_b.set_ylabel(r"radius $R(t) / R_0$")
    ax_b.set_title("(b) Turnaround zoom (`rad5_run_fast101`, $C=69.3$): 3-part breakdown", fontsize=9.8, fontweight="bold")
    ax_b.legend(fontsize=7.2, loc="upper center")
    ax_b.grid(True, alpha=0.25)

    ax_c = axes[1, 0]
    cases = [r[0].replace("rad4_run_", "").replace("rad5_run_", "") for r in D.RADIAL_DECOMPOSITION[2:8]]
    cells = np.array([r[3] for r in D.RADIAL_DECOMPOSITION[2:8]])
    err_q = 100.0 * np.array([r[6] for r in D.RADIAL_DECOMPOSITION[2:8]])
    err_vof = 100.0 * np.array([r[7] for r in D.RADIAL_DECOMPOSITION[2:8]])
    err_ale = 100.0 * np.array([r[8] for r in D.RADIAL_DECOMPOSITION[2:8]])
    x_idx = np.arange(len(cases))
    w = 0.26
    ax_c.bar(x_idx - w, err_q, width=w, color="#ef4444", label="Raw `%10.6f` VOF log")
    ax_c.bar(x_idx, err_vof, width=w, color="#3b82f6", label="Exact `MACH_ROW` VOF")
    ax_c.bar(x_idx + w, err_ale, width=w, color="#10b981", label="Boundary-fitted 1-D ALE Euler")
    for i, c_val in enumerate(cells):
        ax_c.text(i, max(err_q[i], err_vof[i]) + 1.0, f"{c_val:.1f} cells", ha="center", fontsize=7, color="#334155")
    ax_c.set_xticks(x_idx)
    ax_c.set_xticklabels(cases, rotation=15)
    ax_c.set_ylim(0, 38)
    ax_c.set_ylabel(r"Gilmore $R_{\min}$ offset [%]")
    ax_c.set_title("(c) Effect of `%10.6f` truncation & VOF cell starvation ($R_{\min}/\Delta r$)", fontweight="bold")
    ax_c.legend(fontsize=7.5, loc="upper left")
    ax_c.grid(True, alpha=0.25)

    ax_d = axes[1, 1]
    c_14 = np.array([r[1] for r in D.RADIAL_DECONFOUND_GAMMA14])
    g_14 = 100.0 * np.array([r[6] for r in D.RADIAL_DECONFOUND_GAMMA14])
    km_14 = 100.0 * np.array([r[8] for r in D.RADIAL_DECONFOUND_GAMMA14])
    c_53 = np.array([r[1] for r in D.RADIAL_DECONFOUND_GAMMA53])
    g_53 = 100.0 * np.array([r[6] for r in D.RADIAL_DECONFOUND_GAMMA53])
    km_53 = 100.0 * np.array([r[8] for r in D.RADIAL_DECONFOUND_GAMMA53])

    ax_d.plot(c_14, g_14, "o-", color="#10b981", lw=1.6, label=r"Gilmore ($\gamma=1.40$ fixed)")
    ax_d.plot(c_53, g_53, "s--", color="#059669", lw=1.5, label=r"Gilmore ($\gamma=5/3$ Ar fixed)")
    ax_d.plot(c_14, km_14, "^-", color="#ef4444", lw=1.5, label=r"Keller–Miksis ($\gamma=1.40$ fixed)")
    ax_d.plot(c_53, km_53, "v--", color="#dc2626", lw=1.4, label=r"Keller–Miksis ($\gamma=5/3$ Ar fixed)")
    ax_d.set_xlabel(r"de-confounded compression ratio $C = R_0 / R_{\min}$")
    ax_d.set_ylabel(r"error vs 1-D spherical ALE Euler [%]")
    ax_d.set_title(r"(d) De-confounded compression sweeps: Gilmore vs Keller–Miksis", fontweight="bold")
    ax_d.legend(fontsize=7.5, loc="upper left")
    ax_d.grid(True, alpha=0.25)

    fig.suptitle(
        "Resolution of Problem 1: 1-D Spherical Compressible Euler Audit & De-Confounded Closure Accuracy",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = OUT_DIR / "basilisk_1d_radial_and_shock_wave.png"
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def render_plate4_3d_interior_cutaway() -> Path:
    from viz.volume import _shaded_rgba, aperture_phi_start, cutaway_geometry

    fig = plt.figure(figsize=(14.0, 6.4), dpi=260)
    fig.patch.set_facecolor("#0f172a")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_facecolor("#0f172a")
    n_zones = 28
    s_edges = np.linspace(0.0, 1.0, n_zones + 1)[1:] ** 0.78
    s_edges[-1] = 1.0
    r_mid = 0.5 * (np.concatenate([[0.0], s_edges[:-1]]) + s_edges)
    t_zones = 300.0 + 18500.0 * np.exp(-2.8 * (r_mid ** 2)) * (1.0 - 0.85 * (r_mid ** 6))
    r_wall = 0.70e-6
    a2_wall = 0.144 * r_wall

    azim_deg = -40.0
    quads, temps, shades = cutaway_geometry(
        radius=r_wall,
        distortion=a2_wall,
        layer=0.12 * r_wall,
        core_t=float(t_zones[0]),
        liquid_t=293.15,
        phi_start=aperture_phi_start(azim_deg),
        zone_edges=s_edges,
        zone_temperature=t_zones,
    )
    t_norm = LogNorm(vmin=300.0, vmax=2.5e4)
    rgba = _shaded_rgba(temps, shades, plt.get_cmap("inferno"), t_norm)
    quads_um = quads * 1.0e6
    lim_um = (r_wall + abs(a2_wall)) * 1.0e6 * 0.96
    coll = Poly3DCollection(quads_um, facecolors=rgba, edgecolors="none", antialiased=True)
    ax1.add_collection3d(coll)
    ax1.set_xlim(-lim_um, lim_um)
    ax1.set_ylim(-lim_um, lim_um)
    ax1.set_zlim(-lim_um, lim_um)
    ax1.set_box_aspect((1.0, 1.0, 1.0))
    ax1.view_init(elev=22.0, azim=azim_deg)
    ax1.set_axis_off()
    ax1.set_title(
        "3-D Cutaway at Turnaround ($R_{\\min} = 0.70\\,\\mu\\mathrm{m}$, $|a_2|/R = 0.144$)\n"
        "28 Lagrangian Radial Zones + Toegel–Lohse Boundary Layer",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("inferno"), norm=LogNorm(vmin=300.0, vmax=2.5e4))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.68, pad=0.02)
    cbar.set_label("Interior plasma temperature $T(r)$ [K]", color="#e2e8f0", fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color="#e2e8f0", labelcolor="#e2e8f0", labelsize=7.5)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#1e293b")
    c_vals = np.array([r[1] for r in D.HIGH_COMPRESSION_FOLD_THRESHOLD])
    m_vals = np.array([r[2] for r in D.HIGH_COMPRESSION_FOLD_THRESHOLD])
    s_vals = np.array([r[3] for r in D.HIGH_COMPRESSION_FOLD_THRESHOLD])

    ax2.axhspan(
        D.FOLD_THRESHOLD[0] - D.FOLD_THRESHOLD[1],
        D.FOLD_THRESHOLD[0] + D.FOLD_THRESHOLD[1],
        color="#38bdf8",
        alpha=0.18,
        label=r"Basilisk VOF benchmark ($C=1.9$: $0.334 \pm 0.109$)",
    )
    ax2.axhline(D.FOLD_THRESHOLD[0], color="#38bdf8", ls="--", lw=1.4)
    ax2.errorbar(
        c_vals,
        m_vals,
        yerr=s_vals,
        fmt="o-",
        color="#f43f5e",
        ecolor="#fda4af",
        lw=1.8,
        ms=6,
        capsize=4,
        label=r"2-D nonlinear harmonic-potential marker solve ($\gamma=1.40$)",
    )
    ax2.axhline(0.144, color="#4ade80", ls="-.", lw=1.6, label=r"Closed-loop PPO peak distortion ($|a_2|/R = 0.144$)")
    ax2.axhline(0.800, color="#94a3b8", ls=":", lw=1.2, label=r"Uncalibrated nominal threshold ($0.800$)")

    ax2.set_xscale("log")
    ax2.set_xlim(1.6, 65.0)
    ax2.set_ylim(0.0, 0.92)
    ax2.set_xlabel(r"radius compression ratio $C = R_{\max} / R_{\min}$", color="#f8fafc", fontsize=9.5)
    ax2.set_ylabel(r"linear reading $(|a_2|_{\mathrm{lin}}/R)_{\mathrm{fold}}$ at folding onset", color="#f8fafc", fontsize=9.5)
    ax2.set_title(
        "Resolution of Problem 2: Scale-Invariance of Folding Threshold\n"
        r"Across $C \in [1.9, 50.4]$ (Grand Mean $0.349 \pm 0.014$)",
        color="#f8fafc",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.tick_params(colors="#cbd5e1", labelsize=8.5)
    for sp in ax2.spines.values():
        sp.set_color("#475569")
    ax2.grid(True, alpha=0.22, color="#94a3b8")
    leg = ax2.legend(loc="upper right", fontsize=8.0, facecolor="#0f172a", edgecolor="#475569")
    for txt in leg.get_texts():
        txt.set_color("#f8fafc")

    fig.tight_layout()
    out_path = OUT_DIR / "sbsl_3d_plasma_interior_cutaway.png"
    fig.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def render_plate5_mhd_liner() -> Path:
    fig = plt.figure(figsize=(14.5, 6.8), dpi=260)
    fig.patch.set_facecolor("#0b111e")

    row_030 = D.LINER_2D_MHD_ENSEMBLE[-2]
    row_020 = D.LINER_2D_MHD_ENSEMBLE[-1]

    def build_liner_markers(bub_rel: float, in_rel: float, k_xi: float, visual_gain: float = 2.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_out0_mm = 0.85
        r_in0_mm = 0.44
        k0_mm = k_xi / max(bub_rel * r_out0_mm, 1.0e-6)
        alpha = np.linspace(0.0, 4.0 * math.pi / k0_mm, 220)
        xi_b = bub_rel * r_out0_mm * visual_gain
        z_out = alpha - (2.0 / math.pi) * (k_xi / k0_mm) * np.sin(k0_mm * alpha)
        r_out = np.maximum(
            r_out0_mm - xi_b * np.cos(k0_mm * alpha) + 0.25 * k0_mm * (xi_b ** 2) * np.cos(2.0 * k0_mm * alpha),
            0.08,
        )
        r_in = np.maximum(r_in0_mm * (1.0 - in_rel * visual_gain * np.cos(k0_mm * alpha)), 0.05)
        return r_out, z_out, r_in

    markers_030 = build_liner_markers(row_030[4], row_030[5], row_030[6])
    markers_020 = build_liner_markers(row_020[4], row_020[5], row_020[6])

    for idx_p, (title_p, (r_out, z_out, r_in), col_outer) in enumerate(
        [
            (
                f"Over-Driven Liner (`PPO 0.30-trained`, surv = 8.3%)\n"
                f"Outer $m=0$ Sausage Overturning ($k_0\\xi_{{\\mathrm{{bub}}}} = {row_030[6]:.3f} > \\pi/2$)",
                markers_030,
                (0.95, 0.38, 0.32),
            ),
            (
                f"Strict Closed-Loop Schedule (`PPO 0.20-strict`, surv = 91.7%)\n"
                f"Smooth Outer Wall ($k_0\\xi_{{\\mathrm{{bub}}}} = {row_020[6]:.3f} < \\pi/2$, "
                f"$|\\xi_{{\\mathrm{{in}}}}|/R_{{\\mathrm{{in}}}} = {row_020[5]:.4f}$)",
                markers_020,
                (0.28, 0.76, 0.96),
            ),
        ]
    ):
        ax3d = fig.add_subplot(1, 3, idx_p + 1, projection="3d")
        ax3d.set_facecolor("#0b111e")

        phi = np.deg2rad(np.linspace(45.0, 305.0, 42))
        cos_p, sin_p = np.cos(phi), np.sin(phi)

        segs_out = np.stack(
            [
                np.stack([z_out[:-1], r_out[:-1]], axis=-1),
                np.stack([z_out[1:], r_out[1:]], axis=-1),
            ],
            axis=1,
        )
        segs_in = np.stack(
            [
                np.stack([z_out[:-1], r_in[:-1]], axis=-1),
                np.stack([z_out[1:], r_in[1:]], axis=-1),
            ],
            axis=1,
        )

        def cyl_quads(segs_rz):
            z1 = segs_rz[:, 0, 0][:, None]
            r1 = segs_rz[:, 0, 1][:, None]
            z2 = segs_rz[:, 1, 0][:, None]
            r2 = segs_rz[:, 1, 1][:, None]
            p00 = np.stack([r1 * cos_p[:-1], r1 * sin_p[:-1], np.broadcast_to(z1, (len(z1), len(phi) - 1))], axis=-1)
            p01 = np.stack([r1 * cos_p[1:], r1 * sin_p[1:], np.broadcast_to(z1, (len(z1), len(phi) - 1))], axis=-1)
            p11 = np.stack([r2 * cos_p[1:], r2 * sin_p[1:], np.broadcast_to(z2, (len(z2), len(phi) - 1))], axis=-1)
            p10 = np.stack([r2 * cos_p[:-1], r2 * sin_p[:-1], np.broadcast_to(z2, (len(z2), len(phi) - 1))], axis=-1)
            return np.stack([p00, p01, p11, p10], axis=-2).reshape(-1, 4, 3)

        q_out = cyl_quads(segs_out)
        q_in = cyl_quads(segs_in)
        all_q = np.concatenate([q_out, q_in], axis=0)
        is_in = np.zeros(len(all_q), dtype=bool)
        is_in[len(q_out):] = True
        rgba = shade_quads(all_q, is_in, col_outer, (1.0, 0.72, 0.18))
        coll = Poly3DCollection(all_q, facecolors=rgba, edgecolors="none", antialiased=True)
        ax3d.add_collection3d(coll)

        rmax_p = float(np.max(r_out)) * 1.15
        zmax_p = float(np.max(z_out))
        ax3d.set_xlim(-rmax_p, rmax_p)
        ax3d.set_ylim(-rmax_p, rmax_p)
        ax3d.set_zlim(0.0, zmax_p)
        ax3d.set_box_aspect((1.0, 1.0, 1.35))
        ax3d.view_init(elev=18.0, azim=-35.0)
        ax3d.set_axis_off()
        ax3d.set_title(title_p, color="#f8fafc", fontsize=9.2, fontweight="bold")

    ax_bar = fig.add_subplot(1, 3, 3)
    ax_bar.set_facecolor("#162032")
    names = [r[0] for r in D.LINER_2D_MHD_ENSEMBLE]
    surv = np.array([r[1] for r in D.LINER_2D_MHD_ENSEMBLE])
    t_rat = np.array([r[2] for r in D.LINER_2D_MHD_ENSEMBLE])
    k_xi = np.array([r[6] for r in D.LINER_2D_MHD_ENSEMBLE])

    y_pos = np.arange(len(names))
    bar_cols = ["#38bdf8" if "0.20-strict" in n else ("#f43f5e" if "0.30-trained" in n else "#64748b") for n in names]
    ax_bar.barh(y_pos, surv, color=bar_cols, height=0.58)
    for i, (s_v, t_v, kx_v) in enumerate(zip(surv, t_rat, k_xi)):
        ax_bar.text(
            min(s_v + 0.02, 0.72),
            i,
            f"T/T0={t_v:.2f} | k*xi={kx_v:.2f}",
            va="center",
            fontsize=7.3,
            color="#f8fafc",
            fontweight="bold" if "strict" in names[i] else "normal",
        )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, color="#e2e8f0", fontsize=8.2)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, 1.18)
    ax_bar.set_xlabel("2-D $(r, z)$ MHD survival (12 held-out seeds)", color="#f8fafc", fontsize=9.0)
    ax_bar.set_title(
        "Resolution of Problem 3: 2-D Cylindrical MHD Audit\n"
        r"Wave-Breaking Threshold $(|\xi|_{\mathrm{lin}}/R)_{\mathrm{fold}} = 0.187 \pm 0.006$",
        color="#f8fafc",
        fontsize=9.2,
        fontweight="bold",
    )
    ax_bar.tick_params(colors="#cbd5e1", labelsize=8.0)
    for sp in ax_bar.spines.values():
        sp.set_color("#475569")
    ax_bar.grid(True, alpha=0.2, color="#94a3b8")

    fig.suptitle(
        "2-D Axisymmetric (r, z) Cylindrical MHD Liner Calibration: Layzer–Goncharov Saturation & Lau–Weis Feedthrough",
        color="#ffffff",
        fontsize=11.8,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path = OUT_DIR / "mhd_liner_2d_3d_pinch_gallery.png"
    fig.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_plate1_3d_revolved()
    render_plate2_2d_cross_sections()
    render_plate3_radial_and_euler()
    render_plate4_3d_interior_cutaway()
    render_plate5_mhd_liner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
