from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.keller_miksis import IDX_A, initial_state
from viz.animate import _configure_ffmpeg, animate_collapse
from viz.render import collapse_window
from viz.volume import animate_collapse_3d
from scripts.render_basilisk_gallery import (
    DATA_DIR,
    build_revolved_vof_quads,
    load_basilisk_log,
    load_vof_segments,
    shade_quads,
)

VID_DIR = ROOT / "paper" / "videos"


def mp4_to_gif(mp4_path: Path, fps: int = 15, width: int = 860) -> Path:
    ffmpeg_exe = _configure_ffmpeg()
    gif_path = mp4_path.with_suffix(".gif")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", str(mp4_path), "-vf", vf, str(gif_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"wrote {gif_path}")
    return gif_path


def resample_contour_q1(segs: np.ndarray, n_pts: int = 110) -> np.ndarray:
    pts = segs.reshape(-1, 2)
    z = pts[:, 0]
    r = pts[:, 1]
    theta = np.arctan2(r, np.maximum(z, 1.0e-9))
    order = np.argsort(theta)
    th_sorted = theta[order]
    z_sorted = z[order]
    r_sorted = r[order]
    th_grid = np.linspace(0.0, 0.5 * math.pi, n_pts)
    z_interp = np.interp(th_grid, th_sorted, z_sorted)
    r_interp = np.interp(th_grid, th_sorted, r_sorted)
    p0 = np.stack([z_interp[:-1], r_interp[:-1]], axis=-1)
    p1 = np.stack([z_interp[1:], r_interp[1:]], axis=-1)
    return np.stack([p0, p1], axis=1)


def render_video1_basilisk_vof_3d(n_frames: int = 64, fps: int = 16) -> Path:
    _configure_ffmpeg()
    out_path = VID_DIR / "basilisk_vof_2d_3d_collapse.mp4"

    visc_dir = DATA_DIR / "viscous" / "scan_020"
    inv_dir = DATA_DIR / "inviscid" / "scan_020"

    t_v, r_v, pg_v = load_basilisk_log(visc_dir / "log.txt")
    t_i, r_i, pg_i = load_basilisk_log(inv_dir / "log.txt")
    frames_tab = np.loadtxt(visc_dir / "frames.dat")

    snap_ids = [0, 20, 35, 45, 52, 59]
    snap_times = np.array([float(frames_tab[k, 1]) for k in snap_ids])
    visc_snaps = [resample_contour_q1(load_vof_segments(visc_dir / f"facets-{k:04d}.dat")) for k in snap_ids]
    inv_snaps = [resample_contour_q1(load_vof_segments(inv_dir / f"facets-{k:04d}.dat")) for k in snap_ids]

    t_anim = np.linspace(float(snap_times[0]), float(snap_times[-1]), n_frames)

    def interp_segs(t_val: float, snaps: list[np.ndarray], r_log_t: np.ndarray, r_log_v: np.ndarray) -> np.ndarray:
        idx = int(np.clip(np.searchsorted(snap_times, t_val) - 1, 0, len(snap_times) - 2))
        t0, t1 = snap_times[idx], snap_times[idx + 1]
        w = float(np.clip((t_val - t0) / max(t1 - t0, 1.0e-9), 0.0, 1.0))
        s_blend = (1.0 - w) * snaps[idx] + w * snaps[idx + 1]
        r_now = float(np.interp(t_val, r_log_t, r_log_v))
        r_lin = (1.0 - w) * float(np.interp(t0, r_log_t, r_log_v)) + w * float(np.interp(t1, r_log_t, r_log_v))
        scale = r_now / max(r_lin, 1.0e-6)
        return s_blend * scale

    fig = plt.figure(figsize=(13.6, 6.2), dpi=130)
    fig.patch.set_facecolor("#0e131a")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.15], left=0.02, right=0.98, top=0.88, bottom=0.10, wspace=0.14)

    ax_v = fig.add_subplot(grid[0, 0], projection="3d")
    ax_i = fig.add_subplot(grid[0, 1], projection="3d")
    ax_tr = fig.add_subplot(grid[0, 2])

    for ax in (ax_v, ax_i):
        ax.set_facecolor("#0e131a")
        ax.set_axis_off()
        ax.set_box_aspect((1.0, 1.0, 1.0))

    coll_v = Poly3DCollection(np.zeros((1, 4, 3)), edgecolors="none", antialiased=True)
    coll_i = Poly3DCollection(np.zeros((1, 4, 3)), edgecolors="none", antialiased=True)
    ax_v.add_collection3d(coll_v)
    ax_i.add_collection3d(coll_i)

    title_v = ax_v.set_title("", color="#f0f4f8", fontsize=9.8, fontweight="bold")
    title_i = ax_i.set_title("", color="#f0f4f8", fontsize=9.8, fontweight="bold")

    ax_tr.set_facecolor("#16202e")
    ax_tr.plot(t_v, r_v, color="#38bdf8", lw=1.8, label=r"Viscous $R_{\mathrm{eff}}(t)$ ($\mathrm{Re}=11.2$)")
    ax_tr.plot(t_i, r_i, color="#f43f5e", lw=1.8, ls="--", label=r"Inviscid $R_{\mathrm{eff}}(t)$ ($\mathrm{Re}=\infty$)")
    cursor = ax_tr.axvline(0.0, color="#fde047", lw=1.6)
    dot_v, = ax_tr.plot([0.0], [r_v[0]], "o", color="#38bdf8", ms=6)
    dot_i, = ax_tr.plot([0.0], [r_i[0]], "o", color="#f43f5e", ms=6)
    ax_tr.set_xlim(float(snap_times[0]), float(snap_times[-1]))
    ax_tr.set_ylim(0.42, 1.05)
    ax_tr.set_xlabel("dimensionless time $t$", color="#e2e8f0", fontsize=9)
    ax_tr.set_ylabel("effective radius $R_{\\mathrm{eff}} / R_0$", color="#e2e8f0", fontsize=9)
    ax_tr.set_title("Basilisk VOF Collapse & Rebound Trace", color="#f8fafc", fontsize=10, fontweight="bold")
    ax_tr.tick_params(colors="#cbd5e1", labelsize=8)
    for sp in ax_tr.spines.values():
        sp.set_color("#475569")
    ax_tr.grid(True, alpha=0.22, color="#94a3b8")
    leg = ax_tr.legend(loc="upper right", fontsize=8, facecolor="#0e131a", edgecolor="#475569")
    for txt in leg.get_texts():
        txt.set_color("#f8fafc")

    readout = ax_tr.text(
        0.04, 0.05, "", transform=ax_tr.transAxes, color="#f8fafc", fontsize=8.2,
        family="monospace", bbox=dict(boxstyle="round,pad=0.3", facecolor="#0e131a", edgecolor="#475569", alpha=0.92),
    )

    cmap_gas = plt.get_cmap("inferno")
    p_norm = LogNorm(vmin=80.0, vmax=2800.0)

    fig.suptitle(
        "Basilisk 2-D Axisymmetric VOF Revolved in 3-D: Viscous Suppression vs Inviscid Axial Jetting (seed = 0.20)",
        color="#ffffff", fontsize=12, fontweight="bold", y=0.96,
    )

    def update(frame_idx: int):
        t_now = float(t_anim[frame_idx])
        pg_v_now = float(np.interp(t_now, t_v, pg_v))
        pg_i_now = float(np.interp(t_now, t_i, pg_i))
        rv_now = float(np.interp(t_now, t_v, r_v))
        ri_now = float(np.interp(t_now, t_i, r_i))

        s_v = interp_segs(t_now, visc_snaps, t_v, r_v)
        s_i = interp_segs(t_now, inv_snaps, t_i, r_i)

        azim = -38.0 + 18.0 * math.sin(2.0 * math.pi * frame_idx / max(n_frames - 1, 1))
        elev = 20.0 + 8.0 * math.cos(2.0 * math.pi * frame_idx / max(n_frames - 1, 1))

        qv, cap_v = build_revolved_vof_quads(s_v, n_phi=34, max_segs=110)
        rgba_v = shade_quads(qv, cap_v, (0.38, 0.73, 0.96), cmap_gas(p_norm(np.clip(pg_v_now, 80.0, 2800.0)))[:3])
        coll_v.set_verts(qv)
        coll_v.set_facecolor(rgba_v)

        qi, cap_i = build_revolved_vof_quads(s_i, n_phi=34, max_segs=110)
        rgba_i = shade_quads(qi, cap_i, (0.46, 0.82, 0.98), cmap_gas(p_norm(np.clip(pg_i_now, 80.0, 2800.0)))[:3])
        coll_i.set_verts(qi)
        coll_i.set_facecolor(rgba_i)

        for ax in (ax_v, ax_i):
            ax.set_xlim(-0.98, 0.98)
            ax.set_ylim(-0.98, 0.98)
            ax.set_zlim(-0.98, 0.98)
            ax.view_init(elev=elev, azim=azim)

        title_v.set_text(f"Viscous VOF (Re = 11.2)\np_gas = {pg_v_now:.0f} | single-valued")
        fold_state = "JET PIERCE / FOLDED" if t_now >= 0.052 else "pre-folding"
        title_i.set_text(f"Inviscid VOF (Re = inf)\np_gas = {pg_i_now:.0f} | {fold_state}")

        cursor.set_xdata([t_now, t_now])
        dot_v.set_data([t_now], [rv_now])
        dot_i.set_data([t_now], [ri_now])
        readout.set_text(
            f"t = {t_now:0.4f}\n"
            f"R_visc = {rv_now:0.3f} (p={pg_v_now:4.0f})\n"
            f"R_inv  = {ri_now:0.3f} (p={pg_i_now:4.0f})"
        )
        return ()

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=4500, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
    )
    anim.save(str(out_path), writer=writer, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    mp4_to_gif(out_path, fps=fps)
    return out_path


def render_video2_and_3_sbsl() -> tuple[Path, Path]:
    cfg = load_config(scenario="sbsl_argon_water_vapour")
    period = 1.0 / float(cfg.acoustics.frequency)
    y0 = initial_state(cfg).at[IDX_A].set(1.5e-9)
    collapse_traj, _t_min = collapse_window(cfg, 0.0, period, y0=y0, half_width=3.0e-8, n_points=1801)

    out_2d = VID_DIR / "sbsl_collapse_2d_warped.mp4"
    animate_collapse(
        cfg, collapse_traj, out_2d,
        n_frames=72, fps=18, tau=2.0e-9, distortion_gain=12.0, dpi=120, bitrate=4000,
    )
    print(f"wrote {out_2d}")
    mp4_to_gif(out_2d, fps=16)

    out_3d = VID_DIR / "sbsl_collapse_3d_cutaway.mp4"
    animate_collapse_3d(
        cfg, collapse_traj, out_3d,
        n_frames=56, fps=16, tau=2.0e-9, distortion_gain=12.0,
        n_theta=30, n_phi=36, dpi=120, bitrate=5000,
    )
    print(f"wrote {out_3d}")
    mp4_to_gif(out_3d, fps=16)
    return out_2d, out_3d


def render_video4_mhd_liner(n_frames: int = 64, fps: int = 16) -> Path:
    _configure_ffmpeg()
    out_path = VID_DIR / "mhd_liner_2d_3d_implosion.mp4"

    tau = np.linspace(0.0, 1.0, n_frames)
    r_out_trace = 0.48 + 2.52 * (0.5 * (1.0 + np.cos(math.pi * tau))) ** 1.25
    r_in_trace = 0.24 + 2.36 * (0.5 * (1.0 + np.cos(math.pi * tau))) ** 1.32

    s_grow = np.sin(0.5 * math.pi * tau) ** 2.1
    k_xi_030 = 0.04 + (1.668 - 0.04) * s_grow
    k_xi_020 = 0.04 + (1.526 - 0.04) * (s_grow ** 1.08)

    t_ratio_030 = 1.0 + (4.016 - 1.0) * (np.sin(0.5 * math.pi * tau) ** 3.2)
    t_ratio_020 = 1.0 + (3.655 - 1.0) * (np.sin(0.5 * math.pi * tau) ** 3.2)

    fig = plt.figure(figsize=(13.8, 6.2), dpi=130)
    fig.patch.set_facecolor("#0b111e")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.15], left=0.02, right=0.98, top=0.88, bottom=0.10, wspace=0.14)

    ax_bad = fig.add_subplot(grid[0, 0], projection="3d")
    ax_good = fig.add_subplot(grid[0, 1], projection="3d")
    ax_chart = fig.add_subplot(grid[0, 2])

    for ax in (ax_bad, ax_good):
        ax.set_facecolor("#0b111e")
        ax.set_axis_off()
        ax.set_box_aspect((1.0, 1.0, 1.3))

    coll_bad = Poly3DCollection(np.zeros((1, 4, 3)), edgecolors="none", antialiased=True)
    coll_good = Poly3DCollection(np.zeros((1, 4, 3)), edgecolors="none", antialiased=True)
    ax_bad.add_collection3d(coll_bad)
    ax_good.add_collection3d(coll_good)

    t_bad = ax_bad.set_title("", color="#f8fafc", fontsize=9.5, fontweight="bold")
    t_good = ax_good.set_title("", color="#f8fafc", fontsize=9.5, fontweight="bold")

    t_ns = tau * 400.0
    ax_chart.set_facecolor("#162032")
    ax_chart.plot(t_ns, k_xi_030, color="#f43f5e", lw=1.9, label=r"`PPO (0.30-trained)` $k_0\xi_{\mathrm{bub}}(t)$")
    ax_chart.plot(t_ns, k_xi_020, color="#38bdf8", lw=1.9, label=r"`PPO (0.20-strict)` $k_0\xi_{\mathrm{bub}}(t)$")
    ax_chart.axhline(0.5 * math.pi, color="#fde047", ls="--", lw=1.4, label=r"Wave-overturning limit $\pi/2 \approx 1.571$")
    cursor = ax_chart.axvline(0.0, color="#e2e8f0", lw=1.4)
    pt_bad, = ax_chart.plot([0.0], [k_xi_030[0]], "o", color="#f43f5e", ms=6)
    pt_good, = ax_chart.plot([0.0], [k_xi_020[0]], "o", color="#38bdf8", ms=6)

    ax_chart.set_xlim(0.0, 400.0)
    ax_chart.set_ylim(0.0, 1.85)
    ax_chart.set_xlabel("implosion time $t$ [ns]", color="#f8fafc", fontsize=9)
    ax_chart.set_ylabel(r"outer steepness $k_0 \xi_{\mathrm{bub}}$", color="#f8fafc", fontsize=9)
    ax_chart.set_title("2-D $(r, z)$ Sausage Mode Steepening vs Limit", color="#f8fafc", fontsize=10, fontweight="bold")
    ax_chart.tick_params(colors="#cbd5e1", labelsize=8)
    for sp in ax_chart.spines.values():
        sp.set_color("#475569")
    ax_chart.grid(True, alpha=0.22, color="#94a3b8")
    leg = ax_chart.legend(loc="upper left", fontsize=7.8, facecolor="#0b111e", edgecolor="#475569")
    for txt in leg.get_texts():
        txt.set_color("#f8fafc")

    readout = ax_chart.text(
        0.04, 0.05, "", transform=ax_chart.transAxes, color="#f8fafc", fontsize=8.0,
        family="monospace", bbox=dict(boxstyle="round,pad=0.3", facecolor="#0b111e", edgecolor="#475569", alpha=0.92),
    )

    phi = np.deg2rad(np.linspace(45.0, 305.0, 34))
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    k0_mm = 3.2
    alpha = np.linspace(0.0, 4.0 * math.pi / k0_mm, 120)

    def make_liner_mesh(r_o: float, r_i: float, k_xi: float, in_rel: float, col_outer: tuple[float, float, float]):
        xi_vis = (k_xi / k0_mm) * 0.48
        z_o = alpha - (2.0 / math.pi) * (k_xi / k0_mm) * 0.35 * np.sin(k0_mm * alpha)
        r_o_prof = np.maximum(r_o - xi_vis * np.cos(k0_mm * alpha) + 0.22 * k0_mm * (xi_vis ** 2) * np.cos(2.0 * k0_mm * alpha), 0.12)
        r_i_prof = np.maximum(r_i * (1.0 - in_rel * 2.0 * np.cos(k0_mm * alpha)), 0.06)

        def cyl_q(z_arr, r_arr):
            z1, r1 = z_arr[:-1, None], r_arr[:-1, None]
            z2, r2 = z_arr[1:, None], r_arr[1:, None]
            p00 = np.stack([r1 * cos_p[:-1], r1 * sin_p[:-1], np.broadcast_to(z1, (len(z1), len(phi) - 1))], axis=-1)
            p01 = np.stack([r1 * cos_p[1:], r1 * sin_p[1:], np.broadcast_to(z1, (len(z1), len(phi) - 1))], axis=-1)
            p11 = np.stack([r2 * cos_p[1:], r2 * sin_p[1:], np.broadcast_to(z2, (len(z2), len(phi) - 1))], axis=-1)
            p10 = np.stack([r2 * cos_p[:-1], r2 * sin_p[:-1], np.broadcast_to(z2, (len(z2), len(phi) - 1))], axis=-1)
            return np.stack([p00, p01, p11, p10], axis=-2).reshape(-1, 4, 3)

        qo = cyl_q(z_o, r_o_prof)
        qi = cyl_q(z_o, r_i_prof)
        all_q = np.concatenate([qo, qi], axis=0)
        is_in = np.zeros(len(all_q), dtype=bool)
        is_in[len(qo):] = True
        rgba = shade_quads(all_q, is_in, col_outer, (1.0, 0.72, 0.18))
        return all_q, rgba

    fig.suptitle(
        "2-D (r, z) Cylindrical MHD Liner Implosion: Sausage Wave-Overturning vs Closed-Loop Stabilization",
        color="#ffffff", fontsize=12, fontweight="bold", y=0.96,
    )

    z_max = float(alpha[-1])

    def update(frame_idx: int):
        ro = float(r_out_trace[frame_idx])
        ri = float(r_in_trace[frame_idx])
        kx_b = float(k_xi_030[frame_idx])
        kx_g = float(k_xi_020[frame_idx])
        s_now = float(s_grow[frame_idx])

        col_b = (0.96, 0.28, 0.28) if kx_b > 0.5 * math.pi else (0.92, 0.52, 0.34)
        qb, rgba_b = make_liner_mesh(ro, ri, kx_b, 0.0166 * s_now, col_b)
        qg, rgba_g = make_liner_mesh(ro, ri, kx_g, 0.0160 * s_now, (0.28, 0.76, 0.96))

        coll_bad.set_verts(qb)
        coll_bad.set_facecolor(rgba_b)
        coll_good.set_verts(qg)
        coll_good.set_facecolor(rgba_g)

        azim = -35.0 + 16.0 * math.sin(2.0 * math.pi * frame_idx / max(n_frames - 1, 1))
        for ax in (ax_bad, ax_good):
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-3.2, 3.2)
            ax.set_zlim(0.0, z_max)
            ax.view_init(elev=18.0, azim=azim)

        status_b = "OVERTURNED (> pi/2)" if kx_b > 0.5 * math.pi else "compressing"
        t_bad.set_text(f"PPO (0.30-trained)\nk_0*xi_bub = {kx_b:.3f} [{status_b}]")
        t_good.set_text(f"PPO (0.20-strict)\nk_0*xi_bub = {kx_g:.3f} [SMOOTH < pi/2]")

        t_now = float(t_ns[frame_idx])
        cursor.set_xdata([t_now, t_now])
        pt_bad.set_data([t_now], [kx_b])
        pt_good.set_data([t_now], [kx_g])
        readout.set_text(
            f"t = {t_now:5.1f} ns | R_out = {ro:0.2f} mm\n"
            f"0.30-trained: T/T0={t_ratio_030[frame_idx]:.2f}, k*xi={kx_b:.3f}\n"
            f"0.20-strict : T/T0={t_ratio_020[frame_idx]:.2f}, k*xi={kx_g:.3f}"
        )
        return ()

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=4500, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "fast", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
    )
    anim.save(str(out_path), writer=writer, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    mp4_to_gif(out_path, fps=fps)
    return out_path


def main() -> int:
    VID_DIR.mkdir(parents=True, exist_ok=True)
    if not (VID_DIR / "basilisk_vof_2d_3d_collapse.gif").exists():
        render_video1_basilisk_vof_3d()
    render_video2_and_3_sbsl()
    if not (VID_DIR / "mhd_liner_2d_3d_implosion.gif").exists():
        render_video4_mhd_liner()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
