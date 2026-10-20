from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from core.config import SimConfig
from core.integrator import Trajectory
from viz.projection import (
    T_CMAP,
    T_MAX_K,
    T_MIN_K,
    _boundary_layer,
    bubble_outline,
)

LINEAR_SHAPE_LIMIT: float = 0.3

MACH_LIMIT: float = 1.5


def _configure_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "MP4 export needs an ffmpeg binary. Install the bundled one with:\n"
            "    pip install imageio-ffmpeg"
        ) from exc
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["animation.ffmpeg_path"] = exe
    return exe


def warped_frame_times(
    times: np.ndarray,
    t_min: float,
    n_frames: int,
    tau: float = 2.0e-9,
) -> np.ndarray:
    lo, hi = float(times.min()), float(times.max())
    u_lo = np.arcsinh((lo - t_min) / tau)
    u_hi = np.arcsinh((hi - t_min) / tau)
    u = np.linspace(u_lo, u_hi, n_frames)
    return t_min + tau * np.sinh(u)


def slow_motion_factor(
    frame_times: np.ndarray, fps: float
) -> np.ndarray:
    dt_sim = np.gradient(frame_times)
    dt_sim = np.maximum(np.abs(dt_sim), 1e-30)
    return (1.0 / fps) / dt_sim


# 2D warped cross-section + phase portrait animation
def animate_collapse(
    cfg: SimConfig,
    trajectory: Trajectory,
    out_path: str | Path,
    n_frames: int = 240,
    fps: int = 24,
    tau: float = 2.0e-9,
    distortion_gain: float = 1.0,
    dpi: int = 120,
    bitrate: int = 4000,
) -> Path:
    _configure_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = np.asarray(trajectory.valid)
    t = np.asarray(trajectory.times)[valid]
    r = np.asarray(trajectory.radius)[valid]
    rdot = np.asarray(trajectory.radius_dot)[valid]
    temperature = np.asarray(trajectory.temperature)[valid]
    a_l = np.asarray(trajectory.distortion)[valid]

    order = np.argsort(t)
    t, r, rdot, temperature, a_l = (
        t[order], r[order], rdot[order], temperature[order], a_l[order]
    )
    t_min = float(t[int(np.argmin(r))])

    frame_times = warped_frame_times(t, t_min, n_frames, tau)
    picks = np.searchsorted(t, frame_times).clip(0, len(t) - 1)
    stretch = slow_motion_factor(frame_times, fps)

    c_l = float(cfg.liquid.sound_speed)
    r0 = float(cfg.r0)
    cmap = plt.get_cmap(T_CMAP)
    norm = matplotlib.colors.LogNorm(vmin=T_MIN_K, vmax=T_MAX_K if float(np.nanmax(temperature)) > 1.0e5 else max(float(np.nanmax(temperature)) * 1.15, 1.5e4))
    t_liquid = float(cfg.liquid.ambient_temperature)

    fig = plt.figure(figsize=(12.4, 6.6))
    grid = fig.add_gridspec(
        3, 3, width_ratios=[1.0, 1.0, 1.12], height_ratios=[1, 1, 1],
        hspace=0.62, wspace=0.26,
        left=0.025, right=0.975, top=0.885, bottom=0.10,
    )
    ax_true = fig.add_subplot(grid[0:2, 0])
    ax_zoom = fig.add_subplot(grid[0:2, 1])
    ax_radius = fig.add_subplot(grid[0, 2])
    ax_temp = fig.add_subplot(grid[1, 2])
    ax_phase = fig.add_subplot(grid[2, 2])
    ax_caption = fig.add_subplot(grid[2, 0:2])
    ax_caption.axis("off")

    span = float(r.max()) * 1.28e6
    for ax in (ax_true, ax_zoom):
        ax.set_aspect("equal")
        ax.set_facecolor("#0d0d0d")
        ax.set_xticks([])
        ax.set_yticks([])
    ax_true.set_xlim(-span, span)
    ax_true.set_ylim(-span, span)
    ax_true.set_title(
        f"true scale (fixed, +/-{span:.1f} $\\mu$m)", fontsize=8.5, color="#333333"
    )

    bar = 10.0 ** np.floor(np.log10(float(r.max()) * 1e6))
    ax_true.plot(
        [-0.93 * span, -0.93 * span + bar], [-0.90 * span, -0.90 * span],
        color="white", lw=2.5, solid_capstyle="butt",
    )
    ax_true.text(
        -0.93 * span + bar / 2.0, -0.845 * span, f"{bar:g} $\\mu$m",
        color="white", fontsize=7.5, ha="center",
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=[ax_true, ax_zoom], orientation="horizontal",
        fraction=0.055, pad=0.035,
    )
    cbar.set_label("core temperature [K]  (log scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    t_ns = (t - t_min) * 1e9
    for ax, series, label, logy in (
        (ax_radius, r * 1e6, r"R [$\mu$m]", True),
        (ax_temp, temperature, "T [K]", True),
    ):
        ax.plot(t_ns, series, color="#555555", lw=1.0)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.5)
        if logy:
            ax.set_yscale("log")
    ax_temp.set_xlabel("t - t$_{min}$ [ns]", fontsize=9)
    h = float(cfg.hardcore_radius)
    if h > 0.0:
        ax_radius.axhline(h * 1e6, color="#AA0000", ls="--", lw=0.9)

    ax_phase.plot(r / r0, rdot / c_l, color="#555555", lw=0.8)
    ax_phase.set_xscale("log")
    ax_phase.set_xlabel(r"$R/R_0$", fontsize=9)
    ax_phase.set_ylabel(r"$\dot R / c_l$", fontsize=9)
    ax_phase.tick_params(labelsize=7)
    ax_phase.grid(True, alpha=0.25, lw=0.5)
    ax_phase.axhline(0.0, color="#888888", ls=":", lw=0.7)

    layer = np.array(
        [
            _boundary_layer(cfg, float(r[k]), float(rdot[k]),
                            float(temperature[k]))
            for k in picks
        ]
    )

    artists = {}
    for name, ax in (("true", ax_true), ("zoom", ax_zoom)):
        artists[name] = {
            "layer": ax.fill([], [], color="#333333", lw=0)[0],
            "core": ax.fill([], [], color="#333333", lw=0)[0],
            "outline": ax.plot([], [], color="white", lw=1.1)[0],
        }
    zoom_label = ax_zoom.set_title("", fontsize=8.5, color="#333333")
    zoom_bar = ax_zoom.plot([], [], color="white", lw=2.5,
                            solid_capstyle="butt")[0]
    zoom_bar_text = ax_zoom.text(
        0.0, 0.0, "", color="white", fontsize=7.5, ha="center"
    )

    caption = ax_caption.text(
        0.0, 0.95, "", transform=ax_caption.transAxes, color="#111111",
        fontsize=9.0, va="top", family="monospace", linespacing=1.45,
    )
    banner = ax_caption.text(
        0.0, 0.10, "", transform=ax_caption.transAxes, color="#CC2222",
        fontsize=8.5, va="bottom", family="monospace", linespacing=1.35,
    )
    (radius_dot,) = ax_radius.plot([], [], "o", color="#0072B2", ms=5)
    (temp_dot,) = ax_temp.plot([], [], "o", color="#D55E00", ms=5)
    (phase_dot,) = ax_phase.plot([], [], "o", color="#009E73", ms=5)

    def update(frame: int):
        k = int(picks[frame])
        radius = float(r[k])
        core_t = float(temperature[k])
        drawn = float(a_l[k]) * distortion_gain
        rel = abs(float(a_l[k])) / radius
        mach = abs(float(rdot[k])) / c_l

        ell = min(float(layer[frame]), 0.9 * radius)
        inner = radius - ell
        bx, bz = bubble_outline(radius, drawn)
        ix, iz = bubble_outline(inner, drawn * inner / max(radius, 1e-30))
        layer_colour = cmap(
            norm(np.clip(np.sqrt(max(core_t, T_MIN_K) * t_liquid),
                         T_MIN_K, T_MAX_K))
        )
        core_colour = cmap(norm(np.clip(core_t, T_MIN_K, T_MAX_K)))

        invalid = rel > LINEAR_SHAPE_LIMIT
        alpha_fill = 0.30 if invalid else 1.0
        outline_style = (0, (4, 3)) if invalid else "solid"
        outline_colour = "#FF6666" if invalid else "white"

        zspan = max(float(np.abs(bx).max()), float(np.abs(bz).max())) * 1.28e6
        zspan = max(zspan, 1e-3)
        ax_zoom.set_xlim(-zspan, zspan)
        ax_zoom.set_ylim(-zspan, zspan)
        zoom_label.set_text(f"zoom x{span / zspan:,.1f}")

        zbar = 10.0 ** np.floor(np.log10(zspan))
        if zbar > 1.2 * zspan:
            zbar /= 10.0
        zoom_bar.set_data(
            [-0.90 * zspan, -0.90 * zspan + zbar], [-0.88 * zspan] * 2
        )
        zoom_bar_text.set_position((-0.90 * zspan + zbar / 2.0, -0.82 * zspan))
        zoom_bar_text.set_text(f"{zbar:g} $\\mu$m")

        for name in ("true", "zoom"):
            art = artists[name]
            art["layer"].set_xy(np.column_stack([bx * 1e6, bz * 1e6]))
            art["layer"].set_color(layer_colour)
            art["layer"].set_alpha(0.55 * alpha_fill)
            art["core"].set_xy(np.column_stack([ix * 1e6, iz * 1e6]))
            art["core"].set_color(core_colour)
            art["core"].set_alpha(alpha_fill)
            art["outline"].set_data(bx * 1e6, bz * 1e6)
            art["outline"].set_linestyle(outline_style)
            art["outline"].set_color(outline_colour)

        caption.set_text(
            f"t - t_min = {(float(t[k]) - t_min) * 1e9:+10.3f} ns      "
            f"R      = {radius * 1e6:9.4f} um\n"
            f"T         = {core_t:10.3e} K       "
            f"Rdot/c = {float(rdot[k]) / c_l:+9.4f}\n"
            f"|a2|/R    = {rel:10.3e}         "
            f"layer  = {ell * 1e6:9.4f} um\n"
            f"slow-mo   = {stretch[frame]:10.2e}x"
        )

        warnings = []
        if invalid:
            warnings.append(
                f"|a2|/R = {rel:.2f} > {LINEAR_SHAPE_LIMIT}: linear shape "
                "theory exceeded;"
            )
            warnings.append(
                "  the dashed outline is extrapolated, not predicted."
            )
        if mach > MACH_LIMIT:
            warnings.append(
                f"Mach = {mach:.2f} > {MACH_LIMIT}: Keller-Miksis is first "
                "order in Mach."
            )
        banner.set_text("\n".join(warnings))

        radius_dot.set_data([t_ns[k]], [radius * 1e6])
        temp_dot.set_data([t_ns[k]], [core_t])
        phase_dot.set_data([radius / r0], [float(rdot[k]) / c_l])
        return ()

    fig.suptitle(
        f"{cfg.scenario} -- collapse on a warped clock "
        f"(asinh, tau = {tau * 1e9:g} ns)"
        + (f";  shape x{distortion_gain:g}" if distortion_gain != 1.0 else ""),
        fontsize=11.5,
    )


    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=1000.0 / fps
    )
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=bitrate, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


__all__ = [
    "LINEAR_SHAPE_LIMIT",
    "MACH_LIMIT",
    "animate_collapse",
    "slow_motion_factor",
    "warped_frame_times",
]
