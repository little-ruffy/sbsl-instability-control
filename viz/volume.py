from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core.config import SimConfig
from core.integrator import Trajectory
from viz.animate import (
    LINEAR_SHAPE_LIMIT,
    MACH_LIMIT,
    _configure_ffmpeg,
    slow_motion_factor,
    warped_frame_times,
)
from viz.projection import (
    T_CMAP,
    T_MAX_K,
    T_MIN_K,
    _boundary_layer,
    legendre_p2,
)

CUT_FRACTION: float = 0.61

LIGHT_DIRECTION: tuple[float, float, float] = (-0.45, -0.80, 0.40)

AMBIENT: float = 0.38

INTERIOR_SHADE: float = 0.82

INTERFACE_RGB: tuple[float, float, float] = (0.62, 0.72, 0.84)

APERTURE_OFFSET_DEG: float = 52.0

INTERFACE_ALPHA: float = 1.0


def _cell_normals(quads: np.ndarray) -> np.ndarray:
    d1 = quads[:, 2] - quads[:, 0]
    d2 = quads[:, 3] - quads[:, 1]
    n = np.cross(d1, d2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-300)


def _lambert(normals: np.ndarray) -> np.ndarray:
    light = np.asarray(LIGHT_DIRECTION, dtype=float)
    light = light / np.linalg.norm(light)
    lam = np.abs(normals @ light)
    return AMBIENT + (1.0 - AMBIENT) * lam


def shape_radius(
    cos_theta: np.ndarray, radius: float, distortion: float
) -> np.ndarray:
    return np.maximum(radius + distortion * legendre_p2(cos_theta), 0.0)


def profile_temperature(
    fraction: np.ndarray,
    zone_edges: np.ndarray,
    zone_temperature: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(zone_edges, np.clip(fraction, 0.0, 1.0), side="left")
    idx = np.clip(idx, 0, len(zone_temperature) - 1)
    return zone_temperature[idx]


def _layer_temperature(
    fraction: np.ndarray,
    radius: float,
    layer: float,
    core_t: float,
    liquid_t: float,
) -> np.ndarray:
    ell = min(max(layer, 0.0), 0.999 * radius)
    inner_fraction = (radius - ell) / max(radius, 1e-300)
    ramp = np.clip(
        (fraction - inner_fraction) / max(1.0 - inner_fraction, 1e-12), 0.0, 1.0
    )
    return core_t + (liquid_t - core_t) * ramp


def _radial_grid(
    radius: float, layer: float, n_core: int, n_layer: int
) -> np.ndarray:
    ell = min(max(layer, 0.0), 0.999 * radius)
    edge = (radius - ell) / max(radius, 1e-300)
    inner = np.linspace(0.0, edge, n_core + 1)
    outer = np.linspace(edge, 1.0, n_layer + 1)[1:]
    return np.concatenate([inner, outer])


def aperture_phi_start(
    azimuth_deg: float,
    cut_fraction: float = CUT_FRACTION,
    offset_deg: float = APERTURE_OFFSET_DEG,
) -> float:
    removed_centre = np.deg2rad(azimuth_deg + offset_deg)
    return removed_centre - np.pi * (1.0 + cut_fraction)


def cutaway_geometry(
    radius: float,
    distortion: float,
    layer: float,
    core_t: float,
    liquid_t: float,
    n_theta: int = 44,
    n_phi: int = 56,
    n_core: int = 10,
    n_layer: int = 18,
    cut_fraction: float = CUT_FRACTION,
    phi_start: float = 0.0,
    zone_edges: np.ndarray | None = None,
    zone_temperature: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    r_theta = shape_radius(cos_t, radius, distortion)

    phi_end = phi_start + 2.0 * np.pi * cut_fraction
    phi = np.linspace(phi_start, phi_end, n_phi + 1)

    rho = r_theta[:, None] * sin_t[:, None]
    x = rho * np.cos(phi)[None, :]
    y = rho * np.sin(phi)[None, :]
    z = np.repeat((r_theta * cos_t)[:, None], phi.size, axis=1)
    surf = np.stack([x, y, z], axis=-1)

    outer = np.stack(
        [surf[:-1, :-1], surf[:-1, 1:], surf[1:, 1:], surf[1:, :-1]], axis=2
    ).reshape(-1, 4, 3)
    outer_t = np.full(len(outer), np.nan)
    outer_shade = _lambert(_cell_normals(outer))

    resolved = zone_edges is not None and zone_temperature is not None
    if resolved:
        frac = np.concatenate([[0.0], np.asarray(zone_edges, dtype=float)])
    else:
        frac = _radial_grid(radius, layer, n_core, n_layer)
    n_rad = len(frac) - 1
    faces = [outer]
    temps = [outer_t]
    shades = [outer_shade]

    for phi_cut, sign in ((phi_start, -1.0), (phi_end, 1.0)):
        c_phi, s_phi = np.cos(phi_cut), np.sin(phi_cut)
        rr = frac[:, None] * (r_theta[None, :] * sin_t[None, :])
        zz = frac[:, None] * (r_theta[None, :] * cos_t[None, :])
        plane = np.stack([rr * c_phi, rr * s_phi, zz], axis=-1)
        quads = np.stack(
            [plane[:-1, :-1], plane[:-1, 1:], plane[1:, 1:], plane[1:, :-1]],
            axis=2,
        ).reshape(-1, 4, 3)
        mid_frac = np.repeat(
            0.5 * (frac[:-1] + frac[1:])[:, None], n_theta, axis=1
        ).ravel()
        assert len(mid_frac) == n_rad * n_theta
        if resolved:
            cut_t = profile_temperature(
                mid_frac,
                np.asarray(zone_edges, dtype=float),
                np.asarray(zone_temperature, dtype=float),
            )
        else:
            cut_t = _layer_temperature(
                mid_frac, radius, layer, core_t, liquid_t
            )
        normal = np.array([-s_phi * sign, c_phi * sign, 0.0])
        flat = float(_lambert(normal[None, :])[0]) * INTERIOR_SHADE
        faces.append(quads)
        temps.append(cut_t)
        shades.append(np.full(len(quads), flat))

    return (
        np.concatenate(faces, axis=0),
        np.concatenate(temps, axis=0),
        np.concatenate(shades, axis=0),
    )


def _seam_widths(temperature: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(temperature), 0.35, 0.0)


def _shaded_rgba(
    temperature: np.ndarray,
    shade: np.ndarray,
    cmap,
    norm,
) -> np.ndarray:
    is_interface = ~np.isfinite(temperature)
    filled = np.where(is_interface, T_MIN_K, temperature)
    rgba = cmap(norm(np.clip(filled, T_MIN_K, T_MAX_K)))
    rgba[is_interface, :3] = INTERFACE_RGB
    rgba[is_interface, 3] = INTERFACE_ALPHA
    rgba[:, :3] *= shade[:, None]
    return rgba


def animate_collapse_3d(
    cfg: SimConfig,
    trajectory: Trajectory,
    out_path: str | Path,
    n_frames: int = 180,
    fps: int = 24,
    tau: float = 2.0e-9,
    distortion_gain: float = 1.0,
    n_theta: int = 44,
    n_phi: int = 56,
    n_core: int = 10,
    n_layer: int = 18,
    spin_degrees: float = 0.0,
    elev_mid: float = 20.0,
    elev_sweep: float = 44.0,
    azim_view: float = -60.0,
    dpi: int = 120,
    bitrate: int = 6000,
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

    layer = np.array(
        [
            _boundary_layer(cfg, float(r[k]), float(rdot[k]),
                            float(temperature[k]))
            for k in picks
        ]
    )

    c_l = float(cfg.liquid.sound_speed)
    liquid_t = float(cfg.liquid.ambient_temperature)
    cmap = plt.get_cmap(T_CMAP)
    norm = matplotlib.colors.LogNorm(vmin=T_MIN_K, vmax=T_MAX_K if float(np.nanmax(temperature)) > 1.0e5 else max(float(np.nanmax(temperature)) * 1.15, 1.5e4))
    span_true = float(r.max()) * 1.08e6

    internal_mach = _internal_mach_series(cfg, r, rdot, temperature, picks)

    fig = plt.figure(figsize=(13.4, 7.6))
    grid = fig.add_gridspec(
        3, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.045, 0.40],
        hspace=0.42, wspace=0.02,
        left=0.012, right=0.988, top=0.930, bottom=0.020,
    )
    ax_true = fig.add_subplot(grid[0, 0], projection="3d")
    ax_zoom = fig.add_subplot(grid[0, 1], projection="3d")
    ax_cbar = fig.add_subplot(grid[1, :])
    ax_text = fig.add_subplot(grid[2, :])
    ax_text.axis("off")

    for ax in (ax_true, ax_zoom):
        ax.set_facecolor("#0b0b0b")
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.45)
        except TypeError:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except AttributeError:
            pass

    collections: dict[str, Poly3DCollection] = {}
    titles = {}
    for name, ax, label in (
        ("true", ax_true, f"true scale (fixed, ±{span_true:.1f} µm)"),
        ("zoom", ax_zoom, "auto-zoom"),
    ):
        coll = Poly3DCollection(
            np.zeros((1, 4, 3)), zsort="average", antialiased=False
        )
        ax.add_collection3d(coll)
        collections[name] = coll
        titles[name] = ax.set_title(
            label, fontsize=9.5, color="#444444", pad=6
        )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax_cbar, orientation="horizontal",
    )
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label(
        "gas temperature [K]  (log scale)", fontsize=8.5, labelpad=4
    )
    cbar.ax.tick_params(labelsize=7, pad=1)

    caption = ax_text.text(
        0.005, 1.0, "", transform=ax_text.transAxes, color="#111111",
        fontsize=8.8, va="top", family="monospace", linespacing=1.42,
    )
    legend = ax_text.text(
        0.235, 1.0, "", transform=ax_text.transAxes, color="#333333",
        fontsize=8.5, va="top", family="monospace", linespacing=1.42,
    )
    banner = ax_text.text(
        0.605, 1.0, "", transform=ax_text.transAxes, color="#BB2222",
        fontsize=8.5, va="top", family="monospace", linespacing=1.42,
    )
    legend.set_text(
        "grey shell = the interface. It sits at the liquid\n"
        "             temperature, the bottom of the colour\n"
        "             scale, so it is drawn neutral, not black.\n"
        "cut faces  = the interior, on the colour scale.\n"
        "wedge      = removed to expose it. The body is\n"
        "             axisymmetric, so which meridian is\n"
        "             cut carries no information."
    )

    def update(frame: int):
        k = int(picks[frame])
        radius = float(r[k])
        core_t = float(temperature[k])
        true_a2 = float(a_l[k])
        rel = abs(true_a2) / max(radius, 1e-300)
        drawn_a2 = true_a2 * distortion_gain
        mach = abs(float(rdot[k])) / c_l
        ell = float(layer[frame])

        phase = 2.0 * np.pi * frame / max(n_frames - 1, 1)
        elev = elev_mid + 0.5 * elev_sweep * np.sin(phase)
        azim = azim_view + spin_degrees * frame / max(n_frames - 1, 1)

        faces, face_t, shade = cutaway_geometry(
            radius, drawn_a2, ell, core_t, liquid_t,
            n_theta=n_theta, n_phi=n_phi, n_core=n_core, n_layer=n_layer,
            phi_start=aperture_phi_start(azim),
        )
        rgba = _shaded_rgba(face_t, shade, cmap, norm)
        width = _seam_widths(face_t)

        invalid = rel > LINEAR_SHAPE_LIMIT
        if invalid:
            rgba[:, 3] *= 0.55

        reach = float(np.abs(faces).max()) * 1.04e6
        for name, ax, half in (
            ("true", ax_true, span_true),
            ("zoom", ax_zoom, max(reach, 1e-3)),
        ):
            coll = collections[name]
            coll.set_verts(faces * 1e6)
            coll.set_facecolor(rgba)
            coll.set_edgecolor(rgba)
            coll.set_linewidth(width)
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_zlim(-half, half)
            ax.view_init(elev=elev, azim=azim)
        titles["zoom"].set_text(f"auto-zoom ×{span_true / max(reach, 1e-9):,.1f}")

        caption.set_text(
            f"t - t_min  = {(float(t[k]) - t_min) * 1e9:+10.3f} ns\n"
            f"R          = {radius * 1e6:10.4f} um\n"
            f"T_core     = {core_t:10.3e} K\n"
            f"|a2|/R     = {rel:10.3e}\n"
            f"layer l    = {ell * 1e6:10.4f} um  "
            f"({100.0 * ell / max(radius, 1e-300):.1f}% of R)\n"
            f"Rdot/c_l   = {float(rdot[k]) / c_l:+10.4f}\n"
            f"Rdot/c_gas = {internal_mach[frame]:10.4f}\n"
            f"slow-mo    = {stretch[frame]:10.2e}x"
        )

        notes = [
            "interior is ONE uniform zone by construction (REALISM.md R-1):",
            "  the only radial structure drawn is the conduction layer.",
        ]
        if internal_mach[frame] > 1.0:
            notes.append(
                f"Rdot/c_gas = {internal_mach[frame]:.2f} > 1: the wall "
                "outruns its own"
            )
            notes.append(
                "  pressure signal -- a shock forms that this model cannot see,"
            )
            notes.append("  so T_core here is a LOWER bound.")
        elif internal_mach[frame] > 0.3:
            notes.append(
                f"Rdot/c_gas = {internal_mach[frame]:.2f}: interior still "
                "subsonic, uniform core OK."
            )
        if invalid:
            notes.append(
                f"|a2|/R = {rel:.2f} > {LINEAR_SHAPE_LIMIT}: linear shape "
                "theory exceeded;"
            )
            notes.append("  the translucent surface is extrapolated.")
        if mach > MACH_LIMIT:
            notes.append(
                f"Rdot/c_l = {mach:.2f} > {MACH_LIMIT}: Keller-Miksis is "
                "first order in Mach."
            )
        banner.set_text("\n".join(notes))
        return ()

    fig.suptitle(
        f"{cfg.scenario} — 3-D cutaway on a warped clock "
        f"(asinh, τ = {tau * 1e9:g} ns)"
        + (f";  shape ×{distortion_gain:g}" if distortion_gain != 1.0 else ""),
        fontsize=12.0,
    )
    fig.patch.set_facecolor("#F2F2F2")

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=bitrate, codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p", "-preset", "slow",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        ],
    )
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def _internal_mach_series(
    cfg: SimConfig,
    radius: np.ndarray,
    radius_dot: np.ndarray,
    temperature: np.ndarray,
    picks: np.ndarray,
) -> np.ndarray:
    import jax.numpy as jnp

    from core import plasma_core as pc

    out = np.empty(len(picks))
    for i, k in enumerate(picks):
        v_free = pc.free_volume(jnp.asarray(float(radius[k])), cfg)
        temp = jnp.clip(
            jnp.asarray(float(temperature[k])), cfg.guards.t_min, cfg.guards.t_max
        )
        comp = pc.composition(temp, v_free, cfg)
        eos = pc.equation_of_state_and_derivatives(temp, v_free, cfg)
        out[i] = float(
            pc.internal_mach_number(
                jnp.asarray(float(radius_dot[k])), comp, eos, cfg
            )
        )
    return out


def plot_cutaway_still(
    cfg: SimConfig,
    radius: float,
    distortion: float,
    layer: float,
    core_t: float,
    out_path: str | Path,
    elev: float = 18.0,
    azim: float = -60.0,
    dpi: int = 160,
    zone_edges: np.ndarray | None = None,
    zone_temperature: np.ndarray | None = None,
) -> Path:
    faces, face_t, shade = cutaway_geometry(
        radius, distortion, layer, core_t,
        float(cfg.liquid.ambient_temperature),
        phi_start=aperture_phi_start(azim),
        zone_edges=zone_edges,
        zone_temperature=zone_temperature,
    )
    cmap = plt.get_cmap(T_CMAP)
    norm = matplotlib.colors.LogNorm(vmin=T_MIN_K, vmax=T_MAX_K if float(np.nanmax(face_t)) > 1.0e5 else max(float(np.nanmax(face_t)) * 1.15, 1.5e4))

    fig = plt.figure(figsize=(6.4, 6.0))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("#0b0b0b")
    ax.set_axis_off()
    coll = Poly3DCollection(faces * 1e6, zsort="average", antialiased=False)
    rgba = _shaded_rgba(face_t, shade, cmap, norm)
    coll.set_facecolor(rgba)
    coll.set_edgecolor(rgba)
    coll.set_linewidth(_seam_widths(face_t))
    ax.add_collection3d(coll)
    half = float(np.abs(faces).max()) * 1.05e6
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, facecolor="#F2F2F2")
    plt.close(fig)
    return out_path


def animate_collapse_3d_profile(
    cfg: SimConfig,
    times: np.ndarray,
    states: np.ndarray,
    n_zones: int,
    out_path: str | Path,
    n_frames: int = 160,
    fps: int = 24,
    dpi: int = 120,
    elev_mid: float = 20.0,
    elev_sweep: float = 44.0,
    azim_view: float = -60.0,
    bitrate: int = 6000,
    t_reference: float | None = None,
    single_zone: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Path:
    _configure_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times)
    states = np.asarray(states)
    radii = states[:, 0:n_zones]
    temps = states[:, 2 * n_zones : 3 * n_zones]
    wall = states[:, n_zones - 1]

    edges = radii / wall[:, None]
    inner = np.concatenate([np.zeros((len(times), 1)), radii], axis=1)
    volume = (4.0 / 3.0) * np.pi * (inner[:, 1:] ** 3 - inner[:, :-1] ** 3)
    mean_t = np.sum(temps * volume, axis=1) / np.sum(volume, axis=1)

    if t_reference is None:
        t_reference = float(times[int(np.argmin(wall))])

    picks = np.unique(
        np.linspace(0, len(times) - 1, n_frames).astype(int)
    )
    cmap = plt.get_cmap(T_CMAP)
    norm = matplotlib.colors.LogNorm(vmin=T_MIN_K, vmax=T_MAX_K if float(np.nanmax(temps)) > 1.0e5 else max(float(np.nanmax(temps)) * 1.15, 1.5e4))
    span_true = float(wall.max()) * 1.08e6
    liquid_t = float(cfg.liquid.ambient_temperature)

    fig = plt.figure(figsize=(13.4, 7.2))
    grid = fig.add_gridspec(
        3, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 0.05, 0.42],
        hspace=0.46, wspace=0.14,
        left=0.015, right=0.975, top=0.925, bottom=0.020,
    )
    ax_body = fig.add_subplot(grid[0, 0], projection="3d")
    ax_profile = fig.add_subplot(grid[0, 1])
    ax_cbar = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[2, :])
    ax_text.axis("off")

    ax_body.set_facecolor("#0b0b0b")
    ax_body.set_axis_off()
    try:
        ax_body.set_box_aspect((1.0, 1.0, 1.0), zoom=1.45)
    except TypeError:
        ax_body.set_box_aspect((1.0, 1.0, 1.0))

    coll = Poly3DCollection(
        np.zeros((1, 4, 3)), zsort="average", antialiased=False
    )
    ax_body.add_collection3d(coll)
    body_title = ax_body.set_title(
        "", fontsize=9.5, color="#444444", pad=6
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax_cbar, orientation="horizontal",
    )
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.set_label("gas temperature [K]  (log scale)", fontsize=8.5, labelpad=4)
    cbar.ax.tick_params(labelsize=7, pad=1)

    ax_profile.set_xlabel(r"$r / R$", fontsize=10)
    ax_profile.set_ylabel("T [K]", fontsize=10)
    ax_profile.set_yscale("log")
    ax_profile.set_xlim(0.0, 1.0)
    ax_profile.set_ylim(
        max(0.5 * float(temps.min()), T_MIN_K), 2.0 * float(temps.max())
    )
    ax_profile.grid(True, alpha=0.25, lw=0.5)
    ax_profile.tick_params(labelsize=8)
    (profile_line,) = ax_profile.step(
        [], [], where="pre", color="#B2182B", lw=1.6,
        label=f"resolved, {n_zones} zones",
    )
    (profile_pts,) = ax_profile.plot(
        [], [], "o", color="#B2182B", ms=2.6
    )
    mean_line = ax_profile.axhline(
        np.nan, color="#2166AC", ls="--", lw=1.4,
        label="resolved volume mean",
    )
    sz_line = None
    if single_zone is not None:
        sz_t, _sz_r, sz_T = (np.asarray(a) for a in single_zone)
        sz_line = ax_profile.axhline(
            np.nan, color="#444444", ls=":", lw=1.4,
            label="single-zone model",
        )
    ax_profile.legend(fontsize=8, loc="lower left", framealpha=0.9)

    caption = ax_text.text(
        0.005, 1.0, "", transform=ax_text.transAxes, color="#111111",
        fontsize=8.8, va="top", family="monospace", linespacing=1.42,
    )
    note = ax_text.text(
        0.33, 1.0, "", transform=ax_text.transAxes, color="#333333",
        fontsize=8.5, va="top", family="monospace", linespacing=1.42,
    )
    note.set_text(
        "Each cell of the cut face is one Lagrangian zone at its own\n"
        "temperature. Nothing is interpolated between zones: a\n"
        "finite-volume zone IS one temperature over its volume, and\n"
        "smoothing would hide how well the mesh resolves a shock.\n"
        "The dotted line is everything the single-zone model can say\n"
        "about the interior."
    )

    def update(frame: int):
        k = int(picks[frame])
        radius = float(wall[k])
        edge = edges[k]
        zone_t = temps[k]

        azim = azim_view
        phase = 2.0 * np.pi * frame / max(len(picks) - 1, 1)
        elev = elev_mid + 0.5 * elev_sweep * np.sin(phase)

        faces, face_t, shade = cutaway_geometry(
            radius, 0.0, 0.0, float(zone_t[0]), liquid_t,
            phi_start=aperture_phi_start(azim),
            zone_edges=edge, zone_temperature=zone_t,
        )
        rgba = _shaded_rgba(face_t, shade, cmap, norm)
        coll.set_verts(faces * 1e6)
        coll.set_facecolor(rgba)
        coll.set_edgecolor(rgba)
        coll.set_linewidth(_seam_widths(face_t))

        reach = float(np.abs(faces).max()) * 1.04e6
        half = max(reach, 1e-3)
        ax_body.set_xlim(-half, half)
        ax_body.set_ylim(-half, half)
        ax_body.set_zlim(-half, half)
        ax_body.view_init(elev=elev, azim=azim)
        body_title.set_text(
            f"resolved interior, {n_zones} zones  "
            f"(auto-zoom x{span_true / max(reach, 1e-9):,.1f})"
        )

        profile_line.set_data(edge, zone_t)
        profile_pts.set_data(edge, zone_t)
        mean_line.set_ydata([mean_t[k], mean_t[k]])
        hottest = int(np.argmax(zone_t))
        if sz_line is not None:
            j = int(np.argmin(np.abs(sz_t - times[k])))
            sz_line.set_ydata([sz_T[j], sz_T[j]])

        caption.set_text(
            f"t - t_min  = {(times[k] - t_reference) * 1e9:+10.4f} ns\n"
            f"R          = {radius * 1e6:10.5f} um\n"
            f"T centre   = {zone_t[0]:10.4e} K\n"
            f"T hottest  = {zone_t[hottest]:10.4e} K"
            f"  at r/R = {edge[hottest]:.3f}\n"
            f"T vol-mean = {mean_t[k]:10.4e} K\n"
            f"peak/mean  = "
            f"{zone_t[hottest] / max(mean_t[k], 1e-300):10.4f}"
        )
        return ()

    fig.suptitle(
        f"{cfg.scenario} - resolved interior, 3-D cutaway",
        fontsize=12.0,
    )
    fig.patch.set_facecolor("#F2F2F2")

    anim = animation.FuncAnimation(
        fig, update, frames=len(picks), interval=1000.0 / fps, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps, bitrate=bitrate, codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p", "-preset", "slow",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        ],
    )
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


__all__ = [
    "AMBIENT",
    "APERTURE_OFFSET_DEG",
    "aperture_phi_start",
    "INTERFACE_ALPHA",
    "INTERFACE_RGB",
    "CUT_FRACTION",
    "INTERIOR_SHADE",
    "LIGHT_DIRECTION",
    "animate_collapse_3d",
    "animate_collapse_3d_profile",
    "cutaway_geometry",
    "plot_cutaway_still",
    "profile_temperature",
    "shape_radius",
]
