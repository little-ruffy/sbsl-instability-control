from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from core import constants as C
from core import plasma_core as pc
from core.config import SimConfig
from core.integrator import Trajectory

T_MIN_K: float = 300.0
T_MAX_K: float = 1.0e7

T_CMAP: str = "inferno"

_NORM = LogNorm(vmin=T_MIN_K, vmax=T_MAX_K)


def legendre_p2(cos_theta: np.ndarray) -> np.ndarray:
    return 0.5 * (3.0 * cos_theta**2 - 1.0)


def bubble_outline(
    radius: float, distortion: float, n_points: int = 721
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    r = radius + distortion * legendre_p2(np.cos(theta))
    r = np.maximum(r, 0.0)
    return r * np.sin(theta), r * np.cos(theta)


def _boundary_layer(
    cfg: SimConfig, radius: float, radius_dot: float, temperature: float
) -> float:
    v_free = pc.free_volume(jnp.asarray(radius), cfg)
    comp = pc.composition(
        jnp.clip(
            jnp.asarray(temperature), cfg.guards.t_min, cfg.guards.t_max
        ),
        v_free,
        cfg,
    )
    return float(
        pc.boundary_layer_thickness(
            jnp.asarray(radius), jnp.asarray(radius_dot), comp, cfg
        )
    )


# 2D axisymmetric cross-section with boundary-layer annulus
def plot_bubble_projection(
    cfg: SimConfig,
    trajectory: Trajectory,
    n_frames: int = 6,
    times: Sequence[float] | None = None,
    t_reference: float | None = None,
    equal_radius_scale: bool = False,
    distortion_gain: float = 1.0,
) -> Figure:
    valid = np.asarray(trajectory.valid)
    t = np.asarray(trajectory.times)[valid]
    r = np.asarray(trajectory.radius)[valid]
    rdot = np.asarray(trajectory.radius_dot)[valid]
    temperature = np.asarray(trajectory.temperature)[valid]
    a_l = np.asarray(trajectory.distortion)[valid]

    k_min = int(np.argmin(r))
    t_min = float(t[k_min]) if t_reference is None else float(t_reference)

    if times is None:
        before = max((n_frames - 1) // 2, 1)
        after = max(n_frames - 1 - before, 1)
        offsets = np.concatenate(
            [
                -np.geomspace(3e-8, 1e-9, before),
                [0.0],
                np.geomspace(1e-9, 3e-8, after),
            ]
        )
        picks = [
            int(np.argmin(np.abs(t - (t_min + off)))) for off in offsets
        ]
    else:
        picks = [int(np.argmin(np.abs(t - float(ts)))) for ts in times]

    n = len(picks)
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.3 * cols, 2.7 * rows), squeeze=False
    )
    cmap = plt.get_cmap(T_CMAP)
    biggest = max(float(r[k]) for k in picks)
    t_liquid = float(cfg.liquid.ambient_temperature)
    bar = 10.0 ** np.floor(np.log10(biggest * 1e6))

    for panel, k in enumerate(picks):
        ax = axes[panel // cols][panel % cols]
        radius = float(r[k])
        distortion = float(a_l[k])
        core_t = float(temperature[k])
        ell = _boundary_layer(cfg, radius, float(rdot[k]), core_t)
        ell = min(ell, 0.95 * radius)

        t_layer = math.sqrt(max(core_t, T_MIN_K) * t_liquid)
        drawn = distortion * distortion_gain
        bx, bz = bubble_outline(radius, drawn)
        ax.fill(
            bx * 1e6,
            bz * 1e6,
            color=cmap(_NORM(np.clip(t_layer, T_MIN_K, T_MAX_K))),
            alpha=0.6,
            linewidth=0,
        )
        inner = radius - ell
        ix, iz = bubble_outline(inner, drawn * inner / max(radius, 1e-30))
        ax.fill(
            ix * 1e6,
            iz * 1e6,
            color=cmap(_NORM(np.clip(core_t, T_MIN_K, T_MAX_K))),
            linewidth=0,
        )
        ax.plot(bx * 1e6, bz * 1e6, color="black", lw=0.8)

        scale = (biggest if equal_radius_scale else radius) * 1.45e6
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.plot(
            [-0.92 * scale, -0.92 * scale + bar],
            [-0.88 * scale, -0.88 * scale],
            color="white",
            lw=2.2,
            solid_capstyle="butt",
        )
        ax.text(
            -0.92 * scale + bar / 2.0,
            -0.80 * scale,
            f"{bar:g} $\\mu$m",
            color="white",
            fontsize=6.5,
            ha="center",
        )
        ax.set_facecolor("#111111")
        ax.set_title(
            f"t - t$_{{min}}$ = {(float(t[k]) - t_min) * 1e9:+.2f} ns\n"
            f"R = {radius * 1e6:.3f} $\\mu$m,  T = {core_t:.3g} K\n"
            f"|a$_2$|/R = {abs(distortion) / max(radius, 1e-30):.2e}",
            fontsize=7.5,
        )
    for panel in range(n, rows * cols):
        axes[panel // cols][panel % cols].axis("off")

    mappable = plt.cm.ScalarMappable(norm=_NORM, cmap=cmap)
    cax = fig.add_axes((0.945, 0.12, 0.013, 0.68))
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("core temperature [K]  (log scale)", fontsize=9)

    exaggeration = (
        f";  shape exaggerated {distortion_gain:g}x for visibility"
        if distortion_gain != 1.0
        else ""
    )
    fig.suptitle(
        f"Cavity cross-section in the XZ plane -- {cfg.scenario}\n"
        "uniform fill = single-zone core; outer annulus = thermal boundary "
        f"layer{exaggeration}",
        fontsize=10,
        y=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.86))
    return fig


def plot_phase_portrait(
    cfg: SimConfig,
    trajectory: Trajectory,
    n_isobars: int = 7,
    r_range: tuple[float, float] | None = None,
    note: str = "",
) -> Figure:
    from core.acoustics import drive_pressure

    valid = np.asarray(trajectory.valid)
    t = np.asarray(trajectory.times)[valid]
    r = np.asarray(trajectory.radius)[valid]
    rdot = np.asarray(trajectory.radius_dot)[valid]

    r0 = float(cfg.r0)
    c_l = float(cfg.liquid.sound_speed)
    rho = float(cfg.liquid.density)
    sigma = float(cfg.liquid.surface_tension)
    p_v = float(cfg.liquid.vapour_pressure)
    p0 = float(cfg.liquid.ambient_pressure)

    p_inf = np.array(
        [
            float(
                drive_pressure(
                    jnp.asarray(ti), cfg.acoustics, cfg.liquid.ambient_pressure
                )[0]
            )
            for ti in t
        ]
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    lo, hi = r_range if r_range is not None else (
        float(r.min()) / r0 * 0.9,
        float(r.max()) / r0 * 1.05,
    )
    grid = np.geomspace(max(lo, 1e-3), hi, 600) * r0

    gamma = 5.0 / 3.0
    v_hc = float(cfg.n_nuclei * cfg.gas.hardcore_volume_per_nucleus)
    v0 = (4.0 / 3.0) * np.pi * r0**3
    p_g0 = p0 - p_v + 2.0 * sigma / r0
    volume = (4.0 / 3.0) * np.pi * grid**3
    p_gas = p_g0 * ((v0 - v_hc) / np.maximum(volume - v_hc, 1e-300)) ** gamma
    p_wall = p_gas + p_v - 2.0 * sigma / grid

    levels = np.linspace(p_inf.min(), p_inf.max(), n_isobars)
    cmap_iso = plt.get_cmap("coolwarm")
    for i, level in enumerate(levels):
        value = 2.0 * (p_wall - level) / (3.0 * rho)
        ok = value > 0.0
        if not np.any(ok):
            continue
        speed = np.sqrt(value[ok]) / c_l
        colour = cmap_iso(i / max(len(levels) - 1, 1))
        ax.plot(grid[ok] / r0, speed, ls="-", lw=1.0, color=colour, alpha=0.85)
        ax.plot(grid[ok] / r0, -speed, ls="-", lw=1.0, color=colour,
                alpha=0.85)
    ax.plot([], [], ls="-", color="#B08090", lw=1.0,
            label=r"$\ddot R = 0$ isobars (blue $\to$ red: "
                  r"$p_\infty$ min $\to$ max)")

    points = ax.scatter(
        r / r0,
        rdot / c_l,
        c=p_inf / C.ATM_TO_PA,
        cmap="viridis",
        s=3.0,
        zorder=3,
    )
    ax.plot(r / r0, rdot / c_l, color="black", lw=0.35, alpha=0.4, zorder=2)
    cbar = fig.colorbar(points, ax=ax, pad=0.02)
    cbar.set_label(r"drive pressure $p_\infty$ [atm]", fontsize=9)

    ax.axhline(0.0, color="#444444", lw=0.7, ls=":")
    ax.axvline(1.0, color="#444444", lw=0.7, ls=":")
    h = float(cfg.hardcore_radius)
    if h > 0.0:
        ax.axvline(
            h / r0, color="#111111", ls="--", lw=1.6,
            label="van der Waals hard core",
        )
    ax.axhline(-1.0, color="#AA0000", ls="-.", lw=1.2,
               label="Mach 1 (Keller-Miksis validity edge)")

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
    ax.set_xlabel(r"$R / R_0$", fontsize=11)
    ax.set_ylabel(r"$\dot R / c_l$", fontsize=11)
    ax.set_title(
        f"Phase portrait -- {cfg.scenario}\n"
        f"as sampled here: peak inward Mach {abs(rdot.min()) / c_l:.3f}, "
        f"$R_{{\\min}}/R_0$ = {r.min() / r0:.4f}",
        fontsize=11, loc="left",
    )
    ax.grid(True, alpha=0.25, which="both", linewidth=0.5)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_action_spectrogram(
    cfg: SimConfig,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    collapse_times: np.ndarray,
    interval_times: np.ndarray,
    peak_temperatures: np.ndarray | None = None,
    min_radii: np.ndarray | None = None,
) -> Figure:
    from scipy import signal as sp_signal

    amplitudes = np.atleast_2d(amplitudes)
    phases = np.atleast_2d(phases)
    n_intervals, n_harmonics = amplitudes.shape
    f0 = float(cfg.acoustics.frequency)
    period = 1.0 / f0

    fig, axes = plt.subplots(
        3, 1, figsize=(10.5, 10.0),
        gridspec_kw={"height_ratios": [1.0, 1.4, 1.0]},
    )

    ax = axes[0]
    image = ax.imshow(
        amplitudes.T / C.ATM_TO_PA,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0.5, n_intervals + 0.5, 0.5, n_harmonics + 0.5),
    )
    ax.set_yticks(np.arange(1, n_harmonics + 1))
    ax.set_yticklabels(
        [r"$\omega$"] + [rf"${k}\omega$" for k in range(2, n_harmonics + 1)]
    )
    ax.set_xlabel("control interval")
    ax.set_ylabel("harmonic")
    ax.set_title(
        "Commanded harmonic amplitudes $A_k$ [atm]", fontsize=10, loc="left"
    )
    fig.colorbar(image, ax=ax, pad=0.08, fraction=0.04).set_label(
        "$A_k$ [atm]", fontsize=8
    )
    if peak_temperatures is not None:
        twin = ax.twinx()
        twin.plot(
            np.arange(1, n_intervals + 1),
            peak_temperatures,
            color="#00FFCC",
            lw=1.3,
            marker="o",
            ms=3,
        )
        twin.set_yscale("log")
        twin.set_ylabel(
            r"$T_{\rm peak}$ [K]", color="#00AA88", fontsize=9, labelpad=2
        )
        twin.tick_params(axis="y", colors="#00AA88", labelsize=8, pad=1)

    ax = axes[1]
    samples_per_period = 64 * n_harmonics
    dt = period / samples_per_period
    segments = []
    for i in range(n_intervals):
        t_local = interval_times[i] + np.arange(samples_per_period) * dt
        k = np.arange(1, n_harmonics + 1)
        theta = (
            k[None, :] * 2.0 * np.pi * f0 * t_local[:, None] + phases[i][None, :]
        )
        segments.append(-np.sum(amplitudes[i][None, :] * np.sin(theta), axis=1))
    signal = np.concatenate(segments)
    t_axis = interval_times[0] + np.arange(signal.size) * dt

    nperseg = min(2 * samples_per_period, max(signal.size // 4, 32))
    freqs, times_s, sxx = sp_signal.spectrogram(
        signal,
        fs=1.0 / dt,
        nperseg=nperseg,
        noverlap=nperseg * 3 // 4,
        scaling="spectrum",
    )
    floor = max(sxx.max() * 1e-8, 1e-30)
    mesh = ax.pcolormesh(
        t_axis[0] + times_s,
        freqs / f0,
        10.0 * np.log10(np.maximum(sxx, floor) / sxx.max()),
        cmap="viridis",
        shading="nearest",
        vmin=-70.0,
        vmax=0.0,
    )
    for k in range(1, n_harmonics + 1):
        ax.axhline(k, color="white", ls=":", lw=0.7, alpha=0.6)
    finite = np.isfinite(collapse_times)
    if np.any(finite):
        ax.vlines(
            collapse_times[finite],
            0.0,
            n_harmonics + 0.6,
            color="#FF3366",
            lw=1.0,
            alpha=0.85,
            label=r"$R_{\min}$ (collapse)",
        )
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.0, n_harmonics + 0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"frequency / $\omega$")
    ax.set_title(
        r"STFT of the realised drive $p_\infty(t)$ [dB re max], with the "
        r"collapse instants marked",
        fontsize=10, loc="left",
    )
    fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.04).set_label(
        "power [dB]", fontsize=8
    )

    ax = axes[2]
    markers = ["o", "s", "^", "D", "v", "P"]
    for k in range(n_harmonics):
        ax.plot(
            np.arange(1, n_intervals + 1),
            phases[:, k],
            marker=markers[k % len(markers)],
            ms=3.2,
            lw=1.0,
            label=rf"$\phi_{{{k + 1}}}$",
        )
    if min_radii is not None:
        twin = ax.twinx()
        twin.plot(
            np.arange(1, n_intervals + 1),
            np.asarray(min_radii) * 1e6,
            color="#777777",
            lw=1.1,
            ls="--",
        )
        twin.set_ylabel(r"$R_{\min}$ [$\mu$m]", color="#555555", fontsize=9)
        twin.tick_params(axis="y", colors="#555555", labelsize=8)
    ax.set_ylim(-np.pi * 1.05, np.pi * 1.05)
    ax.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel("control interval")
    ax.set_ylabel("phase [rad]")
    ax.set_title("Commanded harmonic phases", fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=n_harmonics, loc="upper left")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.suptitle(
        f"Agent actuation -- {cfg.scenario}, $f$ = {f0 / 1e3:.2f} kHz",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


__all__ = [
    "T_CMAP",
    "T_MAX_K",
    "T_MIN_K",
    "bubble_outline",
    "legendre_p2",
    "plot_action_spectrogram",
    "plot_bubble_projection",
    "plot_phase_portrait",
]
