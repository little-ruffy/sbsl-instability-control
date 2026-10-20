from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from core import constants as C
from core import gilmore as gil
from core import keller_miksis as km
from core import plasma_core as pc
from core.config import SimConfig
from core.integrator import (
    Trajectory,
    diagnostics_along,
    integrate,
)

_COLOURS = {
    "radius": "#0072B2",
    "temperature": "#D55E00",
    "pressure": "#009E73",
    "distortion": "#CC79A7",
    "drive": "#56B4E9",
    "loss": "#E69F00",
    "reference": "#555555",
}


def _valid(trajectory: Trajectory) -> np.ndarray:
    return np.asarray(trajectory.valid)


def _style(ax: plt.Axes, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def collapse_window(
    cfg: SimConfig,
    t0: float,
    t1: float,
    y0: jnp.ndarray | None = None,
    half_width: float = 3.0e-8,
    n_points: int = 2001,
    rhs_fn=km.rhs_only,
) -> tuple[Trajectory, float]:
    scout = integrate(
        cfg,
        t0,
        t1,
        y0=y0,
        rhs_fn=rhs_fn,
        save_at_steps=True,
        max_steps=131072,
        throw=False,
    )
    mask = _valid(scout)
    if not mask.any():
        return scout, 0.5 * (t0 + t1)
    times = np.asarray(scout.times)[mask]
    radii = np.asarray(scout.radius)[mask]
    t_min = float(times[int(np.argmin(radii))])

    lo = max(t_min - half_width, float(t0))
    hi = min(t_min + half_width, float(t1))
    window = jnp.linspace(lo, hi, n_points)
    refined = integrate(
        cfg, t0, t1, y0=y0, rhs_fn=rhs_fn, save_ts=window, throw=False
    )
    return refined, t_min


def plot_cycle(
    cfg: SimConfig,
    trajectory: Trajectory,
    collapse: Trajectory | None = None,
    t_min: float | None = None,
    title: str | None = None,
) -> Figure:
    mask = _valid(trajectory)
    t = np.asarray(trajectory.times)[mask]
    r = np.asarray(trajectory.radius)[mask]
    temperature = np.asarray(trajectory.temperature)[mask]
    rdot = np.asarray(trajectory.radius_dot)[mask]
    a_rel = np.asarray(trajectory.relative_distortion)[mask]
    diag = diagnostics_along(cfg, trajectory)
    p_inf = np.asarray(diag.drive_pressure)[mask]
    q_cond = np.asarray(diag.conduction_loss)[mask]
    q_rad = np.asarray(diag.radiative_loss)[mask]

    period = (
        1.0 / float(cfg.acoustics.frequency)
        if float(cfg.acoustics.frequency) > 0.0
        else (t[-1] - t[0] if t.size > 1 else 1.0)
    )
    t_cycles = (t - t[0]) / period

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.0))
    fig.suptitle(
        title or f"Sonofusion Sim -- {cfg.scenario}", fontsize=12, y=0.985
    )

    ax = axes[0, 0]
    ax.plot(t_cycles, r * 1e6, color=_COLOURS["radius"], lw=1.4, label="R")
    ax.axhline(
        float(cfg.r0) * 1e6,
        color=_COLOURS["reference"],
        ls=":",
        lw=0.9,
        label=r"$R_0$",
    )
    if float(cfg.hardcore_radius) > 0.0:
        ax.axhline(
            float(cfg.hardcore_radius) * 1e6,
            color="#AA0000",
            ls="--",
            lw=0.9,
            label="van der Waals core $h$",
        )
    _style(ax, "time [drive cycles]", r"radius [$\mu$m]", "Bubble radius")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)

    ax2 = ax.twinx()
    ax2.plot(t_cycles, p_inf / C.ATM_TO_PA, color=_COLOURS["drive"], lw=0.8,
             alpha=0.65)
    ax2.set_ylabel(r"$p_\infty$ [atm]", fontsize=9, color=_COLOURS["drive"])
    ax2.tick_params(axis="y", labelsize=8, colors=_COLOURS["drive"])

    ax = axes[0, 1]
    ax.plot(t_cycles, temperature, color=_COLOURS["temperature"], lw=1.3)
    ax.axhline(
        float(cfg.liquid.ambient_temperature),
        color=_COLOURS["reference"],
        ls=":",
        lw=0.9,
        label=r"$T_\infty$",
    )
    ax.set_yscale("log")
    _style(ax, "time [drive cycles]", "core temperature [K]",
           "Core temperature (cycle scale)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    if collapse is not None and t_min is not None:
        cmask = _valid(collapse)
        tc = (np.asarray(collapse.times)[cmask] - t_min) * 1e9
        rc = np.asarray(collapse.radius)[cmask] * 1e6
        tcore = np.asarray(collapse.temperature)[cmask]
        ax.plot(tc, rc, color=_COLOURS["radius"], lw=1.4)
        axt = ax.twinx()
        axt.plot(tc, tcore, color=_COLOURS["temperature"], lw=1.2)
        axt.set_ylabel("T [K]", fontsize=9, color=_COLOURS["temperature"])
        axt.tick_params(axis="y", labelsize=8, colors=_COLOURS["temperature"])
        axt.set_yscale("log")
        _style(
            ax,
            "time from minimum radius [ns]",
            r"radius [$\mu$m]",
            f"Collapse (refined window): "
            f"$R_{{\\min}}$ = {np.min(rc):.4f} $\\mu$m, "
            f"$T_{{\\rm peak}}$ = {np.max(tcore):.3e} K",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "no refined collapse window supplied\n"
            "(a coarse trace misreports the peak\n"
            "by up to a factor of four)",
            ha="center",
            va="center",
            fontsize=9,
            color="#AA0000",
            transform=ax.transAxes,
        )
        _style(ax, "", "", "Collapse")

    ax = axes[1, 1]
    mach = rdot / float(cfg.liquid.sound_speed)
    ax.plot(t_cycles, mach, color=_COLOURS["pressure"], lw=1.1)
    ax.axhline(-1.0, color="#AA0000", ls="--", lw=0.9,
               label="Mach 1 (Keller-Miksis validity edge)")
    ax.axhline(0.0, color=_COLOURS["reference"], ls=":", lw=0.8)
    _style(ax, "time [drive cycles]", r"$\dot R / c$", "Wall Mach number")
    ax.legend(fontsize=7)

    ax = axes[2, 0]
    ax.plot(t_cycles, np.abs(q_cond), color=_COLOURS["loss"], lw=1.1,
            label="conduction")
    ax.plot(t_cycles, np.abs(q_rad), color=_COLOURS["distortion"], lw=1.1,
            label="radiation (bremsstrahlung)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-20)
    _style(ax, "time [drive cycles]", "|power| [W]", "Energy loss channels")
    ax.legend(fontsize=7)

    ax = axes[2, 1]
    threshold = float(cfg.shape.instability_threshold)
    ax.plot(t_cycles, a_rel, color=_COLOURS["distortion"], lw=1.2,
            label=rf"$|a_{cfg.shape.mode_l}| / R$")
    ax.axhline(threshold, color="#AA0000", ls="--", lw=1.0,
               label=f"fragmentation threshold {threshold}")
    ax.axhline(0.3, color=_COLOURS["reference"], ls=":", lw=0.9,
               label="linear-theory limit ~0.3")
    ax.set_yscale("log")
    _style(
        ax,
        "time [drive cycles]",
        r"$|a_l| / R$",
        f"Shape mode $l$ = {cfg.shape.mode_l} "
        f"(max {a_rel.max():.3e})",
    )
    ax.legend(fontsize=7)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def plot_episode(
    cfg: SimConfig, trajectories: Sequence[Trajectory], title: str | None = None
) -> Figure:
    steps = np.arange(1, len(trajectories) + 1)
    peak_t, min_r, max_d, mach, yields = [], [], [], [], []
    for traj in trajectories:
        if not _valid(traj).any():
            peak_t.append(np.nan)
            min_r.append(np.nan)
            max_d.append(np.nan)
            mach.append(np.nan)
            yields.append(np.nan)
            continue
        peak_t.append(float(traj.peak_temperature))
        min_r.append(float(traj.min_radius) * 1e6)
        max_d.append(float(traj.max_relative_distortion))
        mach.append(
            float(traj.peak_wall_speed) / float(cfg.liquid.sound_speed)
        )
        yields.append(float(traj.neutron_yield))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    fig.suptitle(title or f"Episode summary -- {cfg.scenario}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(steps, peak_t, "o-", ms=3.5, color=_COLOURS["temperature"], lw=1.2)
    ax.set_yscale("log")
    _style(ax, "control interval", r"$T_{\rm peak}$ [K]",
           "Peak core temperature per interval")

    ax = axes[0, 1]
    ax.plot(steps, min_r, "o-", ms=3.5, color=_COLOURS["radius"], lw=1.2)
    if float(cfg.hardcore_radius) > 0.0:
        ax.axhline(float(cfg.hardcore_radius) * 1e6, color="#AA0000", ls="--",
                   lw=0.9, label="van der Waals core")
        ax.legend(fontsize=7)
    _style(ax, "control interval", r"$R_{\min}$ [$\mu$m]", "Closest approach")

    ax = axes[1, 0]
    ax.plot(steps, max_d, "o-", ms=3.5, color=_COLOURS["distortion"], lw=1.2)
    ax.axhline(float(cfg.shape.instability_threshold), color="#AA0000",
               ls="--", lw=1.0, label="fragmentation threshold")
    ax.set_yscale("log")
    _style(ax, "control interval", r"$\max |a_l| / R$", "Shape instability")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(steps, mach, "o-", ms=3.5, color=_COLOURS["pressure"], lw=1.2)
    ax.axhline(1.0, color="#AA0000", ls="--", lw=0.9, label="Mach 1")
    _style(ax, "control interval", r"peak $|\dot R| / c$", "Peak wall Mach")
    ax.legend(fontsize=7)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def render_episode(
    cfg: SimConfig,
    trajectories: Sequence[Trajectory],
    mode: str = "rgb_array",
) -> np.ndarray | None:
    fig = plot_episode(cfg, trajectories)
    fig.canvas.draw()
    if mode == "human":
        plt.close(fig)
        return None
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buffer[..., :3].copy()


def save_figures(
    figures: Iterable[tuple[str, Figure]], out_dir: str | Path, dpi: int = 140
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, fig in figures:
        path = out / f"{stem}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


# peak |Rdot|/C_wall against local Tait sound speed
def _local_wall_mach(cfg: SimConfig, trajectory: Trajectory) -> float:
    states = np.asarray(trajectory.states)
    times = np.asarray(trajectory.times)
    valid = np.isfinite(times) & np.all(np.isfinite(states), axis=1)
    if not np.any(valid):
        return float("nan")
    speed = np.where(valid, np.abs(states[:, km.IDX_RDOT]), -np.inf)
    i = int(np.argmax(speed))
    r = max(float(states[i, km.IDX_R]), float(cfg.r_floor))
    rdot = float(states[i, km.IDX_RDOT])
    t_fast = float(times[i])
    n_vap = max(float(states[i, km.IDX_NVAP]), 0.0)
    t = float(
        np.clip(
            states[i, km.IDX_T],
            float(cfg.guards.t_min),
            float(cfg.guards.t_max),
        )
    )
    counts = pc.vapour_counts(cfg, jnp.asarray(n_vap))
    comp = pc.composition(
        jnp.asarray(t),
        pc.free_volume(jnp.asarray(r), cfg, n_per_species=counts),
        cfg,
        n_per_species=counts,
    )
    p_wall = (
        pc.pressure_from_composition(comp, cfg)
        + cfg.liquid.vapour_pressure
        - 2.0 * cfg.liquid.surface_tension / r
        - 4.0 * cfg.liquid.dynamic_viscosity * rdot / r
    )
    p_inf, _ = km.drive_pressure(
        jnp.asarray(t_fast), cfg.acoustics, cfg.liquid.ambient_pressure
    )
    c_wall = float(
        gil.wall_sound_speed(
            gil.enthalpy(p_wall, p_inf, cfg.liquid),
            cfg.liquid,
        )
    )
    return abs(rdot) / max(c_wall, 1.0)


def summarise(cfg: SimConfig, trajectory: Trajectory,
               collapse: Trajectory | None = None) -> dict[str, float]:
    peak_source = collapse if collapse is not None else trajectory
    r_min = float(peak_source.min_radius)
    h = float(cfg.hardcore_radius)
    return {
        "r0_m": float(cfg.r0),
        "hardcore_radius_m": h,
        "r_max_m": float(trajectory.max_radius),
        "r_min_m": r_min,
        "r_min_over_hardcore": r_min / h if h > 0.0 else float("nan"),
        "expansion_ratio": float(trajectory.max_radius) / max(r_min, 1e-300),
        "t_peak_K": float(peak_source.peak_temperature),
        "peak_mach": float(peak_source.peak_wall_speed)
        / float(cfg.liquid.sound_speed),
        "peak_mach_local": _local_wall_mach(cfg, peak_source),
        "max_relative_distortion": float(
            max(
                float(trajectory.max_relative_distortion),
                float(peak_source.max_relative_distortion),
            )
        ),
        "neutron_yield": float(trajectory.neutron_yield),
        "radiated_energy_J": float(
            np.asarray(trajectory.final_state)[km.IDX_ERAD]
        ),
        "solver_steps": float(trajectory.num_steps),
    }


__all__ = [
    "collapse_window",
    "plot_cycle",
    "plot_episode",
    "render_episode",
    "save_figures",
    "summarise",
]
