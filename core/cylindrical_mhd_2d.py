from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from core import mrt

MU0: float = mrt.MU0
CRITICAL_K_XI_FOLD: float = 0.5 * math.pi


@dataclass(frozen=True)
class CylindricalMhdStepResult:
    xi_lin: float
    xidot_lin: float
    xi_bub: float
    xidot_bub: float
    xi_in: float
    xidot_in: float
    k_xi_bub: float
    fold_fraction: float


@dataclass(frozen=True)
class LinerDriveCalibrationRow:
    action: float
    t_fold_ns: float | None
    r_out_ratio_at_fold: float
    bub_rel_at_fold: float
    in_rel_at_fold: float
    lin_rel_at_fold: float
    thin_shell_break_lin: float


# Layzer-Goncharov nonlinear bubble-tip saturation
def layzer_goncharov_alpha(eta: float) -> float:
    e3 = math.exp(-3.0 * max(float(eta), 0.0))
    return (1.0 - e3) / (2.0 + e3)


def mhd_2d_fold_fraction(
    xi_bub: float,
    k0: float,
    *,
    n_markers: int = 128,
) -> float:
    alpha = np.linspace(0.0, math.pi / max(k0, 1.0e-12), n_markers)
    z_coord = alpha - (2.0 / math.pi) * max(float(xi_bub), 0.0) * np.sin(k0 * alpha)
    dz = np.diff(z_coord)
    return float(np.mean(dz <= 0.0))


# 2D (r, z) cylindrical MHD liner evolution with wave-overturning check
def advance_mhd_2d_interval(
    *,
    xi_lin: float,
    xidot_lin: float,
    xi_bub: float,
    xidot_bub: float,
    xi_in: float,
    xidot_in: float,
    r_out: float,
    u_out: float,
    rddot_out: float,
    r_in: float,
    u_in: float,
    rddot_in: float,
    b_z: float,
    rho_fuel: float,
    rho_shell: float,
    k0: float,
    dt_interval: float,
    n_substeps: int = 40,
) -> CylindricalMhdStepResult:
    dt_o = dt_interval / max(n_substeps, 1)
    d_r_shell = max(r_out - r_in, 1.0e-6)
    g_out = max(-rddot_out, 0.0)
    conv_out = u_out / max(r_out, 1.0e-12)
    conv_in = u_in / max(r_in, 1.0e-12)
    atwood_in = (rho_shell - rho_fuel) / (rho_shell + rho_fuel)
    tension_in = (k0 * k0 * b_z * b_z) / (MU0 * (rho_shell + rho_fuel))

    xl, xdl = float(xi_lin), float(xidot_lin)
    xb, xdb = float(xi_bub), float(xidot_bub)
    xi, xdi = float(xi_in), float(xidot_in)

    for _ in range(n_substeps):
        add_lin = (k0 * g_out) * xl - conv_out * xdl
        xl += dt_o * xdl + 0.5 * dt_o * dt_o * add_lin
        xdl += dt_o * add_lin

        alpha_lg = layzer_goncharov_alpha(k0 * xb)
        g_neck = g_out * (1.0 + 2.0 * max(xb, 0.0) / max(r_out, 1.0e-12))
        add_bub = (
            alpha_lg * (g_neck - 3.0 * k0 * abs(xdb) * xdb)
            - conv_out * xdb
        )
        xb += dt_o * xdb + 0.5 * dt_o * dt_o * add_bub
        xdb += dt_o * add_bub

        feed = math.exp(-k0 * d_r_shell) * alpha_lg * g_neck
        add_in = (
            feed
            + (k0 * atwood_in * rddot_in - tension_in) * xi
            - conv_in * xdi
        )
        xi += dt_o * xdi + 0.5 * dt_o * dt_o * add_in
        xdi += dt_o * add_in

    k_xi = k0 * xb
    ff = mhd_2d_fold_fraction(xb, k0)
    return CylindricalMhdStepResult(
        xi_lin=xl,
        xidot_lin=xdl,
        xi_bub=xb,
        xidot_bub=xdb,
        xi_in=xi,
        xidot_in=xdi,
        k_xi_bub=k_xi,
        fold_fraction=ff,
    )


__all__ = [
    "CRITICAL_K_XI_FOLD",
    "CylindricalMhdStepResult",
    "LinerDriveCalibrationRow",
    "MU0",
    "advance_mhd_2d_interval",
    "layzer_goncharov_alpha",
    "mhd_2d_fold_fraction",
]
