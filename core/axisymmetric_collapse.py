from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

FOLD_BINS: int = 40
FOLD_SPREAD: float = 0.05
FOLD_ONSET: float = 0.02


@dataclass(frozen=True)
class AxisymCollapseRun:
    seed: float
    compression: float
    r_min: float
    t_fold: float | None
    lin_at_fold: float | None
    ns_at_fold: float | None
    fold_end: float


@dataclass(frozen=True)
class ThresholdCalibrationRow:
    label: str
    pg0: float
    compression: float
    mean_lin_at_fold: float
    std_lin_at_fold: float
    min_seed_folded: float
    n_folded: int


def fold_fraction_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = FOLD_BINS,
    spread: float = FOLD_SPREAD,
) -> float:
    r = np.hypot(x, np.abs(y))
    mu = np.cos(np.arctan2(np.abs(y), np.maximum(x, 1.0e-30)))
    idx = np.digitize(mu, np.linspace(0.0, 1.0, bins))
    folded = total = 0
    for b in np.unique(idx):
        rr = r[idx == b]
        if len(rr) < 2:
            continue
        total += 1
        if (rr.max() - rr.min()) / max(rr.mean(), 1.0e-30) > spread:
            folded += 1
    return folded / max(total, 1)


def legendre_even_basis(
    mu: np.ndarray, l_modes: list[int] | tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    n_pts = len(mu)
    n_m = len(l_modes)
    p_mat = np.zeros((n_pts, n_m), dtype=np.float64)
    dp_dtheta = np.zeros((n_pts, n_m), dtype=np.float64)
    if n_m == 0:
        return p_mat, dp_dtheta
    l_max = max(l_modes)
    sin_t = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))

    p_prev2 = np.ones(n_pts, dtype=np.float64)
    p_prev1 = mu.copy()
    dp_prev2 = np.zeros(n_pts, dtype=np.float64)
    dp_prev1 = np.ones(n_pts, dtype=np.float64)

    mode_to_col = {l_val: k for k, l_val in enumerate(l_modes)}
    if 0 in mode_to_col:
        p_mat[:, mode_to_col[0]] = p_prev2

    for l_curr in range(2, l_max + 1):
        inv_l = 1.0 / l_curr
        c1 = (2 * l_curr - 1) * inv_l
        c2 = (l_curr - 1) * inv_l
        p_curr = c1 * mu * p_prev1 - c2 * p_prev2
        dp_curr = l_curr * p_prev1 + mu * dp_prev1
        if l_curr in mode_to_col:
            col = mode_to_col[l_curr]
            p_mat[:, col] = p_curr
            dp_dtheta[:, col] = -sin_t * dp_curr
        p_prev2, p_prev1 = p_prev1, p_curr
        dp_prev2, dp_prev1 = dp_prev1, dp_curr

    return p_mat, dp_dtheta


# 2D axisymmetric Lagrangian boundary-integral / mode-coupling solver
def simulate_2d_axisym_collapse(
    seed_eps: float,
    *,
    pg0: float = 100.0,
    pinf: float = 500.0,
    gam_g: float = 1.4,
    gam_l: float = 7.14,
    pi1: float = 30000.0,
    sig: float = 0.0,
    mu_l: float = 0.0,
    r0_init: float = 1.0,
    n_markers: int = 240,
    l_max: int = 12,
    n_frames: int = 60,
) -> AxisymCollapseRun:
    l_modes = list(range(2, l_max + 1, 2))
    n_m = len(l_modes)

    alpha = np.linspace(0.0, 0.5 * math.pi, n_markers)
    mu0 = np.cos(alpha)
    sin0 = np.sin(alpha)
    p2_0 = 0.5 * (3.0 * mu0 * mu0 - 1.0)
    r_init = r0_init * (1.0 + seed_eps * p2_0)
    x = r_init * mu0
    y = r_init * sin0

    r_curr = float(r0_init)
    rdot = 0.0
    a2_lin = float(r0_init * seed_eps)
    a2dot_lin = 0.0

    a_m = np.zeros(n_m, dtype=np.float64)
    adot_m = np.zeros(n_m, dtype=np.float64)
    a_m[0] = r0_init * seed_eps

    c_inf = math.sqrt(gam_l * (pinf + pi1))
    t_rayleigh = 0.915 * r0_init * math.sqrt(1.0 / max(pinf - pg0, 1.0e-12))
    t_end = 2.5 * t_rayleigh
    dt = t_rayleigh / 4000.0

    def tait_hc(p_val: float) -> tuple[float, float]:
        denom = pinf + pi1
        alp = (gam_l - 1.0) / gam_l
        arg = max((p_val - pinf) / denom, -1.0 + 1.0e-12)
        h_val = (
            (gam_l / (gam_l - 1.0))
            * denom
            * math.expm1(alp * math.log1p(arg))
        )
        c_val = math.sqrt(max(c_inf * c_inf + (gam_l - 1.0) * h_val, 1.0e-6))
        return h_val, c_val

    def gilmore_rddot(rr: float, rd: float) -> float:
        def pw(r_v: float) -> float:
            pg = pg0 * (r0_init / max(r_v, 1.0e-12)) ** (3.0 * gam_g)
            return pg - 2.0 * sig / r_v - 4.0 * mu_l * rd / r_v

        h, c = tait_hc(pw(rr))
        e = 1.0e-7 * rr
        hp, _ = tait_hc(pw(rr + e))
        hm, _ = tait_hc(pw(rr - e))
        m = rd / c
        num = (
            h * (1.0 + m)
            + rr * ((hp - hm) / (2.0 * e) * rd) / c * (1.0 - m)
            - 1.5 * rd * rd * (1.0 - m / 3.0)
        )
        return num / (rr * (1.0 - m))

    def shape_ab(l_val: int, rr: float, rd: float, rdd: float) -> tuple[float, float]:
        inv_r = 1.0 / max(rr, 1.0e-12)
        inv_r3 = inv_r * inv_r * inv_r
        cap = (l_val - 1.0) * (l_val + 1.0) * (l_val + 2.0) * sig * inv_r3
        visc_a = (
            2.0
            * (l_val - 1.0)
            * (l_val + 1.0)
            * (l_val + 2.0)
            * mu_l
            * rd
            * inv_r3
        )
        a_coeff = (l_val - 1.0) * (-rdd) * inv_r - cap - visc_a
        b_coeff = (
            3.0 * rd * inv_r
            + 2.0 * (l_val + 2.0) * (2.0 * l_val + 1.0) * mu_l * inv_r * inv_r
        )
        return a_coeff, b_coeff

    t = 0.0
    r_min = r0_init
    fold_onset_t: float | None = None
    fold_onset_lin: float | None = None
    fold_onset_ns: float | None = None

    frame_interval = t_end / max(n_frames, 1)
    next_frame_t = 0.0

    while t < t_end and r_curr > 1.0e-5:
        rddot = gilmore_rddot(r_curr, rdot)
        dt_step = dt
        if r_curr < r_min:
            r_min = r_curr
        elif fold_onset_t is None and rdot > 0.0 and r0_init / max(r_min, 1.0e-30) > 3.0:
            break

        if t >= next_frame_t - 0.5 * dt_step:
            next_frame_t += frame_interval
            ff = fold_fraction_from_coords(x, y)
            if fold_onset_t is None and ff > FOLD_ONSET:
                fold_onset_t = t
                fold_onset_lin = abs(a2_lin) / r_curr
                r_m = np.hypot(x, np.abs(y))
                mu_m = np.cos(np.arctan2(np.abs(y), np.maximum(x, 1.0e-30)))
                order = np.argsort(mu_m)
                p2_m = 0.5 * (3.0 * mu_m[order] ** 2 - 1.0)
                a2_nl = 5.0 * float(np.trapezoid(r_m[order] * p2_m, mu_m[order]))
                r0_nl = float(np.trapezoid(r_m[order], mu_m[order]))
                fold_onset_ns = abs(a2_nl) / max(r0_nl, 1.0e-12)

        a2_coeff, b2_coeff = shape_ab(2, r_curr, rdot, rddot)
        a2ddot_lin = a2_coeff * a2_lin - b2_coeff * a2dot_lin

        if fold_onset_t is None:
            a_ddot_m = np.zeros(n_m, dtype=np.float64)
            for k, l_val in enumerate(l_modes):
                al, bl = shape_ab(l_val, r_curr, rdot, rddot)
                a_ddot_m[k] = al * a_m[k] - bl * adot_m[k]

            inv_r = 1.0 / max(r_curr, 1.0e-12)
            a0_clamped = float(np.clip(a_m[0], -2.0 * r_curr, 2.0 * r_curr))
            adot0_clamped = float(
                np.clip(adot_m[0], -50.0 * c_inf, 50.0 * c_inf)
            )
            q22 = (
                rddot * inv_r * a0_clamped ** 2
                + 0.5 * adot0_clamped ** 2
                + rdot * inv_r * a0_clamped * adot0_clamped
            ) * inv_r
            a_ddot_m[0] += (2.0 / 7.0) * 3.0 * q22
            if n_m > 1:
                a_ddot_m[1] += (18.0 / 35.0) * 5.0 * q22

            r_mk = np.hypot(x, y)
            mu_mk = np.clip(x / np.maximum(r_mk, 1.0e-30), 0.0, 1.0)
            sin_mk = np.clip(y / np.maximum(r_mk, 1.0e-30), 0.0, 1.0)
            p_mat, dp_mat = legendre_even_basis(mu_mk, l_modes)

            ratio_r = np.clip(r_curr / np.maximum(r_mk, 0.2 * r_curr), 0.0, 3.0)
            u_r = (ratio_r ** 2) * rdot
            u_theta = np.zeros_like(r_mk)

            for k, l_val in enumerate(l_modes):
                src_l = adot_m[k] + 2.0 * (rdot / r_curr) * a_m[k]
                rad_factor = ratio_r ** (l_val + 2)
                u_r += adot_m[k] * (ratio_r ** 2) * p_mat[:, k]
                u_theta += -(src_l / (l_val + 1.0)) * rad_factor * dp_mat[:, k]

            u_x = u_r * mu_mk - u_theta * sin_mk
            u_y = u_r * sin_mk + u_theta * mu_mk

            x = np.maximum(x + dt_step * u_x, 0.0)
            y = np.maximum(y + dt_step * u_y, 0.0)

            a_m = np.clip(
                a_m + dt_step * adot_m + 0.5 * dt_step * dt_step * a_ddot_m,
                -2.0 * r_curr,
                2.0 * r_curr,
            )
            adot_m = np.clip(
                adot_m + dt_step * a_ddot_m,
                -50.0 * c_inf,
                50.0 * c_inf,
            )

        r_curr += dt_step * rdot + 0.5 * dt_step * dt_step * rddot
        rdot += dt_step * rddot

        a2_lin += dt_step * a2dot_lin + 0.5 * dt_step * dt_step * a2ddot_lin
        a2dot_lin += dt_step * a2ddot_lin
        t += dt_step

    ff_end = fold_fraction_from_coords(x, y)
    return AxisymCollapseRun(
        seed=float(seed_eps),
        compression=float(r0_init / max(r_min, 1.0e-30)),
        r_min=float(r_min),
        t_fold=fold_onset_t,
        lin_at_fold=fold_onset_lin,
        ns_at_fold=fold_onset_ns,
        fold_end=float(ff_end),
    )


def calibrate_compression_level(
    label: str,
    pg0: float,
    seeds: list[float] | tuple[float, ...],
    *,
    pinf: float = 500.0,
    gam_g: float = 1.4,
    gam_l: float = 7.14,
    pi1: float = 30000.0,
    sig: float = 0.0,
    mu_l: float = 0.0,
    r0_init: float = 1.0,
) -> tuple[ThresholdCalibrationRow, list[AxisymCollapseRun]]:
    runs: list[AxisymCollapseRun] = []
    lin_vals: list[float] = []
    folded_seeds: list[float] = []

    for s in seeds:
        res = simulate_2d_axisym_collapse(
            s,
            pg0=pg0,
            pinf=pinf,
            gam_g=gam_g,
            gam_l=gam_l,
            pi1=pi1,
            sig=sig,
            mu_l=mu_l,
            r0_init=r0_init,
        )
        runs.append(res)
        if res.lin_at_fold is not None:
            lin_vals.append(res.lin_at_fold)
            folded_seeds.append(s)

    arr = np.asarray(lin_vals, dtype=np.float64)
    mean_val = float(arr.mean()) if len(arr) > 0 else 0.0
    std_val = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    min_seed = float(min(folded_seeds)) if folded_seeds else float("nan")
    comp = runs[0].compression if runs else 1.0

    row = ThresholdCalibrationRow(
        label=label,
        pg0=float(pg0),
        compression=float(comp),
        mean_lin_at_fold=mean_val,
        std_lin_at_fold=std_val,
        min_seed_folded=min_seed,
        n_folded=len(lin_vals),
    )
    return row, runs


__all__ = [
    "AxisymCollapseRun",
    "FOLD_BINS",
    "FOLD_ONSET",
    "FOLD_SPREAD",
    "ThresholdCalibrationRow",
    "calibrate_compression_level",
    "fold_fraction_from_coords",
    "legendre_even_basis",
    "simulate_2d_axisym_collapse",
]
