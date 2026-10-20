from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.spherical_euler import (
    PINF_DEFAULT,
    quantized_volume_radius,
    solve_gilmore_polytrope,
    solve_km_polytrope,
    solve_spherical_euler,
    thermodynamic_compliance,
)

DR_BASILISK_L13: float = 50.0 / (2 ** 13)

BASILISK_RADIAL_CASES: tuple[tuple[str, float, float, float, float, float, float, float], ...] = (
    ("collapse.c",       100.0, 1.40, 0.0,  2.0,    0.517440, 0.517440, 0.520590),
    ("rad_p100g14",      100.0, 1.40, 0.0,  0.0,    0.459624, 0.459620, 0.455380),
    ("rad4_run_p30g11",   30.0, 1.10, 0.0,  0.0,    0.102280, 0.102280, 0.102590),
    ("rad4_run_p10g12",   10.0, 1.20, 0.0,  0.0,    0.069540, 0.069540, 0.070580),
    ("rad4_run_p10g11",   10.0, 1.10, 0.0,  0.0,    0.036518, 0.036537, 0.041410),
    ("rad4_run_p10g105",  10.0, 1.05, 0.0,  0.0,    0.024329, 0.024529, 0.027860),
    ("rad5_run_fast102",  10.0, 1.02, 0.0,  0.0,    0.018171, 0.016813, 0.022398),
    ("rad5_run_fast101",  10.0, 1.01, 0.0,  0.0,    0.014422, 0.015311, 0.018818),
    ("rad5_run_mu105",    10.0, 1.05, 0.0,  0.2064, 0.028845, 0.029098, 0.033290),
    ("rad5_run_sig2",     10.0, 1.05, 2.31, 0.0,    0.022894, 0.022634, 0.026890),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-cells",
        type=int,
        default=350,
        help="Radial cells in the co-collapsing ALE Euler grid (default: 350).",
    )
    args = ap.parse_args()
    n_cells = int(args.n_cells)

    print("1. DECOMPOSITION OF THE BASILISK radial_collapse.c DISCREPANCY (MAXLEV = 13)")
    print(
        f"{'case':18s} | {'Rmin_q':>8s} | {'Rmin_VOF':>8s} | {'Rmin/dr':>7s} | "
        f"{'Rmin_ALE':>8s} | {'Rmin_Gil':>8s} | {'err(q)':>8s} | {'err(VOF)':>8s} | {'err(ALE)':>8s}"
    )
    print("-" * 106)

    for label, pg0, gam, sig, mu, r_q, r_vof, r_gil_log in BASILISK_RADIAL_CASES:
        eul = solve_spherical_euler(pg0, gam, sig, mu, n_cells=n_cells)
        gil = solve_gilmore_polytrope(pg0, gam, sig, mu)
        cells_at_rmin = r_vof / DR_BASILISK_L13
        err_q = (r_gil_log - r_q) / r_q
        err_vof = (r_gil_log - r_vof) / r_vof
        err_ale = (gil.r_min - eul.r_min) / eul.r_min
        print(
            f"{label:18s} | {r_q:8.5f} | {r_vof:8.5f} | {cells_at_rmin:7.2f} | "
            f"{eul.r_min:8.5f} | {gil.r_min:8.5f} | {err_q:+7.2%} | {err_vof:+7.2%} | {err_ale:+7.2%}"
        )

    print("\n2. DE-CONFOUNDED SWEEP AT FIXED GAMMA = 1.40 (varying p_g0 at p_inf = 500)")
    print(
        f"{'p_g0':>8s} | {'C_Euler':>8s} | {'M_loc':>6s} | {'M_amb':>6s} | "
        f"{'Rmin_ALE':>9s} | {'Rmin_Gil':>9s} | {'Gil_err':>8s} | {'Rmin_KM':>9s} | {'KM_err':>8s}"
    )
    print("-" * 96)
    for pg0 in (100.0, 30.0, 10.0, 3.0, 1.0, 0.4, 0.2):
        gam = 1.40
        eul = solve_spherical_euler(pg0, gam, n_cells=n_cells)
        gil = solve_gilmore_polytrope(pg0, gam)
        km = solve_km_polytrope(pg0, gam)
        gil_err = (gil.r_min - eul.r_min) / eul.r_min
        km_err = (km.r_min - eul.r_min) / eul.r_min
        print(
            f"{pg0:8.2f} | {eul.compression:8.1f} | {eul.peak_mach_local:6.3f} | "
            f"{eul.peak_mach_ambient:6.3f} | {eul.r_min:9.5f} | {gil.r_min:9.5f} | "
            f"{gil_err:+7.2%} | {km.r_min:9.5f} | {km_err:+7.2%}"
        )

    print("\n3. DE-CONFOUNDED SWEEP AT FIXED GAMMA = 5/3 (monatomic argon, varying p_g0)")
    print(
        f"{'p_g0':>8s} | {'C_Euler':>8s} | {'M_loc':>6s} | {'M_amb':>6s} | "
        f"{'Rmin_ALE':>9s} | {'Rmin_Gil':>9s} | {'Gil_err':>8s} | {'Rmin_KM':>9s} | {'KM_err':>8s}"
    )
    print("-" * 96)
    for pg0 in (100.0, 20.0, 5.0, 1.0, 0.2, 0.05, 0.015):
        gam = 5.0 / 3.0
        eul = solve_spherical_euler(pg0, gam, n_cells=n_cells)
        gil = solve_gilmore_polytrope(pg0, gam)
        km = solve_km_polytrope(pg0, gam)
        gil_err = (gil.r_min - eul.r_min) / eul.r_min
        km_err = (km.r_min - eul.r_min) / eul.r_min
        print(
            f"{pg0:8.3f} | {eul.compression:8.1f} | {eul.peak_mach_local:6.3f} | "
            f"{eul.peak_mach_ambient:6.3f} | {eul.r_min:9.5f} | {gil.r_min:9.5f} | "
            f"{gil_err:+7.2%} | {km.r_min:9.5f} | {km_err:+7.2%}"
        )

    print("\n4. ISO-COMPRESSION SWEEP AT C ~ 40 (varying gamma from 1.05 to 5/3)")
    print(
        f"{'gamma':>6s} | {'p_g0':>8s} | {'C_Euler':>8s} | {'S(gam,C)':>8s} | "
        f"{'Rmin_ALE':>9s} | {'Rmin_Gil':>9s} | {'Gil_err':>8s} | {'Rmin_KM':>9s} | {'KM_err':>8s}"
    )
    print("-" * 98)
    for gam, pg0 in (
        (1.05, 10.0),
        (1.10, 6.2),
        (1.20, 2.4),
        (1.40, 0.34),
        (5.0 / 3.0, 0.022),
    ):
        eul = solve_spherical_euler(pg0, gam, n_cells=n_cells)
        gil = solve_gilmore_polytrope(pg0, gam)
        km = solve_km_polytrope(pg0, gam)
        s_val = thermodynamic_compliance(gam, eul.compression)
        gil_err = (gil.r_min - eul.r_min) / eul.r_min
        km_err = (km.r_min - eul.r_min) / eul.r_min
        print(
            f"{gam:6.3f} | {pg0:8.3f} | {eul.compression:8.1f} | {s_val:8.2f} | "
            f"{eul.r_min:9.5f} | {gil.r_min:9.5f} | {gil_err:+7.2%} | "
            f"{km.r_min:9.5f} | {km_err:+7.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
