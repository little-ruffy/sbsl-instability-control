from __future__ import annotations

import math
import pytest

from core.axisymmetric_collapse import simulate_2d_axisym_collapse
from core.cylindrical_mhd_2d import (
    CRITICAL_K_XI_FOLD,
    advance_mhd_2d_interval,
    layzer_goncharov_alpha,
    mhd_2d_fold_fraction,
)
from core.spherical_euler import (
    quantized_volume_radius,
    solve_gilmore_polytrope,
    solve_km_polytrope,
    solve_spherical_euler,
    thermodynamic_compliance,
)


def test_thermodynamic_compliance_limits() -> None:
    c = 50.0
    s_iso = thermodynamic_compliance(1.0, c)
    s_near_iso = thermodynamic_compliance(1.0 + 1.0e-7, c)
    assert math.isclose(s_iso, math.log(c), rel_tol=1.0e-12)
    assert math.isclose(s_near_iso, math.log(c), rel_tol=1.0e-5)

    s_monatomic = thermodynamic_compliance(5.0 / 3.0, c)
    expected_monatomic = 0.5 * (1.0 - c ** (-2.0))
    assert math.isclose(s_monatomic, expected_monatomic, rel_tol=1.0e-12)
    assert s_iso / s_monatomic > 7.5


def test_bubble_h_volume_quantization_reproduces_radial_collapse_c() -> None:
    _, v_q101, r_q101 = quantized_volume_radius(0.015311, decimals=6)
    assert math.isclose(v_q101, 1.0e-6, rel_tol=1.0e-12)
    assert math.isclose(r_q101, 0.0144224957, rel_tol=1.0e-6)
    r_mdl_log = 0.018818
    err_quant_log = (r_mdl_log - r_q101) / r_q101
    err_exact_log = (r_mdl_log - 0.015311) / 0.015311
    assert 0.300 < err_quant_log < 0.310
    assert 0.225 < err_exact_log < 0.235
    gil_101 = solve_gilmore_polytrope(10.0, 1.01)
    assert (gil_101.r_min - r_q101) / r_q101 - (gil_101.r_min - 0.015311) / 0.015311 > 0.07

    _, v_q102, r_q102 = quantized_volume_radius(0.016813, decimals=6)
    assert math.isclose(v_q102, 2.0e-6, rel_tol=1.0e-12)
    assert math.isclose(r_q102, 0.0181712059, rel_tol=1.0e-6)


def test_spherical_euler_matches_gilmore_and_deconfounds_gamma() -> None:
    eul_mod = solve_spherical_euler(100.0, 1.40, mu_l=2.0, n_cells=350)
    gil_mod = solve_gilmore_polytrope(100.0, 1.40, mu_l=2.0)
    rel_mod = abs(gil_mod.r_min - eul_mod.r_min) / eul_mod.r_min
    assert rel_mod < 0.0015

    eul_hi = solve_spherical_euler(0.2, 5.0 / 3.0, n_cells=350)
    gil_hi = solve_gilmore_polytrope(0.2, 5.0 / 3.0)
    km_hi = solve_km_polytrope(0.2, 5.0 / 3.0)
    assert eul_hi.compression > 18.5
    gil_err = (gil_hi.r_min - eul_hi.r_min) / eul_hi.r_min
    km_err = (km_hi.r_min - eul_hi.r_min) / eul_hi.r_min
    assert 0.0 < gil_err < 0.036
    assert km_err > 3.0 * gil_err


def test_axisymmetric_folding_threshold_scale_invariance() -> None:
    r_low = simulate_2d_axisym_collapse(0.15, pg0=100.0, pinf=500.0, gam_g=1.4)
    r_high = simulate_2d_axisym_collapse(0.15, pg0=1.0, pinf=500.0, gam_g=1.4)
    assert r_low.lin_at_fold is not None
    assert r_high.lin_at_fold is not None
    assert 2.0 < r_low.compression < 2.5
    assert r_high.compression > 20.0
    assert 0.24 < r_low.lin_at_fold < 0.46
    assert 0.24 < r_high.lin_at_fold < 0.46
    assert abs(r_low.lin_at_fold - r_high.lin_at_fold) < 0.05


def test_layzer_goncharov_and_2d_mhd_convergence() -> None:
    eta_small = 1.0e-6
    assert math.isclose(layzer_goncharov_alpha(eta_small) / eta_small, 1.0, rel_tol=2.0e-6)
    assert math.isclose(layzer_goncharov_alpha(50.0), 0.5, rel_tol=1.0e-12)

    k0 = 2.0 * math.pi / 5.0e-4
    xi_crit = CRITICAL_K_XI_FOLD / k0
    assert mhd_2d_fold_fraction(0.98 * xi_crit, k0) == 0.0
    assert mhd_2d_fold_fraction(1.05 * xi_crit, k0) > 0.0

    res = advance_mhd_2d_interval(
        xi_lin=1.0e-10,
        xidot_lin=0.0,
        xi_bub=1.0e-10,
        xidot_bub=0.0,
        xi_in=0.0,
        xidot_in=0.0,
        r_out=3.0e-3,
        u_out=-1.0e4,
        rddot_out=-5.0e11,
        r_in=2.94e-3,
        u_in=-1.0e4,
        rddot_in=-5.0e11,
        b_z=30.0,
        rho_fuel=1.0,
        rho_shell=9000.0,
        k0=k0,
        dt_interval=1.0e-9,
        n_substeps=50,
    )
    assert math.isclose(res.xi_bub, res.xi_lin, rel_tol=1.0e-3)
