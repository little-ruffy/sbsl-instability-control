from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp

from core import constants as C
from core import keller_miksis as km
from core import plasma_core as pc
from core.config import load_config
from core.integrator import integrate

MU0 = 4.0e-7 * math.pi
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
EPS0 = 8.8541878128e-12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="sbsl_argon_unstable")
    ap.add_argument(
        "--b0", type=float, default=30.0, help="applied field [T]"
    )
    ap.add_argument("--cycles", type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_config(scenario=args.scenario)
    period = 1.0 / float(cfg.acoustics.frequency)
    traj = integrate(
        cfg,
        0.0,
        args.cycles * period,
        save_at_steps=True,
        max_steps=20000,
        throw=True,
    )
    valid = np.asarray(traj.valid)
    t = np.asarray(traj.times)[valid]
    y = np.asarray(traj.states)[valid]
    order = np.argsort(t)
    t, y = t[order], y[order]

    r = y[:, km.IDX_R]
    temp = y[:, km.IDX_T]
    rdot = y[:, km.IDX_RDOT]

    k = int(np.argmax(np.abs(rdot)))
    R = float(r[k])
    T = float(temp[k])
    V = float(abs(rdot[k]))
    R0 = float(cfg.r0)
    k_min = int(np.argmin(r))
    k_hot = int(np.argmax(temp))
    print(f"(sampling at peak |Rdot|; for reference R_min = "
          f"{float(r[k_min]) * 1e6:.5f} um and T_peak = "
          f"{float(temp[k_hot]):.4e} K)")

    counts = pc.vapour_counts(cfg, jnp.asarray(y[k, km.IDX_NVAP]))
    v_free = pc.free_volume(jnp.asarray(R), cfg, counts)
    comp = pc.composition(jnp.asarray(T), v_free, cfg, counts)
    eos = pc.equation_of_state_and_derivatives(
        jnp.asarray(T), v_free, cfg, counts
    )
    n_e = float(comp.n_e_density)
    zbar = n_e / max(float(comp.n_heavy_density), 1.0e-30)
    p_gas = float(eos.pressure)
    ln_lambda = float(pc.coulomb_logarithm(comp, cfg.transport))

    print(f"scenario              {args.scenario}")
    print(f"R_0                   {R0 * 1e6:.3f} um")
    print("at peak wall speed:")
    print(f"  R                   {R * 1e6:.5f} um   "
          f"(compression R_0/R = {R0 / R:.2f})")
    print(f"  |Rdot|              {V:.1f} m/s")
    print(f"  T                   {T:.4e} K  ({T / C.KELVIN_PER_EV:.3f} eV)")
    print(f"  n_e                 {n_e:.4e} m^-3")
    print(f"  mean ionisation     {zbar:.4f}")
    print(f"  ln Lambda           {ln_lambda:.3f}")
    print(f"  gas pressure        {p_gas:.4e} Pa  ({p_gas / 1e5:.3e} bar)")

    t_ev = T / C.KELVIN_PER_EV
    n_heavy = max(float(comp.n_heavy_density), 1.0e-30)
    n_neutral = max(n_heavy - n_e, 0.0)

    v_th = math.sqrt(8.0 * C.K_B * T / (math.pi * M_E))

    z_ion = float(np.max(np.asarray(comp.z_mean)))
    z_ion = max(z_ion, 1.0)
    sigma_ei = 1.53e4 * t_ev**1.5 / (z_ion * ln_lambda)
    nu_ei = n_e * E_CHARGE**2 / (M_E * sigma_ei)

    SIGMA_EN = 1.0e-20
    nu_en = n_neutral * SIGMA_EN * v_th

    nu_total = nu_ei + nu_en
    sigma = n_e * E_CHARGE**2 / (M_E * nu_total)
    tau_e = 1.0 / nu_total

    eta = 1.0 / (MU0 * sigma)
    t_diff = R * R / eta
    t_coll = R / max(V, 1.0e-30)
    rm = MU0 * sigma * V * R

    print()
    print(f"  n_neutral           {n_neutral:.4e} m^-3")
    print(f"  nu_ei               {nu_ei:.4e} s^-1")
    print(f"  nu_en               {nu_en:.4e} s^-1   "
          f"({'neutral' if nu_en > nu_ei else 'ion'}-dominated)")
    print(f"  Z (ion charge)      {z_ion:.3f}")
    print(f"  sigma_ei alone      {sigma_ei:.4e} S/m")
    print()
    print(f"conductivity (both)   {sigma:.4e} S/m")
    print(f"magnetic diffusivity  {eta:.4e} m^2/s")
    print(f"electron collision    {tau_e:.4e} s")
    print()
    print(f"magnetic Reynolds     Rm = mu0 sigma |Rdot| R = {rm:.4e}")
    print(f"flux diffusion time   {t_diff:.4e} s")
    print(f"collapse time R/|Rdot|{t_coll:.4e} s")
    print(f"  ratio t_diff/t_coll {t_diff / t_coll:.4e}   "
          f"(needs >> 1 for freezing)")

    r_min = float(r[k_min])
    b_frozen = args.b0 * (R0 / r_min) ** 2
    p_mag = b_frozen**2 / (2.0 * MU0)
    omega_ce = E_CHARGE * b_frozen / M_E
    print()
    print(f"IF the flux were frozen, with B_0 = {args.b0:g} T:")
    print(f"  B at R_min          {b_frozen:.4e} T")
    print(f"  magnetic pressure   {p_mag:.4e} Pa")
    print(f"  p_mag / p_gas       {p_mag / p_gas:.4e}")
    print(f"  omega_ce tau_e      {omega_ce * tau_e:.4e}   "
          f"(needs >> 1 to magnetise)")

    r_needed = eta / max(V, 1.0e-30)
    t_hot = float(temp[k_hot]) / C.KELVIN_PER_EV
    sigma_best = 1.53e4 * t_hot**1.5 / (z_ion * ln_lambda)
    rm_best = MU0 * sigma_best * V * R
    print()
    print("upper bound, ignoring the conductivity model entirely:")
    print(f"  if FULLY ionised at T_peak = {t_hot:.3f} eV,")
    print(f"  sigma <= {sigma_best:.4e} S/m  ->  Rm <= {rm_best:.4e}")
    print(f"  still short of freezing by {1.0 / rm_best:.3e}x")

    print()
    print(f"flux freezing needs R >> eta/|Rdot| = {r_needed:.4e} m "
          f"= {r_needed * 1e3:.4f} mm")
    print(f"  the bubble is       {R:.4e} m, i.e. "
          f"{r_needed / R:.3e}x too small")

    b_ten_pc = math.sqrt(2.0 * MU0 * 0.1 * p_gas)
    print()
    print(f"for p_mag = 0.1 p_gas at R_min you would need B there = "
          f"{b_ten_pc:.3f} T,")
    print(f"  i.e. an applied B_0 of "
          f"{b_ten_pc / (R0 / r_min) ** 2:.4f} T")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
