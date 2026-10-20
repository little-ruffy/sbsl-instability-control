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

FLUIDS = {
    "water (pure)":   dict(sigma=5.5e-6, mu=1.00e-3, rho=998.0,  c=1483.0, st=0.0728),
    "seawater 3.5%":  dict(sigma=5.0,    mu=1.08e-3, rho=1025.0, c=1500.0, st=0.0740),
    "mercury":        dict(sigma=1.04e6, mu=1.53e-3, rho=13534.0, c=1450.0, st=0.4865),
    "Galinstan":      dict(sigma=3.46e6, mu=2.40e-3, rho=6440.0, c=2740.0, st=0.718),
}
Z_PZT = 7500.0 * 4000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="sbsl_argon_unstable")
    ap.add_argument("--b", type=float, default=10.0, help="applied field [T]")
    ap.add_argument("--cycles", type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_config(scenario=args.scenario)
    period = 1.0 / float(cfg.acoustics.frequency)
    traj = integrate(
        cfg, 0.0, args.cycles * period,
        save_at_steps=True, max_steps=20000, throw=True,
    )
    valid = np.asarray(traj.valid)
    t = np.asarray(traj.times)[valid]
    y = np.asarray(traj.states)[valid]
    o = np.argsort(t)
    t, y = t[o], y[o]
    r, temp, rdot = y[:, km.IDX_R], y[:, km.IDX_T], y[:, km.IDX_RDOT]

    k_fast = int(np.argmax(np.abs(rdot)))
    k_max = int(np.argmax(r))
    k_hot = int(np.argmax(temp))
    R_fast, V_fast = float(r[k_fast]), float(abs(rdot[k_fast]))
    R_max = float(r[k_max])
    B = args.b

    print("=" * 72)
    print("ROUTE 1   MHD on the surrounding liquid")
    print("=" * 72)
    print(f"applied field B = {B:g} T, bubble scale L = {R_max * 1e6:.1f} um "
          f"(R_max)\n")
    print(f"{'fluid':>16} {'sigma [S/m]':>12} {'Ha':>10} {'Z/Z_PZT':>9} "
          f"{'j for +1 atm':>14} {'ohmic [W/m^3]':>14}")
    for name, f in FLUIDS.items():
        ha = B * R_max * math.sqrt(f["sigma"] / f["mu"])
        z_ratio = f["rho"] * f["c"] / Z_PZT
        f_need = 101325.0 / 0.01
        j_need = f_need / B
        ohmic = j_need**2 / f["sigma"]
        print(f"{name:>16} {f['sigma']:>12.3e} {ha:>10.3e} {z_ratio:>9.3f} "
              f"{j_need:>14.3e} {ohmic:>14.3e}")

    print()
    print("Two separate objections, and they are not the same one.")
    print("  (a) Ha << 1 for every aqueous option, so Hartmann damping of")
    print("      vortices does nothing at bubble scale.  But Hartmann damping")
    print("      is aimed at the WRONG INSTABILITY in any case: the l = 2 mode")
    print("      here is a capillary-inertial interface mode driven by Rddot")
    print("      (Plesset-Prosperetti, THESIS 2.7), not bulk vorticity.")
    print("      Suppressing turbulence does not address it even in principle.")
    print("  (b) For an electrolyte the ohmic dissipation, not the force, is")
    print("      what forbids active compression.  See the last column.")

    print()
    print("=" * 72)
    print("ROUTE 2   Biermann battery (self-generated field)")
    print("=" * 72)
    counts = pc.vapour_counts(cfg, jnp.asarray(y[k_fast, km.IDX_NVAP]))
    v_free = pc.free_volume(jnp.asarray(R_fast), cfg, counts)
    comp = pc.composition(jnp.asarray(float(temp[k_fast])), v_free, cfg, counts)
    ln_lambda = float(pc.coulomb_logarithm(comp, cfg.transport))
    n_e = float(comp.n_e_density)
    T_K = float(temp[k_fast])
    t_ev = T_K / C.KELVIN_PER_EV
    z_ion = max(float(np.max(np.asarray(comp.z_mean))), 1.0)

    sigma_p = 1.53e4 * t_ev**1.5 / (z_ion * ln_lambda)
    eta = 1.0 / (MU0 * sigma_p)
    L = R_fast

    dbdt = (C.K_B * T_K / E_CHARGE) / L**2
    decay = eta / L**2
    b_sat = dbdt / decay
    print(f"  scale L               {L * 1e6:.3f} um")
    print(f"  T_e                   {t_ev:.3f} eV")
    print(f"  sigma (Spitzer)       {sigma_p:.4e} S/m")
    print(f"  eta                   {eta:.4e} m^2/s")
    print(f"  source dB/dt          {dbdt:.4e} T/s   (most generous estimate)")
    print(f"  resistive decay rate  {decay:.4e} 1/s")
    print(f"  saturated B           {b_sat:.4e} T")
    print(f"  p_mag at that B       {b_sat**2 / (2 * MU0):.4e} Pa")
    print()
    print("  The source is large -- it is the diffusion that wins, by the same")
    print("  factor of L^2 that killed flux freezing.  And the field is driven")
    print("  BY the asymmetry, so it can only appear once the mode it would")
    print("  have to suppress has already grown.")

    print()
    print("=" * 72)
    print("ROUTE 3   ECR pre-ionisation")
    print("=" * 72)
    f_ce = E_CHARGE * B / (2.0 * math.pi * M_E)
    omega = 2.0 * math.pi * f_ce
    n_crit = EPS0 * M_E * omega**2 / E_CHARGE**2

    n_gas_max = float(cfg.n_nuclei) / (4.0 / 3.0 * math.pi * R_max**3)
    frac_cutoff = n_crit / n_gas_max

    print(f"  B                     {B:g} T")
    print(f"  f_ce                  {f_ce / 1e9:.3f} GHz")
    print(f"  critical density      {n_crit:.4e} m^-3")
    print(f"  gas density at R_max  {n_gas_max:.4e} m^-3")
    print(f"  -> cutoff at an ionisation fraction of {frac_cutoff:.3e}")
    print(f"  n_e at peak |Rdot|    {n_e:.4e} m^-3, i.e. "
          f"{n_e / n_crit:.3e}x over cutoff")
    print()
    print(f"  The wave is reflected once {frac_cutoff * 100:.2f} % of the gas is")
    print("  ionised, which happens at the very start of the compression.")
    print("  Microwave heating is therefore available only while the bubble")
    print("  is cold and large, which is when it is least useful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
