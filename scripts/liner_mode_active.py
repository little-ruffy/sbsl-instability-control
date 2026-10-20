from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from core import liner, mrt

R_SPECIFIC = 8.314462618 / 2.014e-3
GAMMA = 5.0 / 3.0


def _rk4(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r0", type=float, default=3.0e-3)
    ap.add_argument("--rho0", type=float, default=1.0,
                    help="fuel density [kg/m^3]; 1.0 is D2 at ~1 mg/cm^3")
    ap.add_argument("--t0-ev", type=float, default=250.0)
    ap.add_argument("--b0", type=float, default=30.0)
    ap.add_argument("--wavelength", type=float, default=5.0e-4,
                    help="axial perturbation wavelength [m]")
    ap.add_argument("--xi0-rel", type=float, default=1.0e-3,
                    help="seed amplitude as a fraction of R_0")
    ap.add_argument("--liner-mass", type=float, default=1.0e-2,
                    help="liner mass per unit length [kg/m]")
    ap.add_argument("--drive", type=float, default=5.0e12,
                    help="drive pressure [Pa]")
    ap.add_argument(
        "--surface", choices=("inner", "outer"), default="outer",
        help="which interface to track.  `inner` is fuel/liner, stabilised "
        "by the compressed axial field; `outer` is liner/vacuum, where the "
        "azimuthal drive field gives no tension for m = 0 and the mode is "
        "live.  Defaults to the one with a control problem on it.",
    )
    ap.add_argument("--n-fuel", type=int, default=12)
    ap.add_argument("--n-liner", type=int, default=8)
    ap.add_argument("--n-zones", type=int, default=24)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--dt", type=float, default=2.0e-12)
    args = ap.parse_args()

    mass = args.rho0 * math.pi * args.r0**2
    outer = args.surface == "outer"
    if outer:
        p = liner.LinerParams(
            gamma=GAMMA, n_zones=args.n_fuel + args.n_liner,
            mass_per_length=mass, conductivity=0.0,
            liner_mass_per_length=args.liner_mass,
            n_fuel=args.n_fuel, liner_density=9000.0,
        )
        frac = liner.two_region_fractions(p, args.r0)
    else:
        p = liner.LinerParams(
            gamma=GAMMA, n_zones=args.n_zones, mass_per_length=mass,
            conductivity=0.0,
            liner_mass_per_length=args.liner_mass,
        )
        frac = None
    st = liner.initial_state(
        p, radius=args.r0, temperature=args.t0_ev * 11604.5,
        b_axial=args.b0, gas_constant=R_SPECIFIC, fractions=frac,
    )

    k = 2.0 * math.pi / args.wavelength
    rho_liner = 9000.0
    atwood = (rho_liner - args.rho0) / (rho_liner + args.rho0)
    m = mrt.MrtParams(
        wavenumber=k,
        atwood=1.0 if outer else atwood,
        density_sum=rho_liner if outer else rho_liner + args.rho0,
    )

    rhs = jax.jit(
        lambda y: liner.rhs(
            y, p, jnp.asarray(args.drive), R_SPECIFIC, fractions=frac
        )
    )
    fields = jax.jit(
        lambda y: liner.zone_fields(y, p, R_SPECIFIC, fractions=frac)
    )

    n = p.n_zones
    xi = args.xi0_rel * args.r0
    xi_dot = 0.0
    xi_seed = xi

    hist = []
    y = st
    stagnated = False
    peak_rel = args.xi0_rel
    peak_bracket = -np.inf
    for i in range(args.steps):
        d = rhs(y)
        r_if = float(y[n - 1])
        u_if = float(y[2 * n - 1])
        a_if = float(d[2 * n - 1])
        f = fields(y)
        b_fuel = float(f["b_field"][-1])

        g = float(
            mrt.effective_gravity(jnp.asarray(a_if), heavy_outside=not outer)
        )
        b_for_tension = 0.0 if outer else b_fuel
        bracket = float(
            mrt.growth_bracket(m, jnp.asarray(g), jnp.asarray(b_for_tension))
        )
        conv = u_if / max(r_if, 1.0e-300)
        xi, xi_dot = (
            xi + args.dt * xi_dot,
            xi_dot + args.dt * (bracket * xi - conv * xi_dot),
        )
        rel = abs(xi) / max(r_if, 1.0e-300)
        peak_rel = max(peak_rel, rel)
        peak_bracket = max(peak_bracket, bracket)
        turning = u_if > 0.0 and not stagnated
        if turning:
            stagnated = True
        if i % max(args.steps // 12, 1) == 0 or turning or i == args.steps - 1:
            hist.append(
                (i * args.dt, r_if, u_if, a_if, b_fuel, bracket, rel, turning)
            )
        y = _rk4(rhs, y, args.dt)
        if r_if < 0.02 * args.r0 or not np.isfinite(float(y[n - 1])):
            hist.append(
                (i * args.dt, r_if, u_if, a_if, b_fuel, bracket, rel, False)
            )
            print(f"  [stopped at step {i}: "
                  f"{'radius floor' if r_if < 0.02 * args.r0 else 'non-finite'}]")
            break

    print(f"fuel: rho_0 = {args.rho0:g} kg/m^3, T_0 = {args.t0_ev:g} eV, "
          f"R_0 = {args.r0 * 1e3:.2f} mm")
    print(f"field: B_0 = {args.b0:g} T    drive = {args.drive:.2e} Pa")
    print(f"surface: {args.surface}"
          f"{'  (no magnetic tension for m=0)' if outer else ''}")
    print(f"mode: lambda = {args.wavelength * 1e3:.3f} mm, k = {k:.3e} 1/m, "
          f"A = {atwood:.4f}, seed |xi|/R = {args.xi0_rel:.1e}\n")

    hdr = (f"{'t [ns]':>9} {'R [um]':>9} {'u [km/s]':>10} {'g [m/s^2]':>12} "
           f"{'B [T]':>10} {'bracket':>12} {'|xi|/R':>11}  phase")
    print(hdr)
    print("-" * len(hdr))
    for t, r, u, a, b, br, rel, turning in hist:
        tag = " <- STAGNATION" if turning else (
            "  decelerating" if a > 0 else "  accelerating")
        print(f"{t * 1e9:>9.3f} {r * 1e6:>9.2f} {u / 1e3:>10.2f} "
              f"{a:>12.3e} {b:>10.2f} {br:>12.3e} {rel:>11.3e}{tag}")

    rels = [h[6] for h in hist]
    brackets = [peak_bracket]
    print()
    print(f"reached stagnation?                     "
          f"{'YES' if stagnated else 'NO -- the unstable phase was never entered'}")
    print(f"growth bracket positive at any STEP?    "
          f"{'YES' if peak_bracket > 0 else 'NO'}  "
          f"(peak {peak_bracket:+.3e})")
    print(f"|xi|/R : seed {args.xi0_rel:.3e} -> peak {peak_rel:.3e}  "
          f"(x{peak_rel / args.xi0_rel:.3e})")

    b_fin = hist[-1][4]
    g_fin = hist[-1][3]
    if g_fin > 0:
        b_crit = float(mrt.field_for_cutoff(m, jnp.asarray(g_fin)))
        print(f"critical field at the last sample: {b_crit:.1f} T "
              f"against an actual {b_fin:.1f} T  -> "
              f"{'field WINS, mode held' if b_fin > b_crit else 'mode grows'}")
    print()
    print("A control problem exists only if the bracket goes positive AND")
    print("|xi|/R moves by orders.  If it does not, this is the R_0 = 4.5 um")
    print("situation of section 7.6.1 and training here would learn nothing")
    print("but to drive harder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
