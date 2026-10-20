from __future__ import annotations

import argparse
import math

MU0 = 4.0e-7 * math.pi
E = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
GAMMA = 5.0 / 3.0
IGNITION = 1.0e24


def spitzer(t_ev: float, z: float = 1.0, ln_lambda: float = 10.0) -> float:
    return 1.53e4 * t_ev**1.5 / (z * ln_lambda)


def compress(
    *,
    t0_ev: float,
    n0: float,
    r0: float,
    b0: float,
    conv: float,
    velocity: float,
    a_mass: float = 2.5,
    liner_rho: float = 9000.0,
) -> dict:
    rf = r0 / conv
    n = n0 * conv**2
    t_ev = t0_ev * conv ** (2.0 * (GAMMA - 1.0))
    b = b0 * conv**2

    c_s = math.sqrt(E * t_ev / (a_mass * M_P))
    tau = rf / c_s
    triple = n * t_ev * tau

    sigma = spitzer(t_ev)
    rm = MU0 * sigma * velocity * rf
    tau_e = 3.44e11 * t_ev**1.5 / (n * 10.0)
    mag = E * b / M_E * tau_e

    p_mag = b * b / (2.0 * MU0)
    p_ram = 0.5 * liner_rho * velocity**2

    return dict(
        rf=rf, n=n, t_ev=t_ev, b=b, tau=tau, triple=triple,
        rm=rm, mag=mag, p_mag=p_mag, p_ram=p_ram,
        ok_flux=rm > 10.0,
        ok_mag=mag > 10.0,
        ok_push=p_mag < p_ram,
        ratio=triple / IGNITION,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r0", type=float, default=3.0e-3,
                    help="initial fuel radius [m]")
    ap.add_argument("--n0", type=float, default=3.0e26,
                    help="initial fuel ion density [m^-3].  3e26 is D2 at about "
                    "1 mg/cm^3, the MagLIF fill; the first version of this "
                    "script defaulted to 1e25 and so understated every "
                    "triple product by 30x.")
    ap.add_argument("--b0", type=float, default=30.0,
                    help="seed axial field [T]")
    ap.add_argument("--velocity", type=float, default=1.0e5,
                    help="implosion velocity [m/s]")
    ap.add_argument("--liner-density", type=float, default=9000.0,
                    help="liner mass density [kg/m^3]")
    args = ap.parse_args()

    print(f"initial fuel: R_0 = {args.r0 * 1e3:.2f} mm, "
          f"n_0 = {args.n0:.2e} m^-3, B_0 = {args.b0:g} T")
    print(f"liner: v = {args.velocity:.2e} m/s, "
          f"rho = {args.liner_density:g} kg/m^3, "
          f"ram pressure = {0.5 * args.liner_density * args.velocity**2:.3e} Pa")
    print(f"target: n T tau >= {IGNITION:.1e} m^-3 eV s\n")

    preheats = (10.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
    convs = (5.0, 10.0, 15.0, 20.0, 30.0)

    print("n T tau / ignition, by preheat temperature and convergence ratio")
    print("  (f = flux leaks, m = electrons unmagnetised, "
          "p = field stops the liner)\n")
    header = f"{'T_0 [eV]':>9} |" + "".join(f"{f'C={c:g}':>13}" for c in convs)
    print(header)
    print("-" * len(header))
    best = None
    for t0 in preheats:
        row = f"{t0:>9.0f} |"
        for conv in convs:
            st = compress(
                t0_ev=t0, n0=args.n0, r0=args.r0, b0=args.b0,
                conv=conv, velocity=args.velocity,
                liner_rho=args.liner_density,
            )
            flags = ""
            if not st["ok_flux"]:
                flags += "f"
            if not st["ok_mag"]:
                flags += "m"
            if not st["ok_push"]:
                flags += "p"
            row += f"{st['ratio']:>10.2e}{flags:<3}"
            if st["ok_flux"] and st["ok_mag"] and st["ok_push"]:
                if best is None or st["ratio"] > best[0]["ratio"]:
                    best = (st, t0, conv)
        print(row)

    print()
    if best is None:
        print("No admissible point in this grid.")
    else:
        st, t0, conv = best
        print(f"best admissible point: preheat {t0:g} eV, convergence {conv:g}")
        print(f"  R_f                 {st['rf'] * 1e6:.1f} um")
        print(f"  n                   {st['n']:.3e} m^-3")
        print(f"  T                   {st['t_ev']:.3e} eV "
              f"({st['t_ev'] / 1e3:.2f} keV)")
        print(f"  B                   {st['b']:.3e} T")
        print(f"  tau                 {st['tau']:.3e} s")
        print(f"  n T tau             {st['triple']:.3e}  "
              f"= {st['ratio']:.3f} x ignition")
        print(f"  Rm                  {st['rm']:.3e}")
        print(f"  omega_ce tau_e      {st['mag']:.3e}")
        print(f"  p_mag / p_ram       {st['p_mag'] / st['p_ram']:.3e}")

    print()
    print("ceiling on the seed field, from p_mag(final) < p_ram:")
    p_ram = 0.5 * args.liner_density * args.velocity**2
    for conv in convs:
        b_max_final = math.sqrt(2.0 * MU0 * p_ram)
        print(f"  C = {conv:>4g}:  B_final <= {b_max_final:.3e} T"
              f"  ->  B_0 <= {b_max_final / conv**2:.3f} T")
    print()
    print("That ceiling is the MIF analogue of the finding in sec 8.5: a")
    print("30 T seed in a bubble would reach p_mag/p_gas = 651 and halt the")
    print("collapse instead of shaping it.  Here the liner's inertia is what")
    print("the field has to be weaker than, and a fast liner buys headroom.")
    t_adiabatic = 250.0 * 20.0 ** (2.0 * (GAMMA - 1.0))
    t_measured = 2.5e3
    over_t = t_adiabatic / t_measured
    over_triple = math.sqrt(over_t)
    print()
    print("calibration against a measured point:")
    print(f"  adiabatic T at T_0 = 250 eV, C = 20   {t_adiabatic / 1e3:.1f} keV")
    print(f"  MagLIF reports                        {t_measured / 1e3:.1f} keV")
    print(f"  so this model overstates T by         x{over_t:.1f}")
    print(f"  and n T tau by (T^1/2)                x{over_triple:.1f}")
    if best is not None:
        print(f"  best point corrected: "
              f"{best[0]['ratio'] / over_triple:.3f} x ignition, "
              f"i.e. a gap of x{over_triple / best[0]['ratio']:.0f}")
    print("  The radiation and conduction losses that cause this are not in")
    print("  the model, so the correction is a calibration and not a result.")

    print()
    print("closed-form scaling:  n T tau  ~  n_0 * T_0 * R_0 * C^(5/3)")
    print("  so convergence is the WEAKEST knob -- a factor of 10 in C buys")
    print("  46, while a factor of 10 in any of n_0, T_0 or R_0 buys 10 each")
    print("  and they multiply.  Reaching ignition scale from the best point")
    print("  above is therefore a target-design problem, not a drive problem.")
    if best is not None:
        need = IGNITION / best[0]["triple"]
        print()
        print(f"  gap from the best admissible point: x{need:.1f}")
        print(f"  e.g. n_0 x{need**0.5:.1f} and R_0 x{need**0.5:.1f} together,")
        print(f"  or C x{need**0.6:.1f} alone (which the field ceiling forbids).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
