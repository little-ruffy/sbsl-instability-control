from __future__ import annotations

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
E = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27


def spitzer(t_ev: float, z: float = 1.0, ln_lambda: float = 10.0) -> float:
    return 1.53e4 * t_ev**1.5 / (z * ln_lambda)


def report(
    label: str,
    *,
    radius: float,
    velocity: float,
    t_ev: float,
    n_i: float,
    b_field: float,
    a_mass: float = 2.5,
) -> dict:
    sigma = spitzer(t_ev)
    eta = 1.0 / (MU0 * sigma)
    rm = MU0 * sigma * velocity * radius

    c_s = math.sqrt(E * t_ev / (a_mass * M_P))
    tau = radius / c_s
    triple = n_i * t_ev * tau

    tau_e = 3.44e11 * t_ev**1.5 / (n_i * 10.0)
    omega_ce = E * b_field / M_E

    print(f"\n{label}")
    print(f"  R                   {radius:.4e} m")
    print(f"  implosion speed     {velocity:.4e} m/s")
    print(f"  T                   {t_ev:.4e} eV")
    print(f"  n_i                 {n_i:.4e} m^-3")
    print(f"  B                   {b_field:.4e} T")
    print(f"  sigma (Spitzer)     {sigma:.4e} S/m")
    print(f"  eta                 {eta:.4e} m^2/s")
    print(f"  Rm                  {rm:.4e}   {'FROZEN' if rm > 10 else 'leaks'}")
    print(f"  c_s                 {c_s:.4e} m/s")
    print(f"  tau = R/c_s         {tau:.4e} s")
    print(f"  n T tau             {triple:.4e} m^-3 eV s")
    print(f"  omega_ce tau_e      {omega_ce * tau_e:.4e}   "
          f"{'MAGNETISED' if omega_ce * tau_e > 10 else 'not magnetised'}")
    return dict(rm=rm, triple=triple, sigma=sigma, t_ev=t_ev,
                radius=radius, velocity=velocity,
                mag=omega_ce * tau_e)


def main() -> int:
    cfg = load_config(scenario="sbsl_argon_unstable")
    period = 1.0 / float(cfg.acoustics.frequency)
    traj = integrate(
        cfg, 0.0, 3.0 * period,
        save_at_steps=True, max_steps=40000, throw=True,
    )
    v = np.asarray(traj.valid)
    y = np.asarray(traj.states)[v]
    y = y[np.argsort(np.asarray(traj.times)[v])]
    r, temp, rdot = y[:, km.IDX_R], y[:, km.IDX_T], y[:, km.IDX_RDOT]

    k = int(np.argmax(np.abs(rdot)))
    counts = pc.vapour_counts(cfg, jnp.asarray(y[k, km.IDX_NVAP]))
    v_free = pc.free_volume(jnp.asarray(float(r[k])), cfg, counts)
    comp = pc.composition(jnp.asarray(float(temp[k])), v_free, cfg, counts)
    n_heavy = float(comp.n_heavy_density)

    print("=" * 70)
    print("MIF / FRC against this simulator's bubble")
    print("=" * 70)

    bub = report(
        "SBSL bubble (measured, at peak wall speed)",
        radius=float(r[k]),
        velocity=float(abs(rdot[k])),
        t_ev=float(temp[k]) / C.KELVIN_PER_EV,
        n_i=n_heavy,
        b_field=30.0 * (float(cfg.r0) / float(r.min())) ** 2,
        a_mass=40.0,
    )

    maglif_pre = report(
        "MagLIF-like, at laser preheat (before implosion)",
        radius=2.5e-3, velocity=1.0e5, t_ev=200.0,
        n_i=1.0e26, b_field=1.0e4,
    )
    maglif = report(
        "MagLIF-like, AT PEAK COMPRESSION",
        radius=1.25e-4, velocity=1.0e5, t_ev=2.5e3,
        n_i=3.0e28, b_field=1.0e4,
    )

    gf_pre = report(
        "liquid-liner FRC, as formed (before compression)",
        radius=0.3, velocity=1.0e3, t_ev=100.0,
        n_i=1.0e22, b_field=10.0,
    )
    gf = report(
        "liquid-liner FRC, AT PEAK COMPRESSION",
        radius=0.03, velocity=1.0e3, t_ev=1.0e4,
        n_i=1.0e25, b_field=1.0e3,
    )

    icf = report(
        "ICF hot spot, for scale (no field)",
        radius=3.0e-5, velocity=3.0e5, t_ev=5.0e3,
        n_i=1.0e31, b_field=0.0,
    )

    print("\n" + "=" * 70)
    print("Where MIF's Rm comes from, against the bubble")
    print("=" * 70)
    for name, tgt in (("MagLIF-like (compressed)", maglif),
                      ("liquid-liner FRC (compressed)", gf)):
        g_t = tgt["sigma"] / bub["sigma"]
        g_r = tgt["radius"] / bub["radius"]
        g_v = tgt["velocity"] / bub["velocity"]
        print(f"\n  {name}:")
        print(f"    from temperature (sigma ~ T^1.5)   x {g_t:.3e}")
        print(f"    from scale       (Rm ~ R)          x {g_r:.3e}")
        print(f"    from speed       (Rm ~ v)          x {g_v:.3e}")
        print(f"    product                            x {g_t * g_r * g_v:.3e}")
        print(f"    bubble Rm x that                   = "
              f"{bub['rm'] * g_t * g_r * g_v:.3e}  "
              f"(direct: {tgt['rm']:.3e})")

    print("\n" + "=" * 70)
    print("Triple product, against the bubble")
    print("=" * 70)
    print(f"  {'target':>42} {'n T tau':>12} {'vs bubble':>12}")
    for name, tgt in (
        ("SBSL bubble, at collapse", bub),
        ("MagLIF-like, at preheat", maglif_pre),
        ("MagLIF-like, compressed", maglif),
        ("liquid-liner FRC, as formed", gf_pre),
        ("liquid-liner FRC, compressed", gf),
        ("ICF hot spot", icf),
    ):
        print(f"  {name:>42} {tgt['triple']:>12.3e} "
              f"{tgt['triple'] / bub['triple']:>12.3e}")
    print()
    print("  A D-T triple product of order 1e21 m^-3 keV s -- i.e. ~1e24 in")
    print("  the m^-3 eV s used here -- is the usual ignition-scale figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
