# Active Closed-Loop Suppression of Hydrodynamic & Plasma Instabilities

**Author:** Timur Shakurov ([shakurtimur@gmail.com](mailto:shakurtimur@gmail.com))  
**Status:** Research & Benchmark Preview *(Full differentiable JAX codebase, 408-test verification suite, and policy checkpoints are held in a private repository prior to provisional patent filing; full 15-page manuscript attached to fellowship application / available upon request).*

---

## 1. Overview

In high-energy-density physics — from single-bubble sonoluminescence (SBSL) to magneto-inertial fusion — compression fails not from a lack of driver energy, but because **Rayleigh–Taylor shape instabilities** grow on nanosecond timescales and shatter symmetry before peak compression is reached. Under stationary constant-amplitude driving, core heating and interface distortion are locked in a single monotone trade-off.

This project builds a **differentiable multi-scale physics engine in JAX** and trains **closed-loop reinforcement learning (PPO) controllers** that actively suppress interface instabilities cycle-by-cycle — sustaining a **2.45 eV (~28,400 K)** plasma core across **2,400 consecutive collapses** with **100% survival** at distortion levels where constant drives fragment, while retaining **95.8% performance** through a physically realizable low-$Q$, high-voltage actuator ($Q = 12.2$, $2.6\text{ kV}$).

<p align="center">
  <img src="assets/basilisk_vof_collapse.gif" width="95%" alt="Basilisk 3D VOF Collapse Audit">
</p>

*Figure A: 3-D revolution of our independent axisymmetric compressible Navier–Stokes VOF audit in Basilisk (`seed = 0.20`), comparing viscous suppression (`single-valued`) vs. inviscid axial microjetting and interface folding (`JET PIERCE / FOLDED`, calibrating the physical breakup threshold at `|a_2|/R = 0.334 ± 0.109`).*

---

## 2. Differentiable Single-Bubble Plasma Simulator (JAX)

To resolve picosecond turnaround dynamics inside a 37.7 µs acoustic period without losing gradient information, the simulator integrates a coupled 9-state ODE/PDE system in `float64` on an $\mathrm{asinh}$-warped clock ($\tau = 2\text{ ns}$):
* **Compressible Liquid Closures:** Keller–Miksis and Gilmore–Kirkwood–Bethe equations with a Tait liquid equation of state and exact analytical Jacobians.
* **Non-Ideal Reacting Plasma Core:** Log-space Saha ionization ladder, diatomic dissociation, truncated vibrational partition functions, van der Waals hard-core excluded volume ($R_0/h \approx 8.86$), Toegel–Lohse thermal boundary layer, flux-limited Spitzer–Härm electron conductivity, optical escape factors, and dynamic interfacial water-vapor transport.
* **Interface Dynamics:** Coupled Plesset–Prosperetti $l=2$ quadrupole shape mode with thermal capillary wave seeding.

<p align="center">
  <img src="assets/sbsl_2d_warped.gif" width="49%" alt="2D Warped Collapse">
  <img src="assets/sbsl_3d_cutaway.gif" width="49%" alt="3D Cutaway Collapse">
</p>

*Figure B: Single-cycle collapse dynamics (`sbsl_argon_water_vapour`) on an $\mathrm{asinh}$-warped clock ($\tau = 2\text{ ns}$). Left: 2-D cross-section showing core plasma heating to $1.9\times 10^4\text{ K}$, the thin Toegel–Lohse conduction boundary layer (`layer ≈ 0.08 µm`), and the $(R/R_0, \dot{R}/c_l)$ phase portrait. Right: 3-D cutaway with $12\times$ exaggerated $l=2$ quadrupole deformation.*

---

## 3. Closed-Loop RL Control Results

### Breaking the Stationary Temperature–Distortion Trade-off
Evaluated across 12 held-out thermal capillary noise realizations over 40-cycle and 200-cycle (2,400-collapse) horizons, the learned PPO policy steps off the stationary baseline curve:

| Drive Regime | Survival Fraction | Episodic Return | Mean Peak $T$ [eV] | Peak Distortion $\vert{}a_2\vert{}/R$ |
| :--- | :---: | :---: | :---: | :---: |
| Constant $A_1 = 1.250\text{ atm}$ | 1.00 | $-0.51 \pm 1.10$ | 1.368 | 0.023 |
| Constant $A_1 = 1.375\text{ atm}$ | 0.67 | $+9.47 \pm 15.81$ | 2.046 | 0.218 |
| Constant $A_1 = 1.450\text{ atm}$ | 0.50 | $+8.07 \pm 20.74$ | 2.375 | 0.314 |
| Constant $A_1 = 1.550\text{ atm}$ | 0.08 | $-7.96 \pm 13.83$ | 2.757 | 0.683 |
| Constant $A_1 = 1.650\text{ atm}$ | 0.00 | $-12.13 \pm 0.65$ | 3.090 | 1.861 |
| **Closed-Loop PPO (Ours)** | **1.00** | **$+28.01 \pm 4.15$** | **2.449** | **0.144** |

<p align="center">
  <img src="assets/fig1_control.png" width="95%" alt="Control Trade-off">
</p>

### Mechanism Ablation & Realizable Actuator Bandwidth ($Q = 12.2$)
* **Feedback vs. Waveform Shaping:** A static 4-harmonic waveform fixed at the policy's mean commands gives $\vert{}a_2\vert{}/R = 0.774$ (0.50 survival), identical to a 1-harmonic drive ($0.780$). Open-loop replay of the policy's time schedule on a different noise seed reaches $0.409$ (0.67 survival). Only closed-loop state feedback reaches **0.144** with **1.00 survival**, modulating the fundamental amplitude ($\sigma(A_1) = 0.2492\text{ atm}$) while keeping higher harmonics nearly fixed.
* **2,400-Collapse Durability & Hardware Lag:** Over 200 control intervals (2,400 collapses), the policy completes 100% of episodes ($+151.56 \pm 22.18$ return at $2.53\text{ eV}$). Decomposing the action trace shows that per-cycle modulation requires a broadband transducer of **$Q = 12.2$** ($2.6\text{ kV}$ drive) rather than a $Q = 1500$ resonant flask; retrained with a first-order complex phasor lag in the loop, the controller retains **95.8% of its return** at $\vert{}a_2\vert{}/R = 0.033$.

<p align="center">
  <img src="assets/fig2_mechanism.png" width="48%" alt="Mechanism Ablation">
  <img src="assets/fig3_durability.png" width="48%" alt="Durability and Bandwidth">
</p>

---

## 4. Multi-Solver PDE Audit (Basilisk VOF, 1-D ALE Euler & 2-D Multimode Potential)

To ensure the RL policy exploits real hydrodynamics rather than ODE closure artifacts, we cross-validated the model against three higher-fidelity PDE/marker solvers:
1. **2-D Axisymmetric Compressible Navier–Stokes VOF (Basilisk):** Verified that one-way shape-to-volume coupling holds within $\Delta R / R \approx 0.047(a_2/R)^2$ (<0.1% at the policy operating point), showed that linear Plesset–Prosperetti theory over-predicts pre-folding amplitude, and calibrated the inviscid folding threshold at $\vert{}a_2\vert{}/R = 0.334 \pm 0.109$.
2. **2-D Nonlinear Multimode Harmonic Potential ($l=2\dots 12$ Clebsch–Gordan):** Confirmed that the linear folding threshold remains scale-invariant ($0.349 \pm 0.014$) across compression ratios up to $C = R_0 / R_{\min} = 50.4$.
3. **1-D Boundary-Fitted Spherical ALE Euler Solver (350 shells):** Decomposed the apparent 30.5% raw VOF radial offset at deep compression ($C = 69.3$), proving that two-thirds stems from 6-decimal volume-log rounding (7.6%) and Eulerian VOF grid starvation (11.7%), leaving a physical $+7.5\%$ to $+9.2\%$ Gilmore radiation-damping residual at realistic $\gamma \in [1.4, 5/3]$.

<p align="center">
  <img src="assets/fig4_audit.png" width="95%" alt="PDE Audit">
</p>

---

## 5. Generalization to Magneto-Inertial Fusion (Cylindrical MHD Liner Control)

To verify that the same closed-loop methodology generalizes beyond acoustic cavitation, the framework also implements a **1-D/2-D cylindrical Lagrangian MHD solver** (`core/liner.py`, `core/mrt.py`) for a magnetized liner implosion (MagLIF geometry) subject to the $m=0$ ("sausage") magneto-Rayleigh–Taylor instability on the outer current-driven surface, where azimuthal drive fields provide zero tension ($\mathbf{k}\cdot\mathbf{B}_\theta = 0$).

<p align="center">
  <img src="assets/mhd_liner_implosion.gif" width="95%" alt="MHD Liner Implosion Control">
</p>

*Figure C: Closed-loop PPO pulse shaping on a cylindrical MHD liner implosion, suppressing $m=0$ sausage wave-overturning (`SMOOTH < pi/2` under strict shaping vs. `OVERTURNED > pi/2` under standard driving).*

---

## Verification & Contact

* **Test Suite:** 408 automated unit, property, and analytical anchor tests (Rayleigh collapse, Lamb frequencies, Saha limits, Rankine–Hugoniot shock conditions, magnetic flux conservation, and bit-identical scalar/batched RL environment checks to $4\times 10^{-15}$).
* **Contact:** For access to the private codebase, full 15-page manuscript, or hardware testbed collaboration, reach out at **[shakurtimur@gmail.com](mailto:shakurtimur@gmail.com)**.
