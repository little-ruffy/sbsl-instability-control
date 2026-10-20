from __future__ import annotations

import math
from typing import Final

# CODATA 2018 fundamental physical constants (SI)
K_B: Final[float] = 1.380649e-23
H_PLANCK: Final[float] = 6.62607015e-34
HBAR: Final[float] = H_PLANCK / (2.0 * math.pi)
N_AVOGADRO: Final[float] = 6.02214076e23
R_GAS: Final[float] = K_B * N_AVOGADRO
ELEMENTARY_CHARGE: Final[float] = 1.602176634e-19
M_ELECTRON: Final[float] = 9.1093837015e-31
M_PROTON: Final[float] = 1.67262192369e-27
M_NEUTRON: Final[float] = 1.67492749804e-27
# nuclear masses and D-D branch energies
M_DEUTERON: Final[float] = 3.3435837724e-27
ATOMIC_MASS_UNIT: Final[float] = 1.66053906660e-27
C_LIGHT: Final[float] = 299792458.0
EPSILON_0: Final[float] = 8.8541878128e-12
# radiation and Spitzer transport constants
SIGMA_SB: Final[float] = 5.670374419e-8
EV_TO_J: Final[float] = ELEMENTARY_CHARGE
KEV_TO_J: Final[float] = 1.0e3 * EV_TO_J
MEV_TO_J: Final[float] = 1.0e6 * EV_TO_J
KELVIN_PER_EV: Final[float] = EV_TO_J / K_B
KELVIN_PER_KEV: Final[float] = 1.0e3 * KELVIN_PER_EV
ATM_TO_PA: Final[float] = 101325.0
BAR_TO_PA: Final[float] = 1.0e5
BREMSSTRAHLUNG_COEFF_CGS: Final[float] = 1.69e-32
BREMSSTRAHLUNG_COEFF_SI: Final[float] = BREMSSTRAHLUNG_COEFF_CGS * 1.0e6 * 1.0e-12
SPITZER_KAPPA_COEFF: Final[float] = 2.64
RAYLEIGH_COLLAPSE_CONST: Final[float] = 0.9146813565019625
Q_DD_N3HE_MEV: Final[float] = 3.269
Q_DD_PT_MEV: Final[float] = 4.033
E_DD_NEUTRON_MEV: Final[float] = 2.45
E_CHARGED_DD_N3HE_MEV: Final[float] = Q_DD_N3HE_MEV - E_DD_NEUTRON_MEV
E_CHARGED_DD_PT_MEV: Final[float] = Q_DD_PT_MEV


__all__ = [
    "K_B",
    "H_PLANCK",
    "HBAR",
    "N_AVOGADRO",
    "R_GAS",
    "ELEMENTARY_CHARGE",
    "M_ELECTRON",
    "M_PROTON",
    "M_NEUTRON",
    "M_DEUTERON",
    "ATOMIC_MASS_UNIT",
    "C_LIGHT",
    "EPSILON_0",
    "SIGMA_SB",
    "EV_TO_J",
    "KEV_TO_J",
    "MEV_TO_J",
    "KELVIN_PER_EV",
    "KELVIN_PER_KEV",
    "ATM_TO_PA",
    "BAR_TO_PA",
    "BREMSSTRAHLUNG_COEFF_CGS",
    "BREMSSTRAHLUNG_COEFF_SI",
    "SPITZER_KAPPA_COEFF",
    "RAYLEIGH_COLLAPSE_CONST",
    "Q_DD_N3HE_MEV",
    "Q_DD_PT_MEV",
    "E_DD_NEUTRON_MEV",
    "E_CHARGED_DD_N3HE_MEV",
    "E_CHARGED_DD_PT_MEV",
]
