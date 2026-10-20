from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

import equinox as eqx
import jax.numpy as jnp
import yaml
from jaxtyping import Array, Float

from core import constants as C


_DEFAULT_CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "config" / "physical_params.yaml"
)

ConductionModel = Literal["none", "neutral_power_law", "blended_spitzer"]
SolverName = Literal["Tsit5", "Dopri8", "Kvaerno3", "Kvaerno5"]
GasPressureMode = Literal["laplace", "ambient", "explicit"]


def _f(x: Any) -> Float[Array, ""]:
    return jnp.asarray(float(x), dtype=jnp.float64)


def _fa(x: Sequence[float]) -> Float[Array, " n"]:
    return jnp.asarray([float(v) for v in x], dtype=jnp.float64)


def _deep_set(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node: Any = tree
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"override path {dotted_key!r}: no such key {p!r}")
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"override path {dotted_key!r}: no such leaf {parts[-1]!r}")
    node[parts[-1]] = value


# host liquid properties (SI)
class LiquidParams(eqx.Module):
    name: str = eqx.field(static=True)
    density: Float[Array, ""]
    sound_speed: Float[Array, ""]
    surface_tension: Float[Array, ""]
    dynamic_viscosity: Float[Array, ""]
    ambient_pressure: Float[Array, ""]
    ambient_temperature: Float[Array, ""]
    vapour_pressure: Float[Array, ""]
    tait_B: Float[Array, ""]
    tait_n: Float[Array, ""]
    thermal_conductivity: Float[Array, ""]
    specific_heat_cp: Float[Array, ""]

    @property
    def kinematic_viscosity(self) -> Float[Array, ""]:
        return self.dynamic_viscosity / self.density

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LiquidParams":
        return cls(
            name=str(d["name"]),
            density=_f(d["density"]),
            sound_speed=_f(d["sound_speed"]),
            surface_tension=_f(d["surface_tension"]),
            dynamic_viscosity=_f(d["dynamic_viscosity"]),
            ambient_pressure=_f(d["ambient_pressure"]),
            ambient_temperature=_f(d["ambient_temperature"]),
            vapour_pressure=_f(d["vapour_pressure"]),
            tait_B=_f(d["tait_B"]),
            tait_n=_f(d["tait_n"]),
            thermal_conductivity=_f(d["thermal_conductivity"]),
            specific_heat_cp=_f(d["specific_heat_cp"]),
        )


# dense per-species thermochemical arrays
class GasMixture(eqx.Module):
    names: tuple[str, ...] = eqx.field(static=True)
    mole_frac: Float[Array, " S"]
    n_atoms: Float[Array, " S"]
    molar_mass_atomic: Float[Array, " S"]
    vdw_b: Float[Array, " S"]
    e_diss: Float[Array, " S"]
    theta_rot: Float[Array, " S"]
    theta_vib: Float[Array, " S"]
    diss_g_ratio: Float[Array, " S"]
    ion_pot: Float[Array, "S Z"]
    ion_mask: Float[Array, "S Z"]
    saha_g_ratio: Float[Array, " S"]
    is_deuteron: Float[Array, " S"]
    kappa_ref: Float[Array, " S"]
    kappa_T_ref: Float[Array, " S"]
    kappa_exponent: Float[Array, " S"]
    free_rot_dof: Float[Array, " S"]
    free_theta_rot: Float[Array, " S"]
    free_theta_vib: Float[Array, "S V"]
    free_vib_mask: Float[Array, "S V"]
    free_e_bind: Float[Array, " S"]

    @property
    def n_species(self) -> int:
        return len(self.names)

    @property
    def mass_per_nucleus(self) -> Float[Array, " S"]:
        return self.molar_mass_atomic / C.N_AVOGADRO

    @property
    def hardcore_volume_per_nucleus_species(self) -> Float[Array, " S"]:
        return self.vdw_b / (self.n_atoms * C.N_AVOGADRO)

    @property
    def molecules_per_nucleus(self) -> Float[Array, ""]:
        return jnp.sum(self.mole_frac / self.n_atoms)

    @property
    def hardcore_volume_per_nucleus(self) -> Float[Array, ""]:
        return jnp.sum(
            self.mole_frac * self.vdw_b / (self.n_atoms * C.N_AVOGADRO)
        )

    @classmethod
    def from_dict(
        cls,
        species_db: Mapping[str, Mapping[str, Any]],
        mixture: Mapping[str, float],
        extra_species: tuple[str, ...] = (),
    ) -> "GasMixture":
        mixture = dict(mixture)
        for name in extra_species:
            if name not in mixture:
                mixture[name] = 0.0
        names = tuple(mixture.keys())
        if not names:
            raise ValueError("gas mixture must contain at least one species")
        missing = [n for n in names if n not in species_db]
        if missing:
            raise KeyError(f"unknown gas species {missing!r}")

        raw = [species_db[n] for n in names]
        total = sum(float(v) for v in mixture.values())
        if total <= 0.0:
            raise ValueError("gas mixture mole fractions must sum to > 0")

        depths = []
        for spec in raw:
            full = len(spec["ionisation_potentials_eV"])
            cap = spec.get("max_ionisation_stages")
            depths.append(full if cap is None else min(int(cap), full))
        z_max = max(depths)
        ion_pot = jnp.zeros((len(names), z_max), dtype=jnp.float64)
        ion_mask = jnp.zeros((len(names), z_max), dtype=jnp.float64)
        for i, (spec, depth) in enumerate(zip(raw, depths, strict=False)):
            pots = [
                float(v) * C.EV_TO_J
                for v in spec["ionisation_potentials_eV"][:depth]
            ]
            ion_pot = ion_pot.at[i, : len(pots)].set(jnp.asarray(pots))
            ion_mask = ion_mask.at[i, : len(pots)].set(1.0)

        vib_lists = [
            [float(v) for v in (spec.get("free_theta_vib_K") or [])]
            for spec in raw
        ]
        n_vib = max(1, max((len(v) for v in vib_lists), default=1))
        free_theta_vib = jnp.ones((len(names), n_vib), dtype=jnp.float64)
        free_vib_mask = jnp.zeros((len(names), n_vib), dtype=jnp.float64)
        for i, vib in enumerate(vib_lists):
            for j, theta in enumerate(vib):
                free_theta_vib = free_theta_vib.at[i, j].set(max(theta, 1.0e-6))
                free_vib_mask = free_vib_mask.at[i, j].set(1.0)

        return cls(
            names=names,
            mole_frac=_fa([float(mixture[n]) / total for n in names]),
            n_atoms=_fa([s["n_atoms_per_molecule"] for s in raw]),
            molar_mass_atomic=_fa([s["molar_mass_atomic"] for s in raw]),
            vdw_b=_fa([s["vdw_b"] for s in raw]),
            e_diss=_fa(
                [float(s["dissociation_energy_eV"]) * C.EV_TO_J for s in raw]
            ),
            theta_rot=jnp.maximum(_fa([s["theta_rot_K"] for s in raw]), 1.0e-6),
            theta_vib=jnp.maximum(_fa([s["theta_vib_K"] for s in raw]), 1.0e-6),
            diss_g_ratio=_fa([s["dissociation_g_ratio"] for s in raw]),
            ion_pot=ion_pot,
            ion_mask=ion_mask,
            saha_g_ratio=_fa([s["degeneracy_ratio"] for s in raw]),
            is_deuteron=_fa(
                [1.0 if bool(s["provides_deuterons"]) else 0.0 for s in raw]
            ),
            free_rot_dof=_fa(
                [float(s.get("free_rotational_dof", 0.0)) for s in raw]
            ),
            free_theta_rot=jnp.maximum(
                _fa([float(s.get("free_theta_rot_K", 0.0)) for s in raw]),
                1.0e-6,
            ),
            free_theta_vib=free_theta_vib,
            free_vib_mask=free_vib_mask,
            free_e_bind=_fa(
                [
                    float(s.get("free_binding_energy_eV", 0.0)) * C.EV_TO_J
                    for s in raw
                ]
            ),
            kappa_ref=_fa([s["kappa_ref"] for s in raw]),
            kappa_T_ref=_fa([s["kappa_T_ref"] for s in raw]),
            kappa_exponent=_fa([s["kappa_exponent"] for s in raw]),
        )


class TransportParams(eqx.Module):
    conduction_enabled: bool = eqx.field(static=True)
    conduction_model: ConductionModel = eqx.field(static=True)
    bremsstrahlung_enabled: bool = eqx.field(static=True)
    blackbody_cap_enabled: bool = eqx.field(static=True)
    dissociation_enabled: bool = eqx.field(static=True)
    ionisation_enabled: bool = eqx.field(static=True)
    vibrational_dof_enabled: bool = eqx.field(static=True)
    flux_limiter_enabled: bool = eqx.field(static=True)
    saha_bisection_iters: int = eqx.field(static=True)
    saha_newton_iters: int = eqx.field(static=True)

    boundary_layer_cap_factor: Float[Array, ""]
    coulomb_log_min: Float[Array, ""]
    coulomb_log_max: Float[Array, ""]
    flux_limiter_factor: Float[Array, ""]
    bremsstrahlung_gaunt: Float[Array, ""]
    blackbody_emissivity: Float[Array, ""]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TransportParams":
        model = str(d["conduction_model"])
        if model not in ("none", "neutral_power_law", "blended_spitzer"):
            raise ValueError(f"unknown conduction_model {model!r}")
        return cls(
            conduction_enabled=bool(d["conduction_enabled"]),
            conduction_model=cast(ConductionModel, model),
            bremsstrahlung_enabled=bool(d["bremsstrahlung_enabled"]),
            blackbody_cap_enabled=bool(d["blackbody_cap_enabled"]),
            dissociation_enabled=bool(d["dissociation_enabled"]),
            ionisation_enabled=bool(d["ionisation_enabled"]),
            vibrational_dof_enabled=bool(d["vibrational_dof_enabled"]),
            flux_limiter_enabled=bool(d["flux_limiter_enabled"]),
            saha_bisection_iters=int(d["saha_bisection_iters"]),
            saha_newton_iters=int(d["saha_newton_iters"]),
            boundary_layer_cap_factor=_f(d["boundary_layer_cap_factor"]),
            coulomb_log_min=_f(d["coulomb_log_min"]),
            coulomb_log_max=_f(d["coulomb_log_max"]),
            flux_limiter_factor=_f(d["flux_limiter_factor"]),
            bremsstrahlung_gaunt=_f(d["bremsstrahlung_gaunt"]),
            blackbody_emissivity=_f(d["blackbody_emissivity"]),
        )


class FusionParams(eqx.Module):
    enabled: bool = eqx.field(static=True)
    reactions: tuple[str, ...] = eqx.field(static=True)
    charged_product_deposition_fraction: Float[Array, ""]
    t_valid_min_keV: Float[Array, ""]
    t_valid_max_keV: Float[Array, ""]
    theta_denominator_floor: Float[Array, ""]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FusionParams":
        return cls(
            enabled=bool(d["enabled"]),
            reactions=tuple(str(r) for r in d["reactions"]),
            charged_product_deposition_fraction=_f(
                d["charged_product_deposition_fraction"]
            ),
            t_valid_min_keV=_f(d["t_valid_min_keV"]),
            t_valid_max_keV=_f(d["t_valid_max_keV"]),
            theta_denominator_floor=_f(d["theta_denominator_floor"]),
        )


class ShapeParams(eqx.Module):
    enabled: bool = eqx.field(static=True)
    mode_l: int = eqx.field(static=True)
    viscous_damping_enabled: bool = eqx.field(static=True)
    damping_coeff_c1: Float[Array, ""]
    damping_coeff_c2: Float[Array, ""]
    a_l_init_rel: Float[Array, ""]
    adot_l_init: Float[Array, ""]
    instability_threshold: Float[Array, ""]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ShapeParams":
        mode_l = int(d["mode_l"])
        if mode_l < 2:
            raise ValueError(
                f"shape.mode_l must be >= 2 (l=0 is the breathing mode and l=1 "
                f"is a rigid translation); got {mode_l}"
            )
        return cls(
            enabled=bool(d["enabled"]),
            mode_l=mode_l,
            viscous_damping_enabled=bool(d["viscous_damping_enabled"]),
            damping_coeff_c1=_f(d["damping_coeff_c1"]),
            damping_coeff_c2=_f(d["damping_coeff_c2"]),
            a_l_init_rel=_f(d["a_l_init_rel"]),
            adot_l_init=_f(d["adot_l_init"]),
            instability_threshold=_f(d["instability_threshold"]),
        )


class VapourParams(eqx.Module):
    enabled: bool = eqx.field(static=True)
    species_index: int = eqx.field(static=True)
    diffusivity_ref: Float[Array, ""]
    density_ref: Float[Array, ""]

    @classmethod
    def from_dict(
        cls, d: Mapping[str, Any] | None, names: tuple[str, ...]
    ) -> "VapourParams":
        if not d or not bool(d.get("enabled", False)):
            return cls(
                enabled=False,
                species_index=-1,
                diffusivity_ref=_f(2.4e-5),
                density_ref=_f(2.5e25),
            )
        species = str(d["species"])
        if species not in names:
            raise KeyError(
                f"vapour.species {species!r} is not in the gas mixture "
                f"{names!r}; add it to the scenario's gas_mixture (a zero "
                f"fraction is fine) or disable vapour"
            )
        return cls(
            enabled=True,
            species_index=names.index(species),
            diffusivity_ref=_f(d["diffusivity_ref"]),
            density_ref=_f(d["density_ref"]),
        )


class ResonatorParams(eqx.Module):
    enabled: bool = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    flask_radius: Float[Array, ""]
    quality_factor: Float[Array, ""]
    piezo_mass: Float[Array, ""]
    piezo_resonance: Float[Array, ""]
    piezo_damping_ratio: Float[Array, ""]
    force_factor: Float[Array, ""]
    drive_area_fraction: Float[Array, ""]

    @property
    def drive_area(self) -> Float[Array, ""]:
        return self.drive_area_fraction * 4.0 * jnp.pi * self.flask_radius**2

    @property
    def piezo_omega(self) -> Float[Array, ""]:
        return 2.0 * jnp.pi * self.piezo_resonance

    @property
    def piezo_stiffness(self) -> Float[Array, ""]:
        return self.piezo_mass * self.piezo_omega**2

    @property
    def piezo_damping(self) -> Float[Array, ""]:
        return 2.0 * self.piezo_damping_ratio * self.piezo_mass * self.piezo_omega

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "ResonatorParams":
        if d is None:
            return cls(
                enabled=False,
                n_modes=1,
                flask_radius=_f(0.03),
                quality_factor=_f(1000.0),
                piezo_mass=_f(0.05),
                piezo_resonance=_f(25000.0),
                piezo_damping_ratio=_f(0.02),
                force_factor=_f(1.0),
                drive_area_fraction=_f(0.02),
            )
        n_modes = int(d["n_modes"])
        if n_modes < 1:
            raise ValueError(f"resonator.n_modes must be >= 1; got {n_modes}")
        return cls(
            enabled=bool(d["enabled"]),
            n_modes=n_modes,
            flask_radius=_f(d["flask_radius"]),
            quality_factor=_f(d["quality_factor"]),
            piezo_mass=_f(d["piezo_mass"]),
            piezo_resonance=_f(d["piezo_resonance"]),
            piezo_damping_ratio=_f(d["piezo_damping_ratio"]),
            force_factor=_f(d["force_factor"]),
            drive_area_fraction=_f(d["drive_area_fraction"]),
        )


class AcousticParams(eqx.Module):
    frequency: Float[Array, ""]
    amplitudes: Float[Array, " K"]
    phases: Float[Array, " K"]
    ramp_cycles: Float[Array, ""]

    @property
    def n_harmonics(self) -> int:
        return int(self.amplitudes.shape[0])

    @property
    def omega(self) -> Float[Array, ""]:
        return 2.0 * jnp.pi * self.frequency

    @property
    def period(self) -> Float[Array, ""]:
        return jnp.where(self.frequency > 0.0, 1.0 / jnp.maximum(
            self.frequency, 1.0e-300), jnp.inf)

    @classmethod
    def from_dict(
        cls, d: Mapping[str, Any], p0_for_atm: float
    ) -> "AcousticParams":
        scale = p0_for_atm if bool(d.get("amplitudes_in_atm", False)) else 1.0
        amps = [float(a) * scale for a in d["amplitudes_init"]]
        phis = [float(p) for p in d["phases_init"]]
        k = int(d.get("n_harmonics", len(amps)))
        if len(amps) != k or len(phis) != k:
            raise ValueError(
                f"acoustics: expected {k} amplitudes and phases, got "
                f"{len(amps)} and {len(phis)}"
            )
        return cls(
            frequency=_f(d["frequency"]),
            amplitudes=_fa(amps),
            phases=_fa(phis),
            ramp_cycles=_f(d.get("ramp_cycles", 0.0)),
        )

    def with_action(
        self,
        amplitudes: Float[Array, " K"],
        phases: Float[Array, " K"],
    ) -> "AcousticParams":
        return eqx.tree_at(
            lambda a: (a.amplitudes, a.phases), self, (amplitudes, phases)
        )


class Guards(eqx.Module):
    r_min_abs: Float[Array, ""]
    r_min_rel_to_hardcore: Float[Array, ""]
    t_min: Float[Array, ""]
    t_max: Float[Array, ""]
    rdot_max_mach: Float[Array, ""]
    km_denominator_floor: Float[Array, ""]
    softmin_sharpness: Float[Array, ""]
    eps_div: Float[Array, ""]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Guards":
        return cls(
            r_min_abs=_f(d["r_min_abs"]),
            r_min_rel_to_hardcore=_f(d["r_min_rel_to_hardcore"]),
            t_min=_f(d["t_min"]),
            t_max=_f(d["t_max"]),
            rdot_max_mach=_f(d["rdot_max_mach"]),
            km_denominator_floor=_f(d["km_denominator_floor"]),
            softmin_sharpness=_f(d["softmin_sharpness"]),
            eps_div=_f(d["eps_div"]),
        )


class IntegratorParams(eqx.Module):
    solver: SolverName = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    pcoeff: float = eqx.field(static=True)
    icoeff: float = eqx.field(static=True)
    dcoeff: float = eqx.field(static=True)
    nondimensionalise: bool = eqx.field(static=True)

    rtol: Float[Array, ""]
    atol_r: Float[Array, ""]
    atol_rdot: Float[Array, ""]
    atol_T: Float[Array, ""]
    atol_a: Float[Array, ""]
    atol_adot: Float[Array, ""]
    atol_yield: Float[Array, ""]
    dt0: Float[Array, ""]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "IntegratorParams":
        solver = str(d["solver"])
        if solver not in ("Tsit5", "Dopri8", "Kvaerno3", "Kvaerno5"):
            raise ValueError(f"unknown solver {solver!r}")
        return cls(
            solver=cast(SolverName, solver),
            max_steps=int(d["max_steps"]),
            pcoeff=float(d["pcoeff"]),
            icoeff=float(d["icoeff"]),
            dcoeff=float(d["dcoeff"]),
            nondimensionalise=bool(d["nondimensionalise"]),
            rtol=_f(d["rtol"]),
            atol_r=_f(d["atol_r"]),
            atol_rdot=_f(d["atol_rdot"]),
            atol_T=_f(d["atol_T"]),
            atol_a=_f(d["atol_a"]),
            atol_adot=_f(d["atol_adot"]),
            atol_yield=_f(d["atol_yield"]),
            dt0=_f(d["dt0"]),
        )


# top-level Equinox pytree configuration
class SimConfig(eqx.Module):
    scenario: str = eqx.field(static=True)
    gas_enabled: bool = eqx.field(static=True)
    r0: Float[Array, ""]
    n_nuclei: Float[Array, ""]
    liquid: LiquidParams
    gas: GasMixture
    transport: TransportParams
    fusion: FusionParams
    shape: ShapeParams
    acoustics: AcousticParams
    vapour: VapourParams
    resonator: ResonatorParams
    guards: Guards
    integrator: IntegratorParams


    @property
    def v0(self) -> Float[Array, ""]:
        return (4.0 / 3.0) * jnp.pi * self.r0**3

    @property
    def hardcore_radius(self) -> Float[Array, ""]:
        v_hc = self.n_nuclei * self.gas.hardcore_volume_per_nucleus
        return jnp.cbrt(3.0 * v_hc / (4.0 * jnp.pi))

    @property
    def r_floor(self) -> Float[Array, ""]:
        return jnp.maximum(
            self.guards.r_min_abs,
            self.guards.r_min_rel_to_hardcore * self.hardcore_radius,
        )

    @property
    def time_scale(self) -> Float[Array, ""]:
        inertial = self.r0 * jnp.sqrt(
            self.liquid.density / self.liquid.ambient_pressure
        )
        return jnp.where(self.acoustics.frequency > 0.0,
                         1.0 / jnp.maximum(self.acoustics.frequency, 1e-300),
                         inertial)

    def with_acoustics(self, acoustics: AcousticParams) -> "SimConfig":
        return eqx.tree_at(lambda c: c.acoustics, self, acoustics)

    def with_integrator(self, **changes: Any) -> "SimConfig":
        fields = {f.name: f for f in dataclasses.fields(IntegratorParams)}
        unknown = set(changes) - set(fields)
        if unknown:
            raise TypeError(
                f"unknown IntegratorParams field(s) {sorted(unknown)}; "
                f"valid: {sorted(fields)}"
            )
        traced = {"rtol", "atol_r", "atol_rdot", "atol_T", "atol_a",
                  "atol_adot", "atol_yield", "dt0"}
        coerced = {
            k: (_f(v) if k in traced else v) for k, v in changes.items()
        }
        new_integrator = dataclasses.replace(self.integrator, **coerced)
        return eqx.tree_at(
            lambda c: c.integrator,
            self,
            new_integrator,
            is_leaf=lambda x: x is self.integrator,
        )


def _initial_gas_pressure(
    mode: GasPressureMode,
    explicit: float | None,
    liquid: Mapping[str, Any],
    r0: float,
) -> float:
    p0 = float(liquid["ambient_pressure"])
    sigma = float(liquid["surface_tension"])
    pv = float(liquid["vapour_pressure"])
    if mode == "laplace":
        return p0 - pv + 2.0 * sigma / r0
    if mode == "ambient":
        return p0
    if mode == "explicit":
        if explicit is None:
            raise ValueError(
                "gas_initial_pressure_mode == 'explicit' requires "
                "gas_initial_pressure to be set"
            )
        return float(explicit)
    raise ValueError(f"unknown gas_initial_pressure_mode {mode!r}")


def _solve_nuclei_count(
    p_g0: float, v0: float, t0: float, mixture: GasMixture
) -> float:
    if p_g0 <= 0.0:
        return 0.0
    beta = float(mixture.molecules_per_nucleus)
    b_tilde = float(mixture.hardcore_volume_per_nucleus)
    return p_g0 * v0 / (beta * C.K_B * t0 + p_g0 * b_tilde)


def load_config(
    path: str | Path | None = None,
    scenario: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SimConfig:
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    if int(doc.get("schema_version", -1)) != 1:
        raise ValueError(
            f"{cfg_path}: expected schema_version 1, got "
            f"{doc.get('schema_version')!r}"
        )

    scen_name = scenario or str(doc["meta"]["scenario"])
    if scen_name not in doc["scenarios"]:
        raise KeyError(
            f"unknown scenario {scen_name!r}; available: "
            f"{sorted(doc['scenarios'])}"
        )
    scen = doc["scenarios"][scen_name]

    liquid_key = str(scen["liquid"])
    if liquid_key not in doc["liquids"]:
        raise KeyError(f"unknown liquid {liquid_key!r}")

    work: dict[str, Any] = {
        "liquid": copy.deepcopy(doc["liquids"][liquid_key]),
        "gas_species": copy.deepcopy(doc["gas_species"]),
        "gas_mixture": copy.deepcopy(scen["gas_mixture"]),
        "transport": copy.deepcopy(doc["transport"]),
        "fusion": copy.deepcopy(doc["fusion"]),
        "shape": copy.deepcopy(doc["shape"]),
        "guards": copy.deepcopy(doc["guards"]),
        "integrator": copy.deepcopy(doc["integrator"]),
        "acoustics": copy.deepcopy(doc["acoustics"]),
        "resonator": copy.deepcopy(doc.get("resonator", {})),
        "vapour": copy.deepcopy(doc.get("vapour", {})),
        "bubble": {
            "r0": float(scen["r0"]),
            "gas_initial_pressure_mode": str(scen["gas_initial_pressure_mode"]),
            "gas_initial_pressure": scen.get("gas_initial_pressure"),
        },
    }
    work["acoustics"].update(copy.deepcopy(scen.get("acoustics", {})))
    work["resonator"].update(copy.deepcopy(scen.get("resonator", {})))
    work["vapour"].update(copy.deepcopy(scen.get("vapour", {})))

    for key, value in (scen.get("overrides") or {}).items():
        _deep_set(work, str(key), value)
    for key, value in (overrides or {}).items():
        _deep_set(work, str(key), value)

    liquid = LiquidParams.from_dict(work["liquid"])
    vapour_block = work["vapour"]
    vapour_on = bool(vapour_block.get("enabled", False))
    extra = (str(vapour_block["species"]),) if vapour_on else ()
    gas = GasMixture.from_dict(
        work["gas_species"], work["gas_mixture"], extra_species=extra
    )
    vapour = VapourParams.from_dict(vapour_block, gas.names)

    r0 = float(work["bubble"]["r0"])
    p_g0 = _initial_gas_pressure(
        cast(GasPressureMode, work["bubble"]["gas_initial_pressure_mode"]),
        work["bubble"]["gas_initial_pressure"],
        work["liquid"],
        r0,
    )
    if vapour_on:
        from core.vapour import saturation_pressure as _p_sat

        p_vapour = float(
            _p_sat(jnp.asarray(float(work["liquid"]["ambient_temperature"])))
        )
        p_g0 = max(p_g0 - p_vapour, 1.0e-6 * p_g0)
    v0 = (4.0 / 3.0) * 3.141592653589793 * r0**3
    n_nuclei = _solve_nuclei_count(
        p_g0, v0, float(work["liquid"]["ambient_temperature"]), gas
    )

    return SimConfig(
        scenario=scen_name,
        gas_enabled=p_g0 > 0.0,
        r0=_f(r0),
        n_nuclei=_f(n_nuclei),
        liquid=liquid,
        gas=gas,
        transport=TransportParams.from_dict(work["transport"]),
        fusion=FusionParams.from_dict(work["fusion"]),
        shape=ShapeParams.from_dict(work["shape"]),
        acoustics=AcousticParams.from_dict(
            work["acoustics"], float(work["liquid"]["ambient_pressure"])
        ),
        vapour=vapour,
        resonator=ResonatorParams.from_dict(work["resonator"] or None),
        guards=Guards.from_dict(work["guards"]),
        integrator=IntegratorParams.from_dict(work["integrator"]),
    )


__all__ = [
    "AcousticParams",
    "ConductionModel",
    "FusionParams",
    "GasMixture",
    "GasPressureMode",
    "ResonatorParams",
    "VapourParams",
    "Guards",
    "IntegratorParams",
    "LiquidParams",
    "ShapeParams",
    "SimConfig",
    "SolverName",
    "TransportParams",
    "load_config",
]
