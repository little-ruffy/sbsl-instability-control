from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
import pytest

from core import keller_miksis as km
from core.config import load_config
from core.integrator import (
    absolute_tolerances,
    diagnostics_along,
    integrate,
    nondim_scaling,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(scenario="sbsl_argon_water_fast")


@pytest.fixture(scope="module")
def period(cfg):
    return 1.0 / float(cfg.acoustics.frequency)


STEPS_BUDGET_SBSL = 4096
STEPS_BUDGET_D2 = 8192
GRID_COARSE = 512
GRID_FINE = 2048
WINDOW_POINTS = 1024


def test_scaling_round_trips(cfg):
    scaling = nondim_scaling(cfg)
    y = km.initial_state(cfg).at[km.IDX_RDOT].set(-734.0)
    y = y.at[km.IDX_ERAD].set(1.7e-13)
    back = scaling.from_nondim(scaling.to_nondim(y))
    np.testing.assert_allclose(np.asarray(back), np.asarray(y), rtol=1e-15)


def test_scaling_brings_the_state_to_order_one(cfg):
    scaling = nondim_scaling(cfg)
    scaled = np.asarray(scaling.to_nondim(km.initial_state(cfg)))
    assert scaled[km.IDX_R] == pytest.approx(1.0, rel=1e-14)
    assert scaled[km.IDX_T] == pytest.approx(1.0, rel=1e-14)
    assert all(abs(v) < 10.0 for v in scaled)
    assert np.all(np.asarray(scaling.state) > 0.0)


def test_tolerances_rescale_consistently(cfg):
    scaling = nondim_scaling(cfg)
    atol_si = np.asarray(absolute_tolerances(cfg, None))
    atol_nd = np.asarray(absolute_tolerances(cfg, scaling))
    np.testing.assert_allclose(
        atol_nd * np.asarray(scaling.state), atol_si, rtol=1e-14
    )
    assert atol_si[km.IDX_R] == float(cfg.integrator.atol_r)


def test_time_scale_falls_back_for_a_dc_configuration():
    cfg = load_config(scenario="rayleigh_validation")
    assert float(cfg.acoustics.frequency) == 0.0
    expected = float(cfg.r0) * np.sqrt(
        float(cfg.liquid.density) / float(cfg.liquid.ambient_pressure)
    )
    assert float(cfg.time_scale) == pytest.approx(expected, rel=1e-14)
    assert np.isfinite(float(cfg.time_scale))


@pytest.mark.parametrize("scenario", ["sbsl_argon_water_fast", "sonofusion_d_acetone"])
def test_nondimensional_and_si_paths_agree(scenario):
    base = load_config(scenario=scenario)
    period_s = 1.0 / float(base.acoustics.frequency)
    grid = jnp.linspace(0.0, period_s, GRID_COARSE)
    results = {}
    for nondim in (True, False):
        traj = integrate(
            base.with_integrator(nondimensionalise=nondim, rtol=1e-10),
            0.0,
            period_s,
            save_ts=grid,
            throw=True,
        )
        results[nondim] = np.asarray(traj.states)

    a, b = results[True], results[False]
    for index, name in enumerate(km.STATE_NAMES):
        scale = max(float(np.max(np.abs(a[:, index]))), 1e-300)
        diff = float(np.max(np.abs(a[:, index] - b[:, index]))) / scale
        assert diff < 1e-5, f"{scenario}: {name} differs by {diff:.3e}"


def test_uniform_grid_misreports_the_collapse(cfg, period):
    coarse = integrate(cfg, 0.0, period, n_save=GRID_COARSE, throw=True)
    fine = integrate(cfg, 0.0, period, n_save=GRID_FINE, throw=True)
    steps = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_SBSL, throw=True
    )

    t_coarse = float(coarse.peak_temperature)
    t_fine = float(fine.peak_temperature)
    t_steps = float(steps.peak_temperature)

    assert float(coarse.max_radius) == pytest.approx(
        float(steps.max_radius), rel=1e-3
    )
    assert t_coarse < 0.2 * t_steps, (
        "the coarse grid is expected to badly under-report T_peak; if it no "
        f"longer does, this test's premise has changed (coarse {t_coarse:.3e} "
        f"vs steps {t_steps:.3e})"
    )
    assert t_fine < t_steps
    assert t_steps > 1.5e4, f"T_peak = {t_steps:.3e} K is implausibly low"
    assert float(steps.min_radius) < float(coarse.min_radius)


def _parabolic_minimum(times: np.ndarray, values: np.ndarray) -> float:
    k = int(np.argmin(values))
    if k == 0 or k == len(values) - 1:
        return float(values[k])
    y0, y1, y2 = values[k - 1], values[k], values[k + 1]
    denominator = y0 - 2.0 * y1 + y2
    if denominator <= 0.0:
        return float(y1)
    offset = 0.5 * (y0 - y2) / denominator
    return float(y1 - 0.25 * (y0 - y2) * offset)


def _parabolic_maximum(values: np.ndarray) -> float:
    return -_parabolic_minimum(np.arange(len(values), dtype=float), -values)


def test_step_resolved_mode_converges_against_a_dense_window(cfg, period):
    steps = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_SBSL, throw=True
    )
    valid = np.asarray(steps.valid)
    times = np.asarray(steps.times)[valid]
    radii = np.asarray(steps.radius)[valid]
    t_min = float(times[int(np.argmin(radii))])

    reference = {}
    for half_width in (3e-8, 3e-9):
        window = jnp.linspace(
            t_min - half_width, t_min + half_width, WINDOW_POINTS
        )
        traj = integrate(cfg, 0.0, period, save_ts=window, throw=True)
        reference[half_width] = (
            _parabolic_maximum(np.asarray(traj.temperature)),
            _parabolic_minimum(
                np.asarray(traj.times), np.asarray(traj.radius)
            ),
        )
    assert reference[3e-8][0] == pytest.approx(reference[3e-9][0], rel=1e-3)
    assert reference[3e-8][1] == pytest.approx(reference[3e-9][1], rel=1e-3)
    assert float(steps.peak_temperature) == pytest.approx(
        reference[3e-9][0], rel=0.02
    )
    assert float(steps.min_radius) == pytest.approx(
        reference[3e-9][1], rel=1e-3
    )


def test_explicit_save_grid_is_respected(cfg, period):
    grid = jnp.linspace(0.3 * period, 0.6 * period, GRID_COARSE)
    traj = integrate(cfg, 0.0, period, save_ts=grid, throw=True)
    np.testing.assert_allclose(
        np.asarray(traj.times), np.asarray(grid), rtol=1e-15
    )
    assert traj.states.shape == (GRID_COARSE, km.STATE_DIM)
    assert bool(np.all(np.asarray(traj.valid)))


def test_masked_reductions_ignore_padding(cfg, period):
    traj = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_SBSL, throw=True
    )
    valid = np.asarray(traj.valid)
    assert valid.sum() < valid.size, "test needs an actually-padded buffer"
    assert not bool(np.all(np.isfinite(np.asarray(traj.times))))

    manual_max_t = float(np.max(np.asarray(traj.temperature)[valid]))
    manual_min_r = float(np.min(np.asarray(traj.radius)[valid]))
    assert float(traj.peak_temperature) == pytest.approx(manual_max_t, rel=1e-15)
    assert float(traj.min_radius) == pytest.approx(manual_min_r, rel=1e-15)
    assert np.isfinite(float(traj.peak_temperature))
    assert np.isfinite(float(traj.max_relative_distortion))
    assert np.isfinite(float(traj.peak_wall_speed))

    last_valid = int(np.nonzero(valid)[0][-1])
    np.testing.assert_allclose(
        np.asarray(traj.final_state),
        np.asarray(traj.states)[last_valid],
        rtol=0,
        atol=0,
    )
    assert int(traj.n_valid) == int(valid.sum())


def test_grid_mode_marks_everything_valid(cfg, period):
    traj = integrate(cfg, 0.0, period, n_save=GRID_COARSE, throw=True)
    assert bool(np.all(np.asarray(traj.valid)))
    assert int(traj.n_valid) == GRID_COARSE


def test_step_budget_exhaustion_is_reported_not_raised(cfg, period):
    traj = integrate(cfg, 0.0, period, n_save=GRID_COARSE, max_steps=16, throw=False)
    assert not bool(traj.success)
    assert int(traj.num_steps) >= 16


def test_step_budget_exhaustion_raises_when_asked(cfg, period):
    with pytest.raises(Exception, match="maximum number of solver steps"):
        integrate(cfg, 0.0, period, n_save=GRID_COARSE, max_steps=16, throw=True)


def test_success_flag_is_true_on_a_normal_run(cfg, period):
    traj = integrate(cfg, 0.0, period, n_save=GRID_COARSE, throw=False)
    assert bool(traj.success)
    assert bool(np.all(np.isfinite(np.asarray(traj.states))))


def test_trajectory_properties_agree_with_manual_reductions(cfg, period):
    traj = integrate(cfg, 0.0, period, n_save=GRID_FINE, throw=True)
    states = np.asarray(traj.states)
    np.testing.assert_allclose(
        np.asarray(traj.radius), states[:, km.IDX_R], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(traj.temperature), states[:, km.IDX_T], rtol=0, atol=0
    )
    assert float(traj.max_radius) == pytest.approx(
        float(states[:, km.IDX_R].max()), rel=1e-15
    )
    assert float(traj.expansion_ratio) == pytest.approx(
        float(states[:, km.IDX_R].max() / states[:, km.IDX_R].min()), rel=1e-12
    )
    expected_distortion = np.abs(states[:, km.IDX_A]) / states[:, km.IDX_R]
    np.testing.assert_allclose(
        np.asarray(traj.relative_distortion), expected_distortion, rtol=1e-12
    )


def test_diagnostics_are_consistent_with_the_states(cfg, period):
    traj = integrate(cfg, 0.0, period, n_save=GRID_COARSE, throw=True)
    diag = diagnostics_along(cfg, traj)
    states = np.asarray(traj.states)

    np.testing.assert_allclose(
        np.asarray(diag.radius_clamped), states[:, km.IDX_R], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(diag.mach),
        states[:, km.IDX_RDOT] / float(cfg.liquid.sound_speed),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(diag.relative_distortion),
        np.abs(states[:, km.IDX_A]) / states[:, km.IDX_R],
        rtol=1e-12,
    )
    assert np.all(np.isfinite(np.asarray(diag.gas_pressure)))
    assert np.all(np.asarray(diag.gas_pressure) > 0.0)
    assert np.all(np.asarray(diag.radiative_loss) >= 0.0)


def test_diagnostics_handle_a_padded_buffer(cfg, period):
    traj = integrate(
        cfg, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_SBSL, throw=True
    )
    diag = diagnostics_along(cfg, traj)
    valid = np.asarray(traj.valid)
    assert np.all(np.isfinite(np.asarray(diag.gas_pressure)[valid]))
    assert np.all(np.isfinite(np.asarray(diag.radius_ddot)[valid]))


def test_repeated_solves_reuse_the_compiled_program(cfg, period):
    integrate(cfg, 0.0, period, n_save=GRID_COARSE, throw=True).states.block_until_ready()

    start = time.perf_counter()
    for _ in range(3):
        integrate(
            cfg, 0.0, period, n_save=GRID_COARSE, throw=True
        ).states.block_until_ready()
    per_call = (time.perf_counter() - start) / 3.0
    assert per_call < 2.0, (
        f"cached solve took {per_call:.3f} s per call, which suggests the "
        "compilation cache is being missed"
    )


def test_changing_a_traced_parameter_does_not_recompile(cfg, period):
    import equinox as eqx

    integrate(cfg, 0.0, period, n_save=GRID_COARSE, throw=True).states.block_until_ready()

    start = time.perf_counter()
    for amplitude in (1.1e5, 1.2e5, 1.3e5, 1.4e5):
        variant = eqx.tree_at(
            lambda c: c.acoustics.amplitudes,
            cfg,
            cfg.acoustics.amplitudes.at[0].set(amplitude),
        ).with_integrator(rtol=1e-8)
        integrate(
            variant, 0.0, period, n_save=GRID_COARSE, throw=True
        ).states.block_until_ready()
    per_call = (time.perf_counter() - start) / 4.0
    assert per_call < 2.0, (
        f"{per_call:.3f} s per call while sweeping traced parameters -- "
        "something that should be traced has become static"
    )


def test_ladder_truncation_does_not_change_the_trajectory(period):
    fast = load_config(scenario="sbsl_argon_water_fast")
    full = load_config(scenario="sbsl_argon_water")
    assert fast.gas.ion_pot.shape == (1, 1)
    assert full.gas.ion_pot.shape == (1, 18)

    out = {}
    for name, c in (("fast", fast), ("full", full)):
        traj = integrate(
            c, 0.0, period, save_at_steps=True, max_steps=STEPS_BUDGET_SBSL, throw=True
        )
        out[name] = (
            float(traj.min_radius),
            float(traj.max_radius),
            float(traj.peak_temperature),
        )

    assert out["fast"][0] == pytest.approx(out["full"][0], rel=1e-4), (
        f"R_min differs: {out['fast'][0]:.6e} vs {out['full'][0]:.6e}"
    )
    assert out["fast"][1] == pytest.approx(out["full"][1], rel=1e-6)
    assert out["fast"][2] == pytest.approx(out["full"][2], rel=5e-3), (
        f"T_peak differs: {out['fast'][2]:.6e} vs {out['full'][2]:.6e}"
    )
