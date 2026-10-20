from __future__ import annotations

import math

import numpy as np
import pytest

from core.config import load_config
from viz.volume import (
    APERTURE_OFFSET_DEG,
    CUT_FRACTION,
    INTERFACE_ALPHA,
    _layer_temperature,
    _radial_grid,
    _seam_widths,
    _shaded_rgba,
    aperture_phi_start,
    cutaway_geometry,
    shape_radius,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(scenario="sbsl_argon_water")


def test_shape_radius_is_the_legendre_surface():
    radius, a2 = 1.0e-6, 1.2e-7
    pole = float(shape_radius(np.array([1.0]), radius, a2)[0])
    equator = float(shape_radius(np.array([0.0]), radius, a2)[0])
    assert pole == pytest.approx(radius + a2, rel=1e-14)
    assert equator == pytest.approx(radius - 0.5 * a2, rel=1e-14)
    assert pole > equator


def test_shape_radius_is_floored_not_wrapped():
    r = shape_radius(np.linspace(-1.0, 1.0, 101), 1.0e-6, 5.0e-6)
    assert float(r.min()) == 0.0
    assert np.all(np.asarray(r) >= 0.0)


def test_sphere_when_undistorted(cfg):
    radius = 2.0e-6
    faces, temperature, _shade = cutaway_geometry(
        radius, 0.0, 0.3e-6, 1.0e4, 293.0
    )
    interface = faces[~np.isfinite(temperature)]
    dist = np.linalg.norm(interface.reshape(-1, 3), axis=1)
    assert np.allclose(dist, radius, rtol=1e-12)


def test_surface_is_the_revolved_legendre_profile():
    radius, a2 = 1.0e-6, 2.5e-7
    faces, temperature, _ = cutaway_geometry(radius, a2, 0.1e-6, 1.0e4, 293.0)
    verts = faces[~np.isfinite(temperature)].reshape(-1, 3)
    dist = np.linalg.norm(verts, axis=1)
    keep = dist > 0.0
    cos_theta = verts[keep, 2] / dist[keep]
    expected = radius + a2 * 0.5 * (3.0 * cos_theta**2 - 1.0)
    assert np.allclose(dist[keep], expected, rtol=1e-12)


@pytest.mark.parametrize("ratio", [0.30, 0.10, 0.03, 0.01, 0.002])
def test_boundary_layer_is_resolved_at_every_thickness(ratio):
    radius = 1.0e-6
    n_core, n_layer = 10, 18
    frac = _radial_grid(radius, ratio * radius, n_core, n_layer)
    edge = 1.0 - ratio
    inside = np.sum(frac > edge + 1e-15)
    assert inside == n_layer
    assert frac[0] == 0.0
    assert frac[-1] == pytest.approx(1.0, abs=1e-15)
    assert np.all(np.diff(frac) > 0.0)


def test_interior_temperature_has_exactly_two_zones():
    radius, ell, core_t, liquid_t = 1.0e-6, 1.0e-7, 2.5e4, 293.0
    frac = np.linspace(0.0, 1.0, 2001)
    temp = _layer_temperature(frac, radius, ell, core_t, liquid_t)

    edge = 1.0 - ell / radius
    core = temp[frac < edge - 1e-6]
    assert np.allclose(core, core_t, rtol=1e-14)
    assert temp[-1] == pytest.approx(liquid_t, rel=1e-12)

    ramp = frac > edge + 1e-6
    slope = np.diff(temp[ramp]) / np.diff(frac[ramp])
    assert np.allclose(slope, slope[0], rtol=1e-9)
    assert slope[0] < 0.0


def test_interior_is_monotone_outward():
    frac = np.linspace(0.0, 1.0, 501)
    for ratio in (0.5, 0.2, 0.05, 0.001):
        temp = _layer_temperature(frac, 1.0e-6, ratio * 1.0e-6, 3.0e4, 293.0)
        assert np.all(np.diff(temp) <= 1e-9)


def test_cut_faces_span_the_full_temperature_range(cfg):
    faces, temperature, _ = cutaway_geometry(
        1.0e-6, 1.0e-7, 1.0e-7, 2.5e4, 293.0
    )
    interior = temperature[np.isfinite(temperature)]
    assert interior.max() == pytest.approx(2.5e4, rel=1e-3)
    assert interior.min() < 1.0e3


def test_interface_is_off_the_temperature_scale():
    faces, temperature, _ = cutaway_geometry(
        1.0e-6, 1.0e-7, 1.0e-7, 2.5e4, 293.0
    )
    n_interface = np.sum(~np.isfinite(temperature))
    assert n_interface > 0
    assert n_interface + np.sum(np.isfinite(temperature)) == len(faces)


def test_shell_is_opaque_and_unstroked():
    temperature = np.array([np.nan, 1.0e4])
    shade = np.array([0.8, 0.8])
    import matplotlib

    from viz.projection import T_CMAP, T_MAX_K, T_MIN_K

    cmap = matplotlib.pyplot.get_cmap(T_CMAP)
    norm = matplotlib.colors.LogNorm(vmin=T_MIN_K, vmax=T_MAX_K)
    rgba = _shaded_rgba(temperature, shade, cmap, norm)
    assert rgba[0, 3] == INTERFACE_ALPHA == 1.0
    assert _seam_widths(temperature)[0] == 0.0
    assert _seam_widths(temperature)[1] > 0.0


def test_aperture_points_the_opening_near_the_camera():
    for azim in (-180.0, -60.0, 0.0, 37.0, 150.0):
        phi0 = aperture_phi_start(azim)
        phi_end = phi0 + 2.0 * math.pi * CUT_FRACTION
        removed_centre = phi_end + math.pi * (1.0 - CUT_FRACTION)
        offset = (removed_centre - math.radians(azim) + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        assert abs(math.degrees(offset) - APERTURE_OFFSET_DEG) < 1e-9
        assert abs(APERTURE_OFFSET_DEG) < 180.0 * (1.0 - CUT_FRACTION)


def test_geometry_scales_linearly_with_radius():
    small = cutaway_geometry(1.0e-6, 1.0e-7, 1.0e-7, 2.5e4, 293.0)[0]
    large = cutaway_geometry(2.0e-6, 2.0e-7, 2.0e-7, 2.5e4, 293.0)[0]
    assert np.allclose(large, 2.0 * small, rtol=1e-12)


def test_face_count_is_what_the_resolution_asks_for():
    n_theta, n_phi, n_core, n_layer = 12, 16, 4, 6
    faces, temperature, shade = cutaway_geometry(
        1.0e-6, 0.0, 1.0e-7, 1.0e4, 293.0,
        n_theta=n_theta, n_phi=n_phi, n_core=n_core, n_layer=n_layer,
    )
    n_rad = n_core + n_layer
    assert len(faces) == n_theta * n_phi + 2 * n_theta * n_rad
    assert len(temperature) == len(faces)
    assert len(shade) == len(faces)
    assert faces.shape[1:] == (4, 3)


def test_shading_never_goes_black():
    _faces, _t, shade = cutaway_geometry(
        1.0e-6, 2.0e-7, 1.0e-7, 2.5e4, 293.0
    )
    assert shade.min() > 0.25
    assert shade.max() <= 1.0
