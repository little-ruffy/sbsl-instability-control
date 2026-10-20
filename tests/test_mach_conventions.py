from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from core import batched as bt
from core import keller_miksis as km
from core.config import load_config
from core.integrator import Trajectory
from viz.render import _local_wall_mach


@pytest.fixture(scope="module")
def cfg():
    return load_config(scenario="sbsl_argon_water_fast")


def _trajectory(cfg, radii, rdots, temperatures):
    n = len(radii)
    states = np.zeros((n, km.STATE_DIM), dtype=np.float64)
    states[:, km.IDX_R] = radii
    states[:, km.IDX_RDOT] = rdots
    states[:, km.IDX_T] = temperatures
    states[:, km.IDX_NVAP] = 0.0
    return Trajectory(
        times=jnp.asarray(np.linspace(0.0, 1.0e-9, n)),
        states=jnp.asarray(states),
        success=jnp.asarray(True),
        num_steps=jnp.asarray(float(n)),
        valid=jnp.asarray(np.ones(n, dtype=bool)),
    )


def test_the_two_implementations_agree(cfg):
    r0 = float(cfg.r0)
    traj = _trajectory(
        cfg,
        radii=[r0, 0.4 * r0, 0.18 * r0, 0.3 * r0],
        rdots=[0.0, -220.0, -980.0, 140.0],
        temperatures=[293.0, 1.2e3, 1.4e4, 6.0e3],
    )
    summary = bt._reduce_one(
        cfg, traj.times, traj.states, traj.success, traj.num_steps
    )
    on_device = float(summary.peak_mach_local)
    scalar = _local_wall_mach(cfg, traj)
    assert scalar == pytest.approx(on_device, rel=1e-9)


def test_an_uncompressed_wall_recovers_the_ambient_convention(cfg):
    r0 = float(cfg.r0)
    rdot = -12.0
    traj = _trajectory(
        cfg,
        radii=[r0, r0],
        rdots=[0.0, rdot],
        temperatures=[float(cfg.liquid.ambient_temperature)] * 2,
    )
    local = _local_wall_mach(cfg, traj)
    ambient = abs(rdot) / float(cfg.liquid.sound_speed)
    assert local == pytest.approx(ambient, rel=0.02)


def test_a_compressed_wall_is_slower_than_the_ambient_figure_suggests(cfg):
    r0 = float(cfg.r0)
    rdot = -1.8e3
    traj = _trajectory(
        cfg,
        radii=[r0, 0.12 * r0],
        rdots=[0.0, rdot],
        temperatures=[293.0, 2.0e4],
    )
    local = _local_wall_mach(cfg, traj)
    ambient = abs(rdot) / float(cfg.liquid.sound_speed)
    assert local < ambient
    assert ambient / local > 1.1
