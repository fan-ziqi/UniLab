"""Cold command reads are finite without changing bound motion or validation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.envs.mdp.observations import generated_commands
from unilab.managers.command_manager import CommandManager
from unilab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)
from unilab.managers.rng import ManagerRng
from unilab.tasks.motion_tracking.common import manager_terms as mt


@pytest.fixture
def motion_env(tmp_path, monkeypatch):
    """Use the real loader/constructor with tiny motion and metadata fixtures."""
    motion_file = tmp_path / "motion.npz"
    positions = np.array([[0.25, -0.5], [0.75, -1.0]], dtype=np.float32)
    velocities = np.array([[1.25, 2.5], [1.75, 3.0]], dtype=np.float32)
    quaternions = np.zeros((2, 1, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    np.savez(
        motion_file,
        fps=np.array([30], dtype=np.int32),
        joint_pos=positions,
        joint_vel=velocities,
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=quaternions,
        body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
    )
    num_envs = 3
    root = torch.zeros((num_envs, 13))
    root[:, 3] = 1.0
    robot = SimpleNamespace(
        name="fixture",
        joint_names=("first", "second"),
        motion_body_ids=np.array([0]),
        find_bodies=lambda names, preserve_order: ([0], list(names)),
        data=SimpleNamespace(
            soft_joint_pos_limits=lambda device: torch.tensor([[-2.0, 2.0], [-2.0, 2.0]]),
            default_joint_pos=lambda device: torch.zeros(2, device=device),
            default_root_state=lambda device: root.to(device),
        ),
    )

    class Scene(dict):
        read_plan = None

        def env_origins(self, device):
            return torch.zeros((num_envs, 3), device=device)

    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        rng=ManagerRng.seeded(37, device="cpu"),
        scene=Scene(robot=robot),
        common_step_counter=0,
        defer_finite_validation=True,
    )
    cfg = mt.MotionCommandCfg(
        entity_name="robot",
        resampling_time_range=(1e9, 1e9),
        params=mt.MotionCommandParamsCfg(
            motion_file=str(motion_file),
            anchor_body_name="base",
            body_names=("base",),
        ),
    )
    # Exercise the same equations eagerly in this constructor/read-phase unit
    # contract; compiler startup and physics stepping are separate acceptance.
    monkeypatch.setattr(mt, "_compiled_motion_relative_state_available", lambda: False)
    monkeypatch.setattr(mt, "_update_motion_relative_state_compiled", None)
    monkeypatch.setattr(mt, "_update_motion_metrics_compiled", None)
    expected = torch.from_numpy(np.concatenate((positions[:1], velocities[:1]), axis=1))
    return env, cfg, expected.expand(num_envs, -1)


def _poison_uninitialized_allocator(monkeypatch):
    allocate = torch.empty

    def empty(*args, **kwargs):
        result = allocate(*args, **kwargs)
        if result.is_floating_point():
            result.fill_(torch.nan)
        return result

    monkeypatch.setattr(torch, "empty", empty)
    assert torch.isnan(torch.empty(1)).all()


@pytest.mark.parametrize("deferred", [False, True])
def test_unbound_observation_read_has_initialized_command(motion_env, monkeypatch, deferred):
    env, cfg, expected = motion_env
    env.defer_finite_validation = deferred
    _poison_uninitialized_allocator(monkeypatch)
    rng_before = env.rng.generator.get_state().clone()
    env.command_manager = CommandManager({"motion": cfg}, env)
    command = env.command_manager.get_term("motion")
    assert not command._bind_read_phase
    torch.testing.assert_close(command._command, torch.zeros_like(expected))

    # This is the actual early generated_commands dimension read from task
    # construction, through both real managers, before read-phase binding.
    observations = ObservationManager(
        {
            "actor": ObservationGroupCfg(
                terms={
                    "command": ObservationTermCfg(
                        func=generated_commands, params={"command_name": "motion"}
                    )
                }
            )
        },
        env,
    )
    assert observations.group_obs_dim["actor"] == (4,)
    assert not command._bind_read_phase
    assert bool(env.command_manager.finite_verdict)
    torch.testing.assert_close(env.rng.generator.get_state(), rng_before)

    command.bind_read_phase()
    assert command._bind_read_phase
    torch.testing.assert_close(env.command_manager.get_command("motion"), expected)
    torch.testing.assert_close(env.rng.generator.get_state(), rng_before)
    assert env.command_manager.poll_finite_validation()


def test_real_nonfinite_command_still_latches_after_repair(motion_env):
    env, cfg, expected = motion_env
    env.command_manager = CommandManager({"motion": cfg}, env)
    command = env.command_manager.get_term("motion")
    command.bind_read_phase()
    command._command[1, 0] = torch.nan
    assert not torch.isfinite(env.command_manager.get_command("motion")).all()
    assert not bool(env.command_manager.finite_verdict)

    # A later valid motion packet must not erase evidence of a real bad read.
    command._refresh_motion_torch()
    torch.testing.assert_close(env.command_manager.get_command("motion"), expected)
    with pytest.raises(ValueError, match="CommandManager returned a non-finite command"):
        env.command_manager.poll_finite_validation()
