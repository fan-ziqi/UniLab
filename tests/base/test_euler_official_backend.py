"""Owner-layer routing for a discovered, Euler-owned UniSim backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import unilab.base.backend_factory as backend_factory
import unilab.base.registry as registry
import unilab.cli as cli
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg


def test_registry_admits_only_declared_third_party_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "_EulerDiscoveredRegistryTest"
    if not registry.contains(name):
        registry.register_env_config(name, EnvCfg)

    monkeypatch.setattr(registry, "_is_declared_third_party_backend", lambda backend: backend == "euler")
    def factory(cfg, *, num_envs=1, backend_type="euler"):
        return None
    registry.register_env(name, factory, "euler")

    unknown = "_EulerUndeclaredRegistryTest"
    if not registry.contains(unknown):
        registry.register_env_config(unknown, EnvCfg)
    with pytest.raises(ValueError, match="Install a UniSim third-party provider"):
        registry.register_env(unknown, factory, "not_a_provider")


def test_env_cfg_rejects_malformed_euler_authority() -> None:
    with pytest.raises(ValueError, match="euler_native_library_path"):
        EnvCfg(euler_native_library_path=" ").validate()
    with pytest.raises(ValueError, match="euler_native_library_path"):
        EnvCfg(euler_native_library_path=7).validate()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("authority", "expected"),
    [
        ({"euler_native_library_path": "/abs/libeuler_unisim_abi.so"}, {"native_library_path": "/abs/libeuler_unisim_abi.so"}),
    ],
)
def test_factory_routes_exactly_one_euler_authority(
    monkeypatch: pytest.MonkeyPatch, authority: dict[str, object], expected: dict[str, object]
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(backend_factory, "ensure_robot_assets_for_paths", lambda *_: None)

    def create(backend_type, scene, num_envs, sim_dt, **kwargs):
        captured.update(
            backend_type=backend_type,
            scene=scene,
            num_envs=num_envs,
            sim_dt=sim_dt,
            kwargs=kwargs,
        )
        return SimpleNamespace()

    monkeypatch.setattr(backend_factory.unisim, "create_backend", create)
    cfg = EnvCfg(**authority)
    cfg.validate()
    backend_factory.create_backend(
        "euler",
        SceneCfg(model_file="Euler-owned-go2.xml"),
        64,
        0.01,
        **backend_factory.env_backend_kwargs(cfg),
    )
    assert captured["backend_type"] == "euler"
    assert captured["num_envs"] == 64
    assert captured["sim_dt"] == 0.01
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert expected.items() <= kwargs.items()
    assert set(kwargs) <= {"base_name", "body_state_required", *expected}
    assert "euler_native_library_path" not in kwargs


def test_factory_requires_the_common_native_library() -> None:
    with pytest.raises(ValueError, match="requires explicit euler_native_library_path"):
        backend_factory.create_backend("euler", SceneCfg(model_file="scene.xml"), 1, 0.01)


def test_removed_worker_options_fail_closed() -> None:
    with pytest.raises(TypeError, match="unknown Euler options"):
        backend_factory.create_backend("euler", SceneCfg(model_file="scene.xml"), 1, 0.01,
                                       euler_native_library_path="/abs/lib",
                                       euler_go2_worker_command=("worker",))


def test_factory_rejects_euler_authority_for_other_backend() -> None:
    with pytest.raises(ValueError, match="valid only for backend_type='euler'"):
        backend_factory.create_backend(
            "mujoco",
            SceneCfg(model_file="scene.xml"),
            1,
            0.01,
            euler_native_library_path="/abs/lib",
        )


def test_euler_factory_never_hydrates_unilab_robot_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fail_if_hydrated(*_args: object) -> None:
        raise AssertionError("Euler must use its explicit asset root, not UniLab hydration")

    def create(backend_type, scene, num_envs, sim_dt, **kwargs):
        captured.update(backend_type=backend_type, scene=scene, kwargs=kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(backend_factory, "ensure_robot_assets_for_paths", fail_if_hydrated)
    monkeypatch.setattr(backend_factory.unisim, "create_backend", create)
    backend_factory.create_backend(
        "euler",
        SceneCfg(model_file="go2_scene.xml"),
        64,
        0.01,
        euler_native_library_path="/abs/libeuler.so",
    )
    assert captured["backend_type"] == "euler"
    assert captured["kwargs"] == {
        "body_state_required": False,
        "native_library_path": "/abs/libeuler.so",
    }


def test_cli_admits_only_discovered_third_party_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "available_sims", lambda: (*cli.SUPPORTED_SIMS, "euler"))
    cli._check_runtime_requirements("ppo", "euler")
    monkeypatch.setattr(cli, "available_sims", lambda: cli.SUPPORTED_SIMS)
    with pytest.raises(SystemExit, match="not an installed"):
        cli._check_runtime_requirements("ppo", "euler")
