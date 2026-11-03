"""Resolve authored backend leaves without importing environments or physics."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).resolve().parents[2] / "src/unilab/conf"
EULER_LEAVES = sorted(CONF_DIR.glob("*/task/**/euler.yaml"))


@pytest.mark.parametrize("leaf", EULER_LEAVES, ids=lambda path: str(path.relative_to(CONF_DIR)))
@pytest.mark.parametrize("explicit_resource_root", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [[], ["algo.num_envs=256", "algo.max_iterations=151", "algo.num_steps_per_env=24"]],
)
def test_backend_leaf_preserves_the_inherited_authored_recipe(
    leaf, overrides, explicit_resource_root, tmp_path,
):
    """Only backend selection and explicit Euler resource authority may differ.

    Discover every Euler owner leaf, require inheritance from a sibling backend
    recipe, and compare the entire resolved config, including rewards, commands,
    randomization, observations, PPO, budgets and play settings.
    """
    source = OmegaConf.to_container(OmegaConf.load(leaf), resolve=True)
    parents = [entry for entry in source["defaults"] if entry != "_self_"]
    assert len(parents) == 1 and isinstance(parents[0], str)
    reference = parents[0].removeprefix("/task/")
    assert reference != parents[0], "Inherit a task owner with an absolute /task/ reference"
    group = leaf.relative_to(CONF_DIR).parts[0]
    canonical = CONF_DIR / group / "task" / (reference + ".yaml")
    assert canonical.is_file() and canonical.parent == leaf.parent
    assert canonical.stem not in ("base", "euler"), "Inherit an authored backend recipe"
    selected = leaf.relative_to(CONF_DIR / group / "task").with_suffix("").as_posix()

    def resolve(task, extra=()):
        with initialize_config_dir(config_dir=str(CONF_DIR / group), version_base="1.3"):
            return OmegaConf.to_container(
                compose("config", overrides=[f"task={task}", *overrides, *extra]), resolve=True
            )

    resource_root = str(tmp_path / "caller-root") if explicit_resource_root else None
    extra = [f"env.euler_mjcf_resource_root={resource_root}"] if explicit_resource_root else []
    actual, expected = resolve(selected, extra), resolve(reference)
    assert actual["training"].pop("sim_backend") == "euler"
    assert expected["training"].pop("sim_backend") == canonical.stem
    # This is a resource requirement, not a task/PPO override. A launch must
    # provide the library; canonical backends do not consume this field.
    assert actual["env"].pop("euler_native_library_path") is None
    assert "euler_native_library_path" not in expected["env"]
    assert actual["env"].pop("euler_mjcf_resource_root") == resource_root
    assert "euler_mjcf_resource_root" not in expected["env"]
    assert actual == expected
