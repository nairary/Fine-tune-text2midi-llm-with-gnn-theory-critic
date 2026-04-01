from __future__ import annotations

from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_teacher import maybe_init_wandb


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        return compose(config_name=config_name, overrides=overrides or [])


def test_baseline_config_resolves_with_wandb_none():
    cfg = _compose("config")
    assert cfg.experiment.name == "baseline"
    assert cfg.wandb.enabled is False
    assert cfg.wandb.mode == "disabled"


def test_full_features_config_resolves():
    cfg = _compose("full_features")
    assert cfg.experiment.name == "full_features"
    assert cfg.model.use_hybrid_graph_scorer is True
    assert cfg.model.local_summary_use_topk_mean is True
    assert cfg.wandb.enabled is True
    assert cfg.wandb.mode == "offline"


def test_wandb_online_and_offline_overrides_resolve():
    offline_cfg = _compose("config", overrides=["wandb=offline"])
    online_cfg = _compose("config", overrides=["wandb=online"])

    assert offline_cfg.wandb.enabled is True
    assert offline_cfg.wandb.mode == "offline"
    assert online_cfg.wandb.enabled is True
    assert online_cfg.wandb.mode == "online"


def test_maybe_init_wandb_skips_import_when_disabled(monkeypatch, tmp_path):
    cfg = OmegaConf.create({"wandb": {"enabled": False}})

    def fail_if_called(_name):
        raise AssertionError("importlib.util.find_spec must not be called when wandb is disabled")

    monkeypatch.setattr("src.training.train_teacher.importlib.util.find_spec", fail_if_called)

    state = maybe_init_wandb(cfg, tmp_path)
    assert state is None
