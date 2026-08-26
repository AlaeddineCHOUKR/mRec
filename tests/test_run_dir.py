"""Run directories carry everything needed to trust the numbers inside them."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from mrec.eval.protocol import EVAL_SEEDS
from mrec.utils.run import RunConfig, log_to_run_dir, new_run_dir


def test_config_defaults_follow_the_protocol():
    cfg = RunConfig(model="gtop")
    assert (cfg.k, cfg.seqlen, cfg.n_users, cfg.seeds) == (10, 30, 3000, EVAL_SEEDS)


def test_yaml_overlay_applies_and_rejects_unknown_keys(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("model: ptop\nsplit: valid\nseeds: [1, 2]\n")
    cfg = RunConfig.from_yaml(path)
    assert (cfg.model, cfg.split, cfg.seeds) == ("ptop", "valid", (1, 2))
    assert cfg.k == 10

    path.write_text("model: ptop\nlearnign_rate: 0.1\n")
    with pytest.raises(ValueError, match="learnign_rate"):
        RunConfig.from_yaml(path)


def test_run_dir_is_named_for_the_run_and_holds_the_resolved_config(tmp_path):
    cfg = RunConfig(model="gtop", split="valid", results_dir=str(tmp_path))
    run_dir = new_run_dir(cfg, stamp="20260826-120000")
    assert run_dir.name == "gtop_valid_20260826-120000"
    # the written config must rebuild the run exactly; yaml has no tuple, so compare configs
    assert RunConfig.from_yaml(run_dir / "config.yaml") == cfg
    assert set(yaml.safe_load((run_dir / "config.yaml").read_text())) == set(
        dataclasses.asdict(cfg)
    )


def test_run_dir_refuses_to_overwrite_an_existing_run(tmp_path):
    cfg = RunConfig(model="gtop", results_dir=str(tmp_path))
    new_run_dir(cfg, stamp="20260826-120000")
    with pytest.raises(FileExistsError):
        new_run_dir(cfg, stamp="20260826-120000")


def test_logging_is_teed_into_the_run_directory(tmp_path):
    import logging

    cfg = RunConfig(model="gtop", results_dir=str(tmp_path))
    run_dir = new_run_dir(cfg, stamp="20260826-120000")
    handler = log_to_run_dir(run_dir)
    try:
        logging.getLogger("mrec.test").warning("a recorded line")
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    assert "a recorded line" in (run_dir / "stdout.log").read_text()
