"""A run must not be able to call itself dirty by writing its own output."""

from __future__ import annotations

import subprocess

import pytest

from mrec.utils.manifest import git_dirty
from mrec.utils.paths import REPO_ROOT


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr("mrec.utils.manifest.REPO_ROOT", tmp_path)
    run = subprocess.run
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
    ):
        run(command, cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "PROJECTS" / "x" / "results").mkdir(parents=True)
    (tmp_path / "tracked.py").write_text("x = 1\n")
    run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_clean_tree_is_clean(repo):
    assert git_dirty() is False


def test_a_new_run_directory_does_not_make_the_tree_dirty(repo):
    run_dir = repo / "PROJECTS" / "x" / "results" / "model_test_20260101-000000"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("seed,metric,k,value\n")
    assert git_dirty() is False


def test_uncommitted_source_still_counts(repo):
    (repo / "tracked.py").write_text("x = 2\n")
    assert git_dirty() is True


def test_a_new_untracked_source_file_still_counts(repo):
    (repo / "untracked.py").write_text("y = 1\n")
    assert git_dirty() is True


def test_an_edited_committed_result_still_counts(repo):
    committed = repo / "PROJECTS" / "x" / "results" / "old_run"
    committed.mkdir()
    (committed / "metrics.csv").write_text("seed,metric,k,value\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "r"], cwd=repo, check=True, capture_output=True)
    assert git_dirty() is False
    (committed / "metrics.csv").write_text("seed,metric,k,value\n1,ndcg_all@10,10,0.9\n")
    assert git_dirty() is True


def test_repo_root_is_importable():
    assert REPO_ROOT.exists()
