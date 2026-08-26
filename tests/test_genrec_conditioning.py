"""The two GLIDE conditioning mechanisms: a control token and a soft prompt.

A control token that cannot move the repeat ratio is decoration, and a soft prompt the
encoder cannot see is a no-op. Both are pinned behaviourally here, not just by shape.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrec.models.genrec import MODES, GenerativeRetriever, GenRecConfig

DIM = 32
LEVELS = 2
CODEBOOK = 8
SEQLEN = 4
SESSION = 3


def make_model(**overrides):
    torch.manual_seed(0)
    cfg = GenRecConfig(
        embedding_dim=DIM,
        seqlen=SEQLEN,
        n_levels=LEVELS,
        codebook_size=CODEBOOK,
        dropout=0.0,
        soft_prompt_dropout=0.0,
        **overrides,
    )
    return GenerativeRetriever(cfg).eval()


def random_history(batch=4, rng=None):
    rng = rng or np.random.default_rng(0)
    codes = torch.as_tensor(
        rng.integers(0, CODEBOOK, size=(batch, SEQLEN, SESSION, LEVELS)), dtype=torch.long
    )
    mask = torch.ones(batch, SEQLEN, dtype=torch.bool)
    return codes, mask


def test_modes_are_the_repeat_explore_axis():
    assert MODES == ("free", "repeat", "explore")


def test_control_token_starts_as_a_no_op():
    """Zero-initialised, so any effect it has was learned rather than assumed."""
    model = make_model(n_modes=len(MODES))
    codes, mask = random_history()
    free = model(codes, mask, mode=torch.zeros(4, dtype=torch.long))
    repeat = model(codes, mask, mode=torch.ones(4, dtype=torch.long))
    torch.testing.assert_close(free, repeat)


def test_control_token_changes_the_query_once_trained():
    model = make_model(n_modes=len(MODES))
    with torch.no_grad():
        model.mode_embedding.weight.normal_()
    codes, mask = random_history()
    free = model(codes, mask, mode=torch.zeros(4, dtype=torch.long))
    explore = model(codes, mask, mode=torch.full((4,), 2, dtype=torch.long))
    assert not torch.allclose(free, explore)


def test_an_unconditioned_model_ignores_a_mode():
    model = make_model()
    codes, mask = random_history()
    assert model.mode_embedding is None
    torch.testing.assert_close(
        model(codes, mask), model(codes, mask, mode=torch.ones(4, dtype=torch.long))
    )


def test_control_token_steers_generation_towards_its_own_targets():
    """The behavioural claim: conditioning on a mode shifts the emitted distribution.

    Repeat targets live in level-0 code 0 and explore targets in level-0 code 1, so a
    control token that works moves the argmax between them from one checkpoint.
    """
    torch.manual_seed(0)
    model = make_model(n_modes=len(MODES)).train()
    rng = np.random.default_rng(0)
    codes, mask = random_history(batch=16, rng=rng)

    targets = torch.zeros(16, SEQLEN, SESSION, LEVELS, dtype=torch.long)
    modes = torch.as_tensor(
        rng.integers(1, 3, size=(16, SEQLEN, SESSION)), dtype=torch.long
    )
    # mode 1 (repeat) -> level-0 token 0; mode 2 (explore) -> level-0 token 1
    targets[..., 0] = (modes == 2).long()
    targets[..., 1] = torch.as_tensor(
        rng.integers(0, CODEBOOK, size=(16, SEQLEN, SESSION)), dtype=torch.long
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(150):
        optimiser.zero_grad()
        loss = model.loss(codes, mask, targets, target_modes=modes)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        flat = torch.zeros(2, LEVELS, dtype=torch.long)
        query = model(codes[:1], mask[:1]).expand(2, -1)
        conditioned = model.condition(query, torch.tensor([1, 2]))
        first_level = model.decode_logits(conditioned, flat)[0]

    assert int(first_level[0].argmax()) == 0, "repeat mode should emit the repeat region"
    assert int(first_level[1].argmax()) == 1, "explore mode should emit the explore region"


def test_soft_prompt_requires_a_user_vector():
    model = make_model(user_dim=6)
    codes, mask = random_history()
    with pytest.raises(ValueError, match="soft prompt"):
        model(codes, mask)


def test_soft_prompt_changes_the_query_and_keeps_the_shape():
    model = make_model(user_dim=6)
    codes, mask = random_history()
    with torch.no_grad():
        reps_a = model.encode_history(codes, mask, torch.zeros(4, 6))
        reps_b = model.encode_history(codes, mask, torch.randn(4, 6))
    assert reps_a.shape == (4, SEQLEN, DIM)
    assert reps_b.shape == (4, SEQLEN, DIM)
    assert not torch.allclose(reps_a, reps_b)


def test_soft_prompt_reaches_the_earliest_history_position():
    """Causal attention means position 0 may only read the prompt -- so it must move."""
    model = make_model(user_dim=6)
    codes, mask = random_history()
    with torch.no_grad():
        first_a = model.encode_history(codes, mask, torch.zeros(4, 6))[:, 0, :]
        first_b = model.encode_history(codes, mask, torch.randn(4, 6) * 5)[:, 0, :]
    assert not torch.allclose(first_a, first_b)


def test_soft_prompt_is_per_user():
    model = make_model(user_dim=6)
    codes, mask = random_history()
    users = torch.zeros(4, 6)
    users[0] = torch.randn(6) * 3
    with torch.no_grad():
        reps = model.encode_history(codes, mask, users)
        shared = model.encode_history(codes, mask, torch.zeros(4, 6))
    assert not torch.allclose(reps[0], shared[0])
    torch.testing.assert_close(reps[1], shared[1])


def test_both_mechanisms_compose():
    model = make_model(n_modes=len(MODES), user_dim=6)
    codes, mask = random_history()
    targets = torch.as_tensor(
        np.random.default_rng(1).integers(0, CODEBOOK, size=(4, SEQLEN, SESSION, LEVELS)),
        dtype=torch.long,
    )
    modes = torch.ones(4, SEQLEN, SESSION, dtype=torch.long)
    loss = model.loss(codes, mask, targets, target_modes=modes, user_vector=torch.randn(4, 6))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.soft_prompt.inner.weight.grad is not None
    assert model.mode_embedding.weight.grad is not None


def build_conditioned(sessions, **overrides):
    from mrec.models import build_model

    params = dict(
        tokenizer="residual_kmeans",
        tokenizer_params={"n_levels": 2, "codebook_size": 3, "seed": 0, "gpu": False},
        model={"embedding_dim": 8, "seqlen": 3, "session_len": 10, "num_blocks": 1},
        training={"epochs": 2, "batch_size": 4, "device": "cpu", "lr": 0.01},
        beam_width=8,
        n_valid_users=2,
        progress=False,
    )
    params.update(overrides)
    return build_model("genrec", **params).fit(sessions)


@pytest.fixture
def built(raw_config):
    from mrec.data.build import build
    from mrec.data.sessions import load_sessions

    return load_sessions(build(raw_config, skip_fetch=True))


def test_conditioned_genrec_runs_end_to_end(built):
    """Both mechanisms on, through tokenize -> train -> beam -> evaluate."""
    from mrec.eval.runner import build_requests, evaluate

    model = build_conditioned(built, control=True, user_prompt="mean_svd", decode_mode="repeat")
    assert model.cfg.n_modes == len(MODES)
    svd_width = np.load(f"{built.proc_dir}/emb_svd.npy").shape[1]
    assert model.cfg.user_dim == svd_width

    requests = build_requests(built, np.arange(built.n_users), "test", seqlen=3)
    scores = model.score(requests[:4])
    assert scores.shape == (4, built.n_items)
    assert np.isfinite(scores).any()

    metrics = evaluate(
        model, built, which="test", k=3, seqlen=3, n_users=2,
        seeds=(1013,), batch_size=2, progress=False,
    )
    assert metrics["value"].is_finite().all()


def test_decode_mode_is_the_only_thing_that_changes_between_passes(built):
    """One checkpoint, three modes: the frontier must not need three models."""
    from mrec.eval.runner import build_requests

    model = build_conditioned(built, control=True)
    requests = build_requests(built, np.arange(built.n_users), "test", seqlen=3)[:4]

    lists = {}
    for mode in MODES:
        model.decode_mode = mode
        lists[mode] = model.score(requests)
    assert set(lists) == set(MODES)
    for mode, scores in lists.items():
        assert scores.shape == (4, built.n_items), mode


def test_an_unknown_decode_mode_is_refused():
    from mrec.models import build_model

    with pytest.raises(ValueError, match="decode mode"):
        build_model("genrec", decode_mode="habitual")


def test_an_unknown_user_prompt_is_refused(built):
    with pytest.raises(ValueError, match="user prompt"):
        build_conditioned(built, user_prompt="telepathy")


def test_control_labels_cover_both_modes_on_real_windows(built):
    """If every training target were a repeat the token would have nothing to learn."""
    from mrec.models.features import repeat_labels, training_windows

    model = build_conditioned(built, control=True)
    windows = training_windows(built, seqlen=model.cfg.seqlen)
    modes, _ = model._conditioning(windows, model.cfg.seqlen)
    assert modes is not None
    assert set(np.unique(modes)).issubset({1, 2})
    assert repeat_labels(built).any(), "the fixture must contain at least one repeat"
