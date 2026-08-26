"""The generative retriever: constrained decoding is the property that must hold."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrec.models.genrec import CodeTrie, GenerativeRetriever, GenRecConfig

LEVELS, BOOK, DIM = 3, 6, 8
CFG = GenRecConfig(
    embedding_dim=DIM, seqlen=4, session_len=3, num_blocks=1, num_heads=2,
    dropout=0.0, n_levels=LEVELS, codebook_size=BOOK,
)


@pytest.fixture
def codes():
    rng = np.random.default_rng(0)
    return np.unique(rng.integers(0, BOOK, size=(40, LEVELS)), axis=0)


@pytest.fixture
def trie(codes):
    return CodeTrie(codes)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GenerativeRetriever(CFG)


def batch(rng, codes, n=3):
    """History and target sessions expressed as semantic IDs, `codebook_size` = padding."""
    L, S = CFG.seqlen, CFG.session_len
    pick = rng.integers(0, len(codes), size=(n, L, S))
    seq = torch.as_tensor(codes[pick])
    seq[:, 0] = BOOK  # a fully padded leading session
    target = torch.as_tensor(codes[rng.integers(0, len(codes), size=(n, L, S))])
    mask = torch.as_tensor((seq[..., 0] != BOOK).any(-1).numpy())
    return seq, mask, target


def test_trie_only_allows_tokens_that_continue_a_real_code(trie, codes):
    assert set(trie.allowed(()).tolist()) == set(codes[:, 0].tolist())
    for row in codes:
        for level in range(LEVELS):
            prefix = tuple(int(t) for t in row[:level])
            assert int(row[level]) in trie.allowed(prefix).tolist()


def test_trie_returns_nothing_for_a_prefix_the_catalog_lacks(trie):
    missing = tuple([BOOK - 1] * (LEVELS - 1))
    if len(trie.allowed(missing)) == 0:
        assert True
    else:  # the prefix happens to exist; construct one that cannot
        assert len(trie.allowed((BOOK + 5,))) == 0


def test_trie_resolves_a_code_back_to_its_items(trie, codes):
    for item, row in enumerate(codes):
        assert item in trie.resolve(tuple(int(t) for t in row))
    assert trie.resolve((BOOK + 1,) * LEVELS) == []


def test_beam_search_only_emits_identifiers_that_exist(model, trie, codes):
    rng = np.random.default_rng(1)
    seq, mask, _ = batch(rng, codes)
    query = model(seq, mask)
    valid = {tuple(int(t) for t in row) for row in codes}
    for beams in model.beam_search(query, trie, beam_width=5):
        assert beams
        for code, _ in beams:
            assert len(code) == LEVELS
            assert code in valid


def test_beam_search_returns_distinct_beams_in_descending_score(model, trie, codes):
    rng = np.random.default_rng(2)
    seq, mask, _ = batch(rng, codes)
    for beams in model.beam_search(model(seq, mask), trie, beam_width=5):
        scores = [score for _, score in beams]
        assert scores == sorted(scores, reverse=True)
        assert len({code for code, _ in beams}) == len(beams)


def test_beam_width_bounds_the_number_of_results(model, trie, codes):
    rng = np.random.default_rng(3)
    seq, mask, _ = batch(rng, codes)
    for width in (1, 3, 8):
        for beams in model.beam_search(model(seq, mask), trie, beam_width=width):
            assert 1 <= len(beams) <= width


def test_padded_history_positions_do_not_break_the_encoder(model, codes):
    rng = np.random.default_rng(4)
    seq, mask, _ = batch(rng, codes)
    out = model.encode_history(seq, mask)
    assert out.shape == (3, CFG.seqlen, DIM)
    assert torch.isfinite(out).all()


def test_loss_is_finite_and_trains(model, codes):
    rng = np.random.default_rng(5)
    seq, mask, target = batch(rng, codes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        loss = model.loss(seq, mask, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_an_unseen_item_still_gets_a_representation(model, codes):
    """Cold start in one line: the representation comes from the codes, not from a row."""
    unseen = torch.as_tensor([[BOOK - 1] * LEVELS])
    assert torch.isfinite(model.item_representation(unseen)).all()


def test_genrec_runs_end_to_end_under_the_protocol(raw_config):
    """Tokenize, train, beam-decode and evaluate on the synthetic fixture, on CPU."""
    from mrec.data.build import build
    from mrec.data.sessions import load_sessions
    from mrec.eval.runner import build_requests, evaluate, top_k
    from mrec.models import build_model

    sessions = load_sessions(build(raw_config, skip_fetch=True))
    model = build_model(
        "genrec",
        tokenizer="residual_kmeans",
        tokenizer_params={"n_levels": 2, "codebook_size": 3, "seed": 0, "gpu": False},
        model={"embedding_dim": 8, "seqlen": 3, "session_len": 10, "num_blocks": 1},
        training={"epochs": 2, "batch_size": 4, "device": "cpu", "lr": 0.01},
        beam_width=8,
        n_valid_users=2,
        progress=False,
    ).fit(sessions)

    requests = build_requests(sessions, np.arange(sessions.n_users), "test", seqlen=3)
    scores = model.score(requests[:4])
    assert scores.shape == (4, sessions.n_items)
    # every scored item must be one the beam actually reached
    assert np.isfinite(scores).any()
    ranked = top_k(scores, 3)
    assert ranked.shape == (4, 3)

    metrics = evaluate(
        model, sessions, which="test", k=3, seqlen=3, n_users=2,
        seeds=(1013,), batch_size=2, progress=False,
    )
    assert metrics["value"].is_finite().all()


def test_genrec_can_reach_items_with_no_interaction_history(raw_config):
    """The payoff of generating identifiers: reachability does not require a trained row."""
    from mrec.data.build import build
    from mrec.data.sessions import load_sessions
    from mrec.eval.runner import Request
    from mrec.models import build_model

    sessions = load_sessions(build(raw_config, skip_fetch=True))
    model = build_model(
        "genrec",
        tokenizer_params={"n_levels": 2, "codebook_size": 3, "seed": 0, "gpu": False},
        model={"embedding_dim": 8, "seqlen": 3, "session_len": 10, "num_blocks": 1},
        training={"epochs": 1, "batch_size": 4, "device": "cpu", "lr": 0.01},
        beam_width=16, n_valid_users=2, progress=False,
    ).fit(sessions)

    target = int(sessions.held_out_session_ids(0, "test")[0])
    reachable = np.flatnonzero(np.isfinite(model.score([Request(0, target, np.array([]))])[0]))
    history = set(sessions.history(0).tolist())
    assert len(reachable) > 0
    assert not set(reachable.tolist()) <= history or len(history) == sessions.n_items


def _reference_beam_search(model, query, trie, beam_width):
    """The per-query beam this module used before it was batched.

    Kept as a test oracle rather than in the module: the batched implementation is
    an optimisation, so it has to return the same beams in the same order, not
    merely beams that are plausible.
    """
    device = query.device
    results = []
    for b in range(len(query)):
        beams = [((), 0.0, query[b], model.level_start)]
        for level in range(model.cfg.n_levels):
            candidates = []
            states = torch.stack([beam[3] for beam in beams])
            hidden = torch.stack([beam[2] for beam in beams])
            new_state = model.decoder(states, hidden)
            logprobs = model.heads[level](new_state).log_softmax(-1)
            for i, (prefix, score, _, _) in enumerate(beams):
                allowed = trie.allowed(prefix)
                if not len(allowed):
                    continue
                allowed_t = torch.as_tensor(allowed, device=device)
                top = torch.topk(logprobs[i, allowed_t], min(beam_width, len(allowed)))
                for value, index in zip(top.values, top.indices, strict=True):
                    token = int(allowed[int(index)])
                    candidates.append(
                        (
                            (*prefix, token),
                            score + float(value),
                            new_state[i],
                            model.code_embedding[level](torch.as_tensor(token, device=device)),
                        )
                    )
            candidates.sort(key=lambda c: -c[1])
            beams = candidates[:beam_width]
        results.append([(prefix, score) for prefix, score, _, _ in beams])
    return results


@pytest.mark.parametrize(
    ("n_levels", "codebook_size", "n_items", "beam_width"),
    [(3, 16, 200, 8), (4, 32, 500, 16), (2, 8, 60, 32)],
)
def test_batched_beam_matches_the_per_query_reference(
    n_levels, codebook_size, n_items, beam_width
):
    torch.manual_seed(0)
    codes = np.random.default_rng(0).integers(0, codebook_size, size=(n_items, n_levels))
    trie = CodeTrie(codes)
    cfg = GenRecConfig(
        embedding_dim=32, n_levels=n_levels, codebook_size=codebook_size, seqlen=30
    )
    model = GenerativeRetriever(cfg).eval()
    query = torch.randn(7, 32)

    with torch.no_grad():
        expected = _reference_beam_search(model, query, trie, beam_width)
    actual = model.beam_search(query, trie, beam_width)

    assert [len(rows) for rows in actual] == [len(rows) for rows in expected]
    for got, want in zip(actual, expected, strict=True):
        assert [code for code, _ in got] == [code for code, _ in want]
        for (_, a), (_, b) in zip(got, want, strict=True):
            assert a == pytest.approx(b, abs=1e-5)


def test_dense_trie_agrees_with_the_prefix_lookup():
    codes = np.random.default_rng(1).integers(0, 12, size=(300, 3))
    trie = CodeTrie(codes)
    allowed, child = trie.dense(12, torch.device("cpu"))

    assert set(np.flatnonzero(allowed[0][0].numpy())) == set(trie.allowed(()).tolist())
    for first in trie.allowed(()):
        node = int(child[0][0, int(first)])
        expected = set(trie.allowed((int(first),)).tolist())
        assert set(np.flatnonzero(allowed[1][node].numpy())) == expected


def test_dense_trie_rejects_a_token_outside_the_codebook():
    trie = CodeTrie(np.array([[0, 1], [1, 5]]))
    with pytest.raises(ValueError, match="beyond codebook"):
        trie.dense(4, torch.device("cpu"))
