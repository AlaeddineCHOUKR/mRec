"""The generative retriever behind the `Recommender` protocol.

Fitting means: tokenize the catalog, build the prefix trie, train the decoder on the
same training windows every other model uses. Scoring means: beam-decode identifiers and
resolve them back to items.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from mrec.data.sessions import Sessions
from mrec.eval.protocol import EVAL_SEEDS, sample_eval_users
from mrec.eval.runner import Request
from mrec.models.features import (
    history_mean_embedding,
    repeat_labels,
    session_item_matrix,
    session_windows,
    training_windows,
)
from mrec.models.genrec import MODES, CodeTrie, GenerativeRetriever, GenRecConfig
from mrec.tokenizers import TOKENIZERS, apply_collision_policy, collisions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Split:
    """One training or validation split: sessions, targets and their conditioning."""

    seq: np.ndarray  # [N, L, S] input item ids
    pos: np.ndarray  # [N, L, S] target item ids
    modes: np.ndarray | None  # [N, L, S] control token per target
    users: np.ndarray | None  # [N, user_dim] soft-prompt vector per window


class GenRecRecommender:
    """`genrec`: semantic IDs plus a constrained decoder."""

    name = "genrec"

    def __init__(
        self,
        tokenizer: str = "residual_kmeans",
        tokenizer_params: dict | None = None,
        collision_policy: str = "extra_token",
        embeddings: str = "svd",
        model: dict | None = None,
        training: dict | None = None,
        beam_width: int = 32,
        n_valid_users: int = 1000,
        control: bool = False,
        decode_mode: str = "free",
        user_prompt: str | None = None,
        progress: bool = True,
    ) -> None:
        self.tokenizer_name = tokenizer
        self.tokenizer_params = tokenizer_params or {"n_levels": 3, "codebook_size": 256}
        self.collision_policy = collision_policy
        self.embeddings = embeddings
        self.model_params = model or {}
        self.training = {
            "lr": 1e-3, "batch_size": 256, "epochs": 30, "device": "cuda", "patience": 0
        }
        self.training.update(training or {})
        self.beam_width = beam_width
        self.n_valid_users = n_valid_users
        # `control` adds the repeat/explore token; `decode_mode` picks which one a
        # scoring pass asks for, so one checkpoint traces the whole frontier
        self.control = control
        if decode_mode not in MODES:
            raise ValueError(f"unknown decode mode `{decode_mode}`; known: {MODES}")
        self.decode_mode = decode_mode
        # `None` disables the soft prompt; "mean_svd" is the plain long-term user vector
        self.user_prompt = user_prompt
        self.progress = progress
        self.user_vectors: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.model: GenerativeRetriever | None = None
        self.trie: CodeTrie | None = None
        self.codes: np.ndarray | None = None
        self.log: list[dict] = []
        self.summary: dict = {}

    def _tokenize(self, sessions: Sessions) -> np.ndarray:
        table = np.load(f"{sessions.proc_dir}/emb_{self.embeddings}.npy")
        tokenizer = TOKENIZERS[self.tokenizer_name](**self.tokenizer_params)
        tokenizer.fit(table)
        codes = apply_collision_policy(tokenizer.encode(table), self.collision_policy)
        report = collisions(tokenizer.encode(table))
        self.summary["collision_rate"] = report.collision_rate
        self.summary["largest_bucket"] = report.largest_bucket
        logger.info(
            "%s: %d levels, collision rate %.3f, largest bucket %d",
            self.tokenizer_name,
            codes.shape[1],
            report.collision_rate,
            report.largest_bucket,
        )
        return codes.astype(np.int64)

    def fit(self, sessions: Sessions) -> GenRecRecommender:
        self.sessions = sessions
        self.codes = self._tokenize(sessions)
        self.trie = CodeTrie(self.codes)
        self.session_matrix = session_item_matrix(sessions)

        if self.user_prompt is not None:
            self.user_vectors = self._user_vectors(sessions)
        if self.control:
            self.labels = repeat_labels(sessions)

        cfg = GenRecConfig(
            n_levels=self.codes.shape[1],
            codebook_size=int(self.codes.max()) + 1,
            n_modes=len(MODES) if self.control else 1,
            user_dim=0 if self.user_vectors is None else self.user_vectors.shape[1],
            **self.model_params,
        )
        self.cfg = cfg
        self.model = GenerativeRetriever(cfg).to(self.training["device"])

        windows = training_windows(sessions, seqlen=cfg.seqlen)
        valid_users = sample_eval_users(
            np.arange(sessions.n_users), self.n_valid_users, EVAL_SEEDS[0]
        )
        valid_windows = [
            (int(u), int(t))
            for u in valid_users
            for t in sessions.held_out_session_ids(int(u), "valid")
        ]
        logger.info(
            "training windows: %d, validation windows: %d", len(windows), len(valid_windows)
        )
        if self.training["epochs"] == 0:
            logger.info("epochs=0: skipping training (checkpoint expected)")
            return self
        train_seq, train_pos = session_windows(sessions, windows, cfg.seqlen)
        valid_seq, valid_pos = session_windows(sessions, valid_windows, cfg.seqlen)
        self._train(
            _Split(train_seq, train_pos, *self._conditioning(windows, cfg.seqlen)),
            _Split(valid_seq, valid_pos, *self._conditioning(valid_windows, cfg.seqlen)),
        )
        return self

    def _user_vectors(self, sessions: Sessions) -> np.ndarray:
        if self.user_prompt == "mean_svd":
            table = np.load(f"{sessions.proc_dir}/emb_svd.npy")
            vectors = history_mean_embedding(sessions, table)
        else:
            raise ValueError(f"unknown user prompt `{self.user_prompt}`")
        logger.info("soft prompt `%s`: %s", self.user_prompt, vectors.shape)
        return vectors

    def _conditioning(self, windows, seqlen: int):
        """Per-target repeat/explore modes and per-window user vectors for a split."""
        users = np.array([u for u, _ in windows])
        modes = None
        if self.labels is not None:
            starts = np.array([target - seqlen for _, target in windows])
            index = starts[:, None] + np.arange(seqlen)[None, :] + 1  # target sessions
            # MODES = (free, repeat, explore): 1 where already heard, else 2
            modes = np.where(self.labels[index], 1, 2).astype(np.int64)
        vectors = None if self.user_vectors is None else self.user_vectors[users]
        return modes, vectors

    def _codes_for(self, items: np.ndarray) -> torch.Tensor:
        """Map item ids (`-1` = padding) to code tokens, padding to `codebook_size`."""
        assert self.codes is not None
        pad = self.cfg.codebook_size
        out = np.full((*items.shape, self.cfg.n_levels), pad, dtype=np.int64)
        present = items >= 0
        out[present] = self.codes[items[present]]
        return torch.as_tensor(out, device=self.training["device"])

    def _batch(self, split: _Split, index) -> dict:
        device = self.training["device"]
        batch = {
            "seq_codes": self._codes_for(split.seq[index]),
            "mask": torch.as_tensor(split.seq[index][..., 0] >= 0, device=device),
            "target_codes": self._codes_for(split.pos[index]),
        }
        if split.modes is not None:
            batch["target_modes"] = torch.as_tensor(split.modes[index], device=device)
        if split.users is not None:
            batch["user_vector"] = torch.as_tensor(split.users[index], device=device)
        return batch

    def _train(self, train: _Split, valid: _Split) -> None:
        assert self.model is not None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training["lr"])
        batch_size = self.training["batch_size"]
        rng = np.random.default_rng(0)
        best = np.inf
        best_epoch = -1
        best_state = None
        patience = self.training.get("patience", 0)
        for epoch in range(self.training["epochs"]):
            started = time.perf_counter()
            self.model.train()
            order = rng.permutation(len(train.seq))
            total = 0.0
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                loss = self.model.loss(**self._batch(train, index))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(index)
            train_loss = total / len(order)

            self.model.eval()
            with torch.no_grad():
                valid_total = 0.0
                for start in range(0, len(valid.seq), batch_size):
                    sl = slice(start, start + batch_size)
                    batch = self._batch(valid, sl)
                    valid_total += float(self.model.loss(**batch)) * len(batch["mask"])
                valid_loss = valid_total / len(valid.seq)

            self.log.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "seconds": round(time.perf_counter() - started, 1),
                }
            )
            improved = valid_loss < best
            if improved:
                best, best_epoch = valid_loss, epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            logger.info(
                "epoch %d  train %.4f  valid %.4f  %.0fs%s",
                epoch, train_loss, valid_loss, self.log[-1]["seconds"], "  *" if improved else "",
            )
            # `patience = 0` keeps the fixed-budget behaviour; a positive value trains each
            # configuration to its own convergence point, which is what makes tokenizers
            # with different code-space sizes comparable
            if patience and epoch - best_epoch >= patience:
                logger.info("no improvement for %d epochs; stopping at %d", patience, epoch)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.summary["best_valid_loss"] = best
        self.summary["best_epoch"] = best_epoch
        self.summary["epochs_run"] = len(self.log)

    def score(self, requests: Sequence[Request]) -> np.ndarray:  # [B, n_items]
        assert self.model is not None and self.trie is not None
        device = self.training["device"]
        starts = np.array([r.target - self.cfg.seqlen for r in requests])
        index = starts[:, None] + np.arange(self.cfg.seqlen)[None, :]
        history = self.session_matrix[index]
        seq = self._codes_for(history)
        mask = torch.as_tensor(history[..., 0] >= 0, device=device)

        conditioning: dict = {}
        if self.user_vectors is not None:
            users = np.array([r.user for r in requests])
            conditioning["user_vector"] = torch.as_tensor(
                self.user_vectors[users], device=device
            )
        if self.control:
            conditioning["mode"] = torch.full(
                (len(requests),), MODES.index(self.decode_mode), device=device
            )

        self.model.eval()
        with torch.no_grad():
            query = self.model(seq, mask, **conditioning)
            beams = self.model.beam_search(query, self.trie, self.beam_width)

        scores = np.full((len(requests), self.sessions.n_items), -np.inf, dtype=np.float32)
        for i, candidates in enumerate(beams):
            for code, score in candidates:
                for item in self.trie.resolve(code):
                    scores[i, item] = max(scores[i, item], score)
        return scores


MODELS = {"genrec": GenRecRecommender}
