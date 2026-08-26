"""PISA behind the `Recommender` protocol: fit once, then score the full catalog."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from mrec.data.sessions import Sessions
from mrec.eval.protocol import EVAL_SEEDS, sample_eval_users
from mrec.eval.runner import Request
from mrec.models.actr import session_cooccurrence
from mrec.models.features import ActrFeatures, training_windows
from mrec.models.pisa import PISA, PisaConfig
from mrec.train.trainer import TrainConfig, train_pisa

logger = logging.getLogger(__name__)


class PisaRecommender:
    """Trains PISA, then answers evaluation requests with catalog-wide scores."""

    name = "pisa"

    def __init__(
        self,
        model: dict | None = None,
        training: dict | None = None,
        embeddings: str = "svd",
        n_valid_users: int = 1000,
        checkpoint: str | None = None,
        progress: bool = True,
    ) -> None:
        self.model_config = PisaConfig(**(model or {}))
        self.train_config = TrainConfig(**(training or {}))
        self.embeddings = embeddings
        self.n_valid_users = n_valid_users
        self.checkpoint = checkpoint
        self.progress = progress
        self.model: PISA | None = None
        self.features: ActrFeatures | None = None
        self.sessions: Sessions | None = None
        self.log: list[dict] = []
        self.summary: dict = {}

    def fit(self, sessions: Sessions, proc_dir: str = "data/proc") -> PisaRecommender:
        self.sessions = sessions
        adjacency = session_cooccurrence(sessions, progress=self.progress)
        self.features = ActrFeatures(
            sessions,
            adjacency,
            seqlen=self.model_config.seqlen,
            num_favs=self.model_config.num_favs,
        )

        table = np.load(f"{sessions.proc_dir}/emb_{self.embeddings}.npy")
        self.model = PISA(sessions.n_items, table, self.model_config)

        if self.checkpoint is not None:
            # Re-decode an existing run without paying for training again. The
            # ACT-R features above are still built because `score` needs them,
            # but the training windows are not: nothing reads them here.
            state = torch.load(self.checkpoint, map_location=self.train_config.device)
            self.model.load_state_dict(state)
            self.model.to(self.train_config.device)
            self.summary = {"checkpoint": self.checkpoint, "trained": False}
            logger.info("loaded PISA checkpoint %s (no training)", self.checkpoint)
            return self

        histories = [sessions.history(u) for u in range(sessions.n_users)]

        windows = training_windows(sessions, seqlen=self.model_config.seqlen)
        logger.info("training windows: %d over %d users", len(windows), sessions.n_users)
        train_features = self.features.precompute(windows, progress=self.progress)

        valid_users = sample_eval_users(
            np.arange(sessions.n_users), self.n_valid_users, EVAL_SEEDS[0]
        )
        valid_windows = [
            (int(u), int(t))
            for u in valid_users
            for t in sessions.held_out_session_ids(int(u), "valid")
        ]
        valid_features = self.features.precompute(valid_windows, progress=self.progress)

        self.log, self.summary = train_pisa(
            self.model,
            train_features,
            valid_features,
            histories,
            sessions.n_items,
            self.train_config,
        )
        return self

    def score(self, requests: Sequence[Request]) -> np.ndarray:  # [B, n_items]
        assert self.model is not None and self.features is not None
        device = self.train_config.device
        examples = [
            self.features.evaluation_example(r.user, r.target) for r in requests
        ]
        batch = {
            "seq_in": _stack([e.seq_in for e in examples], torch.long, device),
            "seq_bll": _stack([e.seq_bll for e in examples], torch.float32, device),
            "seq_spread": _stack([e.seq_spread for e in examples], torch.float32, device),
            "fav_ids": _stack([e.fav_ids for e in examples], torch.long, device),
            "fav_bll": _stack([e.fav_bll for e in examples], torch.float32, device),
        }
        self.model.eval()
        with torch.no_grad():
            short, long = self.model(**batch)
            scores = self.model.score_catalog(short, long)
        # drop the padding column, restoring 0-based item_idx
        return scores[:, 1:].float().cpu().numpy()


def _stack(arrays: list[np.ndarray], dtype, device) -> torch.Tensor:
    return torch.as_tensor(np.stack(arrays), dtype=dtype, device=device)


MODELS = {"pisa": PisaRecommender}
