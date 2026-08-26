"""Training loop for PISA. Batches come from precomputed arrays, negatives are fresh.

Model selection follows the reference: the checkpoint kept is the one with the lowest
*validation loss*, not the best validation NDCG, and training stops after `patience`
epochs without improvement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from mrec.models.pisa import PISA, pisa_loss

logger = logging.getLogger(__name__)

TENSOR_FIELDS = ("seq_in", "seq_bll", "seq_spread", "pos", "pos_bll", "pos_spread")


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 100
    patience: int = 5
    seed: int = 2025
    device: str = "cuda"
    max_steps: int = 150_000


def sample_negatives(
    users: np.ndarray, histories: list[np.ndarray], shape, n_items: int, rng
) -> np.ndarray:
    """Uniform items none of the batch's users has played. `# [B, L, S]`"""
    out = rng.integers(0, n_items, size=shape)
    for i, user in enumerate(users):
        known = np.isin(out[i], histories[user])
        while known.any():
            out[i][known] = rng.integers(0, n_items, size=int(known.sum()))
            known = np.isin(out[i], histories[user])
    return out + 1


def make_batch(
    features: dict[str, np.ndarray],
    index: np.ndarray,
    histories: list[np.ndarray],
    n_items: int,
    rng: np.random.Generator,
    device: str,
) -> dict[str, Tensor]:
    batch = {}
    for field in TENSOR_FIELDS:
        array = features[field][index]
        dtype = torch.long if array.dtype.kind == "i" else torch.float32
        batch[field] = torch.as_tensor(array, dtype=dtype, device=device)
    batch["fav_ids"] = torch.as_tensor(features["fav_ids"][index], dtype=torch.long, device=device)
    batch["fav_bll"] = torch.as_tensor(
        features["fav_bll"][index], dtype=torch.float32, device=device
    )
    users = features["user"][index]
    negatives = sample_negatives(users, histories, features["seq_in"][index].shape, n_items, rng)
    batch["neg"] = torch.as_tensor(negatives, dtype=torch.long, device=device)
    return batch


def _forward_loss(model: PISA, batch: dict[str, Tensor]) -> Tensor:
    short, long = model(
        batch["seq_in"], batch["seq_bll"], batch["seq_spread"], batch["fav_ids"], batch["fav_bll"]
    )
    return pisa_loss(model, short, {**batch, "long": long})


def epoch_loss(
    model: PISA,
    features: dict[str, np.ndarray],
    histories: list[np.ndarray],
    n_items: int,
    cfg: TrainConfig,
    rng: np.random.Generator,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, int]:
    n = len(features["user"])
    order = rng.permutation(n) if train else np.arange(n)
    model.train(train)
    total, steps = 0.0, 0
    for start in range(0, n, cfg.batch_size):
        index = order[start : start + cfg.batch_size]
        batch = make_batch(features, index, histories, n_items, rng, cfg.device)
        if train:
            assert optimizer is not None
            loss = _forward_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = _forward_loss(model, batch)
        total += float(loss.detach()) * len(index)
        steps += 1
    return total / n, steps


def train_pisa(
    model: PISA,
    train_features: dict[str, np.ndarray],
    valid_features: dict[str, np.ndarray],
    histories: list[np.ndarray],
    n_items: int,
    cfg: TrainConfig,
) -> tuple[list[dict], dict]:
    """Fit `model`, returning the per-epoch log and the best checkpoint's state dict."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))

    history: list[dict] = []
    best_loss, best_epoch, best_state = np.inf, -1, None
    total_steps = 0
    for epoch in range(cfg.epochs):
        started = time.perf_counter()
        train_loss, steps = epoch_loss(
            model, train_features, histories, n_items, cfg, rng, train=True, optimizer=optimizer
        )
        valid_loss, _ = epoch_loss(
            model, valid_features, histories, n_items, cfg, rng, train=False
        )
        total_steps += steps
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        improved = valid_loss < best_loss
        if improved:
            best_loss, best_epoch = valid_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(
            "epoch %d  train %.5f  valid %.5f  %.0fs%s",
            epoch,
            train_loss,
            valid_loss,
            history[-1]["seconds"],
            "  *" if improved else "",
        )
        if epoch - best_epoch >= cfg.patience:
            logger.info("no improvement for %d epochs; stopping at %d", cfg.patience, epoch)
            break
        if total_steps >= cfg.max_steps:
            logger.info("reached max_steps=%d", cfg.max_steps)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("best epoch %d, valid loss %.5f", best_epoch, best_loss)
    return history, {"best_epoch": best_epoch, "best_valid_loss": best_loss}
