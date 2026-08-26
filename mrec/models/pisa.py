"""PISA: a session transformer whose attention weights are ACT-R activations.

Ported from `au2actr/models/baselines/pisa.py` (TensorFlow 1.x graph style) to PyTorch.
The architecture in one pass:

1. Each session becomes one vector: a weighted sum of its item embeddings, where the
   weights are ACT-R activations — base-level learning, spreading activation and
   partial matching — combined by learned positive scalars, flattened by a power and
   normalized within the session.
2. The `seqlen` session vectors go through a causal self-attention stack: the
   short-term user representation.
3. A long-term representation is the BLL-weighted sum of the user's favourite tracks.
4. The two are fused by a learned softmax gate, added back to the input as a skip
   connection, and renormalized.
5. Scores are inner products against the item table, with the long-term head mixed in
   at weight `lbda_ls`.

Item ids are 1-based with row 0 of the embedding table pinned to zero (padding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

EPS = 1e-8


@dataclass(frozen=True)
class PisaConfig:
    embedding_dim: int = 128
    seqlen: int = 30
    session_len: int = 10
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.0
    num_favs: int = 20
    flatten_actr: float = 0.5
    lbda_task: float = 0.9
    lbda_pos: float = 0.9
    lbda_ls: float = 0.4
    # the reference attaches an L2 regularizer to the embedding tables but never collects
    # the regularization loss, so `l2_emb: 1e-5` in its config is inert. Kept for the record.
    l2_emb: float = 0.0
    input_scale: bool = True
    causal: bool = True


class FeedForward(nn.Module):
    """The reference's two `conv1d(kernel_size=1)` layers with a residual connection."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.inner = nn.Linear(dim, dim)
        self.outer = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # [B, L, D]
        h = self.dropout(F.relu(self.inner(x)))
        return self.dropout(self.outer(h)) + x


class AttentionBlock(nn.Module):
    """Pre-norm self-attention, exactly as `multi_head_attention_blocks` composes it.

    The reference normalizes the queries but not the keys or values, and adds the
    *unnormalized* queries back as the residual — not the usual pre-norm arrangement,
    and reproduced here rather than tidied.
    """

    def __init__(self, cfg: PisaConfig) -> None:
        super().__init__()
        dim = cfg.embedding_dim
        self.norm_in = nn.LayerNorm(dim, eps=1e-6)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim, eps=1e-6)
        self.ff = FeedForward(dim, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.num_heads = cfg.num_heads
        self.causal = cfg.causal

    def forward(self, seq: Tensor, key_mask: Tensor) -> Tensor:  # [B, L, D], [B, L]
        b, length, dim = seq.shape
        queries = self.norm_in(seq)
        q = self.query(queries).view(b, length, self.num_heads, -1).transpose(1, 2)
        k = self.key(seq).view(b, length, self.num_heads, -1).transpose(1, 2)
        v = self.value(seq).view(b, length, self.num_heads, -1).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dim / self.num_heads)
        scores = scores.masked_fill(~key_mask[:, None, None, :], -(2.0**32) + 1)
        if self.causal:
            causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
            scores = scores.masked_fill(~causal, -(2.0**32) + 1)
        weights = self.dropout(scores.softmax(dim=-1))
        # queries that are pure padding contribute nothing back
        attended = (weights @ v).transpose(1, 2).reshape(b, length, dim)
        query_mask = (queries.abs().sum(-1) > 0).to(seq.dtype)[..., None]
        out = attended * query_mask + queries

        out = self.ff(self.norm_ff(out))
        return out * key_mask[..., None].to(seq.dtype)


class PISA(nn.Module):
    def __init__(self, n_items: int, item_embeddings: np.ndarray, cfg: PisaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.embedding_dim

        # row 0 is padding; the catalog occupies rows 1..n_items
        table = torch.zeros(n_items + 1, dim)
        table[1:] = torch.as_tensor(item_embeddings[:, :dim], dtype=torch.float32)
        self.item_embedding = nn.Parameter(table)
        self.position_embedding = nn.Parameter(torch.randn(cfg.seqlen, dim) / math.sqrt(dim))
        # BLL, spreading activation, partial matching — kept positive at use
        self.actr_weights = nn.Parameter(0.1 * torch.ones(3))
        self.blocks = nn.ModuleList(AttentionBlock(cfg) for _ in range(cfg.num_blocks))
        self.fusion = nn.Linear(2 * dim, 2)

    @property
    def items(self) -> Tensor:
        """The embedding table with padding pinned to zero. `# [n_items + 1, D]`"""
        return torch.cat([self.item_embedding.new_zeros(1, self.cfg.embedding_dim),
                          self.item_embedding[1:]])

    def session_representation(
        self, ids: Tensor, bll: Tensor, spread: Tensor, previous_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Weighted session vectors and their item embeddings. `# [B, L, D], [B, L, S, D]`"""
        table = self.items
        emb = table[ids]  # [B, L, S, D]
        previous = table[previous_ids]  # [B, L, S, D]

        # partial matching: similarity to the previous session, self-similarity removed
        similarity = emb @ previous.transpose(-1, -2)  # [B, L, S, S]
        similarity = similarity - torch.diag_embed(torch.diagonal(similarity, dim1=-2, dim2=-1))
        partial = similarity.sum(-1)  # [B, L, S]

        components = torch.stack([bll, spread, partial], dim=-1)  # [B, L, S, 3]
        weights = (F.relu(components) + EPS) * (F.relu(self.actr_weights) + EPS)
        weights = F.relu(weights.sum(-1)) + EPS  # [B, L, S]
        flattened = weights.pow(self.cfg.flatten_actr)
        weights = flattened / flattened.sum(-1, keepdim=True)
        return (emb * weights[..., None]).sum(2), emb

    def forward(
        self,
        seq_in: Tensor,  # [B, L, S]
        seq_bll: Tensor,
        seq_spread: Tensor,
        fav_ids: Tensor,  # [B, F]
        fav_bll: Tensor,  # [B, F]
    ) -> tuple[Tensor, Tensor]:
        """Short-term and long-term user representations. `# [B, L, D] each`"""
        cfg = self.cfg
        # each session attends to the one before it; the first attends to itself
        previous_ids = torch.cat([seq_in[:, :1], seq_in[:, : cfg.seqlen - 1]], dim=1)
        session_rep, _ = self.session_representation(seq_in, seq_bll, seq_spread, previous_ids)

        n_elems = (seq_in != 0).sum(-1)  # [B, L]
        mask = n_elems != 0
        seq = session_rep * math.sqrt(cfg.embedding_dim) if cfg.input_scale else session_rep
        seq = seq + self.position_embedding
        seq = seq * mask[..., None].to(seq.dtype)

        for block in self.blocks:
            seq = block(seq, mask)
        short = self._normalize(seq, mask)

        favourites = self.items[fav_ids]  # [B, F, D]
        long = (fav_bll[..., None] * favourites).sum(1, keepdim=True)
        long = long.expand(-1, cfg.seqlen, -1)
        long = self._normalize(long, mask)

        gate = self.fusion(torch.cat([long, short], dim=-1)).softmax(dim=-1)
        fused = gate[..., :1] * long + gate[..., 1:] * short
        short = self._normalize(fused + session_rep, mask)
        return short, long

    @staticmethod
    def _normalize(x: Tensor, mask: Tensor) -> Tensor:
        """L2-normalize, with the reference's epsilon on padded rows."""
        guard = (~mask).to(x.dtype)[..., None] * 1e-7
        return x / (x + guard).norm(dim=-1, keepdim=True).clamp_min(EPS)

    def score_catalog(self, short: Tensor, long: Tensor) -> Tensor:
        """Scores over the whole catalog from the last position. `# [B, n_items + 1]`"""
        table = self.items
        scores = short[:, -1, :] @ table.T
        if self.cfg.lbda_ls > 0:
            scores = scores + self.cfg.lbda_ls * (long[:, -1, :] @ table.T)
        return scores


def pisa_loss(
    model: PISA,
    short: Tensor,
    example: dict[str, Tensor],
) -> Tensor:
    """BPR at track level plus a session-level regression, as `_create_loss` combines them."""
    cfg = model.cfg
    pos_ids, neg_ids = example["pos"], example["neg"]
    previous_pos = example["seq_in"]
    weighted_pos, pos_emb = model.session_representation(
        pos_ids, example["pos_bll"], example["pos_spread"], previous_pos
    )
    neg_emb = model.items[neg_ids]
    mask = (pos_ids != 0).to(short.dtype)  # [B, L, S]

    n_elems = (pos_ids != 0).sum(-1)
    weighted_pos = weighted_pos / weighted_pos.norm(dim=-1, keepdim=True).clamp_min(EPS)

    seq_bpr = _bpr(short, pos_emb, neg_emb, mask)
    pos_bpr = _bpr(weighted_pos, pos_emb, neg_emb, mask)
    loss = cfg.lbda_pos * pos_bpr + (1.0 - cfg.lbda_pos) * seq_bpr

    if cfg.lbda_ls > 0:
        loss = loss + cfg.lbda_ls * _bpr(example["long"], pos_emb, neg_emb, mask)

    is_target = (n_elems != 0).to(short.dtype)
    similarity = (weighted_pos * short).sum(-1)
    regression = ((1.0 - similarity) * is_target).mean() / is_target.sum().clamp_min(1.0)
    regression = torch.nan_to_num(regression)
    return cfg.lbda_task * loss + (1.0 - cfg.lbda_task) * regression


def _bpr(seq: Tensor, pos_emb: Tensor, neg_emb: Tensor, mask: Tensor) -> Tensor:
    """`-log sigmoid(pos - neg)` over the items of each target session. `# [B, L, S]`"""
    pos_score = (seq[:, :, None, :] * pos_emb).sum(-1)
    neg_score = (seq[:, :, None, :] * neg_emb).sum(-1)
    loss = -F.logsigmoid(pos_score - neg_score) * mask
    return torch.nan_to_num(loss).mean()
