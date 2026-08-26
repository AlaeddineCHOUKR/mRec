"""Semantic-ID generative retrieval: emit an identifier, token by token.

The decoder never scores a catalog. It generates a semantic ID — one token per
tokenizer level — and the beam is constrained to prefixes that actually exist, so every
completed beam names a real item. That constraint is the whole point: it is what lets
the model reach an item it has never seen a single interaction for, provided the item
has an embedding to tokenize.

Shapes: `L` sessions of history, `S` tracks per session, `C` code levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from mrec.models.pisa import AttentionBlock, PisaConfig

logger = logging.getLogger(__name__)


class CodeTrie:
    """The set of valid semantic IDs, indexed by prefix.

    Built once from the catalog's codes. `allowed(prefix)` answers "which tokens may
    follow?", which is what keeps the beam inside the space of real identifiers.
    """

    def __init__(self, codes: np.ndarray) -> None:
        self.codes = np.asarray(codes, dtype=np.int64)
        self.n_levels = self.codes.shape[1]
        self.children: list[dict[tuple[int, ...], np.ndarray]] = []
        for level in range(self.n_levels):
            mapping: dict[tuple[int, ...], set[int]] = {}
            for row in self.codes:
                prefix = tuple(int(t) for t in row[:level])
                mapping.setdefault(prefix, set()).add(int(row[level]))
            self.children.append({k: np.array(sorted(v)) for k, v in mapping.items()})
        self._dense: dict[tuple[int, str], tuple[list[Tensor], list[Tensor]]] = {}
        self.items: dict[tuple[int, ...], list[int]] = {}
        for item, row in enumerate(self.codes):
            self.items.setdefault(tuple(int(t) for t in row), []).append(item)

    def allowed(self, prefix: tuple[int, ...]) -> np.ndarray:
        """Tokens that may follow `prefix`; empty if the prefix is not in the catalog."""
        return self.children[len(prefix)].get(prefix, np.array([], dtype=np.int64))

    def resolve(self, code: tuple[int, ...]) -> list[int]:
        """Items carrying this exact code — more than one when the policy allows collisions."""
        return self.items.get(code, [])

    def dense(self, codebook_size: int, device: torch.device) -> tuple[list[Tensor], list[Tensor]]:
        """The same constraint as `allowed`, shaped for a batched beam.

        The dict lookup above answers one prefix at a time, which forces beam search
        into a Python loop over queries. Here each distinct prefix becomes a node id,
        so a whole batch of beams is constrained by two gathers:

          `allowed[level][node]` — which tokens may follow, as a boolean row
          `child[level][node, token]` — the node the beam moves to

        Cached per (codebook_size, device); the catalog does not change under us.
        """
        key = (codebook_size, str(device))
        if key in self._dense:
            return self._dense[key]

        allowed, child = [], []
        node = np.zeros(len(self.codes), dtype=np.int64)  # every item starts at the root
        n_nodes = 1
        for level in range(self.n_levels):
            if self.codes[:, level].max() >= codebook_size:
                raise ValueError(f"level {level} has a token beyond codebook {codebook_size}")
            edge = node * codebook_size + self.codes[:, level]
            unique, inverse = np.unique(edge, return_inverse=True)
            rows, tokens = unique // codebook_size, unique % codebook_size

            level_allowed = np.zeros((n_nodes, codebook_size), dtype=bool)
            level_allowed[rows, tokens] = True
            level_child = np.full((n_nodes, codebook_size), -1, dtype=np.int64)
            level_child[rows, tokens] = np.arange(len(unique))

            allowed.append(torch.as_tensor(level_allowed, device=device))
            child.append(torch.as_tensor(level_child, device=device))
            node, n_nodes = inverse.astype(np.int64), len(unique)

        self._dense[key] = (allowed, child)
        return allowed, child


MODES = ("free", "repeat", "explore")
"""Decode-time control over the repeat/explore mix.

GLIDE's control token selects a *discovery horizon* (familiar vs unfamiliar shows,
§4.3.1). The axis that matters on this data is repetition: 84% of held-out targets are
tracks the user has already heard, and Deezer reports RepBias -- the gap between the
recommended repeat ratio and the ground-truth one -- as a headline metric. So the token
selects repeat vs explore, and `free` is the unconditioned mode the model also trains,
which is what GLIDE's single-task variant collapses to.
"""


@dataclass(frozen=True)
class GenRecConfig:
    embedding_dim: int = 128
    seqlen: int = 30
    session_len: int = 10
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    n_levels: int = 3
    codebook_size: int = 256
    # 1 disables the control token entirely, giving the unconditioned model back
    n_modes: int = 1
    # 0 disables the soft prompt; otherwise the width of the user vector fed to it
    user_dim: int = 0
    soft_prompt_dropout: float = 0.05


class SoftPrompt(nn.Module):
    """GLIDE's embedding-to-prefix projection: a user vector as one sequence position.

    GLIDE (§4.1.2, §A) projects a 128-d collaborative-filtering user embedding into the
    LLM hidden size with a two-layer MLP (SiLU, residual, 5% dropout) and inserts it at
    the start of the user context. There is no text prompt here, so the analogue is a
    position prepended to the session sequence: causal attention means every history
    position can read it, which a bias added to the decoder's start state could not do.
    """

    def __init__(self, user_dim: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.inner = nn.Linear(user_dim, dim)
        self.outer = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.position = nn.Parameter(torch.randn(dim) * dim**-0.5)

    def forward(self, user_vector: Tensor) -> Tensor:  # [B, user_dim] -> [B, D]
        h = self.dropout(F.silu(self.inner(user_vector)))
        return self.dropout(self.outer(h)) + h + self.position


class GenerativeRetriever(nn.Module):
    """History encoder plus an autoregressive head over semantic-ID tokens."""

    def __init__(self, cfg: GenRecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.embedding_dim
        # one codebook embedding per level; an item is the sum of its level embeddings,
        # so an unseen item gets a representation from its codes alone
        self.code_embedding = nn.ModuleList(
            nn.Embedding(cfg.codebook_size + 1, dim, padding_idx=cfg.codebook_size)
            for _ in range(cfg.n_levels)
        )
        self.position_embedding = nn.Parameter(torch.randn(cfg.seqlen, dim) * dim**-0.5)
        block_cfg = PisaConfig(
            embedding_dim=dim,
            seqlen=cfg.seqlen,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            causal=True,
        )
        self.blocks = nn.ModuleList(AttentionBlock(block_cfg) for _ in range(cfg.num_blocks))
        self.level_start = nn.Parameter(torch.randn(dim) * dim**-0.5)
        self.decoder = nn.GRUCell(dim, dim)
        self.heads = nn.ModuleList(nn.Linear(dim, cfg.codebook_size) for _ in range(cfg.n_levels))

        self.mode_embedding = (
            nn.Embedding(cfg.n_modes, dim) if cfg.n_modes > 1 else None
        )
        if self.mode_embedding is not None:
            # start unconditioned: at initialisation every mode is the `free` mode, so
            # the control token can only earn its effect from the data
            nn.init.zeros_(self.mode_embedding.weight)
        self.soft_prompt = (
            SoftPrompt(cfg.user_dim, dim, cfg.soft_prompt_dropout) if cfg.user_dim else None
        )

    def item_representation(self, codes: Tensor) -> Tensor:
        """Sum of level embeddings. `# [..., C] -> [..., D]`"""
        out = 0
        for level, table in enumerate(self.code_embedding):
            out = out + table(codes[..., level])
        return out

    def encode_history(
        self, seq_codes: Tensor, mask: Tensor, user_vector: Tensor | None = None
    ) -> Tensor:
        """One vector per history position, causally. `# [B, L, S, C] -> [B, L, D]`

        With a soft prompt the sequence attended over is `L + 1` long; the extra
        position is sliced off again so the output stays aligned with the target
        sessions and the loss below does not have to know about it.
        """
        items = self.item_representation(seq_codes)  # [B, L, S, D]
        present = (seq_codes[..., 0] != self.cfg.codebook_size).to(items.dtype)  # [B, L, S]
        counts = present.sum(-1, keepdim=True).clamp_min(1.0)
        sessions = (items * present[..., None]).sum(2) / counts  # [B, L, D]
        seq = sessions + self.position_embedding
        seq = seq * mask[..., None].to(seq.dtype)

        prefix = 0
        if self.soft_prompt is not None:
            if user_vector is None:
                raise ValueError("this model was built with a soft prompt but got no user vector")
            prompt = self.soft_prompt(user_vector)[:, None, :]  # [B, 1, D]
            seq = torch.cat([prompt, seq], dim=1)
            # the prompt is always present, even for a user whose history is all padding
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
            prefix = 1

        for block in self.blocks:
            seq = block(seq, mask)
        return seq[:, prefix:, :]

    def condition(self, query: Tensor, mode: Tensor | None) -> Tensor:
        """Apply the control token to a query vector. `# [N, D], [N] -> [N, D]`"""
        if self.mode_embedding is None or mode is None:
            return query
        return query + self.mode_embedding(mode)

    def decode_logits(self, query: Tensor, target_codes: Tensor) -> list[Tensor]:
        """Teacher-forced logits for each level. `# [B, D], [B, C] -> C x [B, codebook]`"""
        state = query
        token = self.level_start.expand(len(query), -1)
        logits = []
        for level in range(self.cfg.n_levels):
            state = self.decoder(token, state)
            logits.append(self.heads[level](state))
            token = self.code_embedding[level](target_codes[:, level])
        return logits

    def forward(
        self,
        seq_codes: Tensor,
        mask: Tensor,
        user_vector: Tensor | None = None,
        mode: Tensor | None = None,
    ) -> Tensor:
        """The query vector the decoder conditions on. `# [B, D]`"""
        query = self.encode_history(seq_codes, mask, user_vector)[:, -1, :]
        return self.condition(query, mode)

    def loss(
        self,
        seq_codes: Tensor,
        mask: Tensor,
        target_codes: Tensor,
        target_modes: Tensor | None = None,
        user_vector: Tensor | None = None,
    ) -> Tensor:
        """Cross-entropy over the code tokens of every target session in the window.

        Position `i` predicts the session that follows input `i`, so one window carries
        `L` next-session targets rather than one. Every track of a target session is a
        positive against the same query, so the head learns the session's *distribution*
        over identifiers rather than a single item.

        With a control token each target is scored twice: once under the mode it
        actually belongs to and once under `free`. The second pass is what keeps the
        unconditioned distribution trainable in the same parameters, so a `free` decode
        and a `repeat` decode come from one checkpoint and differ only in the token.
        """
        reps = self.encode_history(seq_codes, mask, user_vector)  # [B, L, D]
        b, length, s, c = target_codes.shape
        pad = self.cfg.codebook_size
        valid = (target_codes[..., 0] != pad) & mask[..., None]  # [B, L, S]

        flat_query = reps[:, :, None, :].expand(-1, -1, s, -1).reshape(b * length * s, -1)
        flat_codes = target_codes.reshape(b * length * s, c)
        weights = valid.reshape(-1).to(reps.dtype)
        # padding carries the reserved token, which is not a class the heads emit; those
        # rows are zero-weighted, so clamping only keeps the index in range
        targets = flat_codes.clamp(max=pad - 1)

        passes: list[Tensor | None] = [None]
        if self.mode_embedding is not None and target_modes is not None:
            free = torch.zeros_like(target_modes.reshape(-1))
            passes = [target_modes.reshape(-1), free]

        total = 0.0
        for mode in passes:
            logits = self.decode_logits(self.condition(flat_query, mode), flat_codes)
            for level, level_logits in enumerate(logits):
                ce = F.cross_entropy(level_logits, targets[:, level], reduction="none")
                total = total + (ce * weights).sum() / weights.sum().clamp_min(1.0)
        return total / (self.cfg.n_levels * len(passes))

    @torch.no_grad()
    def beam_search(self, query: Tensor, trie: CodeTrie, beam_width: int) -> list[list[tuple]]:
        """Top beams per query, constrained to prefixes the catalog actually contains.

        Returns `(code, score)` pairs, most likely first. An unconstrained decoder would
        happily emit identifiers that name nothing.

        The whole batch advances together: one GRU step and one `topk` per level, with
        the trie applied as a gathered boolean mask. Scoring a cohort is otherwise
        dominated by Python overhead rather than by the model.
        """
        device = query.device
        n, width = len(query), beam_width
        codebook = self.cfg.codebook_size
        allowed, child = trie.dense(codebook, device)

        state = query[:, None, :].expand(n, width, -1).reshape(n * width, -1)
        token = self.level_start.expand(n * width, -1)
        # every beam shares the empty prefix at the root, so only one may expand --
        # otherwise the first level returns the same token `width` times
        scores = torch.full((n, width), -torch.inf, device=device)
        scores[:, 0] = 0.0
        nodes = torch.zeros(n, width, dtype=torch.long, device=device)
        chosen_tokens = torch.empty(n, width, 0, dtype=torch.long, device=device)

        for level in range(self.cfg.n_levels):
            state = self.decoder(token, state)
            logprobs = self.heads[level](state).log_softmax(-1).view(n, width, codebook)
            logprobs = logprobs.masked_fill(~allowed[level][nodes], -torch.inf)

            best = (scores[..., None] + logprobs).view(n, width * codebook).topk(width, dim=1)
            beam, chosen, scores = best.indices // codebook, best.indices % codebook, best.values

            dim = state.shape[-1]
            state = state.view(n, width, dim).gather(1, beam[..., None].expand(-1, -1, dim))
            state = state.reshape(n * width, dim)
            # a beam scored -inf has no valid continuation; clamp keeps the gather in
            # range and the beam is dropped below on its score, not on its node
            nodes = child[level][nodes.gather(1, beam), chosen].clamp_min(0)
            kept = chosen_tokens.gather(1, beam[..., None].expand(-1, -1, chosen_tokens.shape[-1]))
            chosen_tokens = torch.cat([kept, chosen[..., None]], dim=-1)
            token = self.code_embedding[level](chosen).view(n * width, -1)

        finite = torch.isfinite(scores).cpu().numpy()
        codes = chosen_tokens.cpu().numpy()
        values = scores.cpu().numpy()
        return [
            [
                (tuple(int(t) for t in codes[i, j]), float(values[i, j]))
                for j in range(width)
                if finite[i, j]
            ]
            for i in range(n)
        ]
