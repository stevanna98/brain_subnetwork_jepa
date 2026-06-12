"""Top-level spatiotemporal BS-JEPA model.

Forward pass (mirrors JEPA):
  1. Tokenize windowed BOLD into RSN-time tokens (shared tokenizer).
  2. Target encoder (EMA copy, no grad) sees the full grid → target latents.
  3. Context encoder sees visible tokens only → context latents.
  4. Predictor predicts target latents from context + (rsn, time) queries.
  5. Return (z_hat, z_tgt, ctx_embs), flattened for the existing jepa_loss.

The tokenizer is shared (like the static feature extractor); only the
Transformer encoder is EMA-tracked. Target latents are detached, so target
information never flows into the context encoder.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.atlas import AtlasMapping
from ..data.windowing import STSample, make_grid_ids
from ..masking.spatiotemporal_masking import STBatch, STMaskOutput
from .st_encoder import RSNTimeTokenizer, SpatioTemporalEncoder
from .st_predictor import SpatioTemporalPredictor


class STBSJEPA(nn.Module):
    """Spatiotemporal Brain Subnetwork JEPA."""

    def __init__(
        self,
        tokenizer: RSNTimeTokenizer,
        context_encoder: SpatioTemporalEncoder,
        predictor: SpatioTemporalPredictor,
        atlas: AtlasMapping,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.atlas = atlas

        self.target_encoder = copy.deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def _grid_ids(self, p: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rsn_ids, time_ids = make_grid_ids(p, self.atlas.num_rsns)
        return rsn_ids.to(device), time_ids.to(device)

    @torch.no_grad()
    def encode(self, sample: STSample) -> torch.Tensor:
        """Full-grid target-encoder latents for one subject → ``(P*K, d)``.

        Returned shape matches the per-token ``sample.rsn_ids`` so the generic
        diagnostics treat each RSN-time token as a "node".
        """
        tokens = self.tokenizer(sample.x_win.unsqueeze(0))      # (1, P, K, d)
        _, p, k, d = tokens.shape
        rsn_ids, time_ids = self._grid_ids(p, tokens.device)
        z = self.target_encoder(
            tokens.reshape(1, p * k, d), rsn_ids.unsqueeze(0), time_ids.unsqueeze(0)
        )
        return z.squeeze(0)                                     # (P*K, d)

    def forward(
        self,
        batch: STBatch,
        masks: STMaskOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = batch.x_win.device
        tokens = self.tokenizer(batch.x_win)                    # (B, P, K, d)
        b, p, k, d = tokens.shape
        tokens_flat = tokens.reshape(b, p * k, d)
        rsn_ids, time_ids = self._grid_ids(p, device)           # (P*K,)

        # ---- Target branch: full grid, EMA encoder, no grad ----
        with torch.no_grad():
            z_tgt_full = self.target_encoder(
                tokens_flat,
                rsn_ids.unsqueeze(0).expand(b, -1),
                time_ids.unsqueeze(0).expand(b, -1),
            )                                                   # (B, P*K, d)

        ctx_mask_flat = masks.context_mask.reshape(b, p * k)
        tgt_mask_flat = masks.target_mask.reshape(b, p * k)

        z_hat_list, z_tgt_list, ctx_list = [], [], []
        for i in range(b):
            ctx_sel = ctx_mask_flat[i]
            tgt_sel = tgt_mask_flat[i]

            # Context encoder sees only visible tokens.
            ctx_tokens = tokens_flat[i][ctx_sel].unsqueeze(0)   # (1, n_ctx, d)
            ctx_rsn = rsn_ids[ctx_sel]
            ctx_time = time_ids[ctx_sel]
            ctx_enc = self.context_encoder(
                ctx_tokens, ctx_rsn.unsqueeze(0), ctx_time.unsqueeze(0)
            ).squeeze(0)                                        # (n_ctx, d)
            ctx_list.append(ctx_enc)

            # Predict target latents.
            tgt_rsn = rsn_ids[tgt_sel]
            tgt_time = time_ids[tgt_sel]
            z_hat_list.append(
                self.predictor(ctx_enc, ctx_rsn, ctx_time, tgt_rsn, tgt_time)
            )
            z_tgt_list.append(z_tgt_full[i][tgt_sel])

        z_hat = torch.cat(z_hat_list, dim=0)                    # (N_tgt_total, d)
        z_tgt = torch.cat(z_tgt_list, dim=0)
        z_tgt = F.layer_norm(z_tgt, (z_tgt.shape[-1],))
        ctx_embs = torch.cat(ctx_list, dim=0)                   # (N_ctx_total, d)
        return z_hat, z_tgt.detach(), ctx_embs


def build_st_bsjepa(
    atlas: AtlasMapping,
    window_length: int,
    embed_dim: int = 256,
    feature_mode: str = "conv1d",
    time_max_windows: int = 64,
    st_encoder_depth: int = 4,
    st_encoder_heads: int = 8,
    st_predictor_dim: int = 128,
    st_predictor_depth: int = 4,
    st_predictor_heads: int = 4,
    dropout: float = 0.0,
) -> STBSJEPA:
    """Factory for :class:`STBSJEPA` from config parameters."""
    tokenizer = RSNTimeTokenizer(
        window_length=window_length,
        embed_dim=embed_dim,
        rsn_ids=atlas.rsn_ids,
        num_rsns=atlas.num_rsns,
        feature_mode=feature_mode,  # type: ignore[arg-type]
    )
    encoder = SpatioTemporalEncoder(
        embed_dim=embed_dim,
        num_rsns=atlas.num_rsns,
        time_max_windows=time_max_windows,
        depth=st_encoder_depth,
        num_heads=st_encoder_heads,
        dropout=dropout,
    )
    predictor = SpatioTemporalPredictor(
        encoder_dim=embed_dim,
        predictor_dim=st_predictor_dim,
        num_rsns=atlas.num_rsns,
        time_max_windows=time_max_windows,
        depth=st_predictor_depth,
        num_heads=st_predictor_heads,
        dropout=dropout,
    )
    return STBSJEPA(tokenizer, encoder, predictor, atlas)
