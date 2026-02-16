"""
mlp_baseline.py — Deep MLP Engagement Prediction Model (v3)
=============================================================
6-layer feedforward network with residual connections for binary
engagement prediction. Produces a 64-dim user representation for
privacy experiments.

Fixes over v2:
  - Width reduced 512->256 (58-dim input doesn't need 512-wide layers;
    over-parameterization caused overfitting gap and wasted capacity)
  - SiLU activation replaces GELU (smoother gradient flow, no dead zone
    at -0.17 that caused 6/64 dead dimensions in v2)
  - Representation layer uses LayerNorm only (no activation), preventing
    dead dimensions by allowing full real-valued range
  - ~230K params (vs 552K in v2, 120K in v1) — sweet spot

Architecture:
    Input (58) -> Linear(256) -> LN -> SiLU -> Drop(0.2)
              -> [ResBlock: Linear(256) -> LN -> SiLU -> Drop(0.2)] + residual
              -> Linear(256) -> LN -> SiLU -> Drop(0.2)
              -> Linear(128) -> LN -> SiLU -> Drop(0.2)
              -> [ResBlock: Linear(128) -> LN -> SiLU -> Drop(0.2)] + residual
              -> Linear(64) -> LayerNorm                    [representation]
              -> Linear(1)                                   [logit output]
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear -> LayerNorm -> SiLU -> Dropout with additive residual."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class MLPEngagementModel(nn.Module):
    """
    Deep MLP for engagement prediction (v3).

    Consumes:
        - agg_features (B, 21): StandardScaled aggregate history features
        - article_cat (B,): category index -> embedding
        - article_type (B,): article type index -> embedding
        - article_cont (B, 2): [premium, sentiment_score]
        - context (B, 3): [device_type, is_subscriber, is_sso_user]

    Produces:
        - logits (B,): raw logits for loss function
        - representations (B, 64): via get_representation()
    """

    def __init__(self, n_categories: int = 32, n_article_types: int = 16,
                 agg_dim: int = 21, cat_emb_dim: int = 16, type_emb_dim: int = 16):
        super().__init__()

        self.cat_embedding = nn.Embedding(n_categories, cat_emb_dim)
        self.type_embedding = nn.Embedding(n_article_types, type_emb_dim)

        # Total input: agg(21) + cat_emb(16) + type_emb(16) + article_cont(2) + context(3) = 58
        input_dim = agg_dim + cat_emb_dim + type_emb_dim + 2 + 3

        # Layer 1: input -> 256
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        # Layer 2: 256 -> 256 + residual
        self.res_block1 = ResidualBlock(256, dropout=0.2)

        # Layer 3: 256 -> 256
        self.layer3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        # Layer 4: 256 -> 128
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        # Layer 5: 128 -> 128 + residual
        self.res_block2 = ResidualBlock(128, dropout=0.2)

        # Layer 6: 128 -> 64 representation (LayerNorm only, no activation)
        # No activation here prevents dead dimensions — full real-valued range
        self.repr_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

        # Classification head: 64 -> 1
        self.head = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization suited for SiLU activation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_input(self, batch: dict) -> torch.Tensor:
        """Concatenate all input features into a single vector."""
        cat_emb = self.cat_embedding(batch["article_cat"])    # (B, 16)
        type_emb = self.type_embedding(batch["article_type"]) # (B, 16)

        x = torch.cat([
            batch["agg_features"],   # (B, 21)
            cat_emb,                 # (B, 16)
            type_emb,                # (B, 16)
            batch["article_cont"],   # (B, 2)
            batch["context"],        # (B, 3)
        ], dim=-1)  # (B, 58)

        return x

    def get_representation(self, batch: dict) -> torch.Tensor:
        """
        Extract the 64-dim representation vector (before classification head).
        Used for re-identification experiments and randomized smoothing.
        """
        x = self._build_input(batch)
        h = self.layer1(x)          # (B, 256)
        h = self.res_block1(h)      # (B, 256) + residual
        h = self.layer3(h)          # (B, 256)
        h = self.layer4(h)          # (B, 128)
        h = self.res_block2(h)      # (B, 128) + residual
        h = self.repr_layer(h)      # (B, 64)
        return h

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass returning raw logits."""
        representation = self.get_representation(batch)
        logits = self.head(representation).squeeze(-1)  # (B,)
        return logits


if __name__ == "__main__":
    model = MLPEngagementModel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    B = 4
    batch = {
        "agg_features": torch.randn(B, 21),
        "article_cat": torch.randint(0, 32, (B,)),
        "article_type": torch.randint(0, 16, (B,)),
        "article_cont": torch.randn(B, 2),
        "context": torch.randn(B, 3),
    }

    logits = model(batch)
    repr_vec = model.get_representation(batch)

    print(f"\nLogits shape:         {logits.shape}  (expected: ({B},))")
    print(f"Representation shape: {repr_vec.shape}  (expected: ({B}, 64))")
    print(f"Logits sample:        {logits.detach().tolist()}")
    print(f"Repr min: {repr_vec.min():.4f}, max: {repr_vec.max():.4f}")
    print(f"\nModel architecture:")
    print(model)
