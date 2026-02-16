"""
lstm_model.py — BiLSTM + Multi-Head Self-Attention Engagement Model (v2)
=========================================================================
2-layer bidirectional LSTM with 4-head self-attention pooling over
the temporal dimension, fused with article/context/aggregate features,
producing a 64-dim user representation for privacy experiments.

v2 changes over v1:
  - Replaced pack_padded_sequence with direct masking (5-10x faster on MPS)
  - Added aggregate features to the context branch (global user profile)
  - Parameterized article_cont_dim (now 5: premium, sentiment, body/title/subtitle len)
  - Reduced lstm_dropout default from 0.3 to 0.15

Architecture:
    history_seq (B, 50, 2) -> InputProjection(2, 64) -> LN -> SiLU
        -> 2-layer BiLSTM(hidden=128, output=256 per timestep)  [masking, no packing]
        -> MultiHeadAttention(4 heads, 256-dim) -> residual + LN -> masked mean pool -> (B, 256)
    article/context/aggregate features -> embeddings + passthrough -> (B, 67)
    Concat(256 + 67 = 323)
        -> Linear(323, 256) -> LN -> SiLU -> Drop(0.2)
        -> [ResidualBlock(256)] + residual
        -> Linear(256, 64) -> LayerNorm                    [representation]
        -> Linear(64, 1)                                    [logit output]

Estimated: ~1.03M trainable parameters.
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


class LSTMEngagementModel(nn.Module):
    """
    BiLSTM + Multi-Head Self-Attention for engagement prediction.

    Consumes:
        - history_seq (B, 50, 2): normalized (read_time, scroll_pct) sequences
        - history_length (B,): actual sequence lengths before padding
        - agg_features (B, 27): aggregate features (global user profile)
        - article_cat (B,): category index -> embedding
        - article_type (B,): article type index -> embedding
        - article_cont (B, 5): [premium, sentiment, body_len_log, title_len_log, subtitle_len_log]
        - context (B, 3): [device_type, is_subscriber, is_sso_user]

    Produces:
        - logits (B,): raw logits for loss function
        - representations (B, 64): via get_representation()
    """

    def __init__(
        self,
        n_categories: int = 32,
        n_article_types: int = 16,
        cat_emb_dim: int = 16,
        type_emb_dim: int = 16,
        input_dim: int = 2,
        proj_dim: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.15,
        n_attention_heads: int = 4,
        mlp_dropout: float = 0.2,
        repr_dim: int = 64,
        article_cont_dim: int = 5,
        agg_dim: int = 27,
    ):
        super().__init__()

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        lstm_output_dim = lstm_hidden * 2  # bidirectional

        # --- Sequence encoder ---

        # Input projection: expand 2-dim behavioral features to proj_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
        )

        # Bidirectional LSTM (no packing -- uses masking for MPS compatibility)
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # Multi-head self-attention pooling over timesteps
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=n_attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_output_dim)

        # --- Article/context features ---

        self.cat_embedding = nn.Embedding(n_categories, cat_emb_dim)
        self.type_embedding = nn.Embedding(n_article_types, type_emb_dim)

        # Context dim: cat_emb + type_emb + article_cont + context
        context_dim = cat_emb_dim + type_emb_dim + article_cont_dim + 3
        # Fusion includes LSTM output + context + aggregate features
        fusion_dim = lstm_output_dim + context_dim + agg_dim

        # --- Fusion MLP ---

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(mlp_dropout),
        )

        self.res_block = ResidualBlock(256, dropout=mlp_dropout)

        # Representation layer: LayerNorm only (no activation) for full-range repr
        self.repr_layer = nn.Sequential(
            nn.Linear(256, repr_dim),
            nn.LayerNorm(repr_dim),
        )

        # Classification head
        self.head = nn.Linear(repr_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if "weight_ih" in param_name:
                        nn.init.kaiming_normal_(param, nonlinearity="linear")
                    elif "weight_hh" in param_name:
                        nn.init.orthogonal_(param)
                    elif "bias" in param_name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1 for stable long-term memory
                        # Only set in bias_ih (not bias_hh) so effective bias = 1, not 2
                        if "bias_ih" in param_name:
                            hidden = self.lstm_hidden
                            param.data[hidden:2*hidden].fill_(1.0)

    def _encode_sequence(self, batch: dict) -> torch.Tensor:
        """
        Encode behavioral sequence via BiLSTM + attention pooling.
        Uses masking instead of pack_padded_sequence for MPS compatibility.

        Returns:
            (B, lstm_hidden*2) pooled sequence representation.
        """
        seq = batch["history_seq"]           # (B, T, 2)
        lengths = batch["history_length"].clamp(min=1)  # (B,) — clamp to avoid NaN from all-masked attention
        B, T, _ = seq.shape

        # Project input features
        proj = self.input_proj(seq)          # (B, T, proj_dim)

        # Run LSTM directly (no packing -- masking is faster on MPS)
        lstm_out, _ = self.lstm(proj)        # (B, T, lstm_hidden*2)

        # Create attention mask for padded positions
        # True = ignore this position (PyTorch convention for key_padding_mask)
        positions = torch.arange(T, device=seq.device).unsqueeze(0)  # (1, T)
        key_padding_mask = positions >= lengths.unsqueeze(1)          # (B, T)

        # Multi-head self-attention: sequence attends to itself
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask,
        )  # (B, T, lstm_hidden*2)

        # Post-attention LayerNorm with residual
        attn_out = self.attn_norm(attn_out + lstm_out)  # (B, T, lstm_hidden*2)

        # Masked mean pooling over valid timesteps
        mask = (~key_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
        pooled = (attn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return pooled  # (B, lstm_hidden*2)

    def _build_context(self, batch: dict) -> torch.Tensor:
        """Build article/context/aggregate feature vector."""
        cat_emb = self.cat_embedding(batch["article_cat"])    # (B, 16)
        type_emb = self.type_embedding(batch["article_type"]) # (B, 16)

        context = torch.cat([
            cat_emb,                 # (B, 16)
            type_emb,                # (B, 16)
            batch["article_cont"],   # (B, 5)
            batch["context"],        # (B, 3)
            batch["agg_features"],   # (B, 27) — global user profile
        ], dim=-1)

        return context

    def get_representation(self, batch: dict) -> torch.Tensor:
        """
        Extract the 64-dim representation vector (before classification head).
        Used for re-identification experiments and randomized smoothing.
        """
        seq_repr = self._encode_sequence(batch)  # (B, 256)
        ctx = self._build_context(batch)          # (B, 67)

        fused = torch.cat([seq_repr, ctx], dim=-1)  # (B, 323)
        h = self.fusion_layer(fused)   # (B, 256)
        h = self.res_block(h)          # (B, 256) + residual
        h = self.repr_layer(h)         # (B, 64)

        return h

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass returning raw logits."""
        representation = self.get_representation(batch)
        logits = self.head(representation).squeeze(-1)  # (B,)
        return logits


if __name__ == "__main__":
    model = LSTMEngagementModel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    B = 4
    batch = {
        "history_seq": torch.randn(B, 50, 2),
        "history_length": torch.tensor([50, 30, 10, 5]),
        "agg_features": torch.randn(B, 27),
        "article_cat": torch.randint(0, 32, (B,)),
        "article_type": torch.randint(0, 16, (B,)),
        "article_cont": torch.randn(B, 5),
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
