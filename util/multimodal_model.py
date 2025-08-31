"""Multi-modal price forecasting model.

This module provides a minimal PyTorch implementation of a price
forecasting model that fuses several modalities at a mid-level as
suggested in the design notes.  It is intentionally light-weight and
serves as a drop-in replacement for the existing single-modality model
used in :mod:`training_step`.

The model uses dedicated encoders for each modality (prices/FX rates,
weather aggregates and news headlines) and combines their outputs with a
learned gated fusion layer.  Each encoder outputs a representation of the
same dimensionality so the fusion layer can compute modality weights and
produce a single vector per time step that feeds a regression head for
multi-horizon forecasts.

The text encoder relies on a pre-trained transformer model from the
``transformers`` library.  By default it is frozen which keeps the number
of trainable parameters small; setting ``finetune=True`` enables gradient
updates.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel

__all__ = [
    "AttentionPool",
    "PriceEncoder",
    "FXEncoder",
    "WeatherEncoder",
    "NewsEncoder",
    "GatedFusion",
    "MultiModalPriceForecast",
]


class AttentionPool(nn.Module):
    """Simple attention-based pooling over the temporal dimension."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # ``x``: [B, T, D]; ``mask``: [B, T] where True denotes valid steps
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class PriceEncoder(nn.Module):
    """Encode historical price series for a single asset."""

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        d_model: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, emb_dim)
        self.proj = nn.Linear(n_features + emb_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=layers, batch_first=True, dropout=dropout)
        self.pool = AttentionPool(d_model)

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # ``x``: [B, T, F]; ``asset_ids``: [B]
        emb = self.asset_emb(asset_ids).unsqueeze(1).expand(-1, x.size(1), -1)
        z = self.proj(torch.cat([x, emb], dim=-1))
        out, _ = self.lstm(z)
        return self.pool(out, mask)


class FXEncoder(nn.Module):
    """Encoder for currency exchange rates."""

    def __init__(self, n_features: int, d_model: int = 64, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=layers, batch_first=True, dropout=dropout)
        self.pool = AttentionPool(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.proj(x)
        out, _ = self.lstm(z)
        return self.pool(out, mask)


class WeatherEncoder(nn.Module):
    """Encoder for weather features aggregated to asset-relevant regions."""

    def __init__(self, n_features: int, d_model: int = 32, layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=layers, batch_first=True, dropout=dropout)
        self.pool = AttentionPool(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.proj(x)
        out, _ = self.lstm(z)
        return self.pool(out, mask)


class NewsEncoder(nn.Module):
    """Encode daily news headlines with a frozen text backbone."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        d_model: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
        finetune: bool = False,
    ) -> None:
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        if not finetune:
            for p in self.text_model.parameters():
                p.requires_grad = False
        txt_dim = self.text_model.config.hidden_size
        self.temporal = nn.LSTM(txt_dim, d_model, num_layers=layers, batch_first=True, dropout=dropout)
        self.pool = AttentionPool(d_model)

    def forward(self, tokens: Dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
        tokens: mapping with ``input_ids`` and ``attention_mask`` of shape [B, T, L].
        mask: optional boolean mask [B, T] indicating which days have news.
        """
        ids = tokens["input_ids"]
        attn = tokens["attention_mask"]
        B, T, L = ids.shape
        ids = ids.view(B * T, L)
        attn = attn.view(B * T, L)
        out = self.text_model(input_ids=ids, attention_mask=attn).last_hidden_state
        # use CLS token (position 0) as representation for each headline
        cls = out[:, 0]  # [B*T, txt_dim]
        cls = cls.view(B, T, -1)
        temp_out, _ = self.temporal(cls)
        return self.pool(temp_out, mask)


class GatedFusion(nn.Module):
    """Gated fusion of modality representations."""

    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        self.gate = nn.Linear(sum(dims), len(dims))

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(inputs, dim=-1)
        weights = torch.softmax(self.gate(concat), dim=-1)
        fused = 0.0
        for i, inp in enumerate(inputs):
            fused = fused + weights[:, i : i + 1] * inp
        return fused


class MultiModalPriceForecast(nn.Module):
    """Full model combining all modalities with a regression head."""

    def __init__(
        self,
        price_encoder: PriceEncoder,
        fx_encoder: FXEncoder,
        weather_encoder: WeatherEncoder,
        news_encoder: NewsEncoder,
        out_dim: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.price_encoder = price_encoder
        self.fx_encoder = fx_encoder
        self.weather_encoder = weather_encoder
        self.news_encoder = news_encoder
        d = price_encoder.lstm.hidden_size  # encoders share dimensionality
        self.fusion = GatedFusion([d, d, d, d])
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        price_x: torch.Tensor,
        asset_ids: torch.Tensor,
        price_mask: torch.Tensor | None,
        fx_x: torch.Tensor,
        fx_mask: torch.Tensor | None,
        weather_x: torch.Tensor,
        weather_mask: torch.Tensor | None,
        news_tokens: Dict[str, torch.Tensor],
        news_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        p = self.price_encoder(price_x, asset_ids, price_mask)
        f = self.fx_encoder(fx_x, fx_mask)
        w = self.weather_encoder(weather_x, weather_mask)
        n = self.news_encoder(news_tokens, news_mask)
        fused = self.fusion([p, f, w, n])
        return self.head(fused)
