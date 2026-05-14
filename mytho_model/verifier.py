"""
Verifier Head — learned quality/confidence assessor for adaptive halting.

Reads both hidden states and scratchpad to predict:
  • confidence:    how certain the model is about current output
  • quality:       estimated answer quality
  • uncertainty:   epistemic uncertainty estimate
  • contradiction: internal consistency check

Used by Uncertainty-Driven ACT to make smarter halting decisions.
"""

import torch
import torch.nn as nn

from .components import RMSNorm


class VerifierHead(nn.Module):
    """
    Multi-signal verifier that reads hidden states + scratchpad.

    Outputs a dict of scalar signals per token, used by ACT
    for uncertainty-driven halting.
    """

    def __init__(self, d_model: int, d_scratch: int = 0):
        super().__init__()
        d_in = d_model + d_scratch
        d_hidden = d_model // 2

        self.input_norm = RMSNorm(d_in)

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )

        # Per-signal heads
        self.confidence_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.SiLU(),
            nn.Linear(d_hidden // 2, 1), nn.Sigmoid(),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.SiLU(),
            nn.Linear(d_hidden // 2, 1), nn.Sigmoid(),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.SiLU(),
            nn.Linear(d_hidden // 2, 1), nn.Sigmoid(),
        )
        self.contradiction_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.SiLU(),
            nn.Linear(d_hidden // 2, 1), nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        scratch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden:  [B, T, d_model]
            scratch: [B, T, d_scratch] or None

        Returns:
            dict with keys (all [B, T, 1]):
                confidence, quality, uncertainty, contradiction
        """
        if scratch is not None:
            x = torch.cat([hidden, scratch], dim=-1)
        else:
            x = hidden

        x = self.input_norm(x)
        features = self.backbone(x)

        return {
            "confidence": self.confidence_head(features),       # [B, T, 1]
            "quality": self.quality_head(features),
            "uncertainty": self.uncertainty_head(features),
            "contradiction": self.contradiction_head(features),
        }

    def should_halt(
        self,
        signals: dict[str, torch.Tensor],
        confidence_threshold: float = 0.85,
        uncertainty_threshold: float = 0.15,
    ) -> torch.Tensor:
        """
        Determine halting based on verifier signals.

        Halt if: confidence >= threshold OR uncertainty <= threshold.

        Returns:
            halt_mask: [B, T, 1] bool
        """
        conf = signals["confidence"]
        unc = signals["uncertainty"]
        return (conf >= confidence_threshold) | (unc <= uncertainty_threshold)

