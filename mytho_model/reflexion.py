"""
Reflexion — Shinn et al. (2023).

A verbal reinforcement-learning loop where the model:
  1. Generates a response (actor).
  2. Evaluates it with a learned critic.
  3. Produces a natural-language *reflection* summarising mistakes.
  4. Stores the reflection in an episodic memory buffer.
  5. Retries with reflections as additional context.

Reference: https://arxiv.org/abs/2303.11366
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MythoModel


class ReflexionCritic(nn.Module):
    """
    Learned self-evaluator that scores a generated sequence.

    Outputs a scalar quality score in [0, 1].
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score a sequence from its mean-pooled hidden states. → [B, 1]"""
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled)


class ReflectionGenerator(nn.Module):
    """
    Produces a *reflection embedding* that encodes what went wrong.

    Takes the original hidden states and the critic score, and outputs
    a reflection vector that can be prepended to future attempts.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score_proj = nn.Linear(1, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, S, D]  sequence hidden states
            score:  [B, 1]     critic score

        Returns:
            reflection: [B, 1, D]  single reflection token embedding
        """
        pooled = hidden.mean(dim=1, keepdim=True)              # [B, 1, D]
        score_emb = self.score_proj(score).unsqueeze(1)         # [B, 1, D]
        fused = self.fuse(torch.cat([pooled, score_emb], dim=-1))
        return self.norm(fused)


class EpisodicMemory:
    """Simple FIFO buffer of past reflection embeddings."""

    def __init__(self, max_reflections: int = 8):
        self.max_reflections = max_reflections
        self.buffer: list[torch.Tensor] = []

    def add(self, reflection: torch.Tensor):
        self.buffer.append(reflection.detach())
        if len(self.buffer) > self.max_reflections:
            self.buffer.pop(0)

    def get_context(self) -> torch.Tensor | None:
        """Return stacked reflections [B, N_refl, D] or None."""
        if not self.buffer:
            return None
        return torch.cat(self.buffer, dim=1)

    def clear(self):
        self.buffer.clear()


class ReflexionController(nn.Module):
    """
    Full Reflexion loop: act → evaluate → reflect → retry.

    Usage:
        controller = ReflexionController(config)
        result = controller.run(model, input_ids, max_trials=3)
    """

    def __init__(self, config):
        super().__init__()
        self.critic = ReflexionCritic(config.d_model)
        self.reflector = ReflectionGenerator(config.d_model)
        self.memory = EpisodicMemory(max_reflections=8)

        # Projection to inject reflections into input embeddings
        self.inject_proj = nn.Linear(config.d_model, config.d_model)
        self.quality_threshold = 0.75

    @torch.no_grad()
    def run(
        self, model: "MythoModel", input_ids: torch.Tensor,
        max_new_tokens: int = 128, max_trials: int = 3,
    ) -> dict:
        """
        Execute the Reflexion loop.

        Returns dict: best_output, best_score, n_trials, reflections
        """
        model.eval()
        self.memory.clear()
        best_output, best_score = None, -1.0
        trial_results = []

        for trial in range(max_trials):
            # Embed input
            x = model.emb_drop(model.token_emb(input_ids))

            # Inject past reflections
            refl_ctx = self.memory.get_context()
            if refl_ctx is not None:
                refl_emb = self.inject_proj(refl_ctx.to(x.device))
                x = torch.cat([refl_emb, x], dim=1)

            # Generate
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=0.7, top_k=50,
            )

            # Evaluate with critic
            with torch.no_grad():
                out_hidden = model.token_emb(output)
                score = self.critic(out_hidden).item()

            trial_results.append({"output": output, "score": score, "trial": trial})

            if score > best_score:
                best_score = score
                best_output = output

            # Good enough?
            if score >= self.quality_threshold:
                break

            # Generate reflection for next trial
            reflection = self.reflector(out_hidden, torch.tensor([[score]], device=x.device))
            self.memory.add(reflection)

        return {
            "best_output": best_output,
            "best_score": best_score,
            "n_trials": len(trial_results),
            "trials": trial_results,
        }

