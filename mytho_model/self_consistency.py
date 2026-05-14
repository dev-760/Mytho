"""
Self-Consistency Decoding — Wang et al. (2023).

Sample *k* independent reasoning paths and aggregate via majority voting
or log-probability weighting to improve reasoning accuracy.

Reference: https://arxiv.org/abs/2203.11171
"""

import torch, torch.nn.functional as F, math
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MythoModel


class SelfConsistencyDecoder:
    """Sample multiple completions, pick the most consistent answer."""

    def __init__(self, model: "MythoModel", n_paths: int = 5,
                 temperature: float = 1.0, top_k: int = 50,
                 top_p: float = 0.95, mode: str = "majority"):
        self.model, self.n_paths = model, n_paths
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.mode = mode  # "majority" | "weighted" | "best_of_n"

    @torch.no_grad()
    def _sample_path(self, input_ids, max_new_tokens):
        gen = input_ids.clone()
        total_lp = 0.0
        for _ in range(max_new_tokens):
            ctx = gen[:, -self.model.config.max_seq_len:]
            logits = self.model(ctx)["logits"][:, -1, :]
            if self.temperature > 0:
                logits = logits / self.temperature
            if self.top_k > 0:
                tv, _ = torch.topk(logits, self.top_k, dim=-1)
                logits[logits < tv[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, 1)
            total_lp += torch.log(probs.gather(1, tok) + 1e-12).item()
            gen = torch.cat([gen, tok], dim=1)
        return gen, total_lp

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, answer_offset=0):
        self.model.eval()
        paths, scores = [], []
        for _ in range(self.n_paths):
            p, s = self._sample_path(input_ids, max_new_tokens)
            paths.append(p); scores.append(s)

        plen = input_ids.shape[1]
        astart = plen + answer_offset

        if self.mode == "best_of_n":
            best_idx = max(range(self.n_paths), key=lambda i: scores[i])
        elif self.mode == "weighted":
            w = torch.tensor([math.exp(s) for s in scores])
            best_idx = w.argmax().item()
        else:
            answers = [tuple(p[0, astart:].tolist()) for p in paths]
            best_ans = Counter(answers).most_common(1)[0][0]
            best_idx = next(i for i, a in enumerate(answers) if a == best_ans)

        best_answer = tuple(paths[best_idx][0, astart:].tolist())
        agree = sum(1 for p in paths if tuple(p[0, astart:].tolist()) == best_answer)
        return {"best_path": paths[best_idx], "all_paths": paths,
                "scores": scores, "consistency": agree / self.n_paths}

