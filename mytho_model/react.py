"""
ReAct — Yao et al. (2023).

Interleaves *Thought* → *Action* → *Observation* traces so the model can
reason about tool usage and incorporate external feedback within the
generation loop.

Reference: https://arxiv.org/abs/2210.03629
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MythoModel


@dataclass
class ToolResult:
    """Return value from an external tool."""
    name: str
    output: str
    success: bool = True


@dataclass
class ReActTrace:
    """One step of the ReAct loop."""
    step: int
    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


class ReActController:
    """
    ReAct reasoning-and-acting loop.

    The controller orchestrates:
      1. **Thought**: model generates internal reasoning.
      2. **Action**: model selects a tool and its arguments.
      3. **Observation**: tool is executed, result appended to context.
      4. Repeat until the model emits a *Finish* action or max steps.

    Tools are registered as callables via ``register_tool(name, fn)``.
    """

    THOUGHT_PREFIX = "Thought:"
    ACTION_PREFIX = "Action:"
    ACTION_INPUT_PREFIX = "Action Input:"
    OBSERVATION_PREFIX = "Observation:"
    FINISH_ACTION = "Finish"

    def __init__(self, model: "MythoModel", max_steps: int = 6,
                 tokenize_fn: Callable | None = None,
                 detokenize_fn: Callable | None = None):
        self.model = model
        self.max_steps = max_steps
        self.tools: dict[str, Callable] = {}

        # Default char-level tokeniser (replace with real one)
        self.tokenize = tokenize_fn or self._default_tokenize
        self.detokenize = detokenize_fn or self._default_detokenize

    def _default_tokenize(self, text: str) -> list[int]:
        v = self.model.config.vocab_size
        return [ord(c) % v for c in text]

    def _default_detokenize(self, ids: list[int]) -> str:
        return "".join(chr(t % 128) if 32 <= (t % 128) < 127 else " " for t in ids)

    def register_tool(self, name: str, fn: Callable):
        """Register an external tool. ``fn`` receives a string input and returns a string."""
        self.tools[name] = fn

    def _generate_text(self, prompt_ids: torch.Tensor, stop_tokens: list[str],
                       max_tokens: int = 64) -> str:
        out = self.model.generate(prompt_ids, max_new_tokens=max_tokens,
                                  temperature=0.4, top_k=40)
        new_tokens = out[0, prompt_ids.shape[1]:].tolist()
        text = self.detokenize(new_tokens)
        for stop in stop_tokens:
            if stop in text:
                text = text[:text.index(stop)]
                break
        return text.strip()

    def _append_to_context(self, context: list[int], text: str) -> list[int]:
        return context + self.tokenize(text)

    @torch.no_grad()
    def run(self, question: str, max_tokens_per_step: int = 64) -> dict:
        """
        Execute the full ReAct loop.

        Returns:
            dict with keys: answer, traces, n_steps, full_context
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        prompt = f"Question: {question}\n"
        context = self.tokenize(prompt)
        traces: list[ReActTrace] = []

        for step in range(1, self.max_steps + 1):
            # ── Thought ─────────────────────────────────────────────
            thought_prompt = context + self.tokenize(f"\n{self.THOUGHT_PREFIX} ")
            ids = torch.tensor([thought_prompt], device=device)
            thought = self._generate_text(ids, ["\n", self.ACTION_PREFIX],
                                          max_tokens_per_step)
            context = self._append_to_context(
                context, f"\n{self.THOUGHT_PREFIX} {thought}"
            )

            # ── Action ──────────────────────────────────────────────
            action_prompt = context + self.tokenize(f"\n{self.ACTION_PREFIX} ")
            ids = torch.tensor([action_prompt], device=device)
            action = self._generate_text(ids, ["\n"], max_tokens_per_step)
            context = self._append_to_context(
                context, f"\n{self.ACTION_PREFIX} {action}"
            )

            trace = ReActTrace(step=step, thought=thought, action=action)

            # Check for finish
            if self.FINISH_ACTION.lower() in action.lower():
                # Extract answer from action input
                ai_prompt = context + self.tokenize(f"\n{self.ACTION_INPUT_PREFIX} ")
                ids = torch.tensor([ai_prompt], device=device)
                answer = self._generate_text(ids, ["\n"], max_tokens_per_step)
                trace.action_input = answer
                traces.append(trace)
                return {"answer": answer, "traces": traces,
                        "n_steps": step, "full_context": self.detokenize(context)}

            # ── Action Input ────────────────────────────────────────
            ai_prompt = context + self.tokenize(f"\n{self.ACTION_INPUT_PREFIX} ")
            ids = torch.tensor([ai_prompt], device=device)
            action_input = self._generate_text(ids, ["\n"], max_tokens_per_step)
            trace.action_input = action_input
            context = self._append_to_context(
                context, f"\n{self.ACTION_INPUT_PREFIX} {action_input}"
            )

            # ── Observation (tool execution) ────────────────────────
            tool_name = action.strip().split("[")[0].strip() if "[" in action else action.strip()
            if tool_name in self.tools:
                try:
                    obs = str(self.tools[tool_name](action_input))
                except Exception as e:
                    obs = f"Error: {e}"
            else:
                obs = f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"

            trace.observation = obs
            context = self._append_to_context(
                context, f"\n{self.OBSERVATION_PREFIX} {obs}"
            )
            traces.append(trace)

        return {"answer": None, "traces": traces,
                "n_steps": self.max_steps, "full_context": self.detokenize(context)}

