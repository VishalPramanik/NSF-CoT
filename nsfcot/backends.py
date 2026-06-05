"""Language-model backends.

The attribution stage (Section 2.5) needs a single capability from the audited
model: the log-probability of a reasoning step given an *ablated* subset of the
context facts,

.. math:: g_k(v) = \\log p_\\text{LM}(y_{[a_k:b_k]} \\mid \\text{Ablate}(F, v), q, y_{<a_k}).

We expose this through the :class:`LMBackend` interface so the rest of the
pipeline is agnostic to whether the model is a local HuggingFace checkpoint
(Pythia-2.8B, LLaMA-3.1-8B, Qwen2.5-72B in the paper) or a deterministic mock
used for offline tests and the bundled demo.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence


class LMBackend(ABC):
    """Interface for the autoregressive model being audited."""

    @abstractmethod
    def step_logprob(
        self,
        retained: Sequence[int],
        facts: Sequence[str],
        question: str,
        prefix_steps: Sequence[str],
        step_text: str,
    ) -> float:
        """Return ``log p(step_text | retained facts, question, prefix_steps)``.

        ``retained`` is the list of indices of facts kept under the ablation
        mask ``v`` (i.e. ``{i : v_i = 1}``).
        """

    def generate_cot(self, facts: Sequence[str], question: str) -> "GenerationResult":
        """Optionally generate a chain-of-thought. Not required for verification
        when a pre-generated trace is supplied."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement CoT generation; supply a "
            "pre-generated chain-of-thought instead."
        )


class GenerationResult:
    """Container for a generated chain-of-thought and final answer."""

    def __init__(self, steps: List[str], answer: str):
        self.steps = steps
        self.answer = answer


# ---------------------------------------------------------------------------
# Deterministic mock backend (offline; used by the demo and unit tests)
# ---------------------------------------------------------------------------
class MockLM(LMBackend):
    """A deterministic stand-in for a real LM.

    The mock implements ``step_logprob`` as a smooth linear function of which
    *relevant* facts are present, so that the genuine LASSO attribution code path
    (:mod:`nsfcot.attribution`) recovers the relied-upon facts exactly as it would
    for a real model -- only without any network access or model weights.

    ``relevance`` maps a 0-based step index to the set of fact indices that the
    (simulated) model relies upon when producing that step.
    """

    def __init__(self, relevance: Dict[int, Sequence[int]], base: float = -8.0,
                 weight: float = 2.0):
        self.relevance = {k: list(v) for k, v in relevance.items()}
        self.base = base
        self.weight = weight

    def step_logprob(self, retained, facts, question, prefix_steps, step_text):  # noqa: D102
        # Locate the step index from the length of the prefix.
        k = len(prefix_steps)
        relevant = self.relevance.get(k, [])
        retained_set = set(retained)
        present = sum(1 for i in relevant if i in retained_set)
        # A step with no relied-upon facts (a confabulation, I_k = empty) must
        # yield a target that is *constant* across ablations, so the LASSO
        # surrogate recovers all-zero weights and hence I_k = empty. When the
        # step does rely on facts, a tiny deterministic perturbation keeps the
        # design well-conditioned without changing the recovered support.
        if not relevant:
            return self.base
        jitter = (_stable_hash(f"{k}:{sorted(retained_set)}") % 1000) / 1e6
        return self.base + self.weight * present + jitter


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


# ---------------------------------------------------------------------------
# HuggingFace backend (real; requires `torch` + `transformers` + weights)
# ---------------------------------------------------------------------------
class HuggingFaceLM(LMBackend):
    """Real backend computing token log-likelihoods from a HuggingFace model.

    This is the backend used for the experiments in the paper. It is imported
    lazily so the package has no hard dependency on ``torch``/``transformers``.

    Example
    -------
    >>> lm = HuggingFaceLM("EleutherAI/pythia-2.8b")          # doctest: +SKIP
    >>> lm.step_logprob([0, 1], facts, question, [], step)    # doctest: +SKIP
    """

    PROMPT_TEMPLATE = (
        "Answer the question using the facts. Think step by step, one step per line.\n"
        "Facts:\n{facts}\nQuestion: {question}\nReasoning:\n{prefix}"
    )

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "HuggingFaceLM requires `torch` and `transformers`. Install with "
                "`pip install torch transformers`."
            ) from exc

        import torch

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, **kwargs
        )
        self.model.eval()

    def _build_prompt(self, retained, facts, question, prefix_steps) -> str:
        kept = "\n".join(f"- {facts[i]}" for i in sorted(retained))
        prefix = "".join(f"{s}\n" for s in prefix_steps)
        return self.PROMPT_TEMPLATE.format(facts=kept, question=question, prefix=prefix)

    def step_logprob(self, retained, facts, question, prefix_steps, step_text):  # noqa: D102
        torch = self._torch
        prompt = self._build_prompt(retained, facts, question, prefix_steps)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        full_ids = self.tokenizer(prompt + step_text, return_tensors="pt").input_ids
        device = next(self.model.parameters()).device
        full_ids = full_ids.to(device)

        with torch.no_grad():
            logits = self.model(full_ids).logits

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = full_ids[:, 1:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        # Only sum over the tokens belonging to the step (the suffix).
        start = prompt_ids.shape[1] - 1
        return float(token_lp[start:].sum().item())
