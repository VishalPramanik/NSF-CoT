"""Stage 3 -- Internal fact attribution (Section 2.5, Appendix B).

We approximate the facts a model *internally relied upon* for each step using
ContextCite (Cohen-Wang et al., 2024): sample binary ablation masks over the
context facts, score each masked configuration with the LM, and fit a sparse
linear surrogate via LASSO (Tibshirani, 1996). Facts with sufficiently large
positive weight form the relied-upon set :math:`I_k` (Eq. 6); normalising the
positive weights yields soft attribution probabilities :math:`\\alpha_{k,i}`
(Eq. 22).

The code below is backend-agnostic: it works unchanged with a real
:class:`~nsfcot.backends.HuggingFaceLM` or the deterministic
:class:`~nsfcot.backends.MockLM`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.linear_model import Lasso

from .backends import LMBackend
from .config import NSFCoTConfig


@dataclass
class StepAttribution:
    """Attribution result for a single reasoning step."""

    weights: np.ndarray          # \hat{w}_k for every fact
    relied_upon: List[int]       # I_k as 0-based fact indices
    soft_probs: np.ndarray       # \alpha_{k,i}


class ContextCiteAttributor:
    """ContextCite + LASSO attribution over context-fact ablation masks."""

    def __init__(self, lm: LMBackend, config: NSFCoTConfig):
        self.lm = lm
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # -- mask sampling ----------------------------------------------------
    def _sample_masks(self, n_facts: int) -> np.ndarray:
        """Sample M masks uniformly from {0,1}^n (Appendix B)."""
        m = self.config.n_ablation_samples
        masks = self._rng.integers(0, 2, size=(m, n_facts))
        # Guarantee the all-ones row is present so the full-context score is
        # always observed; this stabilises the surrogate.
        masks[0, :] = 1
        return masks

    # -- per-step attribution --------------------------------------------
    def attribute_step(
        self,
        step_index: int,
        facts: Sequence[str],
        question: str,
        prefix_steps: Sequence[str],
        step_text: str,
    ) -> StepAttribution:
        n = len(facts)
        masks = self._sample_masks(n)

        # Ablation scores g_k(v) for each sampled mask (Eq. 4 / Eq. 18).
        scores = np.empty(masks.shape[0], dtype=float)
        for m, mask in enumerate(masks):
            retained = [i for i in range(n) if mask[i] == 1]
            scores[m] = self.lm.step_logprob(
                retained, facts, question, prefix_steps, step_text
            )

        # Sparse linear surrogate via LASSO (Eq. 5 / Eq. 19).
        # sklearn's objective is (1/(2M))||Xw - z||^2 + alpha||w||_1, so we
        # rescale lambda accordingly to match the paper's formulation.
        alpha = self.config.lasso_lambda / (2.0 * masks.shape[0])
        lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10_000)
        lasso.fit(masks.astype(float), scores)
        w = lasso.coef_.astype(float)

        relied_upon, soft = self._select(w)
        return StepAttribution(weights=w, relied_upon=relied_upon, soft_probs=soft)

    # -- thresholding & soft probabilities (Eqs. 20-22) ------------------
    def _select(self, w: np.ndarray):
        pos_idx = np.where(w > 0)[0]
        n = len(w)
        if pos_idx.size == 0:
            # I_k = empty; uniform soft probabilities (Appendix B).
            return [], np.full(n, 1.0 / n)
        w_plus = float(w[pos_idx].mean())
        tau = self.config.attribution_threshold
        relied_upon = [int(i) for i in pos_idx if w[i] >= tau * w_plus]
        pos = np.clip(w, 0.0, None)
        denom = pos.sum()
        soft = pos / denom if denom > 0 else np.full(n, 1.0 / n)
        return relied_upon, soft

    # -- whole-chain convenience -----------------------------------------
    def attribute_chain(
        self,
        facts: Sequence[str],
        question: str,
        steps: Sequence[str],
    ) -> List[StepAttribution]:
        results: List[StepAttribution] = []
        for k, step in enumerate(steps):
            results.append(
                self.attribute_step(k, facts, question, list(steps[:k]), step)
            )
        return results
