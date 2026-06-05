"""Configuration for the NSF-CoT verification pipeline.

All defaults match the values reported in the paper (Section 3.1):

* ``n_ablation_samples`` :math:`M = 128`
* ``lasso_lambda`` :math:`\\lambda = 0.01`
* ``attribution_threshold`` :math:`\\tau = 0.5`
* ``beta`` (hybrid SMT/LLM weight) :math:`\\beta = 0.5`
* ``faithfulness_threshold`` :math:`\\tau_\\text{faith} = 0.5`
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NSFCoTConfig:
    """Hyperparameters for the five-stage NSF-CoT pipeline."""

    # --- Stage 3: internal fact attribution (ContextCite + LASSO) --------
    n_ablation_samples: int = 128          # M
    lasso_lambda: float = 0.01             # λ for the LASSO surrogate (Eq. 5)
    attribution_threshold: float = 0.5     # τ used to threshold positive weights (Eq. 6)

    # --- Stage 4: hybrid verification ------------------------------------
    beta: float = 0.5                      # weight on the symbolic signal (Eqs. 11-13)
    use_inference_axioms: bool = True      # toggle φ_rules (transitivity, inheritance, ...)

    # --- Stage 5: faithfulness scoring -----------------------------------
    faithfulness_threshold: float = 0.5    # τ_faith; Fk >= τ_faith => faithful (Eq. 14)

    # --- pruning (Section 4.2) -------------------------------------------
    prune_threshold: float = 0.5           # remove step if min(Gk, Vk) < this

    # --- reproducibility -------------------------------------------------
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if not 0.0 < self.attribution_threshold <= 1.0:
            raise ValueError("attribution_threshold must lie in (0, 1]")
        if self.n_ablation_samples <= 0:
            raise ValueError("n_ablation_samples must be positive")
