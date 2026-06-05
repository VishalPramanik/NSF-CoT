"""Faithfulness-guided CoT pruning (Section 4.2, Eq. 15).

Beyond auditing, NSF-CoT can repair a chain by deleting *decision-harmful* steps
-- those failing groundedness or validity -- and re-decoding the answer:

.. math:: U(x) = \\{ k : \\min(G_k, V_k) < 0.5 \\}.

Utility is intentionally excluded: low utility signals irrelevance, not
incorrectness. Failing steps are replaced by ``[REMOVED]`` while preserving the
chain structure.
"""

from __future__ import annotations

from typing import List

from .pipeline import AuditResult

REMOVED = "[REMOVED]"


def steps_to_prune(result: AuditResult, threshold: float = 0.5) -> List[int]:
    """Indices of steps failing groundedness or validity (Eq. 15)."""
    return [
        s.index
        for s in result.steps
        if min(s.scores.G, s.scores.V) < threshold
    ]


def prune_chain(step_texts: List[str], result: AuditResult,
                threshold: float = 0.5) -> List[str]:
    """Return the pruned chain with failing steps replaced by ``[REMOVED]``."""
    drop = set(steps_to_prune(result, threshold))
    return [REMOVED if k in drop else text for k, text in enumerate(step_texts)]
