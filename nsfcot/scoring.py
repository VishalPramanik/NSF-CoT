"""Stage 5 -- Faithfulness scoring (Section 2.6, Eq. 14).

The per-step faithfulness score is the product of the three criteria,

.. math:: F_k = G_k \\cdot V_k \\cdot U_k,

so a step is faithful only if it is grounded, valid, *and* useful. A step is
flagged faithful when ``F_k >= tau_faith`` (default 0.5).
"""

from __future__ import annotations


def faithfulness_score(G: float, V: float, U: float) -> float:
    """Multiplicative faithfulness score F_k = G_k * V_k * U_k."""
    return G * V * U


def verdict(F: float, tau_faith: float = 0.5) -> str:
    """Return ``"Faithful"`` or ``"Unfaithful"`` for a step score."""
    return "Faithful" if F >= tau_faith else "Unfaithful"
