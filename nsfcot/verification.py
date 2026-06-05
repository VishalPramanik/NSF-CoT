"""Stage 4 (fusion) -- Hybrid scoring (Section 2.6, Eqs. 11-13).

Combine the symbolic indicators ``G^SMT, V^SMT, U^SMT`` with the continuous
judge scores ``G^LLM, V^LLM, U^LLM`` via a weighted average controlled by
``beta``:

.. math::
    G_k = \\beta G^\\text{SMT}_k + (1-\\beta) G^\\text{LLM}_k, \\quad \\text{etc.}

``beta = 1`` recovers pure symbolic verification; ``beta = 0`` relies entirely on
the LLM judge. The paper uses ``beta = 0.5``.

Verification proceeds in two phases so utility can be assessed against the proof
support actually used by the chain:

1. :meth:`grounded_valid` computes ``G_k``, ``V_k`` and the proof support ``R_k``
   for every step.
2. :meth:`utility` computes ``U_k`` for a step given ``R_k`` and the chain-wide
   answer support ``R_ans`` (Eq. 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from .config import NSFCoTConfig
from .llm_judge import EntailmentJudge
from .logic import Atom
from .smt_verifier import SMTVerifier


@dataclass
class StepScores:
    """Per-step hybrid scores and their symbolic/neural components."""

    G: float
    V: float
    U: float
    G_smt: float
    V_smt: float
    U_smt: float
    G_llm: float
    V_llm: float
    U_llm: float
    proof_support: Set[int]


class HybridVerifier:
    """Fuses the SMT verifier and the LLM entailment judge."""

    def __init__(self, smt: SMTVerifier, judge: EntailmentJudge, config: NSFCoTConfig):
        self.smt = smt
        self.judge = judge
        self.config = config

    # -- phase 1: groundedness + validity --------------------------------
    def grounded_valid(
        self,
        claims: Sequence[Atom],
        relied_upon: Sequence[int],
        fact_atoms: Sequence[Sequence[Atom]],
    ) -> Tuple[float, float, float, float, Set[int]]:
        """Return ``(G, V, G_smt, V_smt, R_k)`` for one step."""
        beta = self.config.beta
        g_smt, v_smt, rk = self.smt.groundedness_validity(claims, relied_upon)

        all_atoms: List[Atom] = [a for atoms in fact_atoms for a in atoms]
        relied_atoms: List[Atom] = [a for i in relied_upon for a in fact_atoms[i]]

        g_llm = self.judge.groundedness(all_atoms, claims, g_smt)
        v_llm = self.judge.validity(relied_atoms, claims, v_smt)

        g = beta * g_smt + (1 - beta) * g_llm
        v = beta * v_smt + (1 - beta) * v_llm
        return g, v, g_smt, v_smt, rk

    # -- phase 2: utility -------------------------------------------------
    def utility(
        self,
        claims: Sequence[Atom],
        rk: Set[int],
        answer_claim: Atom,
        answer_support: Set[int],
        fact_atoms: Sequence[Sequence[Atom]],
    ) -> Tuple[float, float]:
        """Return ``(U, U_smt)`` for one step."""
        beta = self.config.beta
        u_smt = self.smt.utility_indicator(rk, answer_support)
        all_atoms: List[Atom] = [a for atoms in fact_atoms for a in atoms]
        u_llm = self.judge.utility(claims, answer_claim, all_atoms, u_smt)
        return beta * u_smt + (1 - beta) * u_llm, u_smt
