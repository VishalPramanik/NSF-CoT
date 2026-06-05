"""Top-level NSF-CoT pipeline (Algorithm 1).

Wires together the five stages:

1. CoT generation / segmentation (here the trace is supplied with the example)
2. Text-to-logic parsing            (:mod:`nsfcot.parsing`)
3. Internal fact attribution        (:mod:`nsfcot.attribution`)
4. Hybrid verification              (:mod:`nsfcot.smt_verifier` + :mod:`nsfcot.llm_judge`)
5. Faithfulness scoring             (:mod:`nsfcot.scoring`)

and returns a per-step audit trail.

Utility (Eq. 10) is evaluated against ``R_ans``, the proof support the chain
actually uses for the answer -- taken as the union of every grounded step's proof
support together with the answer claim's own UNSAT core. This matches the paper's
behaviour, in which evidence-grounded intermediate steps are scored as useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from .attribution import ContextCiteAttributor
from .backends import LMBackend
from .config import NSFCoTConfig
from .data import Example
from .llm_judge import EntailmentJudge
from .logic import Atom
from .parsing import Parser
from .scoring import faithfulness_score, verdict
from .smt_verifier import SMTVerifier
from .verification import HybridVerifier, StepScores


@dataclass
class StepAudit:
    """Full audit record for a single reasoning step."""

    index: int
    text: str
    claims: List[Atom]
    relied_upon: List[int]      # I_k (0-based fact indices)
    scores: StepScores
    F: float
    verdict: str


@dataclass
class AuditResult:
    """Audit trail for an entire chain-of-thought."""

    steps: List[StepAudit]
    answer_claim: Atom
    answer_support: Set[int]

    def faithful_steps(self) -> List[int]:
        return [s.index for s in self.steps if s.verdict == "Faithful"]

    def unfaithful_steps(self) -> List[int]:
        return [s.index for s in self.steps if s.verdict == "Unfaithful"]


class NSFCoT:
    """The neuro-symbolic faithfulness verifier."""

    def __init__(
        self,
        parser: Parser,
        lm: LMBackend,
        judge: EntailmentJudge,
        config: NSFCoTConfig | None = None,
    ):
        self.parser = parser
        self.lm = lm
        self.judge = judge
        self.config = config or NSFCoTConfig()

    def audit(self, example: Example) -> AuditResult:
        cfg = self.config

        # -- Stage 2: parse facts, steps, and the final answer ------------
        fact_atoms = self.parser.parse_all(example.facts)
        step_atoms = [self.parser.parse(s.text) for s in example.steps]
        answer_claims = self.parser.parse(example.answer_text)
        if not answer_claims:
            answer_claims = [Atom.parse(example.answer_atom)]
        answer_claim = answer_claims[0]

        # -- Stage 4 (symbolic) setup -------------------------------------
        smt = SMTVerifier(fact_atoms, cfg)
        hybrid = HybridVerifier(smt, self.judge, cfg)

        # -- Stage 3: attribution ----------------------------------------
        attributor = ContextCiteAttributor(self.lm, cfg)
        step_texts = [s.text for s in example.steps]
        relied = [
            attributor.attribute_step(
                k, example.facts, example.question, step_texts[:k], step_texts[k]
            ).relied_upon
            for k in range(len(step_texts))
        ]

        # -- Phase 1: groundedness + validity + proof support -------------
        phase1 = []
        chain_support: Set[int] = set(smt.answer_support(answer_claim))
        for k, claims in enumerate(step_atoms):
            g, v, g_smt, v_smt, rk = hybrid.grounded_valid(claims, relied[k], fact_atoms)
            phase1.append((g, v, g_smt, v_smt, rk))
            if g_smt > 0:                       # step is symbolically grounded
                chain_support |= rk

        # -- Phase 2: utility + faithfulness scoring ----------------------
        audits: List[StepAudit] = []
        for k, claims in enumerate(step_atoms):
            g, v, g_smt, v_smt, rk = phase1[k]
            u, u_smt = hybrid.utility(claims, rk, answer_claim, chain_support, fact_atoms)

            beta = cfg.beta
            scores = StepScores(
                G=g, V=v, U=u,
                G_smt=g_smt, V_smt=v_smt, U_smt=u_smt,
                G_llm=(g - beta * g_smt) / (1 - beta) if beta < 1 else float("nan"),
                V_llm=(v - beta * v_smt) / (1 - beta) if beta < 1 else float("nan"),
                U_llm=(u - beta * u_smt) / (1 - beta) if beta < 1 else float("nan"),
                proof_support=rk,
            )
            F = faithfulness_score(g, v, u)
            audits.append(
                StepAudit(
                    index=k,
                    text=step_texts[k],
                    claims=claims,
                    relied_upon=relied[k],
                    scores=scores,
                    F=F,
                    verdict=verdict(F, cfg.faithfulness_threshold),
                )
            )

        return AuditResult(steps=audits, answer_claim=answer_claim,
                           answer_support=chain_support)
