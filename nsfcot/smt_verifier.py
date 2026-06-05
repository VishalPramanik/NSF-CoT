"""Stage 4 (symbolic) -- SMT-based verification (Section 2.6, Appendix C).

We encode the knowledge base with one Boolean *assumption literal* ``a_i`` per
context fact (Eq. 7 / Eq. 32) so that facts can be selectively enabled. Entailment
is decided by refutation: a claim ``c`` is entailed iff asserting its negation
yields ``UNSAT`` (Eq. 33). When unsatisfiable, Z3 returns an UNSAT core over the
assumption literals, which we map to the proof-support set ``P(c)`` (Eq. 34).

The universally quantified inference axioms ``phi_rules`` (transitivity,
part-location composition, property inheritance, causal composition; Appendix C)
are *grounded* over the finite entity domain of each problem. Grounding keeps the
encoding decidable and gives clean, fast UNSAT cores, while remaining logically
equivalent to the quantified rules on the relevant domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Sequence, Set

import z3

from .config import NSFCoTConfig
from .logic import Atom, entities_of

# Relations that compose transitively (Appendix C, R_T).
_TRANSITIVE = ["IsA", "PartOf", "AtLocation", "Causes", "HasPrerequisite"]


@dataclass
class SMTResult:
    """Outcome of an entailment query."""

    entailed: bool
    proof_support: Set[int] = field(default_factory=set)  # fact indices in P(c)


class SMTVerifier:
    """Encodes facts + inference axioms in Z3 and answers entailment queries."""

    def __init__(self, fact_atoms: Sequence[Sequence[Atom]], config: NSFCoTConfig):
        self.config = config
        self.fact_atoms: List[List[Atom]] = [list(a) for a in fact_atoms]
        self.n = len(self.fact_atoms)

        self._bools: Dict[str, z3.BoolRef] = {}
        self._assume = [z3.Bool(f"__assume_{i}") for i in range(self.n)]

        self.solver = z3.Solver()
        self._encode_facts()
        if config.use_inference_axioms:
            self._encode_axioms()

    # -- atom <-> Z3 ------------------------------------------------------
    def _var(self, atom: Atom) -> z3.BoolRef:
        key = atom.key
        if key not in self._bools:
            self._bools[key] = z3.Bool(key)
        return self._bools[key]

    def _literal(self, atom: Atom) -> z3.BoolRef:
        var = self._var(atom)
        return z3.Not(var) if atom.negated else var

    # -- knowledge base encoding (Eq. 32) ---------------------------------
    def _encode_facts(self) -> None:
        for i, atoms in enumerate(self.fact_atoms):
            if not atoms:
                continue
            body = z3.And([self._literal(a) for a in atoms])
            self.solver.add(z3.Implies(self._assume[i], body))

    # -- grounded inference axioms (Appendix C) ---------------------------
    def _entity_domain(self) -> List[str]:
        ents: Set[str] = set()
        for atoms in self.fact_atoms:
            ents |= entities_of(atoms)
        return sorted(ents)

    def _atom(self, pred: str, a: str, b: str) -> z3.BoolRef:
        return self._var(Atom(pred, a, b))

    def _encode_axioms(self) -> None:
        E = self._entity_domain()
        if not E:
            return

        # Transitivity: R(x,y) ∧ R(y,z) ⇒ R(x,z)  (Eq. 23)
        for R in _TRANSITIVE:
            for x, y, z in product(E, repeat=3):
                self.solver.add(
                    z3.Implies(
                        z3.And(self._atom(R, x, y), self._atom(R, y, z)),
                        self._atom(R, x, z),
                    )
                )

        # Part-location composition  (Eqs. 24-25)
        for x, y, z in product(E, repeat=3):
            self.solver.add(
                z3.Implies(
                    z3.And(self._atom("PartOf", x, y), self._atom("AtLocation", y, z)),
                    self._atom("AtLocation", x, z),
                )
            )
            self.solver.add(
                z3.Implies(
                    z3.And(self._atom("AtLocation", x, y), self._atom("PartOf", y, z)),
                    self._atom("AtLocation", x, z),
                )
            )

        # Property / capability / has inheritance along IsA  (Eqs. 26-28)
        for child, parent, p in product(E, repeat=3):
            self.solver.add(
                z3.Implies(
                    z3.And(self._atom("IsA", child, parent),
                           self._atom("HasProperty", parent, p)),
                    self._atom("HasProperty", child, p),
                )
            )
            self.solver.add(
                z3.Implies(
                    z3.And(self._atom("IsA", child, parent),
                           self._atom("CapableOf", parent, p)),
                    self._atom("CapableOf", child, p),
                )
            )
            self.solver.add(
                z3.Implies(
                    z3.And(self._atom("IsA", child, parent),
                           self._atom("HasA", parent, p)),
                    self._atom("HasA", child, p),
                )
            )

        # Causal composition: Causes(x,y) ∧ HasEffect(y,z) ⇒ Causes(x,z)  (Eq. 29)
        for x, y, z in product(E, repeat=3):
            self.solver.add(
                z3.Implies(
                    z3.And(self._atom("Causes", x, y), self._atom("HasEffect", y, z)),
                    self._atom("Causes", x, z),
                )
            )

    # -- entailment via refutation (Eq. 33) -------------------------------
    def entails(self, claim: Atom, enabled: Sequence[int]) -> SMTResult:
        """Check ``KB_enabled ⊢_rules claim`` and extract proof support.

        ``enabled`` is the set of fact indices to activate (all facts for
        groundedness; only ``I_k`` for validity). Facts outside ``enabled`` are
        explicitly disabled so they cannot leak into the proof.
        """
        enabled_set = set(enabled)
        assumptions: List[z3.BoolRef] = []
        for i in range(self.n):
            assumptions.append(self._assume[i] if i in enabled_set
                               else z3.Not(self._assume[i]))

        # Assert the negation of the claim and look for a contradiction.
        neg_claim = self._var(claim) if claim.negated else z3.Not(self._var(claim))
        tracker = z3.Bool("__neg_claim")
        assumptions.append(tracker)
        self.solver.push()
        self.solver.add(z3.Implies(tracker, neg_claim))
        status = self.solver.check(assumptions)

        result: SMTResult
        if status == z3.unsat:
            core = self.solver.unsat_core()
            support = {
                i for i in enabled_set if self._assume[i] in core
            }
            result = SMTResult(entailed=True, proof_support=support)
        else:
            result = SMTResult(entailed=False, proof_support=set())
        self.solver.pop()
        return result

    # -- step-level symbolic indicators (Eqs. 8-9) ------------------------
    def groundedness_validity(
        self,
        claims: Sequence[Atom],
        relied_upon: Sequence[int],
    ):
        """Return ``(G^SMT, V^SMT, R_k)`` for one step.

        * G^SMT = 1 iff every claim is entailed by the full context (Eq. 8).
        * V^SMT = 1 iff every claim is entailed by only the relied-upon facts I_k (Eq. 9).
        * R_k    is the aggregated proof support over the step's entailed claims (Eq. 35).
        """
        all_facts = list(range(self.n))
        grounded = True
        rk: Set[int] = set()
        for c in claims:
            res = self.entails(c, all_facts)
            grounded = grounded and res.entailed
            rk |= res.proof_support

        valid = True
        for c in claims:
            valid = valid and self.entails(c, relied_upon).entailed

        g_smt = 1.0 if (claims and grounded) else 0.0
        v_smt = 1.0 if (claims and valid) else 0.0
        return g_smt, v_smt, rk

    @staticmethod
    def utility_indicator(rk: Set[int], answer_support: Set[int]) -> float:
        """U^SMT = 1 iff the step's proof support overlaps the answer's (Eq. 10)."""
        return 1.0 if (rk & answer_support) else 0.0

    def answer_support(self, answer_claim: Atom) -> Set[int]:
        """Proof-support facts for the final answer claim, R_ans = P(c_ans)."""
        return self.entails(answer_claim, list(range(self.n))).proof_support
