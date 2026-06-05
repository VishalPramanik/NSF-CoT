"""Stage 4 (neural) -- LLM entailment judge (Section 2.6, Appendix D).

The judge produces continuous scores ``G^LLM, V^LLM, U^LLM ∈ [0,1]`` to cover
entailments that escape the formal language of the SMT solver (paraphrase,
implicit commonsense, vague modality). SMT results are passed to the judge as
context so it can corroborate or override the symbolic finding.

Implementations:

* :class:`OverlapJudge` -- a transparent, deterministic, offline judge that scores
  entailment by entity/atom overlap between the premise and the claim. Used by the
  demo and tests. It deliberately scores claims whose entities never appear in the
  premise (e.g. an invented physical threshold) near zero -- the exact behaviour a
  faithfulness verifier should exhibit (see Appendix F.4).
* :class:`OpenAIJudge` -- the paper's judge (OpenAI ``o1-preview``) using the exact
  prompts from Appendix D.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from .logic import Atom, entities_of

GROUNDEDNESS_PROMPT = """You are an entailment judge. Given a set of context facts and a claim, determine whether the claim logically follows from the context.

Context Facts:
{facts}

Claim to verify:
{claim}

SMT Solver Result: {smt}
Proof Support (facts used): {support}

Instructions:
1. Provide a step-by-step reasoning trace explaining whether the claim follows from the context
2. Consider both explicit statements and reasonable inferences
3. Output a score between 0 and 1 indicating your confidence that the context entails the claim

Output format:
Trace: <your reasoning>
Score: <0.0 to 1.0>
"""

VALIDITY_PROMPT = """You are an entailment judge. Given a restricted set of facts that the model internally relied upon, determine whether the claim logically follows from ONLY these facts.

Internally Relied-Upon Facts:
{facts}

Claim to verify:
{claim}

SMT Solver Result: {smt}

Instructions:
1. ONLY use the internally relied-upon facts listed above
2. Do NOT use any external knowledge or other context facts
3. Provide a step-by-step reasoning trace
4. Output a score between 0 and 1 indicating your confidence

Output format:
Trace: <your reasoning>
Score: <0.0 to 1.0>
"""

UTILITY_PROMPT = """You are an entailment judge. Determine whether the given claims from a reasoning step help derive the final answer.

Step Claims:
{claim}

Final Answer Claim:
{answer}

SMT Solver Result: {smt}
Answer Proof Support: {support}

Instructions:
1. Analyze whether the step claims are necessary or helpful for reaching the final answer
2. Consider if removing these claims would make the answer harder to derive
3. Provide a step-by-step reasoning trace
4. Output a score between 0 and 1 indicating how much this step contributes to the answer

Output format:
Trace: <your reasoning>
Score: <0.0 to 1.0>
"""


class EntailmentJudge(ABC):
    """Produces continuous groundedness/validity/utility scores for a step."""

    @abstractmethod
    def groundedness(self, all_atoms: Sequence[Atom], claims: Sequence[Atom],
                     smt: float) -> float: ...

    @abstractmethod
    def validity(self, relied_atoms: Sequence[Atom], claims: Sequence[Atom],
                 smt: float) -> float: ...

    @abstractmethod
    def utility(self, claims: Sequence[Atom], answer: Atom,
                context: Sequence[Atom], smt: float) -> float: ...


# ---------------------------------------------------------------------------
# Offline deterministic judge (overlap heuristic)
# ---------------------------------------------------------------------------
class OverlapJudge(EntailmentJudge):
    """Deterministic entailment heuristic based on premise/claim atom overlap.

    Scoring rubric (applied per claim, then averaged):

    * exact atom match (same predicate, args and polarity) in the premise -> ~0.95
    * all claim entities appear in the premise (relation plausibly inferable) -> ~0.78
    * partial entity overlap -> ~0.30
    * no overlap (invented entity / confabulation) -> ~0.10

    A small deterministic jitter is added for realism. This judge requires no
    network access and is what the bundled demo and tests use.
    """

    def __init__(self, jitter: float = 0.03):
        self.jitter = jitter

    def _score_claim(self, premise: Sequence[Atom], claim: Atom) -> float:
        premise_atoms = {(a.predicate, a.arg1, a.arg2, a.negated) for a in premise}
        premise_ents = entities_of(premise)
        key = (claim.predicate, claim.arg1, claim.arg2, claim.negated)
        claim_ents = claim.entities()

        if key in premise_atoms:
            base = 0.95
        elif claim_ents <= premise_ents and premise_ents:
            base = 0.78
        elif claim_ents & premise_ents:
            base = 0.30
        else:
            base = 0.10
        j = (_hash(str(claim)) % 1000) / 1000.0  # in [0,1)
        return _clip(base + self.jitter * (2 * j - 1))

    def _aggregate(self, premise: Sequence[Atom], claims: Sequence[Atom]) -> float:
        if not claims:
            return 0.0
        return sum(self._score_claim(premise, c) for c in claims) / len(claims)

    def groundedness(self, all_atoms, claims, smt):  # noqa: D102
        return self._aggregate(all_atoms, claims)

    def validity(self, relied_atoms, claims, smt):  # noqa: D102
        return self._aggregate(relied_atoms, claims)

    def utility(self, claims, answer, context, smt):  # noqa: D102
        """A step is useful if its claims are grounded in the provided evidence.

        Steps whose entities never appear in the context (confabulations like an
        invented physical threshold) receive low utility; evidence-grounded steps
        receive high utility.
        """
        if not claims:
            return 0.0
        context_ents = entities_of(context)
        scores = []
        for c in claims:
            ents = c.entities()
            if ents <= context_ents and context_ents:
                scores.append(0.85)
            elif ents & context_ents:
                scores.append(0.6)
            else:
                scores.append(0.25)
        j = (_hash("util" + str(claims[0])) % 1000) / 1000.0
        return _clip(sum(scores) / len(scores) + self.jitter * (2 * j - 1))


# ---------------------------------------------------------------------------
# OpenAI judge (paper configuration: o1-preview)
# ---------------------------------------------------------------------------
class OpenAIJudge(EntailmentJudge):
    """LLM entailment judge backed by OpenAI ``o1-preview`` (Appendix D prompts)."""

    def __init__(self, model: str = "o1-preview"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("OpenAIJudge requires `openai`: pip install openai") from exc
        self.client = OpenAI()
        self.model = model

    def _ask(self, prompt: str) -> float:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_score(resp.choices[0].message.content or "")

    def groundedness(self, all_atoms, claims, smt):  # noqa: D102
        return self._ask(GROUNDEDNESS_PROMPT.format(
            facts=_fmt(all_atoms), claim=_fmt(claims), smt=smt, support=""))

    def validity(self, relied_atoms, claims, smt):  # noqa: D102
        return self._ask(VALIDITY_PROMPT.format(
            facts=_fmt(relied_atoms), claim=_fmt(claims), smt=smt))

    def utility(self, claims, answer, context, smt):  # noqa: D102
        return self._ask(UTILITY_PROMPT.format(
            claim=_fmt(claims), answer=str(answer), smt=smt, support=_fmt(context)))


# -- helpers ----------------------------------------------------------------
def _fmt(atoms: Sequence[Atom]) -> str:
    return "\n".join(str(a) for a in atoms) if atoms else "(none)"


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_score(text: str) -> float:
    import re
    m = re.search(r"Score:\s*([01](?:\.\d+)?)", text)
    if m:
        return _clip(float(m.group(1)))
    nums = re.findall(r"\b0?\.\d+\b|\b[01]\b", text)
    return _clip(float(nums[-1])) if nums else 0.0
