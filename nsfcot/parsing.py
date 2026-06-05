"""Stage 2 -- Text-to-logic parsing (Section 2.4, Appendix A).

A :class:`Parser` maps a natural-language sentence to a set of ground
:class:`~nsfcot.logic.Atom` over the ConceptNet-inspired predicate vocabulary.

Two implementations are provided:

* :class:`DictionaryParser` -- looks up pre-parsed atoms supplied with an example.
  Used by the offline demo and tests so the pipeline is fully reproducible without
  any external model.
* :class:`OpenAIParser` -- the parser used in the paper (OpenAI ``o3``) with the
  exact structured prompt from Appendix A.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from .logic import Atom, parse_atoms

# The exact text-to-logic prompt from Appendix A of the paper.
PARSER_PROMPT = """You are a semantic parser that converts natural language sentences into first-order logic predicates.

Available predicates (from ConceptNet):
- IsA(X, Y): X is a type/instance of Y
- PartOf(X, Y): X is a part of Y
- AtLocation(X, Y): X is located in/at Y
- HasProperty(X, Y): X has property Y
- CapableOf(X, Y): X is capable of Y
- UsedFor(X, Y): X is used for Y
- MadeOf(X, Y): X is made of Y
- HasA(X, Y): X has/possesses Y
- Causes(X, Y): X causes Y
- HasPrerequisite(X, Y): X requires Y to happen first
- HasEffect(X, Y): X results in Y

Rules:
1. Output ONLY predicates, one per line
2. Use lowercase for all entity arguments
3. Extract the core semantic relation(s) from the sentence
4. If a sentence contains multiple relations, output multiple predicates
5. If the input is already in FOL format (e.g., "IsA(dog, mammal)"), preserve it exactly as given
6. If no predicate applies, output "None"
7. Do not add information not present in the sentence

Examples:
Input: "The Eiffel Tower is in Paris"
Output: AtLocation(eiffel_tower, paris)

Input: "Dogs are mammals that have fur"
Output: IsA(dog, mammal)
HasProperty(dog, fur)

Input: "Heavy rain causes flooding"
Output: Causes(heavy_rain, flooding)

Now parse the following sentence:
Input: "{sentence}"
"""


class Parser(ABC):
    """Maps a sentence to a list of ground atoms ``Parse(s)``."""

    @abstractmethod
    def parse(self, sentence: str) -> List[Atom]:
        ...

    def parse_all(self, sentences: Sequence[str]) -> List[List[Atom]]:
        return [self.parse(s) for s in sentences]


class DictionaryParser(Parser):
    """Returns pre-parsed atoms keyed by the *verbatim* sentence text.

    This makes the offline demo deterministic: the example file ships the
    ``o3``-style parses alongside each fact/step, and this parser simply looks
    them up. Unknown sentences fall back to attempting to parse the text itself
    as an atom (useful when steps are already written in FOL form, per rule 5).
    """

    def __init__(self, table: Dict[str, Sequence[str]]):
        self.table = {k: list(v) for k, v in table.items()}

    def parse(self, sentence: str) -> List[Atom]:
        key = sentence.strip()
        if key in self.table:
            return parse_atoms(self.table[key])
        # Fall back: maybe the sentence is itself an atom string.
        try:
            return [Atom.parse(sentence)]
        except ValueError:
            return []


class OpenAIParser(Parser):
    """Text-to-logic parser backed by OpenAI ``o3`` (the paper's configuration).

    Requires the ``openai`` package and an ``OPENAI_API_KEY`` environment
    variable. Imported lazily to avoid a hard dependency.
    """

    def __init__(self, model: str = "o3", temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "OpenAIParser requires the `openai` package: pip install openai"
            ) from exc
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def parse(self, sentence: str) -> List[Atom]:
        prompt = PARSER_PROMPT.format(sentence=sentence)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return parse_atoms(text.splitlines())
