"""Data model for contextual-QA instances and their (optionally pre-generated)
chain-of-thought traces.

An :class:`Example` bundles everything the verifier needs:

* the context facts ``F`` (text, and optional pre-parsed atoms);
* the question / task specification ``t``;
* a chain-of-thought ``S`` (text, and optional pre-parsed claims per step);
* the final answer and its parsed target claim ``c_ans``;
* optional ``relevance`` annotations used only by the offline :class:`MockLM`.

Examples can be loaded from JSON (see ``examples/*.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class Step:
    text: str
    atoms: List[str] = field(default_factory=list)   # pre-parsed FOL (optional)


@dataclass
class Example:
    question: str
    facts: List[str]
    steps: List[Step]
    answer_text: str
    answer_atom: str
    gold: Optional[str] = None
    fact_atoms: List[List[str]] = field(default_factory=list)   # pre-parsed (optional)
    relevance: Dict[int, List[int]] = field(default_factory=dict)  # for MockLM
    dataset: Optional[str] = None

    # -- pre-parse lookup table for the DictionaryParser ------------------
    def parse_table(self) -> Dict[str, Sequence[str]]:
        table: Dict[str, Sequence[str]] = {}
        for fact, atoms in zip(self.facts, self.fact_atoms):
            table[fact.strip()] = atoms
        for step in self.steps:
            table[step.text.strip()] = step.atoms
        table[self.answer_text.strip()] = [self.answer_atom]
        return table

    # -- (de)serialisation ------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        steps = [Step(text=s["text"], atoms=s.get("atoms", [])) for s in d["steps"]]
        relevance = {int(k): list(v) for k, v in d.get("relevance", {}).items()}
        return cls(
            question=d["question"],
            facts=list(d["facts"]),
            steps=steps,
            answer_text=d["answer_text"],
            answer_atom=d["answer_atom"],
            gold=d.get("gold"),
            fact_atoms=[list(a) for a in d.get("fact_atoms", [])],
            relevance=relevance,
            dataset=d.get("dataset"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Example":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
