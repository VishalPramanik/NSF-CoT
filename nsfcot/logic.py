"""First-order-logic primitives for NSF-CoT.

We restrict the logical language to *ground atoms* over a fixed set of binary
predicates inspired by ConceptNet (Speer et al., 2017), exactly as described in
Section 2.4 and Appendix A of the paper. An atom has the form ``P(a, b)`` with an
optional leading negation, e.g. ``AtLocation(paris, france)`` or
``¬AtLocation(flower, antarctica)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

#: The ConceptNet-inspired predicate vocabulary P (Eq. 2 in the paper).
PREDICATES: Set[str] = {
    "IsA",
    "PartOf",
    "AtLocation",
    "HasProperty",
    "CapableOf",
    "UsedFor",
    "MadeOf",
    "HasA",
    "Causes",
    "HasPrerequisite",
    "HasEffect",
}

_ATOM_RE = re.compile(r"^\s*(¬|~|!|not\s+)?\s*([A-Za-z]+)\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*$")


@dataclass(frozen=True)
class Atom:
    """A ground atom ``P(arg1, arg2)`` with an optional negation flag."""

    predicate: str
    arg1: str
    arg2: str
    negated: bool = False

    def __post_init__(self) -> None:
        if self.predicate not in PREDICATES:
            raise ValueError(
                f"Unknown predicate {self.predicate!r}; expected one of {sorted(PREDICATES)}"
            )

    # --- convenience -----------------------------------------------------
    @property
    def positive(self) -> "Atom":
        """Return the same atom without negation (the underlying proposition)."""
        return Atom(self.predicate, self.arg1, self.arg2, negated=False)

    @property
    def key(self) -> str:
        """Stable identifier ignoring negation (used as a Z3 variable name)."""
        return f"{self.predicate}__{self.arg1}__{self.arg2}"

    def entities(self) -> Set[str]:
        return {self.arg1, self.arg2}

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        prefix = "¬" if self.negated else ""
        return f"{prefix}{self.predicate}({self.arg1}, {self.arg2})"

    # --- parsing ---------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Atom":
        """Parse a textual atom such as ``"AtLocation(paris, france)"``.

        Accepts ``¬``, ``~``, ``!`` or a leading ``not`` as negation markers.
        Entity arguments are lower-cased and whitespace-normalised to snake_case,
        matching the parser conventions in Appendix A.
        """
        m = _ATOM_RE.match(text)
        if not m:
            raise ValueError(f"Could not parse atom from {text!r}")
        neg, pred, a, b = m.groups()
        return cls(
            predicate=pred,
            arg1=_norm_entity(a),
            arg2=_norm_entity(b),
            negated=neg is not None,
        )


def _norm_entity(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"\s+", "_", token)
    return token


def entities_of(atoms: Iterable[Atom]) -> Set[str]:
    """Collect every entity mentioned by a collection of atoms."""
    out: Set[str] = set()
    for atom in atoms:
        out |= atom.entities()
    return out


def parse_atoms(lines: Sequence[str]) -> List[Atom]:
    """Parse a list of textual atoms, skipping blanks and ``None`` lines."""
    atoms: List[Atom] = []
    for line in lines:
        line = line.strip()
        if not line or line.lower() == "none":
            continue
        atoms.append(Atom.parse(line))
    return atoms
