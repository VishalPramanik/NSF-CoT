"""Unit tests for the Z3 SMT verifier (groundedness, validity, inference axioms,
and UNSAT-core proof-support extraction).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsfcot.config import NSFCoTConfig  # noqa: E402
from nsfcot.logic import Atom  # noqa: E402
from nsfcot.smt_verifier import SMTVerifier  # noqa: E402


def _verifier(fact_atom_strings, **cfg):
    fact_atoms = [[Atom.parse(s) for s in atoms] for atoms in fact_atom_strings]
    return SMTVerifier(fact_atoms, NSFCoTConfig(**cfg))


def test_direct_fact_is_entailed():
    smt = _verifier([["IsA(dog, mammal)"]])
    res = smt.entails(Atom.parse("IsA(dog, mammal)"), enabled=[0])
    assert res.entailed
    assert res.proof_support == {0}


def test_unsupported_claim_not_entailed():
    smt = _verifier([["IsA(dog, mammal)"]])
    res = smt.entails(Atom.parse("IsA(cat, reptile)"), enabled=[0])
    assert not res.entailed


def test_transitivity_multi_hop():
    # Causes(a,b), Causes(b,c), Causes(c,d)  =>  Causes(a,d) by transitivity.
    smt = _verifier(
        [["Causes(a, b)"], ["Causes(b, c)"], ["Causes(c, d)"]]
    )
    res = smt.entails(Atom.parse("Causes(a, d)"), enabled=[0, 1, 2])
    assert res.entailed
    # Proof support should reference the chained facts.
    assert res.proof_support == {0, 1, 2}


def test_transitivity_disabled_by_axiom_toggle():
    smt = _verifier(
        [["Causes(a, b)"], ["Causes(b, c)"]], use_inference_axioms=False
    )
    res = smt.entails(Atom.parse("Causes(a, c)"), enabled=[0, 1])
    assert not res.entailed          # without phi_rules the hop is not derivable


def test_validity_restricts_to_relied_facts():
    smt = _verifier([["Causes(a, b)"], ["Causes(b, c)"]])
    # With only f0 enabled, Causes(a, c) cannot be proven.
    assert not smt.entails(Atom.parse("Causes(a, c)"), enabled=[0]).entailed
    # With both enabled it can.
    assert smt.entails(Atom.parse("Causes(a, c)"), enabled=[0, 1]).entailed


def test_property_inheritance_along_isa():
    smt = _verifier([["IsA(dog, mammal)"], ["HasProperty(mammal, warm_blooded)"]])
    res = smt.entails(Atom.parse("HasProperty(dog, warm_blooded)"), enabled=[0, 1])
    assert res.entailed


def test_negative_fact_entailment():
    smt = _verifier([["not CapableOf(penguin, fly)"]])
    res = smt.entails(Atom.parse("not CapableOf(penguin, fly)"), enabled=[0])
    assert res.entailed
    assert res.proof_support == {0}


def test_groundedness_validity_indicators():
    smt = _verifier([["Causes(a, b)"], ["Causes(b, c)"]])
    claims = [Atom.parse("Causes(a, c)")]
    g_smt, v_smt, rk = smt.groundedness_validity(claims, relied_upon=[0, 1])
    assert g_smt == 1.0 and v_smt == 1.0 and rk == {0, 1}
    # Relying on only f0 should fail validity but keep groundedness.
    g_smt, v_smt, _ = smt.groundedness_validity(claims, relied_upon=[0])
    assert g_smt == 1.0 and v_smt == 0.0
