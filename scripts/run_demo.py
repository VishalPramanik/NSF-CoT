#!/usr/bin/env python3
"""Offline demo for NSF-CoT.

Runs the full five-stage neuro-symbolic faithfulness audit on a bundled example
using deterministic, dependency-light backends (no model weights, no network):

* :class:`~nsfcot.backends.MockLM`        -- deterministic ablation log-probs,
  so the *real* LASSO attribution code path recovers the relied-upon facts.
* :class:`~nsfcot.parsing.DictionaryParser` -- looks up the ``o3``-style parses
  shipped with the example.
* :class:`~nsfcot.llm_judge.OverlapJudge`   -- transparent entailment heuristic.

The Z3 SMT verifier is the *real* component in this path -- groundedness,
validity, proof-support extraction and multi-hop inference all run through Z3.

Usage
-----
    python scripts/run_demo.py                         # QASC evaporation example
    python scripts/run_demo.py examples/hummingbird.json

To run with real models instead, see ``scripts/run_eval.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from a source checkout without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsfcot import (  # noqa: E402
    NSFCoT,
    NSFCoTConfig,
    DictionaryParser,
    MockLM,
    OverlapJudge,
    prune_chain,
    steps_to_prune,
)
from nsfcot.data import Example  # noqa: E402

DEFAULT_EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "qasc_evaporation.json",
)


def _fmt_atoms(atoms) -> str:
    return ", ".join(str(a) for a in atoms) if atoms else "-"


def run(example_path: str) -> int:
    example = Example.load(example_path)

    parser = DictionaryParser(example.parse_table())
    lm = MockLM(example.relevance)
    judge = OverlapJudge()
    config = NSFCoTConfig()  # paper defaults: M=128, lambda=0.01, tau=0.5, beta=0.5

    nsf = NSFCoT(parser=parser, lm=lm, judge=judge, config=config)
    result = nsf.audit(example)

    print("=" * 78)
    print(f"NSF-CoT faithfulness audit  |  dataset: {example.dataset or 'n/a'}")
    print("=" * 78)
    print(f"Question : {example.question}")
    if example.gold:
        print(f"Gold     : {example.gold}")
    print()
    print("Context facts (F):")
    for i, fact in enumerate(example.facts):
        print(f"  f{i + 1}: {fact}")
    print()

    header = f"{'Step':<5}{'I_k':<12}{'G':>6}{'V':>6}{'U':>6}{'F':>9}  {'Verdict':<11}Parsed claim"
    print(header)
    print("-" * len(header))
    for s in result.steps:
        sc = s.scores
        ik = "{" + ",".join(f"f{i + 1}" for i in s.relied_upon) + "}" if s.relied_upon else "{}"
        print(
            f"s{s.index + 1:<4}{ik:<12}"
            f"{sc.G:6.2f}{sc.V:6.2f}{sc.U:6.2f}{s.F:9.4f}  "
            f"{s.verdict:<11}{_fmt_atoms(s.claims)}"
        )
    print()
    print(f"Answer claim   : {result.answer_claim}")
    print(f"Faithful steps : {[i + 1 for i in result.faithful_steps()]}")
    print(f"Unfaithful     : {[i + 1 for i in result.unfaithful_steps()]}")
    print()

    # Faithfulness-guided pruning (Section 4.2).
    drop = steps_to_prune(result, config.prune_threshold)
    pruned = prune_chain([s.text for s in example.steps], result, config.prune_threshold)
    print("Faithfulness-guided pruning  (remove steps with min(G, V) < 0.5):")
    print(f"  Removed steps: {[i + 1 for i in drop]}")
    for k, text in enumerate(pruned):
        marker = "  x" if k in drop else "   "
        print(f"{marker} s{k + 1}: {text}")
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the offline NSF-CoT demo.")
    ap.add_argument(
        "example",
        nargs="?",
        default=DEFAULT_EXAMPLE,
        help="Path to an example JSON file (defaults to the QASC evaporation example).",
    )
    args = ap.parse_args()
    return run(args.example)


if __name__ == "__main__":
    raise SystemExit(main())
