#!/usr/bin/env python3
"""Reference evaluation entry point using the *real* backends from the paper.

This wires the paper's production configuration:

* parser : OpenAI ``o3``               (:class:`~nsfcot.parsing.OpenAIParser`)
* model  : an open-weight HF checkpoint (:class:`~nsfcot.backends.HuggingFaceLM`)
           -- Pythia-2.8B, LLaMA-3.1-8B, or Qwen2.5-72B in the paper
* judge  : OpenAI ``o1-preview``        (:class:`~nsfcot.llm_judge.OpenAIJudge`)

It expects examples in the same JSON schema as ``examples/*.json`` but does not
require the ``fact_atoms`` / ``relevance`` annotations -- those are produced by
the real parser and attribution stages at run time.

Requirements (not installed by default; see ``requirements.txt`` extras):

    pip install torch transformers openai
    export OPENAI_API_KEY=...

Because model weights are downloaded from the Hugging Face Hub and the OpenAI
API is called over the network, this script does **not** run in a sandboxed or
offline environment -- use ``scripts/run_demo.py`` for a self-contained demo.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsfcot import NSFCoT, NSFCoTConfig  # noqa: E402
from nsfcot.data import Example  # noqa: E402


def build_pipeline(model_name: str, parser_model: str, judge_model: str) -> NSFCoT:
    """Instantiate NSF-CoT with the real parser, LM, and judge backends."""
    try:
        from nsfcot.backends import HuggingFaceLM
        from nsfcot.parsing import OpenAIParser
        from nsfcot.llm_judge import OpenAIJudge
    except Exception as exc:  # pragma: no cover - import surface
        raise SystemExit(f"Failed to import a real backend: {exc}")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. The OpenAI parser (o3) and judge "
            "(o1-preview) require API access. Set the variable or use "
            "scripts/run_demo.py for the offline demo."
        )

    parser = OpenAIParser(model=parser_model)
    lm = HuggingFaceLM(model_name)         # downloads weights from the HF Hub
    judge = OpenAIJudge(model=judge_model)
    return NSFCoT(parser=parser, lm=lm, judge=judge, config=NSFCoTConfig())


def main() -> int:
    ap = argparse.ArgumentParser(description="Run NSF-CoT with real backends.")
    ap.add_argument("example", help="Path to an example JSON file.")
    ap.add_argument(
        "--model",
        default="EleutherAI/pythia-2.8b",
        help="HuggingFace model id for the audited LM (default: Pythia-2.8B).",
    )
    ap.add_argument("--parser-model", default="o3", help="OpenAI text-to-logic parser.")
    ap.add_argument("--judge-model", default="o1-preview", help="OpenAI entailment judge.")
    args = ap.parse_args()

    nsf = build_pipeline(args.model, args.parser_model, args.judge_model)
    result = nsf.audit(Example.load(args.example))

    for s in result.steps:
        sc = s.scores
        print(
            f"s{s.index + 1}  G={sc.G:.2f} V={sc.V:.2f} U={sc.U:.2f} "
            f"F={s.F:.4f}  I_k={[i + 1 for i in s.relied_upon]}  {s.verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
