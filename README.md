<div align="center">

# NSF-CoT

### Neuro-Symbolic Formal Verification of Chain-of-Thought Faithfulness in Contextual Question Answering

[![Paper](https://img.shields.io/badge/ACL%20Findings-2026-b31b1b.svg)](paper/NSF-CoT_ACL2026.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[**Paper (PDF)**](paper/NSF-CoT_ACL2026.pdf) &nbsp;•&nbsp; [**arXiv**](https://arxiv.org/abs/XXXX.XXXXX) &nbsp;•&nbsp; [**Code**](https://github.com/VishalPramanik/NSF-CoT) &nbsp;•&nbsp; [**Quickstart**](#quickstart)

</div>

---


## Overview

Chain-of-thought (CoT) prompting makes language models write step-by-step
explanations, but those steps need not reflect what the model actually relied
upon to choose its answer. **NSF-CoT** is a neuro-symbolic method that audits CoT
faithfulness **step by step** for contextual question answering. For every step it
produces an interpretable verdict backed by a symbolic proof and a calibrated
neural judgement.

Given context facts `F = {f1, ..., fn}` and a question `t`, an autoregressive LM
produces a chain-of-thought `S = <s1, ..., sK>` and a final answer `y`. NSF-CoT
then scores each step on three criteria and combines them multiplicatively:

| Symbol | Criterion | Question it answers |
| :----: | :-------- | :------------------ |
| `G_k`  | **Groundedness** | Are the step's claims derivable from the *full* context `F`? |
| `V_k`  | **Validity**     | Are they derivable from only the *internally relied-upon* facts `I_k`? |
| `U_k`  | **Utility**      | Does the step contribute to deriving the final answer? |

The per-step faithfulness score is their product, `F_k = G_k · V_k · U_k`
(Eq. 14); a step is faithful iff `F_k >= 0.5`. The multiplicative form means a
step is faithful only if it is grounded **and** valid **and** useful.

## Method

The pipeline (Algorithm 1 in the paper) has five stages:

1. **CoT generation** — the LM emits a step-segmented reasoning trace (§2.3).
2. **Text-to-logic parsing** — each fact, step, and the answer is parsed into
   ground first-order-logic atoms over 11 ConceptNet-inspired binary predicates
   `{IsA, PartOf, AtLocation, HasProperty, CapableOf, UsedFor, MadeOf, HasA,
   Causes, HasPrerequisite, HasEffect}` (§2.4, Appendix A).
3. **Internal fact attribution** — ContextCite estimates the relied-upon set
   `I_k` by fitting a LASSO surrogate over `M = 128` fact-ablation masks (§2.5,
   Appendix B).
4. **Hybrid verification** — a Z3 SMT solver checks entailment by refutation and
   returns an UNSAT-core proof support, while an LLM entailment judge supplies a
   continuous score for cases that escape the formal language. The two are fused
   as `G_k = β·G_k^SMT + (1−β)·G_k^LLM` (and likewise for `V_k`, `U_k`), with
   `β = 0.5` (§2.6, Appendix C/D).
5. **Faithfulness scoring** — combine into `F_k` and emit a verdict (§2.6).

The SMT layer grounds the inference axioms (transitivity, part–location
composition, property/capability/has inheritance, causal composition;
Appendix C) over each problem's finite entity domain, which keeps the encoding
decidable and yields clean proof-support cores.

## Repository structure

```
NSF-CoT/
├── nsfcot/                  # the verification library
│   ├── logic.py            #   FOL atoms + predicate vocabulary (§2.4)
│   ├── config.py           #   hyperparameters (paper defaults)
│   ├── backends.py         #   LMBackend: MockLM (offline) + HuggingFaceLM (real)
│   ├── parsing.py          #   DictionaryParser (offline) + OpenAIParser (o3)
│   ├── attribution.py      #   ContextCite + LASSO attribution (§2.5)
│   ├── smt_verifier.py     #   Z3 encoding, axioms, proof support (§2.6, App. C)
│   ├── llm_judge.py        #   OverlapJudge (offline) + OpenAIJudge (o1-preview)
│   ├── verification.py     #   hybrid SMT/LLM fusion (Eqs. 11-13)
│   ├── scoring.py          #   faithfulness score + verdict (Eq. 14)
│   ├── pruning.py          #   faithfulness-guided pruning (§4.2)
│   ├── data.py             #   Example / Step data model + JSON loading
│   └── pipeline.py         #   NSFCoT orchestrator (Algorithm 1)
├── scripts/
│   ├── run_demo.py         # offline, self-contained demo (no weights, no network)
│   └── run_eval.py         # reference wiring for the real backends
├── examples/
│   ├── qasc_evaporation.json
│   └── hummingbird.json
├── tests/                  # pytest suite (pipeline + SMT)
├── paper/NSF-CoT_ACL2026.pdf
├── requirements.txt
├── setup.py / pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone https://github.com/VishalPramanik/NSF-CoT.git
cd NSF-CoT
pip install -r requirements.txt          # core deps: z3-solver, scikit-learn, numpy
pip install -e .                         # optional: install the package
```

The core install is lightweight and runs the full offline demo and test suite.
The real backends are optional extras (see [Real backends](#real-backends)).

## Quickstart

Run the self-contained demo. It uses deterministic offline backends so it needs
**no model weights and no network access**, while the Z3 SMT verifier runs for
real:

```bash
python scripts/run_demo.py                       # QASC evaporation example
python scripts/run_demo.py examples/hummingbird.json
```

Expected output (QASC) reproduces the paper's step-level audit pattern — the
confabulated step `s4` is flagged unfaithful and pruned, the evidence-grounded
steps are faithful, and `s5` is proven by multi-hop `Causes`-transitivity:

```
Step I_k              G     V     U        F  Verdict    Parsed claim
---------------------------------------------------------------------
s1   {f4}          0.97  0.97  0.91   0.8580  Faithful   HasA(clothes, liquid_water)
s2   {f3}          0.97  0.97  0.92   0.8748  Faithful   Causes(sunlight, heat)
s3   {f2}          0.98  0.98  0.92   0.8740  Faithful   Causes(heat, evaporation)
s4   {}            0.04  0.04  0.13   0.0002  Unfaithful HasProperty(wind, always_dominates)
s5   {f1,f2,f3}    0.90  0.90  0.94   0.7572  Faithful   Causes(sunlight, water_to_vapor)
```

### Library usage

```python
from nsfcot import NSFCoT, NSFCoTConfig, DictionaryParser, MockLM, OverlapJudge
from nsfcot.data import Example

example = Example.load("examples/qasc_evaporation.json")
nsf = NSFCoT(
    parser=DictionaryParser(example.parse_table()),
    lm=MockLM(example.relevance),
    judge=OverlapJudge(),
    config=NSFCoTConfig(),           # M=128, lambda=0.01, tau=0.5, beta=0.5
)
result = nsf.audit(example)
for s in result.steps:
    print(s.index + 1, s.verdict, round(s.F, 4), s.relied_upon)
```

## Real backends

The library is backend-agnostic. To reproduce the paper's setup, swap in the
real components (these require extra dependencies and network/API access, so they
do not run in an offline sandbox):

```bash
pip install torch transformers openai
export OPENAI_API_KEY=...
python scripts/run_eval.py examples/qasc_evaporation.json \
    --model EleutherAI/pythia-2.8b --parser-model o3 --judge-model o1-preview
```

| Component | Offline (demo/tests) | Paper configuration |
| :-------- | :------------------- | :------------------ |
| Audited LM | `MockLM` (deterministic ablation log-probs) | `HuggingFaceLM` — Pythia-2.8B, LLaMA-3.1-8B, Qwen2.5-72B |
| Text-to-logic parser | `DictionaryParser` (shipped parses) | `OpenAIParser` — OpenAI `o3` |
| Entailment judge | `OverlapJudge` (overlap heuristic) | `OpenAIJudge` — OpenAI `o1-preview` |
| SMT solver | **Z3 (real)** | **Z3 (real)** |

The attribution stage (`nsfcot/attribution.py`) and SMT verifier
(`nsfcot/smt_verifier.py`) are identical across both paths — only the model,
parser, and judge change. This makes the offline demo a faithful exercise of the
real attribution and verification code.

## Datasets

The paper evaluates on three contextual-QA benchmarks requiring multi-step
reasoning with explicit supporting facts: **OpenBookQA**, **QASC**, and
**HotpotQA**. The two bundled examples mirror the qualitative case studies in
the paper (Tables 1 and 2). To evaluate on full datasets, format each instance
into the `examples/*.json` schema (or supply raw text and let the real parser
produce the atoms) and pass it to `scripts/run_eval.py`.

## Testing

```bash
pip install pytest
pytest -q
```

The suite checks both the end-to-end pipeline (correct verdicts, attribution
recovers `I_k = ∅` for confabulations, pruning removes only unfaithful steps)
and the SMT verifier in isolation (transitivity, inheritance, negative-fact
entailment, and UNSAT-core proof-support extraction).

## Citation

If you use NSF-CoT in your research, please cite:

```bibtex
@inproceedings{pramanik2026nsfcot,
  title     = {{NSF-CoT}: Neuro-Symbolic Formal Verification of Chain-of-Thought
               Faithfulness in Contextual Question Answering},
  author    = {Pramanik, Vishal and Maliha, Maisha and Bastian, Nathaniel D. and
               Velasquez, Alvaro and Jha, Susmit and Jha, Sumit Kumar},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
