"""NSF-CoT: Neuro-Symbolic Formal Verification of Chain-of-Thought Faithfulness.

Reference implementation of the ACL Findings 2026 paper. See the README for
usage and the paper for the method (Pramanik et al., 2026).
"""

from .attribution import ContextCiteAttributor, StepAttribution
from .backends import GenerationResult, HuggingFaceLM, LMBackend, MockLM
from .config import NSFCoTConfig
from .data import Example, Step
from .llm_judge import EntailmentJudge, OpenAIJudge, OverlapJudge
from .logic import PREDICATES, Atom
from .parsing import DictionaryParser, OpenAIParser, Parser
from .pipeline import AuditResult, NSFCoT, StepAudit
from .pruning import prune_chain, steps_to_prune
from .scoring import faithfulness_score, verdict
from .smt_verifier import SMTResult, SMTVerifier
from .verification import HybridVerifier, StepScores

__version__ = "1.0.0"

__all__ = [
    "Atom",
    "PREDICATES",
    "NSFCoTConfig",
    "Example",
    "Step",
    "Parser",
    "DictionaryParser",
    "OpenAIParser",
    "LMBackend",
    "MockLM",
    "HuggingFaceLM",
    "GenerationResult",
    "ContextCiteAttributor",
    "StepAttribution",
    "SMTVerifier",
    "SMTResult",
    "EntailmentJudge",
    "OverlapJudge",
    "OpenAIJudge",
    "HybridVerifier",
    "StepScores",
    "faithfulness_score",
    "verdict",
    "prune_chain",
    "steps_to_prune",
    "NSFCoT",
    "AuditResult",
    "StepAudit",
]
