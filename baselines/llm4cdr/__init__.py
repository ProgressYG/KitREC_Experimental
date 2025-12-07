"""
LLM4CDR: LLM-based Cross-Domain Recommendation

Paper: Liu, X., et al. (2025). Uncovering Cross-Domain Recommendation Ability of LLMs. ACM RecSys.

Key Features:
- 3-stage prompting pipeline
- Domain gap analysis
- User interest reasoning
- Candidate re-ranking

WARNING: Original paper uses 3+20~30 candidates, KitREC uses 1+99.
Must align candidate set for fair comparison.
"""

from .prompts import LLM4CDRPrompts
from .evaluator import LLM4CDREvaluator

__all__ = ["LLM4CDRPrompts", "LLM4CDREvaluator"]
