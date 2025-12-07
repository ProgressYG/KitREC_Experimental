"""
DTCDR: Dual-Target Cross-Domain Recommendation

Paper: Zhu, F., et al. (2019). DTCDR: A Framework for Dual-Target Cross-Domain Recommendation. ACM CIKM.

Key Features:
- Multi-task learning for both source and target domains
- Shared user embedding with domain adaptation
- Embedding mapping between domains
"""

from .model import DTCDR
from .data_converter import DTCDRDataConverter
from .trainer import DTCDRTrainer
from .evaluator import DTCDREvaluator

__all__ = ["DTCDR", "DTCDRDataConverter", "DTCDRTrainer", "DTCDREvaluator"]
