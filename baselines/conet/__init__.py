"""
CoNet: Collaborative Cross Networks

Paper: Hu, G., Zhang, Y., & Yang, Q. (2018). CoNet: Collaborative Cross Networks for Cross-Domain Recommendation. ACM CIKM.

Key Features:
- Cross-stitch Network for hidden layer information sharing
- Dual-domain collaborative filtering
"""

from .model import CoNet
from .data_converter import CoNetDataConverter
from .trainer import CoNetTrainer
from .evaluator import CoNetEvaluator

__all__ = ["CoNet", "CoNetDataConverter", "CoNetTrainer", "CoNetEvaluator"]
