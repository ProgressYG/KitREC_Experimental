"""
Model loading and wrapper modules.

- KitRECModel: Fine-tuned KitREC model loader
- BaseModel: Untuned Qwen3-14B loader
- Baseline wrappers: CoNet, DTCDR, LLM4CDR
"""

from .kitrec_model import KitRECModel
from .base_model import BaseModel

__all__ = ["KitRECModel", "BaseModel"]
