"""
Base Evaluator Class

모든 베이스라인 평가기의 공통 베이스 클래스
- Candidate Set 검증 (100 items + GT 포함)
- 공통 메트릭 계산 (Hit@K, MRR, NDCG@K)

Updated 2025-12-07:
- Logging 추가 for raise_on_error=False
- per-sample metrics 수집 지원
- RQ4 평가 대상 명확화: Baseline은 RQ4(Explainability) 평가 대상이 아님
  - Baseline은 Confidence Score/MAE/RMSE 계산 불필요
  - normalize_confidence()는 레거시 호환용으로 유지
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np
import torch
import logging

# Setup logging
logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """모든 베이스라인 평가기의 공통 베이스 클래스"""

    EXPECTED_CANDIDATE_COUNT = 100  # KitREC 표준: 1 GT + 99 Negatives

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

    def validate_candidate_set(
        self,
        candidates: List,
        gt_id: Optional[str] = None,
        raise_on_error: bool = True
    ) -> Dict[str, bool]:
        """
        Candidate Set 검증

        CLAUDE.md Critical:
        - 모든 Baseline 동일 Candidate Set (100개) 사용 필수
        - Ground Truth가 Candidate Set에 포함되어야 함

        Args:
            candidates: Candidate item ID 리스트
            gt_id: Ground truth item ID (검증용)
            raise_on_error: True면 오류 시 예외 발생

        Returns:
            {"valid_count": bool, "gt_included": bool, "is_valid": bool}
        """
        result = {
            "candidate_count": len(candidates),
            "expected_count": self.EXPECTED_CANDIDATE_COUNT,
            "valid_count": len(candidates) == self.EXPECTED_CANDIDATE_COUNT,
            "gt_included": True if gt_id is None else gt_id in candidates,
            "is_valid": False
        }
        result["is_valid"] = result["valid_count"] and result["gt_included"]

        if not result["is_valid"]:
            msg_parts = []
            if not result["valid_count"]:
                msg_parts.append(
                    f"candidate_count={len(candidates)} (expected {self.EXPECTED_CANDIDATE_COUNT})"
                )
            if not result["gt_included"] and gt_id is not None:
                msg_parts.append(f"gt_id '{gt_id}' not in candidates")

            full_msg = f"Invalid candidate set: {', '.join(msg_parts)}"

            if raise_on_error:
                raise ValueError(full_msg)
            else:
                # Log warning when not raising (A-3 fix: passive validation → active logging)
                logger.warning(full_msg)

        return result

    def normalize_confidence(self, raw_score: float) -> float:
        """
        Confidence score를 1-10 범위로 정규화

        ⚠️ 참고: RQ4(Explainability)는 KitREC 모델만 평가 대상이므로
        Baseline에서는 이 메서드를 실제로 사용하지 않습니다.
        레거시 호환용으로 유지됩니다.

        Args:
            raw_score: Raw model score

        Returns:
            Normalized confidence score in [1, 10] range
        """
        # Sigmoid then scale to [1, 10]
        sigmoid = 1 / (1 + np.exp(-raw_score))
        return sigmoid * 9 + 1  # [0, 1] -> [1, 10]

    def calculate_metrics(self, gt_rank: int) -> Dict[str, float]:
        """
        공통 메트릭 계산

        Args:
            gt_rank: Ground truth 아이템의 순위 (1-indexed)

        Returns:
            {hit@1, hit@5, hit@10, mrr, ndcg@5, ndcg@10}
        """
        if gt_rank <= 0:
            return {
                "hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0,
                "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0,
            }

        return {
            "hit@1": 1.0 if gt_rank <= 1 else 0.0,
            "hit@5": 1.0 if gt_rank <= 5 else 0.0,
            "hit@10": 1.0 if gt_rank <= 10 else 0.0,
            "mrr": 1.0 / gt_rank,
            "ndcg@5": 1.0 / np.log2(gt_rank + 1) if gt_rank <= 5 else 0.0,
            "ndcg@10": 1.0 / np.log2(gt_rank + 1) if gt_rank <= 10 else 0.0,
        }

    @abstractmethod
    def evaluate_sample(self, sample) -> Dict:
        """
        단일 샘플 평가

        Args:
            sample: 평가 샘플

        Returns:
            메트릭 딕셔너리
        """
        pass

    @abstractmethod
    def evaluate(self, samples: List, **kwargs) -> Dict[str, float]:
        """
        전체 샘플 평가

        Args:
            samples: 평가 샘플 리스트

        Returns:
            집계된 메트릭
        """
        pass
