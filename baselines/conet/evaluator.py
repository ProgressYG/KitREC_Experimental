"""
CoNet Evaluator

동일 Candidate Set을 사용한 평가
CLAUDE.md: 모든 Baseline 동일 Candidate Set 사용 필수
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

from .model import CoNet
from .data_converter import CoNetSample, CoNetDataConverter
from ..base_evaluator import BaseEvaluator


class CoNetEvaluator(BaseEvaluator):
    """CoNet 평가 클래스"""

    def __init__(
        self,
        model: CoNet,
        data_converter: CoNetDataConverter,
        device: str = "cuda"
    ):
        super().__init__(device)
        self.model = model.to(device)
        self.model.eval()
        self.data_converter = data_converter

    def evaluate_sample(
        self,
        sample: CoNetSample
    ) -> Dict[str, float]:
        """
        단일 샘플 평가

        CLAUDE.md Critical:
        - 동일 Candidate Set (100개) 사용
        - 순위 기반 메트릭 계산

        Args:
            sample: CoNetSample

        Returns:
            {hit@1, hit@5, hit@10, mrr, ndcg@5, ndcg@10}
        """
        if not sample.candidate_item_ids:
            return {
                "hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0,
                "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0
            }

        # Validate candidate set (100 items required)
        self.validate_candidate_set(
            sample.candidate_item_ids,
            gt_id=sample.ground_truth_id if hasattr(sample, 'ground_truth_id') else None,
            raise_on_error=False  # Log warning but don't crash
        )

        # Get scores for all candidates
        candidate_ids = torch.tensor(
            sample.candidate_item_ids,
            dtype=torch.long,
            device=self.device
        )

        scores = self.model.get_candidate_scores(
            sample.user_id,
            candidate_ids,
            device=self.device
        )

        # Rank candidates by score (descending)
        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()

        # Find GT position
        gt_idx = sample.ground_truth_idx
        if gt_idx < 0:
            # GT not in candidates (should not happen if data is correct)
            return {
                "hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0,
                "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0
            }

        # GT rank (1-indexed)
        gt_rank = np.where(sorted_indices == gt_idx)[0][0] + 1

        return {
            "hit@1": 1.0 if gt_rank <= 1 else 0.0,
            "hit@5": 1.0 if gt_rank <= 5 else 0.0,
            "hit@10": 1.0 if gt_rank <= 10 else 0.0,
            "mrr": 1.0 / gt_rank,
            "ndcg@5": 1.0 / np.log2(gt_rank + 1) if gt_rank <= 5 else 0.0,
            "ndcg@10": 1.0 / np.log2(gt_rank + 1) if gt_rank <= 10 else 0.0,
            "rank": gt_rank
        }

    def evaluate(
        self,
        samples: List[CoNetSample],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        전체 샘플 평가

        Args:
            samples: CoNetSample 리스트
            batch_size: 배치 크기 (unused, for consistency)

        Returns:
            평균 메트릭 (per_sample 포함 for statistical testing)
        """
        all_metrics = []
        # Per-sample metrics for statistical significance testing (A-2 fix)
        per_sample = {"hit@10": [], "ndcg@10": [], "mrr": []}

        for sample in tqdm(samples, desc="Evaluating CoNet"):
            metrics = self.evaluate_sample(sample)
            all_metrics.append(metrics)

            # Collect per-sample metrics for t-test
            for key in per_sample:
                per_sample[key].append(metrics.get(key, 0.0))

        # Aggregate
        if not all_metrics:
            return {}

        aggregated = {}
        for key in all_metrics[0].keys():
            if key != "rank":
                values = [m[key] for m in all_metrics]
                aggregated[key] = np.mean(values)

        # Mean rank
        ranks = [m.get("rank", float('inf')) for m in all_metrics]
        aggregated["mean_rank"] = np.mean([r for r in ranks if r != float('inf')])

        aggregated["total_samples"] = len(samples)

        # Add per-sample metrics for statistical significance testing
        aggregated["per_sample"] = per_sample

        return aggregated

    def evaluate_by_user_type(
        self,
        samples: List[CoNetSample],
        user_type_mapping: Dict[int, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        User Type별 평가

        Args:
            samples: CoNetSample 리스트
            user_type_mapping: {user_id: user_type}

        Returns:
            {user_type: metrics}
        """
        from collections import defaultdict

        grouped = defaultdict(list)

        for sample in samples:
            user_type = user_type_mapping.get(sample.user_id, "unknown")
            metrics = self.evaluate_sample(sample)
            grouped[user_type].append(metrics)

        results = {}
        for user_type, metrics_list in grouped.items():
            aggregated = {}
            for key in metrics_list[0].keys():
                if key != "rank":
                    values = [m[key] for m in metrics_list]
                    aggregated[key] = np.mean(values)
            aggregated["sample_count"] = len(metrics_list)
            results[user_type] = aggregated

        return results

    def get_predictions(
        self,
        sample: CoNetSample,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Top-K 예측 결과 반환 (KitREC 형식과 호환)

        Args:
            sample: CoNetSample
            top_k: 반환할 상위 아이템 수

        Returns:
            [{"rank": 1, "item_id": "...", "confidence_score": float}, ...]
        """
        if not sample.candidate_item_ids:
            return []

        # Get scores
        candidate_ids = torch.tensor(
            sample.candidate_item_ids,
            dtype=torch.long,
            device=self.device
        )

        scores = self.model.get_candidate_scores(
            sample.user_id,
            candidate_ids,
            device=self.device
        )

        # Rank
        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
        sorted_scores = scores[sorted_indices].cpu().numpy()

        # Get reverse vocab for item IDs
        reverse_vocab = self.data_converter.get_reverse_vocab("target_item")

        predictions = []
        for i in range(min(top_k, len(sorted_indices))):
            item_idx = sample.candidate_item_ids[sorted_indices[i]]
            item_id_str = reverse_vocab.get(item_idx, f"UNK_{item_idx}")

            # Normalize score to 1-10 range (sigmoid then scale)
            # KitREC uses 1-10 range, so we use sigmoid * 9 + 1
            raw_score = float(sorted_scores[i])
            confidence = (1 / (1 + np.exp(-raw_score))) * 9 + 1  # [0,1] -> [1,10]

            predictions.append({
                "rank": i + 1,
                "item_id": item_id_str,
                "confidence_score": confidence,
                "rationale": "CoNet collaborative filtering prediction"
            })

        return predictions
