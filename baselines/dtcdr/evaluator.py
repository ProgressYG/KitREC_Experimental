"""
DTCDR Evaluator

CoNet과 동일한 평가 로직 사용
"""

import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from .model import DTCDR
from .data_converter import DTCDRDataConverter, DTCDRSample
from ..base_evaluator import BaseEvaluator


class DTCDREvaluator(BaseEvaluator):
    """DTCDR 평가 클래스"""

    def __init__(
        self,
        model: DTCDR,
        data_converter: DTCDRDataConverter,
        device: str = "cuda"
    ):
        super().__init__(device)
        self.model = model.to(device)
        self.model.eval()
        self.data_converter = data_converter

    def evaluate_sample(self, sample: DTCDRSample) -> Dict[str, float]:
        """단일 샘플 평가"""
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

        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()

        gt_idx = sample.ground_truth_idx
        if gt_idx < 0:
            return {
                "hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0,
                "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0
            }

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

    def evaluate(self, samples: List[DTCDRSample], batch_size: int = 32) -> Dict[str, float]:
        """
        전체 평가

        Returns:
            평균 메트릭 (per_sample 포함 for statistical testing)
        """
        all_metrics = []
        # Per-sample metrics for statistical significance testing (A-2 fix)
        per_sample = {"hit@10": [], "ndcg@10": [], "mrr": []}

        for sample in tqdm(samples, desc="Evaluating DTCDR"):
            metrics = self.evaluate_sample(sample)
            all_metrics.append(metrics)

            # Collect per-sample metrics for t-test
            for key in per_sample:
                per_sample[key].append(metrics.get(key, 0.0))

        if not all_metrics:
            return {}

        aggregated = {}
        for key in all_metrics[0].keys():
            if key != "rank":
                values = [m[key] for m in all_metrics]
                aggregated[key] = np.mean(values)

        ranks = [m.get("rank", float('inf')) for m in all_metrics]
        aggregated["mean_rank"] = np.mean([r for r in ranks if r != float('inf')])
        aggregated["total_samples"] = len(samples)

        # Add per-sample metrics for statistical significance testing
        aggregated["per_sample"] = per_sample

        return aggregated

    def evaluate_by_user_type(
        self,
        samples: List[DTCDRSample],
        user_type_mapping: Dict[int, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        User Type별 평가 (RQ3: Cold-start analysis)

        Args:
            samples: DTCDRSample 리스트
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

    def get_predictions(self, sample: DTCDRSample, top_k: int = 10) -> List[Dict]:
        """Top-K 예측 반환"""
        if not sample.candidate_item_ids:
            return []

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

        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
        sorted_scores = scores[sorted_indices].cpu().numpy()

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
                "rationale": "DTCDR dual-target prediction"
            })

        return predictions
