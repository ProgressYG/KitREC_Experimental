"""
랭킹 평가 지표

CLAUDE.md Evaluation Metrics 참조:
- Hit@K: Top-K 안에 정답 포함 여부
- MRR: Mean Reciprocal Rank
- NDCG@K: Normalized Discounted Cumulative Gain
"""

import numpy as np
from typing import List, Dict, Optional


class RankingMetrics:
    """랭킹 평가 지표 계산 클래스"""

    @staticmethod
    def hit_at_k(predictions: List[Dict], ground_truth_id: str, k: int) -> float:
        """
        Hit@K: Top-K 안에 정답 포함 여부

        - Hit@1: 정확히 1위로 예측
        - Hit@5: Top-5 안에 정답 포함
        - Hit@10: Top-10 안에 정답 포함

        Args:
            predictions: 예측 결과 리스트 (rank 순서)
            ground_truth_id: 정답 item_id
            k: Top-K

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        if not predictions:
            return 0.0

        top_k_ids = [p.get("item_id", "") for p in predictions[:k]]
        return 1.0 if ground_truth_id in top_k_ids else 0.0

    @staticmethod
    def mrr(predictions: List[Dict], ground_truth_id: str) -> float:
        """
        Mean Reciprocal Rank

        - 1위 = 1.0
        - 2위 = 0.5
        - 3위 = 0.33
        - 10위 = 0.1
        - 정답 없음 = 0.0

        Args:
            predictions: 예측 결과 리스트
            ground_truth_id: 정답 item_id

        Returns:
            1/rank if found, 0.0 otherwise
        """
        for i, pred in enumerate(predictions):
            if pred.get("item_id", "") == ground_truth_id:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(predictions: List[Dict], ground_truth_id: str, k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain

        - log2 기반
        - 1위 = 1.0
        - 2위 = 0.631
        - 10위 = 0.289

        Args:
            predictions: 예측 결과 리스트
            ground_truth_id: 정답 item_id
            k: Top-K

        Returns:
            NDCG score (0~1)
        """
        dcg = 0.0
        for i, pred in enumerate(predictions[:k]):
            if pred.get("item_id", "") == ground_truth_id:
                # relevance = 1 for ground truth
                dcg = 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                break

        # IDCG: 이상적인 경우 (정답이 1위)
        idcg = 1.0 / np.log2(2)  # = 1.0

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def get_rank(predictions: List[Dict], ground_truth_id: str) -> int:
        """
        정답의 순위 반환

        Args:
            predictions: 예측 결과 리스트
            ground_truth_id: 정답 item_id

        Returns:
            1-based rank, or infinity if not found
        """
        for i, pred in enumerate(predictions):
            if pred.get("item_id", "") == ground_truth_id:
                return i + 1
        return float('inf')

    @staticmethod
    def calculate_all(
        predictions: List[Dict],
        ground_truth_id: str
    ) -> Dict[str, float]:
        """
        모든 랭킹 메트릭 계산

        Args:
            predictions: 예측 결과 리스트
            ground_truth_id: 정답 item_id

        Returns:
            {
                "hit@1": float,
                "hit@5": float,
                "hit@10": float,
                "mrr": float,
                "ndcg@5": float,
                "ndcg@10": float,
                "rank": int
            }
        """
        return {
            "hit@1": RankingMetrics.hit_at_k(predictions, ground_truth_id, 1),
            "hit@5": RankingMetrics.hit_at_k(predictions, ground_truth_id, 5),
            "hit@10": RankingMetrics.hit_at_k(predictions, ground_truth_id, 10),
            "mrr": RankingMetrics.mrr(predictions, ground_truth_id),
            "ndcg@5": RankingMetrics.ndcg_at_k(predictions, ground_truth_id, 5),
            "ndcg@10": RankingMetrics.ndcg_at_k(predictions, ground_truth_id, 10),
            "rank": RankingMetrics.get_rank(predictions, ground_truth_id),
        }

    @staticmethod
    def aggregate_metrics(
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        여러 샘플의 메트릭 집계 (평균)

        Args:
            metrics_list: 각 샘플의 메트릭 딕셔너리 리스트

        Returns:
            평균 메트릭
        """
        if not metrics_list:
            return {}

        aggregated = {}
        metric_names = metrics_list[0].keys()

        for name in metric_names:
            if name == "rank":
                # rank는 평균 대신 중앙값 사용
                ranks = [m[name] for m in metrics_list if m[name] != float('inf')]
                aggregated["mean_rank"] = np.mean(ranks) if ranks else float('inf')
                aggregated["median_rank"] = np.median(ranks) if ranks else float('inf')
            else:
                values = [m[name] for m in metrics_list]
                aggregated[name] = np.mean(values)

        return aggregated

    @staticmethod
    def calculate_with_confidence_interval(
        metrics_list: List[Dict[str, float]],
        confidence: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        신뢰구간과 함께 메트릭 계산

        Args:
            metrics_list: 각 샘플의 메트릭 딕셔너리 리스트
            confidence: 신뢰 수준 (기본 95%)

        Returns:
            {metric_name: {"mean": float, "std": float, "ci_lower": float, "ci_upper": float}}
        """
        from scipy import stats

        if not metrics_list:
            return {}

        result = {}
        metric_names = [k for k in metrics_list[0].keys() if k != "rank"]

        for name in metric_names:
            values = [m[name] for m in metrics_list]
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)

            # t-분포 기반 신뢰구간
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_value * (std / np.sqrt(n))

            result[name] = {
                "mean": mean,
                "std": std,
                "ci_lower": mean - margin,
                "ci_upper": mean + margin,
                "n_samples": n,
            }

        return result
