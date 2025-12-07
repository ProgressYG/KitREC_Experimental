"""
Stratified 분석

CLAUDE.md: User Type별, Metadata별 분리 분석
- RQ3: Cold-start 성능 분석 (1-core ~ 10-core)
- Sub-group: Movies Metadata 가용성별 분리
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np


class StratifiedAnalysis:
    """Stratified 분석 클래스"""

    # User Type 정의 (CLAUDE.md 참조)
    # Extended with granular core levels (5-core, 6-9 core, 10+ core)
    USER_TYPE_MAPPING = {
        # Movies domain
        "source_only_movies": {"core": 1, "core_group": "1-core", "model": "SingleFT-Movies", "domain": "Movies"},
        "cold_start_2core_movies": {"core": 2, "core_group": "2-core", "model": "DualFT-Movies", "domain": "Movies"},
        "cold_start_3core_movies": {"core": 3, "core_group": "3-core", "model": "DualFT-Movies", "domain": "Movies"},
        "cold_start_4core_movies": {"core": 4, "core_group": "4-core", "model": "DualFT-Movies", "domain": "Movies"},
        "overlapping_5core_movies": {"core": 5, "core_group": "5-core", "model": "DualFT-Movies", "domain": "Movies"},
        "overlapping_6to9core_movies": {"core": "6-9", "core_group": "6-9 core", "model": "DualFT-Movies", "domain": "Movies"},
        "overlapping_10core_movies": {"core": "10+", "core_group": "10+-core", "model": "DualFT-Movies", "domain": "Movies"},
        "overlapping_books_movies": {"core": "5+", "core_group": "5+-core", "model": "DualFT-Movies", "domain": "Movies"},

        # Music domain
        "source_only_music": {"core": 1, "core_group": "1-core", "model": "SingleFT-Music", "domain": "Music"},
        "cold_start_2core_music": {"core": 2, "core_group": "2-core", "model": "DualFT-Music", "domain": "Music"},
        "cold_start_3core_music": {"core": 3, "core_group": "3-core", "model": "DualFT-Music", "domain": "Music"},
        "cold_start_4core_music": {"core": 4, "core_group": "4-core", "model": "DualFT-Music", "domain": "Music"},
        "overlapping_5core_music": {"core": 5, "core_group": "5-core", "model": "DualFT-Music", "domain": "Music"},
        "overlapping_6to9core_music": {"core": "6-9", "core_group": "6-9 core", "model": "DualFT-Music", "domain": "Music"},
        "overlapping_10core_music": {"core": "10+", "core_group": "10+-core", "model": "DualFT-Music", "domain": "Music"},
        "overlapping_books_music": {"core": "5+", "core_group": "5+-core", "model": "DualFT-Music", "domain": "Music"},
    }

    # Core level order for sorting
    CORE_LEVEL_ORDER = ["1-core", "2-core", "3-core", "4-core", "5-core", "6-9 core", "10+-core", "5+-core", "unknown"]

    def analyze_by_user_type(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict]:
        """
        User Type별 성능 분석

        Args:
            results: [{"metrics": {...}, "metadata": {"user_type": "..."}}]

        Returns:
            {user_type: {"hit@10": float, "ndcg@10": float, ...}}
        """
        grouped = defaultdict(list)

        for result in results:
            user_type = result.get("metadata", {}).get("user_type", "unknown")
            metrics = result.get("metrics", {})
            grouped[user_type].append(metrics)

        analysis = {}
        for user_type, metrics_list in grouped.items():
            analysis[user_type] = self._aggregate_metrics(metrics_list)

            # Core level 정보 추가
            type_info = self.USER_TYPE_MAPPING.get(user_type, {})
            analysis[user_type]["core_level"] = type_info.get("core", "unknown")
            analysis[user_type]["recommended_model"] = type_info.get("model", "unknown")
            analysis[user_type]["sample_count"] = len(metrics_list)

        return analysis

    def analyze_by_core_level(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Core Level별 성능 분석 (RQ3: Cold-start)

        1-core (source_only) ~ 5+ core (overlapping)

        Args:
            results: 결과 리스트

        Returns:
            {"1-core": {...}, "2-core": {...}, ..., "5+-core": {...}}
        """
        grouped = defaultdict(list)

        for result in results:
            user_type = result.get("metadata", {}).get("user_type", "unknown")
            core = self.USER_TYPE_MAPPING.get(user_type, {}).get("core", "unknown")
            core_label = f"{core}-core" if core != "unknown" else "unknown"

            metrics = result.get("metrics", {})
            grouped[core_label].append(metrics)

        analysis = {}
        for level, metrics_list in grouped.items():
            analysis[level] = {
                **self._aggregate_metrics(metrics_list),
                "sample_count": len(metrics_list)
            }

        # 정렬 (1-core, 2-core, ..., 5-core, 6-9 core, 10+-core)
        sorted_analysis = {}
        for key in self.CORE_LEVEL_ORDER:
            if key in analysis:
                sorted_analysis[key] = analysis[key]

        return sorted_analysis

    def analyze_by_granular_core_level(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict]:
        """
        세분화된 Core Level별 성능 분석

        5-core, 6-9 core, 10+-core를 분리하여 더 정밀한 분석

        Args:
            results: 결과 리스트

        Returns:
            {"1-core": {...}, "2-core": {...}, ..., "5-core": {...}, "6-9 core": {...}, "10+-core": {...}}
        """
        grouped = defaultdict(list)

        for result in results:
            user_type = result.get("metadata", {}).get("user_type", "unknown")
            core_group = self.USER_TYPE_MAPPING.get(user_type, {}).get("core_group", "unknown")

            metrics = result.get("metrics", {})
            grouped[core_group].append(metrics)

        analysis = {}
        for level, metrics_list in grouped.items():
            analysis[level] = {
                **self._aggregate_metrics(metrics_list),
                "sample_count": len(metrics_list)
            }

        # 정렬
        sorted_analysis = {}
        for key in self.CORE_LEVEL_ORDER:
            if key in analysis:
                sorted_analysis[key] = analysis[key]

        return sorted_analysis

    def analyze_by_domain(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Target Domain별 성능 분석 (Movies vs Music)

        Args:
            results: 결과 리스트

        Returns:
            {"Movies": {...}, "Music": {...}}
        """
        grouped = defaultdict(list)

        for result in results:
            domain = result.get("metadata", {}).get("target_domain", "unknown")
            metrics = result.get("metrics", {})
            grouped[domain].append(metrics)

        return {
            domain: {
                **self._aggregate_metrics(metrics_list),
                "sample_count": len(metrics_list)
            }
            for domain, metrics_list in grouped.items()
        }

    def analyze_by_metadata_availability(
        self,
        results: List[Dict],
        metadata_lookup: Dict[str, bool]
    ) -> Dict[str, Dict]:
        """
        Movies Metadata 분리 평가 (CLAUDE.md Sub-group Analysis)

        - Group A: Target Items with Metadata (Title/Category 존재)
        - Group B: Target Items without Metadata (Unknown)

        예상 결과: Group A 성능 >> Group B 성능
        (메타데이터 없으면 추천 어려움)

        Args:
            results: 결과 리스트
            metadata_lookup: {item_id: has_metadata}

        Returns:
            {"group_a_with_metadata": {...}, "group_b_unknown": {...}}
        """
        group_a = []  # Metadata 있음
        group_b = []  # Metadata 없음 (Unknown)

        for result in results:
            gt_item_id = result.get("ground_truth", {}).get("item_id", "")
            has_metadata = metadata_lookup.get(gt_item_id, False)
            metrics = result.get("metrics", {})

            if has_metadata:
                group_a.append(metrics)
            else:
                group_b.append(metrics)

        return {
            "group_a_with_metadata": {
                **self._aggregate_metrics(group_a),
                "count": len(group_a),
                "percentage": len(group_a) / max(len(group_a) + len(group_b), 1) * 100
            },
            "group_b_unknown": {
                **self._aggregate_metrics(group_b),
                "count": len(group_b),
                "percentage": len(group_b) / max(len(group_a) + len(group_b), 1) * 100
            }
        }

    def analyze_by_candidate_set(
        self,
        results: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Candidate Set별 성능 분석 (Set A: Hard Negatives vs Set B: Random)

        Args:
            results: 결과 리스트

        Returns:
            {"set_a": {...}, "set_b": {...}}
        """
        grouped = defaultdict(list)

        for result in results:
            candidate_set = result.get("metadata", {}).get("candidate_set", "unknown")
            metrics = result.get("metrics", {})
            grouped[candidate_set.lower()].append(metrics)

        return {
            set_name: {
                **self._aggregate_metrics(metrics_list),
                "sample_count": len(metrics_list)
            }
            for set_name, metrics_list in grouped.items()
        }

    def compare_models(
        self,
        results_dict: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        모델 간 비교 분석

        Args:
            results_dict: {model_name: results}

        Returns:
            {model_name: aggregated_metrics}
        """
        comparison = {}

        for model_name, results in results_dict.items():
            metrics_list = [r.get("metrics", {}) for r in results]
            comparison[model_name] = {
                **self._aggregate_metrics(metrics_list),
                "sample_count": len(results)
            }

        return comparison

    def _aggregate_metrics(
        self,
        metrics_list: List[Dict]
    ) -> Dict[str, float]:
        """
        메트릭 집계 (평균)

        Args:
            metrics_list: 개별 메트릭 딕셔너리 리스트

        Returns:
            평균 메트릭
        """
        if not metrics_list:
            return {}

        aggregated = defaultdict(list)
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value != float('inf'):
                    aggregated[key].append(value)

        return {
            key: np.mean(values) if values else 0.0
            for key, values in aggregated.items()
        }

    def generate_comparison_table(
        self,
        analysis: Dict[str, Dict],
        metrics: List[str] = None
    ) -> str:
        """
        비교 테이블 생성 (Markdown 형식)

        Args:
            analysis: 분석 결과
            metrics: 포함할 메트릭 (기본: hit@10, ndcg@10, mrr)

        Returns:
            Markdown 테이블 문자열
        """
        if metrics is None:
            metrics = ["hit@10", "ndcg@10", "mrr"]

        # 헤더
        header = "| Category | " + " | ".join(metrics) + " | Count |"
        separator = "|" + "---|" * (len(metrics) + 2)

        rows = [header, separator]

        for category, data in analysis.items():
            values = [f"{data.get(m, 0):.4f}" for m in metrics]
            count = data.get("sample_count", data.get("count", 0))
            row = f"| {category} | " + " | ".join(values) + f" | {count} |"
            rows.append(row)

        return "\n".join(rows)

    def get_cold_start_improvement(
        self,
        analysis: Dict[str, Dict],
        baseline_category: str = "1-core"
    ) -> Dict[str, Dict[str, float]]:
        """
        Cold-start 개선 비율 계산

        baseline 대비 각 core level의 개선 비율

        Args:
            analysis: Core level별 분석 결과
            baseline_category: 기준 카테고리

        Returns:
            {category: {metric: improvement_percentage}}
        """
        if baseline_category not in analysis:
            return {}

        baseline = analysis[baseline_category]
        improvements = {}

        for category, data in analysis.items():
            if category == baseline_category:
                continue

            improvements[category] = {}
            for metric in ["hit@10", "ndcg@10", "mrr"]:
                baseline_value = baseline.get(metric, 0)
                current_value = data.get(metric, 0)

                if baseline_value > 0:
                    improvement = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    improvement = 0.0

                improvements[category][metric] = improvement

        return improvements
