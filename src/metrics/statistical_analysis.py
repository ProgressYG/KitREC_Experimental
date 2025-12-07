"""
통계적 유의성 검정

CLAUDE.md: 논문 발표를 위해 모든 Baseline 비교에서 통계적 유의성 보고 필수
- Paired t-test
- Bootstrap confidence interval
- Effect size (Cohen's d)
- Multiple comparison correction (Bonferroni, Holm, FDR)
"""

from scipy import stats
from scipy.stats import norm
import numpy as np
from typing import List, Dict, Tuple, Optional


# Global random seed for reproducibility
RANDOM_SEED = 42


class StatisticalAnalysis:
    """통계적 분석 클래스"""

    @staticmethod
    def paired_t_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict:
        """
        Paired t-test for per-sample metric comparison

        CLAUDE.md 수정사항:
        - RQ1: KitREC-Full vs Ablation models
        - RQ2: KitREC vs Baselines

        Args:
            scores_a: 첫 번째 모델의 점수 (per-sample)
            scores_b: 두 번째 모델의 점수 (per-sample)

        Returns:
            {
                "t_statistic": float,
                "p_value": float,
                "significant_at_0.05": bool,
                "significant_at_0.01": bool,
                "significant_at_0.001": bool,
                "effect_size_cohens_d": float,
                "mean_diff": float,
                "n_samples": int
            }
        """
        if len(scores_a) != len(scores_b):
            raise ValueError(f"Score lists must have same length: {len(scores_a)} vs {len(scores_b)}")

        if len(scores_a) < 2:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant_at_0.05": False,
                "significant_at_0.01": False,
                "significant_at_0.001": False,
                "effect_size_cohens_d": 0.0,
                "mean_diff": 0.0,
                "n_samples": len(scores_a)
            }

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        # Cohen's d effect size
        diff = np.array(scores_a) - np.array(scores_b)
        std_diff = np.std(diff, ddof=1)
        effect_size = np.mean(diff) / std_diff if std_diff > 0 else 0.0

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "significant_at_0.001": p_value < 0.001,
            "effect_size_cohens_d": float(effect_size),
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(std_diff),
            "n_samples": len(scores_a)
        }

    @staticmethod
    def bootstrap_ci(
        scores: List[float],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        random_seed: int = 42
    ) -> Dict:
        """
        Bootstrap confidence interval for single metric

        Args:
            scores: 점수 리스트
            n_bootstrap: 부트스트랩 반복 횟수
            ci: 신뢰 수준 (기본 95%)
            random_seed: 랜덤 시드

        Returns:
            {
                "mean": float,
                "std": float,
                "ci_lower": float,
                "ci_upper": float,
                "ci_level": float
            }
        """
        np.random.seed(random_seed)

        if len(scores) < 2:
            return {
                "mean": np.mean(scores) if scores else 0.0,
                "std": 0.0,
                "ci_lower": np.mean(scores) if scores else 0.0,
                "ci_upper": np.mean(scores) if scores else 0.0,
                "ci_level": ci
            }

        bootstrapped = [
            np.mean(np.random.choice(scores, size=len(scores), replace=True))
            for _ in range(n_bootstrap)
        ]

        alpha = (1 - ci) / 2
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci_lower": float(np.percentile(bootstrapped, alpha * 100)),
            "ci_upper": float(np.percentile(bootstrapped, (1 - alpha) * 100)),
            "ci_level": ci
        }

    @staticmethod
    def compare_all_baselines(
        kitrec_scores: Dict[str, List[float]],
        baseline_scores: Dict[str, Dict[str, List[float]]]
    ) -> Dict:
        """
        Compare KitREC against all baselines for all metrics

        Args:
            kitrec_scores: {metric_name: [per_sample_scores]}
            baseline_scores: {baseline_name: {metric_name: [per_sample_scores]}}

        Returns:
            {baseline_name: {metric: t_test_result}}
        """
        results = {}

        for baseline_name, baseline_metrics in baseline_scores.items():
            results[baseline_name] = {}

            for metric, kitrec_metric_scores in kitrec_scores.items():
                if metric in baseline_metrics:
                    results[baseline_name][metric] = StatisticalAnalysis.paired_t_test(
                        kitrec_metric_scores,
                        baseline_metrics[metric]
                    )

        return results

    @staticmethod
    def format_for_paper(
        result: Dict,
        metric_value: Optional[float] = None
    ) -> str:
        """
        Format t-test result for paper table

        표기법:
        - *: p < 0.05
        - **: p < 0.01
        - ***: p < 0.001

        Args:
            result: t-test 결과
            metric_value: 표시할 메트릭 값 (없으면 mean_diff 사용)

        Returns:
            "0.85***" 형식의 문자열
        """
        value = metric_value if metric_value is not None else result.get("mean_diff", 0)

        if result.get("significant_at_0.001"):
            return f"{value:.4f}***"
        elif result.get("significant_at_0.01"):
            return f"{value:.4f}**"
        elif result.get("significant_at_0.05"):
            return f"{value:.4f}*"
        else:
            return f"{value:.4f}"

    @staticmethod
    def interpret_effect_size(cohens_d: float) -> str:
        """
        Cohen's d 해석

        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
        """
        d = abs(cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    @staticmethod
    def generate_significance_table(
        comparison_results: Dict[str, Dict[str, Dict]],
        metrics: List[str] = None
    ) -> str:
        """
        유의성 검정 결과 테이블 생성 (Markdown)

        Args:
            comparison_results: {baseline: {metric: t_test_result}}
            metrics: 포함할 메트릭 리스트

        Returns:
            Markdown 테이블
        """
        if metrics is None:
            metrics = ["hit@10", "ndcg@10", "mrr"]

        # 헤더
        header = "| Baseline | " + " | ".join(metrics) + " |"
        separator = "|" + "---|" * (len(metrics) + 1)

        rows = [header, separator]

        for baseline, metric_results in comparison_results.items():
            values = []
            for m in metrics:
                if m in metric_results:
                    result = metric_results[m]
                    formatted = StatisticalAnalysis.format_for_paper(
                        result,
                        result.get("mean_diff")
                    )
                    values.append(formatted)
                else:
                    values.append("-")

            row = f"| {baseline} | " + " | ".join(values) + " |"
            rows.append(row)

        # 범례 추가
        legend = "\n*p < 0.05, **p < 0.01, ***p < 0.001"
        rows.append(legend)

        return "\n".join(rows)

    @staticmethod
    def wilcoxon_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict:
        """
        Wilcoxon signed-rank test (비모수 대안)

        정규성 가정이 위배될 때 사용

        Args:
            scores_a: 첫 번째 모델의 점수
            scores_b: 두 번째 모델의 점수

        Returns:
            Wilcoxon test 결과
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")

        if len(scores_a) < 10:
            # 샘플이 너무 적으면 의미 없음
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant_at_0.05": False,
                "warning": "Sample size too small for Wilcoxon test"
            }

        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
            return {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant_at_0.05": p_value < 0.05,
                "significant_at_0.01": p_value < 0.01,
            }
        except ValueError as e:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant_at_0.05": False,
                "error": str(e)
            }

    @staticmethod
    def normality_test(
        scores: List[float]
    ) -> Dict:
        """
        Shapiro-Wilk 정규성 검정

        p > 0.05이면 정규 분포 가정 유지

        Args:
            scores: 점수 리스트

        Returns:
            정규성 검정 결과
        """
        if len(scores) < 3:
            return {"is_normal": False, "warning": "Sample too small"}

        stat, p_value = stats.shapiro(scores)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05
        }

    # =========================================================================
    # Multiple Comparison Correction (Critical #5)
    # =========================================================================

    @staticmethod
    def bonferroni_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        Bonferroni correction for multiple comparisons

        가장 보수적인 다중 비교 보정 방법
        adjusted_alpha = alpha / n_comparisons

        Args:
            p_values: 각 비교의 p-value 리스트
            alpha: 유의 수준 (기본 0.05)

        Returns:
            보정된 결과
        """
        n = len(p_values)
        if n == 0:
            return {"error": "Empty p_values list"}

        adjusted_alpha = alpha / n
        adjusted_p_values = [min(p * n, 1.0) for p in p_values]

        return {
            "method": "bonferroni",
            "n_comparisons": n,
            "original_alpha": alpha,
            "adjusted_alpha": adjusted_alpha,
            "original_p_values": p_values,
            "adjusted_p_values": adjusted_p_values,
            "significant": [p < adjusted_alpha for p in p_values],
            "n_significant": sum(p < adjusted_alpha for p in p_values)
        }

    @staticmethod
    def holm_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        Holm-Bonferroni step-down correction

        Bonferroni보다 덜 보수적이면서도 FWER 제어
        순차적으로 검정하여 더 많은 유의한 결과 발견 가능

        Args:
            p_values: 각 비교의 p-value 리스트
            alpha: 유의 수준 (기본 0.05)

        Returns:
            보정된 결과
        """
        n = len(p_values)
        if n == 0:
            return {"error": "Empty p_values list"}

        # p-value를 정렬하고 인덱스 기억
        sorted_indices = np.argsort(p_values)
        sorted_p = [p_values[i] for i in sorted_indices]

        # Holm procedure - compute all adjusted p-values first
        significant = [False] * n
        adjusted_p = [1.0] * n

        # Calculate adjusted p-values for all comparisons
        # Holm adjusted p: p_i * (n - i), with step-up enforcement
        max_adjusted = 0.0
        for i, idx in enumerate(sorted_indices):
            raw_adjusted = sorted_p[i] * (n - i)
            # Step-up enforcement: adjusted p-value cannot be smaller than previous
            max_adjusted = max(max_adjusted, raw_adjusted)
            adjusted_p[idx] = min(max_adjusted, 1.0)

        # Determine significance: reject if adjusted p < alpha
        # Stop rejecting once first non-significant is found (step-down)
        reject = True
        for i, idx in enumerate(sorted_indices):
            if reject and adjusted_p[idx] < alpha:
                significant[idx] = True
            else:
                reject = False  # Stop rejecting from this point

        return {
            "method": "holm",
            "n_comparisons": n,
            "original_alpha": alpha,
            "original_p_values": p_values,
            "adjusted_p_values": adjusted_p,
            "significant": significant,
            "n_significant": sum(significant)
        }

    @staticmethod
    def fdr_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        Benjamini-Hochberg False Discovery Rate (FDR) correction

        FWER 대신 FDR을 제어하므로 더 많은 발견 가능
        탐색적 연구에 적합

        Args:
            p_values: 각 비교의 p-value 리스트
            alpha: FDR 수준 (기본 0.05)

        Returns:
            보정된 결과
        """
        n = len(p_values)
        if n == 0:
            return {"error": "Empty p_values list"}

        # p-value를 정렬
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array([p_values[i] for i in sorted_indices])

        # BH procedure
        adjusted_p = np.zeros(n)
        significant = [False] * n

        # Calculate adjusted p-values (cumulative minimum from right)
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                adjusted_p[sorted_indices[i]] = sorted_p[i]
            else:
                adjusted_p[sorted_indices[i]] = min(
                    adjusted_p[sorted_indices[i + 1]],
                    sorted_p[i] * n / (i + 1)
                )

        adjusted_p = np.minimum(adjusted_p, 1.0)

        # Determine significance
        for i, idx in enumerate(sorted_indices):
            bh_threshold = (i + 1) * alpha / n
            if sorted_p[i] <= bh_threshold:
                significant[idx] = True

        return {
            "method": "benjamini_hochberg_fdr",
            "n_comparisons": n,
            "original_alpha": alpha,
            "original_p_values": p_values,
            "adjusted_p_values": adjusted_p.tolist(),
            "significant": significant,
            "n_significant": sum(significant)
        }

    @staticmethod
    def apply_multiple_correction(
        comparison_results: Dict[str, Dict[str, Dict]],
        method: str = "holm",
        alpha: float = 0.05
    ) -> Dict:
        """
        비교 결과에 다중 비교 보정 일괄 적용

        Args:
            comparison_results: {baseline: {metric: t_test_result}}
            method: 보정 방법 ("bonferroni", "holm", "fdr")
            alpha: 유의 수준

        Returns:
            보정된 비교 결과
        """
        # 모든 p-value 수집
        all_p_values = []
        p_value_map = []  # (baseline, metric, p_value)

        for baseline, metrics in comparison_results.items():
            for metric, result in metrics.items():
                if "p_value" in result:
                    p_val = result["p_value"]
                    all_p_values.append(p_val)
                    p_value_map.append((baseline, metric, p_val))

        if not all_p_values:
            return {"error": "No p-values found in comparison results"}

        # 보정 적용
        if method == "bonferroni":
            correction = StatisticalAnalysis.bonferroni_correction(all_p_values, alpha)
        elif method == "holm":
            correction = StatisticalAnalysis.holm_correction(all_p_values, alpha)
        elif method == "fdr":
            correction = StatisticalAnalysis.fdr_correction(all_p_values, alpha)
        else:
            raise ValueError(f"Unknown correction method: {method}")

        # 결과에 보정된 유의성 추가
        corrected_results = {}
        for i, (baseline, metric, _) in enumerate(p_value_map):
            if baseline not in corrected_results:
                corrected_results[baseline] = {}

            original_result = comparison_results[baseline][metric].copy()
            original_result["corrected_significant"] = correction["significant"][i]
            original_result["corrected_p_value"] = correction["adjusted_p_values"][i]
            original_result["correction_method"] = method

            corrected_results[baseline][metric] = original_result

        return {
            "corrected_results": corrected_results,
            "correction_summary": correction
        }

    # =========================================================================
    # Bootstrap BCa (Bias-Corrected and Accelerated) - Enhanced
    # =========================================================================

    @staticmethod
    def bootstrap_bca_ci(
        scores: List[float],
        n_bootstrap: int = 2000,
        ci: float = 0.95,
        random_seed: int = RANDOM_SEED
    ) -> Dict:
        """
        Bias-Corrected and Accelerated (BCa) Bootstrap Confidence Interval

        기본 percentile 방법보다 더 정확한 신뢰구간 제공

        Args:
            scores: 점수 리스트
            n_bootstrap: 부트스트랩 반복 횟수
            ci: 신뢰 수준 (기본 95%)
            random_seed: 랜덤 시드

        Returns:
            BCa 신뢰구간
        """
        np.random.seed(random_seed)
        scores = np.array(scores)
        n = len(scores)

        if n < 3:
            return {
                "mean": float(np.mean(scores)) if n > 0 else 0.0,
                "ci_lower": float(np.mean(scores)) if n > 0 else 0.0,
                "ci_upper": float(np.mean(scores)) if n > 0 else 0.0,
                "method": "bca",
                "warning": "Sample size too small for BCa"
            }

        theta_hat = np.mean(scores)

        # Bootstrap samples
        boot_means = np.array([
            np.mean(np.random.choice(scores, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])

        # Bias correction factor (z0)
        prop_less = np.mean(boot_means < theta_hat)
        if prop_less == 0:
            prop_less = 1 / (n_bootstrap + 1)
        elif prop_less == 1:
            prop_less = n_bootstrap / (n_bootstrap + 1)
        z0 = norm.ppf(prop_less)

        # Acceleration factor (a) using jackknife
        jackknife_means = np.array([
            np.mean(np.delete(scores, i)) for i in range(n)
        ])
        jk_mean = np.mean(jackknife_means)

        num = np.sum((jk_mean - jackknife_means) ** 3)
        denom = 6 * (np.sum((jk_mean - jackknife_means) ** 2) ** 1.5)

        if denom == 0:
            a = 0
        else:
            a = num / denom

        # BCa percentiles
        alpha = (1 - ci) / 2
        z_alpha = norm.ppf(alpha)
        z_1_alpha = norm.ppf(1 - alpha)

        # Adjusted percentiles
        def bca_percentile(z):
            return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

        p_lower = bca_percentile(z_alpha)
        p_upper = bca_percentile(z_1_alpha)

        # Clip to valid range
        p_lower = max(0.001, min(p_lower, 0.999))
        p_upper = max(0.001, min(p_upper, 0.999))

        return {
            "mean": float(theta_hat),
            "std": float(np.std(scores)),
            "ci_lower": float(np.percentile(boot_means, p_lower * 100)),
            "ci_upper": float(np.percentile(boot_means, p_upper * 100)),
            "ci_level": ci,
            "method": "bca",
            "bias_correction": float(z0),
            "acceleration": float(a)
        }

    # =========================================================================
    # Robust Paired Test (auto-select based on normality)
    # =========================================================================

    @staticmethod
    def robust_paired_test(
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        정규성에 따라 적절한 검정 자동 선택

        정규 분포 → Paired t-test
        비정규 분포 → Wilcoxon signed-rank test

        Args:
            scores_a: 첫 번째 모델의 점수
            scores_b: 두 번째 모델의 점수
            alpha: 정규성 검정 유의 수준

        Returns:
            선택된 검정 결과
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")

        if len(scores_a) < 3:
            return {
                "test_used": "none",
                "warning": "Sample size too small",
                "p_value": 1.0,
                "significant_at_0.05": False
            }

        # 차이의 정규성 검정
        diff = np.array(scores_a) - np.array(scores_b)
        normality = StatisticalAnalysis.normality_test(diff.tolist())

        if normality.get("is_normal", False):
            # 정규 분포 → Paired t-test
            result = StatisticalAnalysis.paired_t_test(scores_a, scores_b)
            result["test_used"] = "paired_t_test"
            result["normality_p_value"] = normality.get("p_value")
        else:
            # 비정규 분포 → Wilcoxon
            result = StatisticalAnalysis.wilcoxon_test(scores_a, scores_b)
            result["test_used"] = "wilcoxon"
            result["normality_p_value"] = normality.get("p_value")

        return result

    # =========================================================================
    # ANOVA for multiple group comparison
    # =========================================================================

    @staticmethod
    def one_way_anova(
        groups: Dict[str, List[float]]
    ) -> Dict:
        """
        One-way ANOVA for comparing multiple independent groups

        3개 이상의 모델을 비교할 때 사용

        Args:
            groups: {group_name: [scores]}

        Returns:
            ANOVA 결과
        """
        group_scores = list(groups.values())

        if len(group_scores) < 2:
            return {"error": "Need at least 2 groups for ANOVA"}

        if any(len(g) < 2 for g in group_scores):
            return {"error": "Each group needs at least 2 samples"}

        f_stat, p_value = stats.f_oneway(*group_scores)

        return {
            "test": "one_way_anova",
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "groups": list(groups.keys()),
            "group_sizes": [len(g) for g in group_scores]
        }

    @staticmethod
    def friedman_test(
        groups: Dict[str, List[float]]
    ) -> Dict:
        """
        Friedman test for repeated measures (non-parametric)

        동일 샘플에서 여러 조건을 비교할 때 사용 (반복 측정)
        ANOVA의 비모수 대안

        Args:
            groups: {group_name: [scores]} - 모든 그룹이 같은 길이여야 함

        Returns:
            Friedman 검정 결과
        """
        group_scores = list(groups.values())

        if len(group_scores) < 3:
            return {"error": "Need at least 3 groups for Friedman test"}

        # 모든 그룹이 같은 길이인지 확인
        lengths = [len(g) for g in group_scores]
        if len(set(lengths)) > 1:
            return {"error": f"All groups must have same length. Got: {lengths}"}

        try:
            stat, p_value = stats.friedmanchisquare(*group_scores)
            return {
                "test": "friedman",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant_at_0.05": p_value < 0.05,
                "significant_at_0.01": p_value < 0.01,
                "groups": list(groups.keys()),
                "n_samples": lengths[0]
            }
        except ValueError as e:
            return {"error": str(e)}
