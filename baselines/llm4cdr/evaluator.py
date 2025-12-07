"""
LLM4CDR Evaluator

3-stage pipeline 실행 및 평가
WARNING: Candidate Set을 KitREC과 동일하게 맞춰야 함 (1 GT + 99 Neg)

Updated 2025-12-07:
- BaseEvaluator 상속 추가
- Candidate Set 검증 (100개 + GT 포함) 추가
- Confidence Score 1-10 정규화 추가
- per-sample 메트릭 수집 추가
- 무효 예측 로깅 추가
"""

import json
import re
import logging
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import numpy as np

from .prompts import LLM4CDRPrompts, VanillaZeroShotPrompt
from ..base_evaluator import BaseEvaluator

# Setup logging
logger = logging.getLogger(__name__)


class LLM4CDREvaluator(BaseEvaluator):
    """
    LLM4CDR 평가 클래스

    BaseEvaluator를 상속하여 공통 검증/정규화 로직 활용
    """

    def __init__(
        self,
        inference_engine: Any,  # VLLMInference or TransformersInference
        use_3stage: bool = True,
        cache_domain_analysis: bool = True,
        device: str = "cuda"
    ):
        """
        Args:
            inference_engine: 추론 엔진
            use_3stage: 3-stage pipeline 사용 여부 (False면 Vanilla)
            cache_domain_analysis: Stage 1 결과 캐싱 여부
            device: Device (inherited from BaseEvaluator)
        """
        super().__init__(device)

        self.inference_engine = inference_engine
        self.use_3stage = use_3stage
        self.cache_domain_analysis = cache_domain_analysis

        self.prompts = LLM4CDRPrompts()
        self.vanilla_prompts = VanillaZeroShotPrompt()

        # Domain analysis cache
        self._domain_cache: Dict[str, str] = {}

        # Per-sample metrics for user_type analysis (RQ3)
        self.per_sample_metrics: Dict[str, Dict[str, float]] = {}

        # Statistics tracking
        self._validation_failures = 0
        self._total_invalid_predictions = 0

    def evaluate_sample(
        self,
        sample: Dict,
        source_domain: str = "Books",
        target_domain: str = "Movies"
    ) -> Dict:
        """
        단일 샘플 평가

        Args:
            sample: KitREC 형식 샘플
            source_domain: Source domain name
            target_domain: Target domain name

        Returns:
            {predictions: [...], metrics: {...}}
        """
        prompt = sample.get("input") or sample.get("instruction", "")
        gt = sample.get("ground_truth", {})
        if isinstance(gt, str):
            gt = json.loads(gt)
        gt_item_id = gt.get("item_id") or sample.get("gt_item_id", "")
        user_id = sample.get("user_id", "")

        # Extract data from prompt
        user_history = self.prompts.extract_user_history_from_kitrec_prompt(prompt)
        candidate_list = self.prompts.extract_candidate_list_from_kitrec_prompt(prompt)

        # Extract target history (KitREC extension for fair comparison)
        target_history = self.prompts.extract_target_history_from_kitrec_prompt(prompt)

        # Extract candidate IDs for validation
        candidate_ids = re.findall(r'\(ID:\s*([A-Z0-9]+)\)', candidate_list)

        # Validate candidate set (100 items + GT included)
        validation = self.validate_candidate_set(
            candidate_ids,
            gt_id=gt_item_id,
            raise_on_error=False
        )

        if not validation["is_valid"]:
            self._validation_failures += 1
            logger.warning(
                f"Invalid candidate set for sample {user_id}: "
                f"count={validation['candidate_count']} (expected {self.EXPECTED_CANDIDATE_COUNT}), "
                f"gt_included={validation['gt_included']}"
            )

        if self.use_3stage:
            predictions = self._run_3stage_pipeline(
                source_domain, target_domain,
                user_history, candidate_list,
                target_history=target_history
            )
        else:
            predictions = self._run_vanilla(
                source_domain, target_domain,
                user_history, candidate_list,
                target_history=target_history
            )

        # Normalize confidence scores to 1-10 range
        predictions = self._normalize_predictions(predictions)

        # Validate and filter predictions
        valid_predictions = []
        invalid_items = []
        for p in predictions:
            item_id = p.get("item_id", "")
            # Case-insensitive matching
            if item_id.upper() in [cid.upper() for cid in candidate_ids]:
                valid_predictions.append(p)
            else:
                invalid_items.append(item_id)

        num_invalid = len(invalid_items)
        if num_invalid > 0:
            self._total_invalid_predictions += num_invalid
            logger.warning(
                f"Filtered {num_invalid}/{len(predictions)} invalid predictions "
                f"(item_ids not in candidate set): {invalid_items[:5]}..."
            )

        if len(valid_predictions) < 10:
            logger.warning(
                f"Only {len(valid_predictions)} valid predictions returned "
                f"(expected at least 10 for Top-10 ranking)"
            )

        # Calculate metrics
        metrics = self._calculate_metrics(valid_predictions, gt_item_id)

        # Store per-sample metrics for user_type analysis
        if user_id:
            self.per_sample_metrics[user_id] = metrics

        return {
            "predictions": valid_predictions,
            "metrics": metrics,
            "raw_predictions": predictions,
            "num_invalid": num_invalid,
            "validation_passed": validation["is_valid"]
        }

    def _normalize_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Normalize confidence scores to 1-10 range

        LLM may output confidence scores in various formats.
        This ensures all scores are within the expected 1-10 range.
        """
        for pred in predictions:
            if "confidence_score" in pred:
                raw_score = pred["confidence_score"]
                if isinstance(raw_score, (int, float)):
                    # Clamp to 1-10 range
                    pred["confidence_score"] = max(1.0, min(10.0, float(raw_score)))
                else:
                    # Default to middle value if not a number
                    pred["confidence_score"] = 5.0
            else:
                # Add default confidence if missing
                pred["confidence_score"] = 5.0

        return predictions

    def _run_3stage_pipeline(
        self,
        source_domain: str,
        target_domain: str,
        user_history: str,
        candidate_list: str,
        target_history: str = ""
    ) -> List[Dict]:
        """
        3-stage pipeline 실행

        Note: target_history is a KitREC extension for fair comparison.
        Original LLM4CDR does not use target history.
        """

        # Stage 1: Domain Gap Analysis (캐싱)
        cache_key = f"{source_domain}_{target_domain}"
        if self.cache_domain_analysis and cache_key in self._domain_cache:
            domain_analysis = self._domain_cache[cache_key]
        else:
            stage1_prompt = self.prompts.build_stage1_prompt(source_domain, target_domain)
            domain_analysis = self.inference_engine.generate(stage1_prompt)

            if self.cache_domain_analysis:
                self._domain_cache[cache_key] = domain_analysis

        # Stage 2: User Interest Reasoning (with target history for fair comparison)
        stage2_prompt = self.prompts.build_stage2_prompt(
            source_domain, target_domain, user_history, domain_analysis,
            target_history=target_history
        )
        user_profile = self.inference_engine.generate(stage2_prompt)

        # Stage 3: Candidate Re-ranking
        stage3_prompt = self.prompts.build_stage3_prompt(
            target_domain, user_profile, domain_analysis, candidate_list
        )
        ranking_output = self.inference_engine.generate(stage3_prompt)

        # Parse output
        return self._parse_output(ranking_output)

    def _run_vanilla(
        self,
        source_domain: str,
        target_domain: str,
        user_history: str,
        candidate_list: str,
        target_history: str = ""
    ) -> List[Dict]:
        """
        Vanilla Zero-shot 실행

        Note: target_history is available but not used in vanilla mode
        to maintain the simple zero-shot baseline behavior.
        """
        prompt = self.vanilla_prompts.build_prompt(
            source_domain, target_domain, user_history, candidate_list
        )
        output = self.inference_engine.generate(prompt)
        return self._parse_output(output)

    def _parse_output(self, output: str) -> List[Dict]:
        """모델 출력 파싱"""
        # JSON 블록 추출
        pattern = r'```json\s*([\[\{][\s\S]*?[\]\}])\s*```'
        match = re.search(pattern, output)

        if not match:
            # 코드 블록 없이 JSON 찾기
            pattern = r'\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]'
            match = re.search(pattern, output)

        if not match:
            logger.warning("JSON block not found in LLM output")
            return []

        json_str = match.group(1) if '```' in output else match.group(0)

        # Trailing comma 제거
        json_str = re.sub(r',\s*\]', ']', json_str)
        json_str = re.sub(r',\s*\}', '}', json_str)

        try:
            predictions = json.loads(json_str)
            if isinstance(predictions, dict):
                predictions = [predictions]
            return predictions
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []

    def _calculate_metrics(
        self,
        predictions: List[Dict],
        gt_item_id: str
    ) -> Dict[str, float]:
        """메트릭 계산"""
        if not predictions or not gt_item_id:
            return {
                "hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0,
                "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0
            }

        # Find GT rank (case-insensitive)
        gt_rank = float('inf')
        for i, pred in enumerate(predictions):
            if pred.get("item_id", "").upper() == gt_item_id.upper():
                gt_rank = i + 1
                break

        if gt_rank == float('inf'):
            return {
                "hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0,
                "mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0
            }

        return {
            "hit@1": 1.0 if gt_rank <= 1 else 0.0,
            "hit@5": 1.0 if gt_rank <= 5 else 0.0,
            "hit@10": 1.0 if gt_rank <= 10 else 0.0,
            "mrr": 1.0 / gt_rank,
            "ndcg@5": 1.0 / np.log2(gt_rank + 1) if gt_rank <= 5 else 0.0,
            "ndcg@10": 1.0 / np.log2(gt_rank + 1) if gt_rank <= 10 else 0.0,
        }

    def evaluate(
        self,
        samples: List[Dict],
        source_domain: str = "Books",
        target_domain: str = "Movies"
    ) -> Dict[str, float]:
        """
        전체 샘플 평가

        Returns:
            Aggregated metrics including per_sample for statistical testing
        """
        all_metrics = []
        per_sample = {"hit@10": [], "ndcg@10": [], "mrr": []}
        total_invalid = 0
        validation_failures = 0

        # Reset per-sample metrics
        self.per_sample_metrics = {}
        self._validation_failures = 0
        self._total_invalid_predictions = 0

        for sample in tqdm(samples, desc="Evaluating LLM4CDR"):
            result = self.evaluate_sample(sample, source_domain, target_domain)
            all_metrics.append(result["metrics"])
            total_invalid += result["num_invalid"]

            if not result.get("validation_passed", True):
                validation_failures += 1

            # Collect per-sample metrics for t-test
            for key in per_sample:
                per_sample[key].append(result["metrics"].get(key, 0.0))

        if not all_metrics:
            return {}

        # Aggregate
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = np.mean(values)

        aggregated["total_samples"] = len(samples)
        aggregated["total_invalid_items"] = total_invalid
        aggregated["invalid_rate"] = total_invalid / (len(samples) * 10)  # Top-10 기준
        aggregated["validation_failures"] = validation_failures
        aggregated["validation_failure_rate"] = validation_failures / len(samples) if samples else 0

        # Add per-sample metrics for statistical significance testing
        aggregated["per_sample"] = per_sample

        return aggregated

    def evaluate_by_user_type(
        self,
        samples: List[Dict],
        user_type_mapping: Dict[str, str],
        source_domain: str = "Books",
        target_domain: str = "Movies"
    ) -> Dict[str, Dict[str, float]]:
        """
        User Type별 평가 (RQ3: Cold-start analysis)

        Args:
            samples: 평가 샘플 리스트
            user_type_mapping: {user_id: user_type}
            source_domain: Source domain
            target_domain: Target domain

        Returns:
            {user_type: {metric: value}}
        """
        from collections import defaultdict

        grouped = defaultdict(list)

        for sample in tqdm(samples, desc="Evaluating LLM4CDR by User Type"):
            user_id = sample.get("user_id", "")
            user_type = user_type_mapping.get(user_id, "unknown")

            result = self.evaluate_sample(sample, source_domain, target_domain)
            grouped[user_type].append(result["metrics"])

        results = {}
        for user_type, metrics_list in grouped.items():
            if not metrics_list:
                continue

            aggregated = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                aggregated[key] = np.mean(values)
            aggregated["sample_count"] = len(metrics_list)
            results[user_type] = aggregated

        return results

    def get_predictions(
        self,
        sample: Dict,
        source_domain: str = "Books",
        target_domain: str = "Movies"
    ) -> List[Dict]:
        """예측 결과 반환 (KitREC 호환 형식)"""
        result = self.evaluate_sample(sample, source_domain, target_domain)
        return result["predictions"]

    def clear_cache(self):
        """도메인 분석 캐시 초기화"""
        self._domain_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            "validation_failures": self._validation_failures,
            "total_invalid_predictions": self._total_invalid_predictions,
            "samples_evaluated": len(self.per_sample_metrics)
        }
