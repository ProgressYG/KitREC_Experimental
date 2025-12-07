"""
설명력 평가 지표

CLAUDE.md Evaluation Metrics:
- Confidence Score 정규화 필수 (1-10 → 0-5)
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- Perplexity: rationale 부분만 계산 (CLAUDE.md 수정사항)
- GPT-4.1 Rationale Quality Evaluation (RQ4)
"""

import numpy as np
import json
import random
from typing import List, Dict, Optional, Tuple
import torch
from tqdm import tqdm


class ExplainabilityMetrics:
    """설명력 평가 지표 계산 클래스"""

    def __init__(self, confidence_divisor: float = 2.0):
        """
        Args:
            confidence_divisor: Confidence Score 정규화 비율

        CLAUDE.md Critical Notes:
        - Model 출력: 1~10 float (NOT 0-10)
        - Ground Truth: 0~5 rating
        - 정규화: confidence / 2
        """
        self.confidence_divisor = confidence_divisor

    def normalize_confidence(self, confidence: float) -> float:
        """
        Confidence Score 정규화

        Model output (1-10) → GT scale (0-5)
        """
        return confidence / self.confidence_divisor

    def mae(
        self,
        predicted_confidences: List[float],
        ground_truth_ratings: List[float]
    ) -> float:
        """
        Mean Absolute Error

        예측 신뢰도와 실제 Rating 비교

        Args:
            predicted_confidences: 모델 출력 confidence scores (1-10)
            ground_truth_ratings: 실제 rating (0-5)

        Returns:
            MAE value
            
        Note:
            confidence_score = 0은 파싱 오류로 처리하여 통계에서 제외됨
            (CLAUDE.md 참조: ⚠️ confidence_score = 0 은 파싱 오류로 처리)
        """
        if not predicted_confidences or not ground_truth_ratings:
            return 0.0

        errors = []
        skipped = 0
        for conf, gt_rating in zip(predicted_confidences, ground_truth_ratings):
            # confidence_score = 0은 파싱 오류로 처리하여 제외
            if conf <= 0:
                skipped += 1
                continue
            normalized_conf = self.normalize_confidence(conf)
            errors.append(abs(normalized_conf - gt_rating))

        if skipped > 0:
            import logging
            logging.getLogger(__name__).warning(
                f"MAE: Skipped {skipped} samples with confidence_score <= 0 (parsing error)"
            )

        return np.mean(errors) if errors else 0.0

    def rmse(
        self,
        predicted_confidences: List[float],
        ground_truth_ratings: List[float]
    ) -> float:
        """
        Root Mean Squared Error

        Args:
            predicted_confidences: 모델 출력 confidence scores (1-10)
            ground_truth_ratings: 실제 rating (0-5)

        Returns:
            RMSE value
            
        Note:
            confidence_score = 0은 파싱 오류로 처리하여 통계에서 제외됨
            (CLAUDE.md 참조: ⚠️ confidence_score = 0 은 파싱 오류로 처리)
        """
        if not predicted_confidences or not ground_truth_ratings:
            return 0.0

        squared_errors = []
        skipped = 0
        for conf, gt_rating in zip(predicted_confidences, ground_truth_ratings):
            # confidence_score = 0은 파싱 오류로 처리하여 제외
            if conf <= 0:
                skipped += 1
                continue
            normalized_conf = self.normalize_confidence(conf)
            squared_errors.append((normalized_conf - gt_rating) ** 2)

        if skipped > 0:
            import logging
            logging.getLogger(__name__).warning(
                f"RMSE: Skipped {skipped} samples with confidence_score <= 0 (parsing error)"
            )

        return np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

    def perplexity(
        self,
        model,
        tokenizer,
        rationales: List[str],
        device: str = "cuda"
    ) -> float:
        """
        Perplexity: 추천 설명의 언어적 품질 평가

        CLAUDE.md 수정사항: rationale 부분만 계산
        - 전체 출력이 아닌 rationale 텍스트만 평가
        - 낮을수록 모델이 확신을 가지고 생성

        Args:
            model: Language model
            tokenizer: Tokenizer
            rationales: Rationale 텍스트 리스트
            device: 디바이스 (cuda/cpu)

        Returns:
            Average perplexity
        """
        if not rationales:
            return 0.0

        total_loss = 0.0
        total_tokens = 0

        model.eval()

        for rationale in rationales:
            if not rationale or not rationale.strip():
                continue

            inputs = tokenizer(
                rationale,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                num_tokens = inputs["input_ids"].size(1)

                total_loss += loss * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        return np.exp(avg_loss)

    def perplexity_batch(
        self,
        model,
        tokenizer,
        rationales: List[str],
        batch_size: int = 8,
        device: str = "cuda"
    ) -> Tuple[float, List[float]]:
        """
        배치 Perplexity 계산

        Args:
            model: Language model
            tokenizer: Tokenizer
            rationales: Rationale 텍스트 리스트
            batch_size: 배치 크기
            device: 디바이스

        Returns:
            (average_perplexity, per_sample_perplexities)
        """
        if not rationales:
            return 0.0, []

        per_sample_ppl = []
        model.eval()

        for i in range(0, len(rationales), batch_size):
            batch = rationales[i:i + batch_size]

            # 빈 rationale 필터링
            valid_batch = [(j, r) for j, r in enumerate(batch) if r and r.strip()]
            if not valid_batch:
                per_sample_ppl.extend([float('inf')] * len(batch))
                continue

            indices, texts = zip(*valid_batch)

            inputs = tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])

                # 개별 샘플 perplexity 계산
                logits = outputs.logits
                labels = inputs["input_ids"]

                for k, idx in enumerate(indices):
                    # 마스킹된 토큰 제외
                    mask = inputs["attention_mask"][k].bool()
                    sample_labels = labels[k][mask]
                    sample_logits = logits[k][mask]

                    # Cross-entropy loss
                    shift_logits = sample_logits[:-1]
                    shift_labels = sample_labels[1:]

                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(shift_logits, shift_labels)

                    ppl = np.exp(loss.item())
                    per_sample_ppl.append(ppl)

        avg_ppl = np.mean([p for p in per_sample_ppl if p != float('inf')])
        return avg_ppl, per_sample_ppl

    def calculate_all(
        self,
        predictions: List[Dict],
        ground_truth_ratings: List[float],
        model=None,
        tokenizer=None,
        calculate_perplexity: bool = False
    ) -> Dict[str, float]:
        """
        모든 설명력 메트릭 계산

        Args:
            predictions: 예측 결과 (confidence_score, rationale 포함)
            ground_truth_ratings: 실제 rating 리스트
            model: Perplexity 계산용 모델
            tokenizer: Perplexity 계산용 토크나이저
            calculate_perplexity: Perplexity 계산 여부

        Returns:
            {
                "mae": float,
                "rmse": float,
                "perplexity": float (optional)
            }
        """
        # Confidence scores 추출
        confidences = [
            float(p.get("confidence_score", 5.0))
            for p in predictions
        ]

        results = {
            "mae": self.mae(confidences, ground_truth_ratings),
            "rmse": self.rmse(confidences, ground_truth_ratings),
        }

        # Perplexity (선택적)
        if calculate_perplexity and model is not None and tokenizer is not None:
            rationales = [p.get("rationale", "") for p in predictions]
            results["perplexity"] = self.perplexity(model, tokenizer, rationales)

        return results

    @staticmethod
    def confidence_distribution(
        predictions: List[Dict]
    ) -> Dict[str, float]:
        """
        Confidence Score 분포 통계

        Returns:
            mean, std, min, max, median
        """
        confidences = [
            float(p.get("confidence_score", 5.0))
            for p in predictions
        ]

        if not confidences:
            return {}

        return {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "median": np.median(confidences),
        }

    @staticmethod
    def rationale_length_distribution(
        predictions: List[Dict]
    ) -> Dict[str, float]:
        """
        Rationale 길이 분포 통계

        Returns:
            mean, std, min, max (문자 수 기준)
        """
        lengths = [
            len(p.get("rationale", ""))
            for p in predictions
        ]

        if not lengths:
            return {}

        return {
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "empty_count": sum(1 for l in lengths if l == 0),
        }


class GPTRationaleEvaluator:
    """
    GPT-4.1 API를 통한 Rationale 품질 평가 (RQ4)

    User Type별 균등 추출(Stratified Sampling)로 모델당 50개 샘플을 평가합니다.
    - 10개 User Type × 5개/Type = 50개/모델
    - 비용 효율적이면서 User Type별 균형 잡힌 평가

    ⚠️ RQ4는 KitREC 모델만 평가 대상 (Baseline 제외)

    평가 기준:
    1. 논리성 (logic): 추천 이유가 논리적인가?
    2. 구체성 (specificity): 구체적인 근거를 제시하는가?
    3. Cross-domain 연결성 (cross_domain): Source→Target 연결이 명확한가?
    4. 사용자 선호 반영 (preference): 히스토리를 잘 반영했는가?
    """

    EVALUATION_PROMPT = '''You are an expert evaluator for recommendation system explanations.

Evaluate the following recommendation rationale on a 1-10 scale for each criterion.

## User's Source Domain History:
{source_history}

## Recommended Item:
{recommended_item}

## Recommendation Rationale:
{rationale}

## Evaluation Criteria:
1. **Logic (1-10)**: Is the reasoning logically sound and coherent?
2. **Specificity (1-10)**: Does the explanation provide specific, concrete evidence?
3. **Cross-domain Connection (1-10)**: Is the source→target domain connection clearly explained?
4. **Preference Alignment (1-10)**: Does the explanation reflect the user's historical preferences?

## Response Format:
Respond ONLY with a JSON object (no markdown, no explanation):
{{"logic": <1-10>, "specificity": <1-10>, "cross_domain": <1-10>, "preference": <1-10>, "overall": <1-10>}}'''

    def __init__(
        self,
        api_key: Optional[str] = None,
        samples_per_type: int = 5,
        model: str = "gpt-4.1",
        random_seed: int = 42
    ):
        """
        Args:
            api_key: OpenAI API key (can also use OPENAI_API_KEY env var)
            samples_per_type: Number of samples per User Type for stratified sampling
                              (default 5, total 50 samples for 10 User Types)
            model: GPT model to use
            random_seed: Random seed for reproducibility
        """
        self.samples_per_type = samples_per_type
        self.model = model
        self.random_seed = random_seed
        self.client = None

        # Try to initialize OpenAI client
        try:
            from openai import OpenAI
            import os
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            pass  # Will handle in evaluate methods

    def is_available(self) -> bool:
        """Check if GPT evaluation is available (API key set)"""
        return self.client is not None

    def evaluate_sample(self, sample: Dict) -> Optional[Dict[str, float]]:
        """
        단일 샘플 평가

        Args:
            sample: {
                "source_history": str,
                "recommended_item": str,
                "rationale": str
            }

        Returns:
            {"logic": float, "specificity": float, "cross_domain": float,
             "preference": float, "overall": float} or None on error
        """
        if not self.client:
            return None

        prompt = self.EVALUATION_PROMPT.format(
            source_history=sample.get("source_history", "N/A"),
            recommended_item=sample.get("recommended_item", "N/A"),
            rationale=sample.get("rationale", "N/A")
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle potential markdown wrapping
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)

            # Validate and clamp scores to 1-10
            for key in ["logic", "specificity", "cross_domain", "preference", "overall"]:
                if key in result:
                    result[key] = max(1, min(10, float(result[key])))
                else:
                    result[key] = 5.0  # Default if missing

            return result

        except Exception as e:
            print(f"GPT evaluation error: {e}")
            return None

    def _stratified_sample(
        self,
        results: List[Dict],
        samples_per_type: int
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        User Type별 균등 추출 (Stratified Sampling)

        Args:
            results: 전체 결과 리스트
            samples_per_type: 각 User Type당 샘플 수

        Returns:
            (sampled_results, sampling_stats)
        """
        from collections import defaultdict

        # User Type별 그룹화
        grouped = defaultdict(list)
        for r in results:
            user_type = r.get("metadata", {}).get("user_type", "unknown")
            grouped[user_type].append(r)

        # 각 User Type에서 samples_per_type개 랜덤 추출
        sampled = []
        sampling_stats = {}

        for user_type, samples in grouped.items():
            k = min(samples_per_type, len(samples))
            selected = random.sample(samples, k)
            sampled.extend(selected)
            sampling_stats[user_type] = {
                "total": len(samples),
                "sampled": k
            }

        return sampled, sampling_stats

    def evaluate_batch(
        self,
        results: List[Dict],
        random_seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        User Type별 균등 추출 (Stratified Sampling) 후 GPT-4.1 평가

        - 10개 User Type × 5개/Type = 50개/모델 (기본값)
        - 비용 효율적이면서 User Type별 균형 잡힌 평가

        Args:
            results: 평가할 결과 리스트 (각각 source_history, recommended_item, rationale, metadata 포함)
            random_seed: Random seed (default: self.random_seed)

        Returns:
            {
                "logic": mean_score,
                "specificity": mean_score,
                "cross_domain": mean_score,
                "preference": mean_score,
                "overall": mean_score,
                "n_evaluated": int,
                "n_total": int,
                "errors": int,
                "sampling_stats": {user_type: {total, sampled}}
            }
        """
        if not self.client:
            return {
                "error": "OpenAI client not available. Set OPENAI_API_KEY or pass api_key.",
                "n_evaluated": 0
            }

        seed = random_seed if random_seed is not None else self.random_seed
        random.seed(seed)

        # Stratified sampling by User Type
        sampled, sampling_stats = self._stratified_sample(results, self.samples_per_type)

        scores = {
            "logic": [],
            "specificity": [],
            "cross_domain": [],
            "preference": [],
            "overall": []
        }
        errors = 0

        for sample in tqdm(sampled, desc="GPT-4.1 Rationale Evaluation"):
            eval_result = self.evaluate_sample(sample)
            if eval_result:
                for key in scores:
                    if key in eval_result:
                        scores[key].append(eval_result[key])
            else:
                errors += 1

        # Aggregate results
        aggregated = {
            "n_evaluated": len(sampled) - errors,
            "n_total": len(results),
            "samples_per_type": self.samples_per_type,
            "sampling_stats": sampling_stats,
            "errors": errors
        }

        for key, values in scores.items():
            if values:
                aggregated[key] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
            else:
                aggregated[key] = 0.0
                aggregated[f"{key}_std"] = 0.0

        return aggregated

    def prepare_sample_from_kitrec(
        self,
        input_prompt: str,
        predictions: List[Dict],
        top_k: int = 1
    ) -> List[Dict]:
        """
        KitREC 출력에서 평가용 샘플 준비

        Args:
            input_prompt: KitREC 입력 프롬프트 (user history 포함)
            predictions: 예측 결과 리스트
            top_k: 상위 몇 개 예측을 평가할지

        Returns:
            GPT 평가용 샘플 리스트
        """
        import re

        # Extract source history from prompt
        source_pattern = r"### User's \w+ History \(Source Domain\):\s*(.*?)(?=###|## List)"
        source_match = re.search(source_pattern, input_prompt, re.DOTALL)
        source_history = source_match.group(1).strip() if source_match else "N/A"

        samples = []
        for pred in predictions[:top_k]:
            item_id = pred.get("item_id", "Unknown")
            title = pred.get("title", "")
            rationale = pred.get("rationale", "")

            recommended_item = f"{title} (ID: {item_id})" if title else f"ID: {item_id}"

            samples.append({
                "source_history": source_history,
                "recommended_item": recommended_item,
                "rationale": rationale
            })

        return samples
