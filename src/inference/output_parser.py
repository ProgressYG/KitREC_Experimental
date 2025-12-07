"""
모델 출력 파싱

CLAUDE.md Critical Notes #3: Output Parsing 주의사항 적용
- <think>...</think> 블록 분리
- JSON 블록 추출 (개선된 regex)
- trailing comma 제거
- item_id 검증: candidate_ids에 없으면 fail
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ParseResult:
    """파싱 결과 구조체"""
    thinking: Optional[str]
    predictions: List[Dict]
    parse_errors: List[str] = field(default_factory=list)
    invalid_items: List[str] = field(default_factory=list)  # 후보군 외 item_id
    raw_output: str = ""
    json_extracted: bool = False


class OutputParser:
    """모델 출력 파싱 클래스"""

    def parse(
        self,
        raw_output: str,
        candidate_ids: List[str],
        validate_items: bool = True
    ) -> ParseResult:
        """
        모델 출력 파싱

        1. <think>...</think> 블록 분리
        2. JSON 블록 추출 (```json ... ```)
        3. trailing comma 제거
        4. item_id 검증: candidate_ids에 없으면 fail (rank=∞)
        5. 오류율 통계 반환

        Args:
            raw_output: 모델 원본 출력
            candidate_ids: 유효한 candidate item_id 리스트
            validate_items: item_id 검증 여부

        Returns:
            ParseResult 객체
        """
        errors = []
        invalid_items = []

        # 1. Thinking 블록 분리
        thinking = self._extract_thinking(raw_output)

        # 2. JSON 블록 추출
        json_str = self._extract_json(raw_output)
        if not json_str:
            errors.append("JSON block not found in output")
            return ParseResult(
                thinking=thinking,
                predictions=[],
                parse_errors=errors,
                invalid_items=[],
                raw_output=raw_output,
                json_extracted=False
            )

        # 3. Trailing comma 제거
        json_str = self._remove_trailing_comma(json_str)

        # 4. JSON 파싱
        try:
            predictions = json.loads(json_str)

            # 단일 객체인 경우 리스트로 변환
            if isinstance(predictions, dict):
                predictions = [predictions]

        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {str(e)}")
            return ParseResult(
                thinking=thinking,
                predictions=[],
                parse_errors=errors,
                invalid_items=[],
                raw_output=raw_output,
                json_extracted=False
            )

        # 5. item_id 검증 (case-insensitive comparison for robustness)
        if validate_items:
            valid_predictions = []
            # Create uppercase lookup set for case-insensitive matching
            candidate_ids_upper = {cid.upper(): cid for cid in candidate_ids}
            for pred in predictions:
                item_id = pred.get("item_id", "")
                # Check with case-insensitive matching
                item_id_upper = item_id.upper() if item_id else ""
                if item_id_upper in candidate_ids_upper:
                    # Normalize item_id to the canonical form from candidate set
                    pred["item_id"] = candidate_ids_upper[item_id_upper]
                    valid_predictions.append(pred)
                elif item_id in candidate_ids:
                    # Direct match (original behavior)
                    valid_predictions.append(pred)
                else:
                    invalid_items.append(item_id)
                    errors.append(f"Invalid item_id: {item_id} (not in candidate set)")
            predictions = valid_predictions

        return ParseResult(
            thinking=thinking,
            predictions=predictions,
            parse_errors=errors,
            invalid_items=invalid_items,
            raw_output=raw_output,
            json_extracted=True
        )

    def _extract_thinking(self, output: str) -> Optional[str]:
        """<think>...</think> 블록 추출"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, output, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_json(self, output: str) -> Optional[str]:
        """
        JSON 블록 추출 (개선된 버전)

        detail_task_plan.md 수정 사항:
        - 다중 라인 JSON 처리 위해 [\\s\\S] 사용
        - 배열 및 객체 형식 모두 지원
        """
        # Priority 1: ```json 코드 블록 (배열 또는 객체)
        pattern = r'```json\s*([\[\{][\s\S]*?[\]\}])\s*```'
        match = re.search(pattern, output)
        if match:
            return match.group(1).strip()

        # Priority 2: ``` 코드 블록 (json 태그 없음)
        pattern = r'```\s*([\[\{][\s\S]*?[\]\}])\s*```'
        match = re.search(pattern, output)
        if match:
            return match.group(1).strip()

        # Priority 3: 코드 블록 없이 JSON 배열 직접 찾기
        # Improved pattern to avoid greedy matching across multiple JSON blocks
        pattern = r'\[(?:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*,?\s*)+\]'
        match = re.search(pattern, output)
        if match:
            return match.group(0)

        # Priority 3.5: Fallback - simpler pattern for basic JSON arrays
        pattern = r'\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]'
        match = re.search(pattern, output)
        if match:
            return match.group(0)

        # Priority 4: 단일 JSON 객체
        pattern = r'\{[\s\S]*?"item_id"[\s\S]*?\}'
        match = re.search(pattern, output)
        return match.group(0) if match else None

    def _remove_trailing_comma(self, json_str: str) -> str:
        """trailing comma 제거"""
        # },] 또는 }, ] 패턴 처리
        json_str = re.sub(r',\s*\]', ']', json_str)
        json_str = re.sub(r',\s*\}', '}', json_str)
        return json_str

    def extract_confidence_scores(
        self,
        predictions: List[Dict]
    ) -> List[float]:
        """Confidence Score 추출"""
        return [
            float(pred.get("confidence_score", 5.0))
            for pred in predictions
        ]

    def extract_rationales(self, predictions: List[Dict]) -> List[str]:
        """Rationale 추출"""
        return [
            pred.get("rationale", "")
            for pred in predictions
        ]

    def get_top_k_predictions(
        self,
        predictions: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """Top-K 예측 반환 (rank 기준 정렬)"""
        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get("rank", float('inf'))
        )
        return sorted_preds[:k]


class ErrorStatistics:
    """파싱 오류 통계 클래스"""

    def __init__(self):
        self.total_samples = 0
        self.parse_failures = 0
        self.json_extraction_failures = 0
        self.invalid_item_count = 0
        self.invalid_item_ids = []
        self.error_messages = []

    def update(self, result: ParseResult):
        """통계 업데이트"""
        self.total_samples += 1

        if not result.json_extracted:
            self.json_extraction_failures += 1

        if result.parse_errors:
            self.parse_failures += 1
            self.error_messages.extend(result.parse_errors)

        self.invalid_item_count += len(result.invalid_items)
        self.invalid_item_ids.extend(result.invalid_items)

    def get_summary(self) -> Dict:
        """통계 요약 반환"""
        total = max(self.total_samples, 1)  # Avoid division by zero

        return {
            "total_samples": self.total_samples,
            "parse_failure_count": self.parse_failures,
            "parse_failure_rate": self.parse_failures / total,
            "json_extraction_failure_count": self.json_extraction_failures,
            "json_extraction_failure_rate": self.json_extraction_failures / total,
            "invalid_item_count": self.invalid_item_count,
            "invalid_item_rate": self.invalid_item_count / total,
            "unique_invalid_items": len(set(self.invalid_item_ids)),
            "top_invalid_items": self._get_top_invalid_items(10),
        }

    def _get_top_invalid_items(self, n: int) -> List[Tuple[str, int]]:
        """가장 많이 발생한 invalid item_id"""
        from collections import Counter
        counter = Counter(self.invalid_item_ids)
        return counter.most_common(n)

    def get_error_samples(self, limit: int = 100) -> List[str]:
        """에러 메시지 샘플 반환"""
        return self.error_messages[:limit]

    def is_acceptable(self, max_error_rate: float = 0.05) -> bool:
        """
        에러율이 허용 범위 내인지 확인

        eval_config.yaml: max_error_rate: 0.05 (5%)
        """
        return (self.parse_failures / max(self.total_samples, 1)) <= max_error_rate

    def reset(self):
        """통계 초기화"""
        self.__init__()
