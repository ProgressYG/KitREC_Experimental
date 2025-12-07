"""
Candidate Set 처리 및 검증

CLAUDE.md Baseline 공정성 조건:
- 모든 모델은 동일한 Candidate Set (1 GT + 99 Neg) 사용 필수
- Baseline 모델도 KitREC과 동일한 후보군 사용
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CandidateValidationResult:
    """Candidate 검증 결과"""
    is_valid: bool
    predicted_id: str
    is_in_candidate_set: bool
    candidate_count: int
    error_message: Optional[str] = None


class CandidateHandler:
    """Candidate Set 처리 및 검증 클래스"""

    EXPECTED_CANDIDATE_SIZE = 100  # 1 GT + 99 Negatives

    def extract_candidate_ids(self, prompt: str) -> List[str]:
        """
        프롬프트에서 Candidate item_id 추출

        Pattern: (ID: xxx) 형식
        """
        pattern = r'\(ID:\s*([A-Z0-9]+)\)'
        return re.findall(pattern, prompt)

    def validate_prediction(
        self,
        predicted_id: str,
        candidate_ids: List[str]
    ) -> CandidateValidationResult:
        """
        예측된 item_id가 Candidate Set에 있는지 검증

        CLAUDE.md Critical Notes #3:
        후보군 외 item_id 출력 시 → 자동 fail 처리 (rank = ∞)
        """
        is_valid = predicted_id in candidate_ids

        return CandidateValidationResult(
            is_valid=is_valid,
            predicted_id=predicted_id,
            is_in_candidate_set=is_valid,
            candidate_count=len(candidate_ids),
            error_message=None if is_valid else f"Item {predicted_id} not in candidate set"
        )

    def validate_predictions_batch(
        self,
        predictions: List[Dict],
        candidate_ids: List[str]
    ) -> Tuple[List[Dict], List[str]]:
        """
        배치 예측 검증

        Returns:
            valid_predictions: 유효한 예측 리스트
            invalid_ids: 무효한 item_id 리스트
        """
        valid_predictions = []
        invalid_ids = []

        for pred in predictions:
            item_id = pred.get("item_id", "")
            if item_id in candidate_ids:
                valid_predictions.append(pred)
            else:
                invalid_ids.append(item_id)

        return valid_predictions, invalid_ids

    def validate_candidate_set_size(self, candidate_ids: List[str]) -> bool:
        """Candidate Set 크기 검증 (100개 필수)"""
        return len(candidate_ids) == self.EXPECTED_CANDIDATE_SIZE

    def validate_gt_in_candidates(
        self,
        ground_truth_id: str,
        candidate_ids: List[str]
    ) -> bool:
        """Ground Truth가 Candidate Set에 포함되어 있는지 검증"""
        return ground_truth_id in candidate_ids

    def convert_to_id_matrix(
        self,
        user_history: List[str],
        item_vocab: Dict[str, int]
    ) -> np.ndarray:
        """
        Baseline 모델용: 텍스트 History → ID matrix 변환

        CLAUDE.md Critical Notes #4:
        - KitREC에 들어가는 History와 동일한 시점의 데이터 사용 필수
        - 동일 User History 시퀀스 보장

        Args:
            user_history: item_id 리스트
            item_vocab: item_id → integer index 매핑

        Returns:
            ID matrix (numpy array)
        """
        return np.array([
            item_vocab.get(item_id, 0)  # Unknown은 0
            for item_id in user_history
        ])

    def convert_candidates_to_ids(
        self,
        candidate_ids: List[str],
        item_vocab: Dict[str, int]
    ) -> np.ndarray:
        """
        Baseline 모델용: Candidate Set → ID array 변환

        CLAUDE.md: 반드시 KitREC과 동일한 100개 후보 (1 GT + 99 Neg) 사용
        """
        return np.array([
            item_vocab.get(cid, 0)
            for cid in candidate_ids
        ])

    def extract_history_from_prompt(self, prompt: str) -> Dict[str, List[str]]:
        """
        프롬프트에서 User History item_id 추출

        Returns:
            {"source": [source_item_ids], "target": [target_item_ids]}
        """
        result = {"source": [], "target": []}

        # Source Domain History 추출
        source_pattern = r"### User's \w+ History \(Source Domain\):\s*(.*?)(?=###|## List)"
        source_match = re.search(source_pattern, prompt, re.DOTALL)
        if source_match:
            source_text = source_match.group(1)
            result["source"] = re.findall(r'\(ID:\s*([A-Z0-9]+)\)', source_text)

        # Target Domain History 추출
        target_pattern = r"### User's \w+ History \(Target Domain\):\s*(.*?)(?=## List)"
        target_match = re.search(target_pattern, prompt, re.DOTALL)
        if target_match:
            target_text = target_match.group(1)
            result["target"] = re.findall(r'\(ID:\s*([A-Z0-9]+)\)', target_text)

        return result

    def compare_candidate_sets(
        self,
        set1: List[str],
        set2: List[str]
    ) -> Dict[str, any]:
        """
        두 Candidate Set 비교 (Baseline 동기화 검증용)

        Returns:
            {
                "identical": bool,
                "set1_size": int,
                "set2_size": int,
                "common_items": int,
                "only_in_set1": list,
                "only_in_set2": list
            }
        """
        s1 = set(set1)
        s2 = set(set2)

        return {
            "identical": s1 == s2,
            "set1_size": len(set1),
            "set2_size": len(set2),
            "common_items": len(s1 & s2),
            "only_in_set1": list(s1 - s2),
            "only_in_set2": list(s2 - s1)
        }

    def build_item_vocab(
        self,
        all_item_ids: List[str],
        start_index: int = 1
    ) -> Dict[str, int]:
        """
        Item vocabulary 생성 (Baseline 모델용)

        Args:
            all_item_ids: 모든 item_id 리스트
            start_index: 시작 인덱스 (0은 Unknown용으로 예약)

        Returns:
            item_id → integer index 매핑
        """
        unique_ids = sorted(set(all_item_ids))
        return {
            item_id: idx + start_index
            for idx, item_id in enumerate(unique_ids)
        }

    def build_reverse_vocab(self, item_vocab: Dict[str, int]) -> Dict[int, str]:
        """역방향 vocabulary (index → item_id)"""
        return {idx: item_id for item_id, idx in item_vocab.items()}
