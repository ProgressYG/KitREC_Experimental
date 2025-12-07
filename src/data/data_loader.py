"""
HuggingFace Hub에서 Test Set 로딩

CLAUDE.md Critical Notes #1: Template Schema Difference 적용
- Val/Test: `input` 필드에 전체 프롬프트
- Training: `instruction` 필드에 전체 프롬프트
"""

import re
import json
from typing import Dict, List, Optional, Any, Union
from datasets import load_dataset
from dataclasses import dataclass


# Global random seed for reproducibility
RANDOM_SEED = 42


@dataclass
class GroundTruth:
    """Ground Truth 정보 구조체"""
    item_id: str
    title: str
    rating: float
    category: Optional[str] = None


@dataclass
class TestSample:
    """테스트 샘플 구조체"""
    user_id: str
    prompt: str
    ground_truth: GroundTruth
    candidate_ids: List[str]
    user_type: str
    target_domain: str
    source_domain: str
    metadata: Dict[str, Any]


class DataLoader:
    """HuggingFace Hub에서 Test 데이터 로딩"""

    def __init__(self, dataset_name: str, hf_token: Optional[str] = None):
        """
        Args:
            dataset_name: HuggingFace Hub dataset name (e.g., "Younggooo/kitrec-test-seta")
            hf_token: HuggingFace token for private datasets
        """
        self.dataset_name = dataset_name
        self.hf_token = hf_token
        self._dataset = None

    def load_test_data(self, split: str = "test") -> Any:
        """
        Test 데이터 로딩

        Args:
            split: Dataset split (test data is usually in "train" split)

        Returns:
            HuggingFace Dataset object
        """
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name,
                token=self.hf_token,
                split=split
            )
        return self._dataset

    def extract_prompt(self, sample: Dict) -> str:
        """
        프롬프트 추출

        CLAUDE.md Template Schema Difference:
        - Val/Test: `input` 필드에 전체 프롬프트
        - Training: `instruction` 필드에 전체 프롬프트
        """
        # Priority: input > instruction
        if sample.get("input") and sample["input"].strip():
            return sample["input"]
        elif sample.get("instruction") and sample["instruction"].strip():
            return sample["instruction"]
        else:
            raise ValueError(f"No valid prompt field found in sample: {sample.get('user_id', 'unknown')}")

    def extract_ground_truth(self, sample: Dict) -> GroundTruth:
        """
        Ground Truth 아이템 정보 추출 (인스턴스 메서드)
        """
        return DataLoader._extract_ground_truth_static(sample)

    @staticmethod
    def _extract_ground_truth_static(sample: Dict) -> GroundTruth:
        """
        Ground Truth 아이템 정보 추출 (정적 메서드)

        Expected fields:
        - ground_truth: dict or JSON string with item_id, title, rating
        - Or: gt_item_id, gt_title, gt_rating as separate fields
        """
        gt_data = sample.get("ground_truth")

        # Case 1: ground_truth as dict
        if gt_data and isinstance(gt_data, dict):
            return GroundTruth(
                item_id=gt_data.get("item_id", ""),
                title=gt_data.get("title", ""),
                rating=float(gt_data.get("rating", 0.0)),
                category=gt_data.get("category")
            )

        # Case 2: ground_truth as JSON string
        if gt_data and isinstance(gt_data, str):
            try:
                gt_parsed = json.loads(gt_data)
                return GroundTruth(
                    item_id=gt_parsed.get("item_id", ""),
                    title=gt_parsed.get("title", ""),
                    rating=float(gt_parsed.get("rating", 0.0)),
                    category=gt_parsed.get("category")
                )
            except json.JSONDecodeError:
                pass  # Fall through to Case 3

        # Case 3: Separate fields
        return GroundTruth(
            item_id=sample.get("gt_item_id", ""),
            title=sample.get("gt_title", ""),
            rating=float(sample.get("gt_rating", sample.get("confidence_score", 0.0))),
            category=sample.get("gt_category")
        )

    def extract_candidate_ids(self, sample: Dict) -> List[str]:
        """
        Candidate Set의 item_id 리스트 추출 (인스턴스 메서드)
        """
        return DataLoader._extract_candidate_ids_static(sample)

    @staticmethod
    def _extract_candidate_ids_static(sample: Dict) -> List[str]:
        """
        Candidate Set의 item_id 리스트 추출 (정적 메서드)

        Pattern: (ID: xxx) 형식
        """
        prompt = DataLoader._extract_prompt_static(sample)
        pattern = r'\(ID:\s*([A-Z0-9]+)\)'
        return re.findall(pattern, prompt)

    @staticmethod
    def _extract_prompt_static(sample: Dict) -> str:
        """프롬프트 추출 (정적 메서드)"""
        if sample.get("input") and sample["input"].strip():
            return sample["input"]
        elif sample.get("instruction") and sample["instruction"].strip():
            return sample["instruction"]
        else:
            return ""

    def extract_metadata(self, sample: Dict) -> Dict[str, Any]:
        """샘플 메타데이터 추출"""
        return {
            "user_type": sample.get("user_type", "unknown"),
            "target_domain": sample.get("target_domain", "unknown"),
            "source_domain": sample.get("source_domain", "Books"),
            "candidate_set": sample.get("candidate_set", "unknown"),
            "thinking_length": sample.get("thinking_length", 0),
        }

    def parse_sample(self, sample: Dict) -> TestSample:
        """전체 샘플 파싱"""
        return TestSample(
            user_id=sample.get("user_id", ""),
            prompt=self.extract_prompt(sample),
            ground_truth=self.extract_ground_truth(sample),
            candidate_ids=self.extract_candidate_ids(sample),
            user_type=sample.get("user_type", "unknown"),
            target_domain=sample.get("target_domain", "unknown"),
            source_domain=sample.get("source_domain", "Books"),
            metadata=self.extract_metadata(sample)
        )

    def get_samples_by_user_type(self, user_type: str) -> List[Dict]:
        """특정 User Type의 샘플만 필터링"""
        dataset = self.load_test_data()
        return [s for s in dataset if s.get("user_type") == user_type]

    def get_samples_by_domain(self, target_domain: str) -> List[Dict]:
        """특정 Target Domain의 샘플만 필터링"""
        dataset = self.load_test_data()
        return [s for s in dataset if s.get("target_domain") == target_domain]

    def validate_candidate_set(self, sample: Dict) -> bool:
        """
        Candidate Set 검증

        Expected: 100 candidates (1 GT + 99 Negatives)
        """
        candidate_ids = self.extract_candidate_ids(sample)
        gt = self.extract_ground_truth(sample)

        # Check size
        if len(candidate_ids) != 100:
            return False

        # Check GT is in candidates
        if gt.item_id not in candidate_ids:
            return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 반환"""
        dataset = self.load_test_data()

        user_type_counts = {}
        domain_counts = {}

        for sample in dataset:
            # User type distribution
            ut = sample.get("user_type", "unknown")
            user_type_counts[ut] = user_type_counts.get(ut, 0) + 1

            # Domain distribution
            td = sample.get("target_domain", "unknown")
            domain_counts[td] = domain_counts.get(td, 0) + 1

        return {
            "total_samples": len(dataset),
            "user_type_distribution": user_type_counts,
            "domain_distribution": domain_counts,
        }
