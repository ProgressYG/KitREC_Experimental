"""
CoNet Data Converter

Convert KitREC text data to ID matrix format for CoNet
CLAUDE.md Critical Notes #4: 동일 User History 시퀀스 사용 필수
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CoNetSample:
    """CoNet 학습/평가 샘플"""
    user_id: int
    source_item_ids: List[int]  # Source domain history
    target_item_ids: List[int]  # Target domain history
    candidate_item_ids: List[int]  # Candidate items (100)
    ground_truth_idx: int  # GT item index in candidates
    ground_truth_id: int  # GT item ID


class CoNetDataConverter:
    """KitREC 데이터를 CoNet 형식으로 변환"""

    def __init__(self):
        self.user_vocab: Dict[str, int] = {}
        self.source_item_vocab: Dict[str, int] = {}
        self.target_item_vocab: Dict[str, int] = {}

        self._user_counter = 1  # 0 is reserved for unknown
        self._source_item_counter = 1
        self._target_item_counter = 1

    def build_vocabulary(self, samples: List[Dict]):
        """
        어휘 사전 구축

        Args:
            samples: KitREC 샘플 리스트
        """
        for sample in samples:
            # User ID
            user_id = sample.get("user_id", "")
            if user_id and user_id not in self.user_vocab:
                self.user_vocab[user_id] = self._user_counter
                self._user_counter += 1

            # Source domain items (from prompt)
            prompt = sample.get("input") or sample.get("instruction", "")
            source_items = self._extract_source_items(prompt)
            for item_id in source_items:
                if item_id not in self.source_item_vocab:
                    self.source_item_vocab[item_id] = self._source_item_counter
                    self._source_item_counter += 1

            # Target domain items (candidates + history)
            target_items = self._extract_target_items(prompt)
            for item_id in target_items:
                if item_id not in self.target_item_vocab:
                    self.target_item_vocab[item_id] = self._target_item_counter
                    self._target_item_counter += 1

        print(f"Vocabulary built:")
        print(f"  Users: {len(self.user_vocab)}")
        print(f"  Source items: {len(self.source_item_vocab)}")
        print(f"  Target items: {len(self.target_item_vocab)}")

    def convert_sample(self, sample: Dict) -> Optional[CoNetSample]:
        """
        단일 샘플 변환

        CLAUDE.md: KitREC에 들어가는 History와 동일한 시점의 데이터 사용 필수

        Args:
            sample: KitREC 샘플

        Returns:
            CoNetSample or None if conversion fails
        """
        try:
            # User ID
            user_str = sample.get("user_id", "")
            user_id = self.user_vocab.get(user_str, 0)

            # Prompt
            prompt = sample.get("input") or sample.get("instruction", "")

            # Source domain history
            source_items = self._extract_source_items(prompt)
            source_item_ids = [
                self.source_item_vocab.get(item_id, 0)
                for item_id in source_items
            ]

            # Target domain history
            target_history = self._extract_target_history(prompt)
            target_item_ids = [
                self.target_item_vocab.get(item_id, 0)
                for item_id in target_history
            ]

            # Candidate items (동일 candidate set 사용 필수)
            candidate_items = self._extract_candidates(prompt)
            candidate_item_ids = [
                self.target_item_vocab.get(item_id, 0)
                for item_id in candidate_items
            ]

            # Ground truth
            gt = sample.get("ground_truth", {})
            gt_item_str = gt.get("item_id") or sample.get("gt_item_id", "")
            gt_item_id = self.target_item_vocab.get(gt_item_str, 0)

            # GT index in candidates
            gt_idx = -1
            if gt_item_str in candidate_items:
                gt_idx = candidate_items.index(gt_item_str)

            return CoNetSample(
                user_id=user_id,
                source_item_ids=source_item_ids,
                target_item_ids=target_item_ids,
                candidate_item_ids=candidate_item_ids,
                ground_truth_idx=gt_idx,
                ground_truth_id=gt_item_id
            )

        except Exception as e:
            print(f"Error converting sample: {e}")
            return None

    def convert_dataset(self, samples: List[Dict]) -> List[CoNetSample]:
        """전체 데이터셋 변환"""
        converted = []
        for sample in samples:
            result = self.convert_sample(sample)
            if result is not None:
                converted.append(result)

        print(f"Converted {len(converted)}/{len(samples)} samples")
        return converted

    def _extract_source_items(self, prompt: str) -> List[str]:
        """Source domain history에서 item ID 추출"""
        # Pattern: User's Books History (Source Domain)
        pattern = r"### User's \w+ History \(Source Domain\):\s*(.*?)(?=###|## List)"
        match = re.search(pattern, prompt, re.DOTALL)

        if match:
            history_text = match.group(1)
            return re.findall(r'\(ID:\s*([A-Z0-9]+)\)', history_text)

        return []

    def _extract_target_history(self, prompt: str) -> List[str]:
        """Target domain history에서 item ID 추출"""
        # Pattern: User's Movies/Music History (Target Domain)
        pattern = r"### User's \w+ History \(Target Domain\):\s*(.*?)(?=## List)"
        match = re.search(pattern, prompt, re.DOTALL)

        if match:
            history_text = match.group(1)
            return re.findall(r'\(ID:\s*([A-Z0-9]+)\)', history_text)

        return []

    def _extract_target_items(self, prompt: str) -> List[str]:
        """Target domain의 모든 item ID 추출 (history + candidates)"""
        # Candidate list
        pattern = r'## List of Available Candidate Items.*?\n(.*?)(?=## (?:Reasoning|Output))'
        match = re.search(pattern, prompt, re.DOTALL)

        items = []
        if match:
            candidate_text = match.group(1)
            items.extend(re.findall(r'\(ID:\s*([A-Z0-9]+)\)', candidate_text))

        # Target history
        items.extend(self._extract_target_history(prompt))

        return items

    def _extract_candidates(self, prompt: str) -> List[str]:
        """Candidate list에서 item ID 추출"""
        pattern = r'\(ID:\s*([A-Z0-9]+)\)'

        # Find candidate section
        candidate_section = re.search(
            r'## List of Available Candidate Items.*?(?=## (?:Reasoning|Output))',
            prompt, re.DOTALL
        )

        if candidate_section:
            return re.findall(pattern, candidate_section.group(0))

        return []

    def save_vocabulary(self, path: str):
        """어휘 사전 저장"""
        vocab = {
            "user_vocab": self.user_vocab,
            "source_item_vocab": self.source_item_vocab,
            "target_item_vocab": self.target_item_vocab,
        }
        with open(path, "w") as f:
            json.dump(vocab, f, indent=2)

    def load_vocabulary(self, path: str):
        """어휘 사전 로드"""
        with open(path, "r") as f:
            vocab = json.load(f)

        self.user_vocab = vocab["user_vocab"]
        self.source_item_vocab = vocab["source_item_vocab"]
        self.target_item_vocab = vocab["target_item_vocab"]

    def get_vocab_sizes(self) -> Dict[str, int]:
        """어휘 크기 반환"""
        return {
            "num_users": len(self.user_vocab) + 1,
            "num_source_items": len(self.source_item_vocab) + 1,
            "num_target_items": len(self.target_item_vocab) + 1,
        }

    def get_reverse_vocab(self, vocab_type: str = "target_item") -> Dict[int, str]:
        """역방향 어휘 사전 (ID → string)"""
        if vocab_type == "user":
            vocab = self.user_vocab
        elif vocab_type == "source_item":
            vocab = self.source_item_vocab
        else:
            vocab = self.target_item_vocab

        return {idx: item_id for item_id, idx in vocab.items()}
