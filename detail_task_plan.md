# KitREC Experimental_test 상세 작업 계획서

**작성일:** 2025-12-06
**최종 수정:** 2025-12-07
**목적:** KitREC 모델 평가 및 Baseline 비교 실험을 위한 구현 명세 + 실험 로드맵
**참조 문서:** CLAUDE.md

---

## 🎯 Implementation Status Summary

### ✅ 코드 구현 완료 (2025-12-07)

| Phase | 상태 | 주요 내용 |
|-------|------|----------|
| **Phase 1: Critical** | ✅ 완료 | Confidence 1-10 범위, Train/Test 분리, User Type Mapping |
| **Phase 2: High** | ✅ 완료 | Device 수정, Gradient Clipping, Statistical Testing |
| **Phase 3: Medium** | ✅ 완료 | LLM4CDR Target History, LR Scheduler, Holm Correction |
| **Phase 4: Explain** | ✅ 완료 | GPT-4.1 Rationale Evaluation, MAE/RMSE, Perplexity |
| **Bug Fixes** | ✅ 완료 | 무한루프 방지, Path validation, Empty list 처리 |
| **Phase 6: Refinements** | ✅ 완료 | LLM4CDR BaseEvaluator, per-sample metrics, Logging |

### ✅ 추가 개선 (2025-12-07 Updated)

| Issue ID | 파일 | 수정 내용 | 목적 |
|----------|------|----------|------|
| **A-1** | `llm4cdr/evaluator.py` | BaseEvaluator 상속, Candidate 검증, Confidence 정규화 | 공통 인프라 활용 |
| **A-2** | `baselines/*/evaluator.py` | per-sample metrics 수집 | t-test 통계 검정 지원 |
| **A-3** | `base_evaluator.py` | 검증 실패 시 logging 추가 | 능동적 오류 탐지 |

---

## 📊 파일 구현 현황

### ✅ 구현 완료 파일

| 카테고리 | 파일 | 상태 |
|---------|------|------|
| **src/data/** | `data_loader.py`, `prompt_builder.py`, `candidate_handler.py` | ✅ |
| **src/inference/** | `vllm_inference.py`, `output_parser.py`, `batch_inference.py` | ✅ |
| **src/metrics/** | `ranking_metrics.py`, `explainability_metrics.py`, `stratified_analysis.py`, `statistical_analysis.py` | ✅ |
| **src/models/** | `kitrec_model.py`, `base_model.py` | ✅ |
| **src/utils/** | `logger.py`, `io_utils.py`, `visualization.py` | ✅ |
| **baselines/conet/** | `model.py`, `trainer.py`, `evaluator.py`, `data_converter.py` | ✅ |
| **baselines/dtcdr/** | `model.py`, `trainer.py`, `evaluator.py`, `data_converter.py` | ✅ |
| **baselines/llm4cdr/** | `prompts.py`, `evaluator.py` | ✅ |
| **baselines/** | `base_evaluator.py` | ✅ |
| **scripts/** | `run_kitrec_eval.py`, `run_ablation_study.py`, `run_baseline_eval.py` | ✅ |
| **scripts/** | `verify_environment.py`, `verify_env_and_data.py` | ✅ |

### ❌ 미구현 파일 (선택사항)

| 카테고리 | 파일 | 필요성 |
|---------|------|--------|
| **configs/** | `eval_config.yaml`, `model_paths.yaml`, `baseline_config.yaml` | 선택 (하드코딩 대체 가능) |
| **src/models/** | `conet_wrapper.py`, `dtcdr_wrapper.py`, `llm4cdr_wrapper.py` | 선택 (직접 호출 가능) |
| **scripts/** | `run_stratified_analysis.py`, `run_metadata_subgroup.py`, `generate_report.py` | 선택 (수동 분석 가능) |

---

## 🧪 실험 실행 현황

| 실험 Phase | 상태 | 설명 |
|------------|------|------|
| **Phase 1: 환경 설정** | ⏳ 대기 | vLLM, GPU, HF Token 설정 필요 |
| **Phase 2: KitREC 평가** | ❌ 미실행 | 8개 모델 × 30,000 샘플 |
| **Phase 3: RQ1 Ablation** | ❌ 미실행 | 2×2 교차 검증 |
| **Phase 4: Baseline 평가** | ❌ 미실행 | CoNet, DTCDR, LLM4CDR 학습/평가 |
| **Phase 5: Stratified 분석** | ❌ 미실행 | User Type별, Metadata별 |
| **Phase 6: 리포트 생성** | ❌ 미실행 | 최종 논문용 테이블/그래프 |

---

### 구현된 베이스라인

| Baseline | 상태 | 파일 위치 |
|----------|------|----------|
| **CoNet** | ✅ 구현 완료 | `baselines/conet/` |
| **DTCDR** | ✅ 구현 완료 | `baselines/dtcdr/` |
| **LLM4CDR** | ✅ 구현 완료 | `baselines/llm4cdr/` |
| **BaseEvaluator** | ✅ 공통 클래스 | `baselines/base_evaluator.py` |

### 구현된 메트릭

| 메트릭 | 상태 | 파일 위치 |
|--------|------|----------|
| Ranking Metrics | ✅ | `src/metrics/ranking_metrics.py` |
| Explainability Metrics | ✅ | `src/metrics/explainability_metrics.py` |
| Statistical Analysis | ✅ | `src/metrics/statistical_analysis.py` |
| GPT-4.1 Evaluation | ✅ | `src/metrics/explainability_metrics.py` | User Type별 균등 추출, 모델당 50개 |

> ⚠️ **RQ4 평가 대상**: KitREC 모델만 (Baseline 제외)
> - Baseline(CoNet, DTCDR, LLM4CDR)은 Confidence Score/MAE/RMSE 계산 불필요

---

## 1. 프로젝트 폴더 구조

```
Experimental_test/
│
├── CLAUDE.md                          # Claude Code 가이드 문서
├── detail_task_plan.md                # 본 작업 계획서
│
├── configs/                           # 설정 파일
│   ├── eval_config.yaml               # 평가 공통 설정 (metrics, batch_size 등)
│   ├── model_paths.yaml               # HuggingFace 모델 경로 매핑
│   └── baseline_config.yaml           # Baseline 모델별 설정
│
├── src/                               # 소스 코드
│   ├── __init__.py
│   ├── data/                          # 데이터 로딩 및 전처리
│   │   ├── __init__.py
│   │   ├── data_loader.py             # HuggingFace Hub 데이터 로딩
│   │   ├── prompt_builder.py          # Inference 프롬프트 생성
│   │   └── candidate_handler.py       # Candidate Set 처리 (1 GT + 99 Neg)
│   │
│   ├── models/                        # 모델 로딩 및 추론
│   │   ├── __init__.py
│   │   ├── kitrec_model.py            # KitREC (Full/Direct) 모델 로더
│   │   ├── base_model.py              # Base-CoT/Base-Direct 모델 로더
│   │   ├── conet_wrapper.py           # CoNet 베이스라인 래퍼
│   │   ├── dtcdr_wrapper.py           # DTCDR 베이스라인 래퍼
│   │   └── llm4cdr_wrapper.py         # LLM4CDR 베이스라인 래퍼
│   │
│   ├── inference/                     # 추론 엔진
│   │   ├── __init__.py
│   │   ├── vllm_inference.py          # vLLM 기반 LLM 추론
│   │   ├── batch_inference.py         # 배치 추론 관리
│   │   └── output_parser.py           # 모델 출력 파싱 (<think>, JSON)
│   │
│   ├── metrics/                       # 평가 지표 계산
│   │   ├── __init__.py
│   │   ├── ranking_metrics.py         # Hit@K, MRR, NDCG@K
│   │   ├── explainability_metrics.py  # MAE, RMSE, Perplexity
│   │   └── stratified_analysis.py     # User Type별, Metadata별 분석
│   │
│   └── utils/                         # 유틸리티
│       ├── __init__.py
│       ├── logger.py                  # 로깅 설정
│       ├── io_utils.py                # 파일 I/O
│       └── visualization.py           # 결과 시각화
│
├── scripts/                           # 실행 스크립트
│   ├── run_kitrec_eval.py             # KitREC 모델 평가 실행
│   ├── run_ablation_study.py          # RQ1: 2×2 Ablation Study
│   ├── run_baseline_eval.py           # Baseline 모델 평가 실행
│   ├── run_stratified_analysis.py     # User Type별 분석
│   ├── run_metadata_subgroup.py       # Movies Metadata 분리 평가
│   └── generate_report.py             # 최종 리포트 생성
│
├── baselines/                         # Baseline 모델 코드
│   ├── conet/                         # CoNet 구현체
│   │   ├── model.py
│   │   ├── data_converter.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── dtcdr/                         # DTCDR 구현체
│   │   ├── model.py
│   │   ├── data_converter.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── llm4cdr/                       # LLM4CDR 구현체
│       ├── prompts.py                 # 3-stage prompts
│       └── evaluator.py
│
├── results/                           # 평가 결과 저장
│   ├── kitrec/                        # KitREC 결과
│   │   ├── dualft_movies_seta/
│   │   ├── dualft_movies_setb/
│   │   ├── dualft_music_seta/
│   │   ├── dualft_music_setb/
│   │   ├── singleft_movies_seta/
│   │   ├── singleft_movies_setb/
│   │   ├── singleft_music_seta/
│   │   └── singleft_music_setb/
│   ├── ablation/                      # RQ1 Ablation 결과
│   │   ├── kitrec_full/
│   │   ├── kitrec_direct/
│   │   ├── base_cot/
│   │   └── base_direct/
│   ├── baselines/                     # Baseline 결과
│   │   ├── conet/
│   │   ├── dtcdr/
│   │   ├── llm4cdr/
│   │   └── vanilla_zeroshot/
│   └── reports/                       # 최종 분석 리포트
│       ├── rq1_ablation_report.md
│       ├── rq2_cdr_comparison.md
│       ├── rq3_coldstart_analysis.md
│       ├── rq4_explainability.md
│       └── final_paper_tables.md
│
├── logs/                              # 실행 로그
│   └── {timestamp}_eval.log
│
└── notebooks/                         # 분석 노트북 (선택)
    ├── result_analysis.ipynb
    └── visualization.ipynb
```

---

## 2. 설정 파일 상세 (`configs/`)

### 2.1 `eval_config.yaml`

```yaml
# 평가 공통 설정
evaluation:
  batch_size: 8
  max_new_tokens: 2048
  temperature: 0.0  # Greedy decoding

metrics:
  ranking:
    - hit@1
    - hit@5
    - hit@10
    - mrr
    - ndcg@5
    - ndcg@10
  explainability:
    - mae
    - rmse
    - perplexity

# ⚠️ Confidence Score 정규화 (CLAUDE.md 참조)
confidence_normalization:
  model_scale: 10  # 모델 출력: 0~10
  gt_scale: 5      # Ground Truth: 0~5
  divisor: 2       # confidence / 2

# 후보군 외 item_id 처리
invalid_item_handling:
  action: "fail"   # rank = ∞
  log_errors: true
```

### 2.2 `model_paths.yaml`

```yaml
# HuggingFace 모델 경로 매핑
base_model:
  name: "Qwen/Qwen3-14B"

kitrec_models:
  dualft_movies_seta: "Younggooo/kitrec-dualft-movies-seta-model"
  dualft_movies_setb: "Younggooo/kitrec-dualft-movies-setb-model"
  dualft_music_seta: "Younggooo/kitrec-dualft-music-seta-model"
  dualft_music_setb: "Younggooo/kitrec-dualft-music-setb-model"
  singleft_movies_seta: "Younggooo/kitrec-singleft-movies-seta-model"
  singleft_movies_setb: "Younggooo/kitrec-singleft-movies-setb-model"
  singleft_music_seta: "Younggooo/kitrec-singleft-music-seta-model"
  singleft_music_setb: "Younggooo/kitrec-singleft-music-setb-model"

datasets:
  test_seta: "Younggooo/kitrec-test-seta"
  test_setb: "Younggooo/kitrec-test-setb"
```

### 2.3 `baseline_config.yaml`

```yaml
# Baseline 모델 설정
conet:
  hidden_dim: 256
  num_layers: 3
  learning_rate: 0.001

dtcdr:
  embedding_dim: 128
  mlp_layers: [256, 128]

llm4cdr:
  model: "Qwen/Qwen3-14B"
  stages:
    - domain_gap_analysis
    - user_interest_reasoning
    - candidate_reranking
```

---

## 3. 소스 코드 상세 명세 (`src/`)

### 3.1 `src/data/data_loader.py`

```python
"""
HuggingFace Hub에서 Test Set 로딩
⚠️ CLAUDE.md Critical Notes #1: Template Schema Difference 적용
"""

from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name: str, hf_token: str = None):
        self.dataset_name = dataset_name
        self.hf_token = hf_token

    def load_test_data(self):
        """Test 데이터 로딩 및 프롬프트 추출"""
        dataset = load_dataset(self.dataset_name, token=self.hf_token)
        return dataset["train"]  # test split

    def extract_prompt(self, sample: dict) -> str:
        """
        ⚠️ Template Schema Difference (CLAUDE.md 필수 패턴):
        - Val/Test: `input` 필드에 전체 프롬프트
        - Training: `instruction` 필드에 전체 프롬프트
        """
        prompt = sample["input"] if sample.get("input") else sample["instruction"]
        return prompt

    def extract_ground_truth(self, sample: dict) -> dict:
        """Ground Truth 아이템 정보 추출"""
        return sample["ground_truth"]

    def extract_candidate_ids(self, sample: dict) -> list:
        """Candidate Set의 item_id 리스트 추출 (검증용)"""
        # 프롬프트에서 (ID: xxx) 패턴 추출
        import re
        prompt = self.extract_prompt(sample)
        pattern = r'\(ID:\s*([A-Z0-9]+)\)'
        return re.findall(pattern, prompt)
```

### 3.2 `src/data/prompt_builder.py`

```python
"""
Inference 프롬프트 생성
⚠️ CLAUDE.md RQ1 Note: Direct 모델용 Reasoning Guidelines 제거 버전 필요
"""

class PromptBuilder:
    # CLAUDE.md의 Inference Prompt Template 전체
    THINKING_TEMPLATE = '''# Expert Cross-Domain Recommendation System

You are a specialized recommendation system with expertise in cross-domain knowledge transfer.
Your task is to leverage comprehensive user interaction patterns from source and target domains to rank the **Top 10** most suitable items from the candidate list.

## Input Parameters
- Source Domain: {source_domain}
- Target Domain: {target_domain}
- Task: Rank the top 10 items based on user preference alignment.

## User Interaction History
{user_history}

## List of Available Candidate Items (Total 100):
{candidate_list}

## Reasoning Guidelines (Thinking Process)
Before generating the final JSON output, you must engage in a deep reasoning process.
Think step-by-step using the following phases:

### Phase 1: Pattern Recognition (Source Domain Analysis)
- Analyze the user's `{source_domain}` history to identify core preference signals.
- Extract key genres, thematic interests, content complexity, and stylistic preferences.
- Identify high-rated items (Rating > 4.0) to understand what the user truly values.

### Phase 2: Cross-Domain Knowledge Transfer
- Apply domain knowledge to map preferences from `{source_domain}` to `{target_domain}`.
- Consider semantic connections, author/director styles, and emotional tone.

### Phase 3: Candidate Evaluation & Selection
- Evaluate the 100 candidate items against the transferred profile.
- Select the Top 10 items that best match the inferred preferences.
- Ensure diversity in the selection while maintaining high relevance.
- Formulate a rationale for each selected item.

## Output Format
After your reasoning process, provide results **ONLY** as a JSON array containing the **Top-10** recommended items.

```json
[
   {{ "rank": 1, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..." }},
   ...
   {{ "rank": 10, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..." }}
]
```
'''

    # Direct 모델용 (Reasoning Guidelines 제거)
    DIRECT_TEMPLATE = '''# Expert Cross-Domain Recommendation System

You are a specialized recommendation system with expertise in cross-domain knowledge transfer.
Your task is to leverage comprehensive user interaction patterns from source and target domains to rank the **Top 10** most suitable items from the candidate list.

## Input Parameters
- Source Domain: {source_domain}
- Target Domain: {target_domain}
- Task: Rank the top 10 items based on user preference alignment.

## User Interaction History
{user_history}

## List of Available Candidate Items (Total 100):
{candidate_list}

## Output Format
Provide results **ONLY** as a JSON array containing the **Top-10** recommended items.
Do NOT include any reasoning or thinking process.

```json
[
   {{ "rank": 1, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..." }},
   ...
   {{ "rank": 10, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..." }}
]
```
'''

    def build_thinking_prompt(self, sample: dict) -> str:
        """
        KitREC-Full, Base-CoT용 (Reasoning Guidelines 포함)
        - 학습된 모델 또는 Zero-shot CoT 평가에 사용
        """
        # Val/Test 데이터는 이미 완성된 프롬프트가 input에 있음
        return sample["input"] if sample.get("input") else sample["instruction"]

    def build_direct_prompt(self, sample: dict) -> str:
        """
        KitREC-Direct, Base-Direct용 (Reasoning Guidelines 제거)
        - Thinking Process 없이 바로 JSON 출력
        """
        original_prompt = sample["input"] if sample.get("input") else sample["instruction"]

        # "## Reasoning Guidelines" 섹션 제거
        import re
        pattern = r'## Reasoning Guidelines.*?(?=## Output Format)'
        modified = re.sub(pattern, '', original_prompt, flags=re.DOTALL)

        # Output Format 수정 (reasoning 없이 바로 출력)
        modified = modified.replace(
            "After your reasoning process, provide results",
            "Provide results directly"
        )

        return modified
```

### 3.3 `src/data/candidate_handler.py`

```python
"""
Candidate Set 처리 및 검증
⚠️ CLAUDE.md Baseline 공정성 조건: 모든 모델 동일 Candidate Set 사용
"""

import re
from typing import List, Set

class CandidateHandler:
    def extract_candidate_ids(self, prompt: str) -> List[str]:
        """프롬프트에서 Candidate item_id 추출"""
        pattern = r'\(ID:\s*([A-Z0-9]+)\)'
        return re.findall(pattern, prompt)

    def validate_prediction(self, predicted_id: str, candidate_ids: List[str]) -> bool:
        """
        예측된 item_id가 Candidate Set에 있는지 검증
        ⚠️ 후보군 외 item 출력 시 → 자동 fail 처리 (rank = ∞)
        """
        return predicted_id in candidate_ids

    def convert_to_id_matrix(self, user_history: List[str], item_vocab: dict) -> List[int]:
        """
        Baseline 모델용: 텍스트 History → ID matrix 변환
        ⚠️ CLAUDE.md Critical Notes #4: 동일 시점 데이터 사용 필수
        """
        return [item_vocab.get(item_id, 0) for item_id in user_history]
```

### 3.4 `src/inference/output_parser.py`

```python
"""
모델 출력 파싱
⚠️ CLAUDE.md Critical Notes #3: Output Parsing 주의사항 적용
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ParseResult:
    thinking: Optional[str]
    predictions: List[Dict]
    parse_errors: List[str]
    invalid_items: List[str]  # 후보군 외 item_id

class OutputParser:
    def parse(self, raw_output: str, candidate_ids: List[str]) -> ParseResult:
        """
        모델 출력 파싱

        1. <think>...</think> 블록 분리
        2. JSON 블록 추출 (```json ... ```)
        3. trailing comma 제거
        4. item_id 검증: candidate_ids에 없으면 fail (rank=∞)
        5. 오류율 통계 반환
        """
        errors = []
        invalid_items = []

        # 1. Thinking 블록 분리
        thinking = self._extract_thinking(raw_output)

        # 2. JSON 블록 추출
        json_str = self._extract_json(raw_output)
        if not json_str:
            errors.append("JSON block not found")
            return ParseResult(thinking, [], errors, [])

        # 3. Trailing comma 제거
        json_str = self._remove_trailing_comma(json_str)

        # 4. JSON 파싱
        try:
            predictions = json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {str(e)}")
            return ParseResult(thinking, [], errors, [])

        # 5. item_id 검증
        valid_predictions = []
        for pred in predictions:
            item_id = pred.get("item_id", "")
            if item_id in candidate_ids:
                valid_predictions.append(pred)
            else:
                invalid_items.append(item_id)
                errors.append(f"Invalid item_id: {item_id} (not in candidate set)")

        return ParseResult(thinking, valid_predictions, errors, invalid_items)

    def _extract_thinking(self, output: str) -> Optional[str]:
        """<think>...</think> 블록 추출"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, output, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_json(self, output: str) -> Optional[str]:
        """
        ```json ... ``` 블록 추출 (개선된 버전)
        - 다중 라인 JSON 처리 위해 [\s\S] 사용
        - 배열 및 객체 형식 모두 지원
        """
        # Priority 1: ```json 코드 블록 (배열)
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
        pattern = r'\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]'
        match = re.search(pattern, output)
        if match:
            return match.group(0)

        # Priority 4: 단일 JSON 객체
        pattern = r'\{[\s\S]*?\}'
        match = re.search(pattern, output)
        return match.group(0) if match else None

    def _remove_trailing_comma(self, json_str: str) -> str:
        """trailing comma 제거"""
        # },] 또는 }, ] 패턴 처리
        json_str = re.sub(r',\s*\]', ']', json_str)
        json_str = re.sub(r',\s*\}', '}', json_str)
        return json_str


class ErrorStatistics:
    """파싱 오류 통계"""
    def __init__(self):
        self.total_samples = 0
        self.parse_failures = 0
        self.invalid_item_count = 0
        self.invalid_item_ids = []

    def update(self, result: ParseResult):
        self.total_samples += 1
        if result.parse_errors:
            self.parse_failures += 1
        self.invalid_item_count += len(result.invalid_items)
        self.invalid_item_ids.extend(result.invalid_items)

    def get_summary(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "parse_failure_rate": self.parse_failures / max(self.total_samples, 1),
            "invalid_item_rate": self.invalid_item_count / max(self.total_samples, 1),
            "unique_invalid_items": len(set(self.invalid_item_ids))
        }
```

### 3.5 `src/metrics/ranking_metrics.py`

```python
"""
랭킹 평가 지표
⚠️ CLAUDE.md Evaluation Metrics 참조
"""

import numpy as np
from typing import List, Dict

class RankingMetrics:
    @staticmethod
    def hit_at_k(predictions: List[Dict], ground_truth_id: str, k: int) -> float:
        """
        Hit@K: Top-K 안에 정답 포함 여부
        - Hit@1: 정확히 1위로 예측
        - Hit@5: Top-5 안에 정답 포함
        - Hit@10: Top-10 안에 정답 포함
        """
        top_k_ids = [p["item_id"] for p in predictions[:k]]
        return 1.0 if ground_truth_id in top_k_ids else 0.0

    @staticmethod
    def mrr(predictions: List[Dict], ground_truth_id: str) -> float:
        """
        Mean Reciprocal Rank
        - 1위=1.0, 2위=0.5, 3위=0.33, 10위=0.1
        - 정답이 없으면 0
        """
        for i, pred in enumerate(predictions):
            if pred["item_id"] == ground_truth_id:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(predictions: List[Dict], ground_truth_id: str, k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        - 1위=1.0, 2위=0.631, 10위=0.289 (log2 기반)
        """
        dcg = 0.0
        for i, pred in enumerate(predictions[:k]):
            if pred["item_id"] == ground_truth_id:
                # relevance = 1 for ground truth
                dcg = 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                break

        # IDCG: 이상적인 경우 (정답이 1위)
        idcg = 1.0 / np.log2(2)  # = 1.0

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_all(predictions: List[Dict], ground_truth_id: str) -> Dict[str, float]:
        """모든 랭킹 메트릭 계산"""
        return {
            "hit@1": RankingMetrics.hit_at_k(predictions, ground_truth_id, 1),
            "hit@5": RankingMetrics.hit_at_k(predictions, ground_truth_id, 5),
            "hit@10": RankingMetrics.hit_at_k(predictions, ground_truth_id, 10),
            "mrr": RankingMetrics.mrr(predictions, ground_truth_id),
            "ndcg@5": RankingMetrics.ndcg_at_k(predictions, ground_truth_id, 5),
            "ndcg@10": RankingMetrics.ndcg_at_k(predictions, ground_truth_id, 10),
        }
```

### 3.6 `src/metrics/explainability_metrics.py`

```python
"""
설명력 평가 지표
⚠️ CLAUDE.md Evaluation Metrics: Confidence Score 정규화 필수
"""

import numpy as np
from typing import List, Dict
import torch

class ExplainabilityMetrics:
    def __init__(self, confidence_divisor: float = 2.0):
        """
        ⚠️ Confidence Score 정규화:
        - Model 출력: 0~10 float
        - Ground Truth: 0~5 rating
        - 정규화: confidence / 2
        """
        self.confidence_divisor = confidence_divisor

    def normalize_confidence(self, confidence: float) -> float:
        """Confidence Score 정규화"""
        return confidence / self.confidence_divisor

    def mae(self, predictions: List[Dict], ground_truth_ratings: List[float]) -> float:
        """
        Mean Absolute Error
        예측 신뢰도와 실제 Rating 비교
        """
        errors = []
        for pred, gt_rating in zip(predictions, ground_truth_ratings):
            normalized_conf = self.normalize_confidence(pred.get("confidence_score", 5.0))
            errors.append(abs(normalized_conf - gt_rating))
        return np.mean(errors) if errors else 0.0

    def rmse(self, predictions: List[Dict], ground_truth_ratings: List[float]) -> float:
        """
        Root Mean Squared Error
        """
        squared_errors = []
        for pred, gt_rating in zip(predictions, ground_truth_ratings):
            normalized_conf = self.normalize_confidence(pred.get("confidence_score", 5.0))
            squared_errors.append((normalized_conf - gt_rating) ** 2)
        return np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

    def perplexity(self, model, tokenizer, rationales: List[str]) -> float:
        """
        Perplexity: 추천 설명의 언어적 품질 평가
        낮을수록 모델이 확신을 가지고 생성
        """
        total_loss = 0.0
        total_tokens = 0

        for rationale in rationales:
            inputs = tokenizer(rationale, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return np.exp(avg_loss)
```

### 3.7 `src/metrics/stratified_analysis.py`

```python
"""
Stratified 분석
⚠️ CLAUDE.md: User Type별, Metadata별 분리 분석
"""

from typing import List, Dict
from collections import defaultdict

class StratifiedAnalysis:
    # User Type 정의 (CLAUDE.md 참조)
    USER_TYPE_MAPPING = {
        "source_only_movies": {"core": 1, "model": "SingleFT-Movies"},
        "source_only_music": {"core": 1, "model": "SingleFT-Music"},
        "cold_start_2core_movies": {"core": 2, "model": "DualFT-Movies"},
        "cold_start_2core_music": {"core": 2, "model": "DualFT-Music"},
        "cold_start_3core_movies": {"core": 3, "model": "DualFT-Movies"},
        "cold_start_3core_music": {"core": 3, "model": "DualFT-Music"},
        "cold_start_4core_movies": {"core": 4, "model": "DualFT-Movies"},
        "cold_start_4core_music": {"core": 4, "model": "DualFT-Music"},
        "overlapping_books_movies": {"core": "5+", "model": "DualFT-Movies"},
        "overlapping_books_music": {"core": "5+", "model": "DualFT-Music"},
    }

    def analyze_by_user_type(self, results: List[Dict]) -> Dict[str, Dict]:
        """User Type별 성능 분석"""
        grouped = defaultdict(list)

        for result in results:
            user_type = result["metadata"]["user_type"]
            grouped[user_type].append(result["metrics"])

        analysis = {}
        for user_type, metrics_list in grouped.items():
            analysis[user_type] = self._aggregate_metrics(metrics_list)
            analysis[user_type]["core_level"] = self.USER_TYPE_MAPPING.get(
                user_type, {}
            ).get("core", "unknown")

        return analysis

    def analyze_by_core_level(self, results: List[Dict]) -> Dict[str, Dict]:
        """Core Level별 성능 분석 (1-core ~ 10-core)"""
        grouped = defaultdict(list)

        for result in results:
            user_type = result["metadata"]["user_type"]
            core = self.USER_TYPE_MAPPING.get(user_type, {}).get("core", "unknown")
            grouped[f"{core}-core"].append(result["metrics"])

        return {
            level: self._aggregate_metrics(metrics_list)
            for level, metrics_list in grouped.items()
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
        """
        group_a = []  # Metadata 있음
        group_b = []  # Metadata 없음 (Unknown)

        for result in results:
            gt_item_id = result["ground_truth"]["item_id"]
            has_metadata = metadata_lookup.get(gt_item_id, False)

            if has_metadata:
                group_a.append(result["metrics"])
            else:
                group_b.append(result["metrics"])

        return {
            "group_a_with_metadata": {
                **self._aggregate_metrics(group_a),
                "count": len(group_a)
            },
            "group_b_unknown": {
                **self._aggregate_metrics(group_b),
                "count": len(group_b)
            }
        }

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """메트릭 집계 (평균)"""
        if not metrics_list:
            return {}

        aggregated = defaultdict(list)
        for metrics in metrics_list:
            for key, value in metrics.items():
                aggregated[key].append(value)

        return {
            key: sum(values) / len(values)
            for key, values in aggregated.items()
        }
```

### 3.8 `src/metrics/statistical_analysis.py`

```python
"""
통계적 유의성 검정
⚠️ CLAUDE.md: 논문 발표를 위해 모든 Baseline 비교에서 통계적 유의성 보고 필수
"""

from scipy import stats
import numpy as np
from typing import List, Dict

class StatisticalAnalysis:
    @staticmethod
    def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict:
        """
        Paired t-test for per-sample metric comparison
        - RQ1: KitREC-Full vs Ablation models
        - RQ2: KitREC vs Baselines
        """
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        # Cohen's d effect size
        diff = np.array(scores_a) - np.array(scores_b)
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "significant_at_0.001": p_value < 0.001,
            "effect_size_cohens_d": float(effect_size),
            "mean_diff": float(np.mean(diff)),
            "n_samples": len(scores_a)
        }

    @staticmethod
    def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
        """Bootstrap confidence interval for single metric"""
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
    def compare_all_baselines(kitrec_scores: Dict[str, List[float]],
                               baseline_scores: Dict[str, Dict[str, List[float]]]) -> Dict:
        """
        Compare KitREC against all baselines for all metrics
        Returns: {baseline_name: {metric: t_test_result}}
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
    def format_for_paper(result: Dict) -> str:
        """Format t-test result for paper table (e.g., 0.85* or 0.85**)"""
        mean_diff = result["mean_diff"]
        if result["significant_at_0.001"]:
            return f"{mean_diff:+.3f}***"
        elif result["significant_at_0.01"]:
            return f"{mean_diff:+.3f}**"
        elif result["significant_at_0.05"]:
            return f"{mean_diff:+.3f}*"
        else:
            return f"{mean_diff:+.3f}"
```

---

## 4. Baseline 모델 상세 (`baselines/`)

### 🔒 4.0 Baseline Train/Test 데이터 분리 (Data Leakage 방지)

> **⚠️ 핵심 원칙:** Baseline 모델은 반드시 KitREC Training 데이터로 학습하고, KitREC Test 데이터로 평가해야 합니다.
> 기존 HuggingFace 데이터셋은 이미 Train/Test가 엄격히 분리되어 있으므로 Data Leakage 문제가 없습니다.

#### Baseline 학습용 데이터셋 매핑

| Target Domain | Candidate Set | Training Dataset | Samples |
|---------------|---------------|------------------|---------|
| Movies | Set A | `Younggooo/kitrec-dualft_movies-seta` | 12,000 |
| Movies | Set B | `Younggooo/kitrec-dualft_movies-setb` | 12,000 |
| Music | Set A | `Younggooo/kitrec-dualft_music-seta` | 12,000 |
| Music | Set B | `Younggooo/kitrec-dualft_music-setb` | 12,000 |

#### Baseline 평가용 데이터셋

| Candidate Set | Test Dataset | Samples | 내용 |
|---------------|--------------|---------|------|
| Set A | `Younggooo/kitrec-test-seta` | 30,000 | Hard Negatives |
| Set B | `Younggooo/kitrec-test-setb` | 30,000 | Random Negatives |

#### 올바른 Baseline 실험 코드

```python
# ✅ 올바른 방법: Train/Test 완전 분리
def run_baseline_experiment(args):
    # Step 1: 학습 데이터 로드 (KitREC Training Dataset)
    train_dataset = f"Younggooo/kitrec-dualft_{args.target_domain}-{args.candidate_set}"
    train_loader = DataLoader(train_dataset, hf_token=args.hf_token)
    train_data = train_loader.load_test_data()

    # Step 2: Baseline 모델 학습
    if args.train_baseline:
        trainer.train(train_data)
        trainer.save_checkpoint(f"checkpoints/{args.baseline}_{args.target_domain}_{args.candidate_set}.pt")

    # Step 3: 평가 데이터 로드 (KitREC Test Dataset) - 별도 단계
    test_dataset = f"Younggooo/kitrec-test-{args.candidate_set}"
    test_loader = DataLoader(test_dataset, hf_token=args.hf_token)
    test_data = test_loader.load_test_data()

    # Step 4: Baseline 평가 (Test 데이터만 사용)
    evaluator.evaluate(test_data)
```

```python
# ❌ 금지: Test 데이터를 분할하여 학습에 사용 (Data Leakage)
# 아래와 같은 코드는 절대 사용하면 안 됩니다!
train_samples = converted_samples[:-1000]  # 잘못됨!
val_samples = converted_samples[-1000:]    # 잘못됨!
```

#### Baseline 실험 실행 예시

```bash
# CoNet 학습 (Train 데이터)
python scripts/run_baseline_train.py \
    --baseline conet \
    --target_domain movies \
    --candidate_set seta \
    --train_dataset Younggooo/kitrec-dualft_movies-seta

# CoNet 평가 (Test 데이터)
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain movies \
    --candidate_set seta \
    --baseline_checkpoint checkpoints/conet_movies_seta.pt
```

### 4.1 `baselines/conet/data_converter.py`

```python
"""
CoNet용 데이터 변환
⚠️ CLAUDE.md Critical Notes #4: 동일 User History 시퀀스 사용 필수
"""

class CoNetDataConverter:
    def __init__(self, item_vocab: Dict[str, int]):
        self.item_vocab = item_vocab

    def convert_history(self, text_history: List[str]) -> np.ndarray:
        """
        텍스트 History → ID matrix 변환
        ⚠️ KitREC에 들어가는 History와 동일한 시점의 데이터 사용
        """
        ids = [self.item_vocab.get(item_id, 0) for item_id in text_history]
        return np.array(ids)

    def convert_candidates(self, candidate_ids: List[str]) -> np.ndarray:
        """
        Candidate Set 변환
        ⚠️ 반드시 KitREC과 동일한 100개 후보 (1 GT + 99 Neg) 사용
        """
        return np.array([self.item_vocab.get(cid, 0) for cid in candidate_ids])
```

### 4.2 `baselines/llm4cdr/prompts.py`

```python
"""
LLM4CDR 3-stage 파이프라인 프롬프트
"""

class LLM4CDRPrompts:
    STAGE1_DOMAIN_GAP = """
Analyze the relationship between {source_domain} and {target_domain} domains.
What semantic connections exist between these two domains?
"""

    STAGE2_USER_INTEREST = """
Based on the user's {source_domain} history:
{user_history}

Describe the user's preferences and interests that might transfer to {target_domain}.
"""

    STAGE3_RERANKING = """
Given the user's inferred preferences and the following candidate items:
{candidates}

Re-rank these items based on the user's likely preferences.
Return the top 10 items in JSON format.
"""
```

---

## 5. 실행 스크립트 상세 (`scripts/`)

### 5.1 `scripts/run_kitrec_eval.py`

```python
"""
KitREC 모델 평가 실행
8개 모델 × 30,000 샘플 평가
"""

import argparse
from src.data.data_loader import DataLoader
from src.models.kitrec_model import KitRECModel
from src.inference.vllm_inference import VLLMInference
from src.inference.output_parser import OutputParser, ErrorStatistics
from src.metrics.ranking_metrics import RankingMetrics
from src.metrics.explainability_metrics import ExplainabilityMetrics

def main(args):
    # 1. 데이터 로딩
    loader = DataLoader(f"Younggooo/kitrec-test-{args.set}")
    test_data = loader.load_test_data()

    # 2. 모델 로딩
    model = KitRECModel.load(args.model_name)

    # 3. 추론 엔진 초기화
    inference = VLLMInference(model)
    parser = OutputParser()
    error_stats = ErrorStatistics()

    # 4. 평가 실행
    results = []
    for sample in tqdm(test_data):
        prompt = loader.extract_prompt(sample)
        candidate_ids = loader.extract_candidate_ids(sample)
        gt = loader.extract_ground_truth(sample)

        # 추론
        output = inference.generate(prompt)

        # 파싱
        parse_result = parser.parse(output, candidate_ids)
        error_stats.update(parse_result)

        # 메트릭 계산
        metrics = RankingMetrics.calculate_all(
            parse_result.predictions,
            gt["item_id"]
        )

        results.append({
            "sample_id": sample["user_id"],
            "predictions": parse_result.predictions,
            "ground_truth": gt,
            "metrics": metrics,
            "metadata": sample["metadata"]
        })

    # 5. 결과 저장
    save_results(results, args.output_dir)
    print(f"Error Statistics: {error_stats.get_summary()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--set", choices=["seta", "setb"], required=True)
    parser.add_argument("--output_dir", default="results/kitrec")
    args = parser.parse_args()
    main(args)
```

### 5.2 `scripts/run_ablation_study.py`

```python
"""
RQ1: 2×2 Ablation Study
⚠️ CLAUDE.md RQ1 상세 정의 참조
⚠️ KitREC-Direct는 Option A (별도 학습) 권장
"""

import re
from src.data.prompt_builder import PromptBuilder

def run_ablation():
    """
    4개 모델 비교:
    ① KitREC-Full (제안 모델) - Thinking + Fine-tuned
    ② KitREC-Direct (Ablation) - No Thinking + Fine-tuned (Option A: 별도 학습)
    ③ Base-CoT (Strong Baseline) - Thinking + Untuned
    ④ Base-Direct (Weak Baseline) - No Thinking + Untuned
    """
    prompt_builder = PromptBuilder()

    models = {
        "kitrec_full": {
            "model": load_kitrec_model("kitrec-full"),  # 기존 학습 모델
            "prompt_fn": prompt_builder.build_thinking_prompt,
            "description": "Fine-tuned + Thinking"
        },
        "kitrec_direct": {
            # ⚠️ Option A: <think> 제거된 데이터로 별도 학습한 모델 사용
            "model": load_kitrec_model("kitrec-direct"),  # 별도 학습 필요
            "prompt_fn": prompt_builder.build_direct_prompt,
            "description": "Fine-tuned (No Thinking) + Direct Output"
        },
        "base_cot": {
            "model": load_base_model(),  # Qwen3-14B-Instruct
            "prompt_fn": prompt_builder.build_thinking_prompt,
            "description": "Untuned + Thinking"
        },
        "base_direct": {
            "model": load_base_model(),  # Qwen3-14B-Instruct
            "prompt_fn": prompt_builder.build_direct_prompt,
            "description": "Untuned + No Thinking"
        }
    }

    for model_name, config in models.items():
        print(f"Evaluating {model_name}: {config['description']}")
        evaluate_model(config["model"], config["prompt_fn"])


# ============================================================================
# KitREC-Direct Option A: 학습 데이터 준비 (별도 학습용)
# ============================================================================

def prepare_kitrec_direct_training_data(original_data_path: str, output_path: str):
    """
    KitREC-Direct 학습용 데이터 생성 (Option A)
    - Training 데이터의 output에서 <think>...</think> 블록 제거
    - JSON 출력만 남김

    ⚠️ 이 함수로 생성된 데이터로 별도 모델을 학습해야 함
    """
    import json

    def remove_thinking_block(output: str) -> str:
        """<think>...</think> 블록 제거"""
        pattern = r'<think>[\s\S]*?</think>\s*'
        return re.sub(pattern, '', output).strip()

    with open(original_data_path, 'r') as f:
        original_data = [json.loads(line) for line in f]

    direct_data = []
    for sample in original_data:
        direct_sample = {
            "instruction": sample["instruction"],
            "input": sample.get("input", ""),
            "output": remove_thinking_block(sample["output"]),  # <think> 제거
            "metadata": sample.get("metadata", {})
        }
        direct_data.append(direct_sample)

    with open(output_path, 'w') as f:
        for sample in direct_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"KitREC-Direct training data saved: {output_path}")
    print(f"Original samples: {len(original_data)}")
    print(f"Direct samples: {len(direct_data)}")

# Usage:
# prepare_kitrec_direct_training_data(
#     "data/kitrec-dualft_music-seta/train.jsonl",
#     "data/kitrec-direct_music-seta/train.jsonl"
# )
```

### 5.3 `scripts/run_metadata_subgroup.py`

```python
"""
Movies Metadata 분리 평가
⚠️ CLAUDE.md Sub-group Analysis: Group A/B 분리 필수
"""

from src.metrics.stratified_analysis import StratifiedAnalysis

def analyze_metadata_subgroups(results_path: str, metadata_path: str):
    """
    Movies 도메인 43.3% 메타데이터 누락 문제 분석

    Group A: GT item에 Title/Category 존재 → KitREC 실제 성능
    Group B: GT item이 Unknown → 메타데이터 의존도 측정

    예상 결과: Group A 성능 >> Group B 성능
    """
    results = load_results(results_path)
    metadata_lookup = load_metadata_availability(metadata_path)

    analyzer = StratifiedAnalysis()
    subgroup_analysis = analyzer.analyze_by_metadata_availability(
        results,
        metadata_lookup
    )

    print("=== Movies Metadata Sub-group Analysis ===")
    print(f"Group A (with metadata): {subgroup_analysis['group_a_with_metadata']}")
    print(f"Group B (unknown): {subgroup_analysis['group_b_unknown']}")

    # 성능 차이 계산
    diff = {
        metric: subgroup_analysis['group_a_with_metadata'].get(metric, 0) -
                subgroup_analysis['group_b_unknown'].get(metric, 0)
        for metric in ['hit@10', 'ndcg@10', 'mrr']
    }
    print(f"Performance Gap (A - B): {diff}")
```

---

## 6. 실험 로드맵 (Experiment Roadmap)

### Phase 1: 환경 설정 및 데이터 준비
| 순서 | 작업 | 완료 기준 |
|------|------|----------|
| 1.1 | vLLM 환경 설치 | `python -c "import vllm"` 성공 |
| 1.2 | GPU 확인 (Nvidia 5090, 36GB) | `nvidia-smi` 메모리 확인 |
| 1.3 | HuggingFace 토큰 설정 | Private 모델 접근 테스트 |
| 1.4 | Test Set 로딩 검증 | 30,000 samples × 2 sets 확인 |
| 1.5 | Candidate Set 동기화 검증 | 100개 후보 일치 확인 |

### Phase 2: KitREC 모델 평가
| 순서 | 모델 | 샘플 수 | 예상 시간 |
|------|------|---------|----------|
| 2.1 | DualFT-Movies (Set A) | 15,000 | ~3h |
| 2.2 | DualFT-Movies (Set B) | 15,000 | ~3h |
| 2.3 | DualFT-Music (Set A) | 15,000 | ~3h |
| 2.4 | DualFT-Music (Set B) | 15,000 | ~3h |
| 2.5 | SingleFT-Movies (Set A) | 15,000 | ~3h |
| 2.6 | SingleFT-Movies (Set B) | 15,000 | ~3h |
| 2.7 | SingleFT-Music (Set A) | 15,000 | ~3h |
| 2.8 | SingleFT-Music (Set B) | 15,000 | ~3h |

### Phase 3: RQ1 Ablation Study (2×2)
| 순서 | 모델 | 설명 |
|------|------|------|
| 3.1 | KitREC-Full | Phase 2 결과 활용 |
| 3.2 | KitREC-Direct | Thinking 제거 버전 평가 |
| 3.3 | Base-CoT | Zero-shot + Thinking |
| 3.4 | Base-Direct | Vanilla Zero-shot |

### Phase 4: Baseline 모델 평가
| 순서 | 모델 | 작업 내용 |
|------|------|----------|
| 4.1 | CoNet | 학습 + 동일 Candidate Set 평가 |
| 4.2 | DTCDR | 학습 + 동일 Candidate Set 평가 |
| 4.3 | LLM4CDR | 3-stage pipeline 평가 |
| 4.4 | Vanilla Zero-shot | Base-Direct와 동일 |

### Phase 5: Stratified Analysis
| 순서 | 분석 | 목적 |
|------|------|------|
| 5.1 | User Type별 (1~10 core) | RQ3: Cold-start 성능 |
| 5.2 | Movies Metadata Sub-group | Group A/B 분리 |
| 5.3 | Set A vs Set B | Hard vs Random 비교 |

### Phase 6: 최종 리포트 생성
| 순서 | 산출물 | 내용 |
|------|--------|------|
| 6.1 | rq1_ablation_report.md | 2×2 Ablation 결과 |
| 6.2 | rq2_cdr_comparison.md | Baseline 비교 테이블 |
| 6.3 | rq3_coldstart_analysis.md | Core Level별 그래프 |
| 6.4 | rq4_explainability.md | MAE/RMSE/PPL 결과 |
| 6.5 | final_paper_tables.md | 논문용 LaTeX 테이블 |

---

## 7. 핵심 체크포인트 (CLAUDE.md 참조)

| # | 체크포인트 | CLAUDE.md 섹션 | 검증 방법 |
|---|-----------|---------------|----------|
| CP1 | `input` 필드에서 프롬프트 추출 | Critical Notes #1 | 단위 테스트 |
| CP2 | Direct 모델용 Reasoning 제거 | RQ1 Note | 출력에 `<think>` 없음 확인 |
| CP3 | Confidence Score ÷ 2 정규화 | Evaluation Metrics | MAE/RMSE 범위 검증 |
| CP4 | 후보군 외 item_id → fail | Critical Notes #3 | 오류율 로깅 |
| CP5 | 모든 Baseline 동일 Candidate Set | Baseline 공정성 | ID 리스트 비교 |
| CP6 | Baseline History 동일 시점 | Critical Notes #4 | 데이터 검증 |
| CP7 | Movies Group A/B 분리 | Sub-group Analysis | 분리 카운트 확인 |

---

## 8. 예상 산출물

```
results/
├── kitrec/
│   └── {model}_{set}/
│       ├── predictions.jsonl          # 전체 예측 결과
│       ├── metrics_summary.json       # 집계 메트릭
│       └── error_statistics.json      # 파싱 오류 통계
│
├── ablation/
│   └── comparison_table.md            # 2×2 비교 테이블
│
├── baselines/
│   └── {model}/
│       ├── predictions.jsonl
│       └── metrics_summary.json
│
└── reports/
    ├── rq1_ablation_report.md
    ├── rq2_cdr_comparison.md
    ├── rq3_coldstart_analysis.md
    ├── rq4_explainability.md
    └── final_paper_tables.md           # LaTeX 테이블 포함
```

---

## 9. 구현 상세 (Implementation Details)

### 9.1 Baseline 공통 인프라 (`baselines/base_evaluator.py`)

```python
"""
모든 베이스라인이 공유하는 공통 평가 인프라

주요 기능:
1. Candidate Set 검증 (100개 + GT 포함)
2. Confidence Score 정규화 ([0,1] → [1,10])
3. 공통 메트릭 계산
"""

class BaseEvaluator(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.metrics = RankingMetrics()

    def validate_candidate_set(
        self,
        candidates: list,
        gt_id: Optional[int] = None,
        raise_on_error: bool = True
    ) -> bool:
        """
        Candidate Set 검증
        - 100개 후보 확인
        - GT 포함 여부 확인
        """
        if len(candidates) != 100:
            msg = f"Expected 100 candidates, got {len(candidates)}"
            if raise_on_error:
                raise ValueError(msg)
            print(f"Warning: {msg}")
            return False

        if gt_id is not None and gt_id not in candidates:
            msg = f"Ground truth {gt_id} not in candidate set"
            if raise_on_error:
                raise ValueError(msg)
            print(f"Warning: {msg}")
            return False

        return True

    def normalize_confidence(self, raw_score: float) -> float:
        """
        Confidence Score 정규화

        KitREC 범위: 1-10
        Baseline 출력: sigmoid(raw_score) ∈ [0,1]
        변환: sigmoid * 9 + 1 → [1,10]
        """
        sigmoid = 1 / (1 + np.exp(-raw_score))
        return sigmoid * 9 + 1
```

### 9.2 User Type Mapping (RQ3 Cold-start Analysis)

```python
"""
User Type별 분석을 위한 매핑 구축

baselines/dtcdr/evaluator.py에 추가된 메서드:
"""

def evaluate_by_user_type(
    self,
    samples: List[DTCDRSample],
    user_type_mapping: Dict[int, str]
) -> Dict[str, Dict[str, float]]:
    """
    User Type별 평가 (RQ3: Cold-start analysis)

    Args:
        samples: 평가 샘플 리스트
        user_type_mapping: {user_id: user_type}

    Returns:
        {user_type: {metric: value}}
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
```

### 9.3 GPT-4.1 Rationale Evaluation (RQ4)

> ⚠️ **RQ4는 KitREC 모델만 평가 대상** (Baseline 제외)

```python
"""
GPT-4.1 API를 통한 Rationale 품질 평가

src/metrics/explainability_metrics.py의 GPTRationaleEvaluator 클래스:
"""

class GPTRationaleEvaluator:
    """
    User Type별 균등 추출(Stratified Sampling)로 모델당 50개 샘플 평가
    - 10개 User Type × 5개/Type = 50개/모델
    - 비용 효율적이면서 User Type별 균형 잡힌 평가

    평가 기준 (1-10점):
    1. 논리성 (logic): 추천 이유가 논리적인가?
    2. 구체성 (specificity): 구체적인 근거를 제시하는가?
    3. Cross-domain 연결성 (cross_domain): Source→Target 연결이 명확한가?
    4. 사용자 선호 반영 (preference): 히스토리를 잘 반영했는가?
    """

    EVALUATION_PROMPT = '''You are an expert evaluator...
    Respond ONLY with a JSON object:
    {"logic": <1-10>, "specificity": <1-10>, "cross_domain": <1-10>,
     "preference": <1-10>, "overall": <1-10>}'''

    def __init__(
        self,
        api_key: Optional[str] = None,
        samples_per_type: int = 5,  # 각 User Type당 5개 (총 50개/모델)
        model: str = "gpt-4.1",
        random_seed: int = 42
    ):
        self.samples_per_type = samples_per_type
        self.model = model
        # OpenAI client 초기화...

    def evaluate_batch(self, results: List[Dict]) -> Dict[str, float]:
        """
        User Type별 균등 추출 (Stratified Sampling) 후 GPT-4.1 평가

        Returns:
            {
                "logic": mean_score,
                "specificity": mean_score,
                "cross_domain": mean_score,
                "preference": mean_score,
                "overall": mean_score,
                "n_evaluated": int,
                "n_total": int,
                "sampling_stats": {user_type: {total, sampled}}
            }
        """
```

### 9.4 Statistical Significance Testing

```python
"""
통계적 유의성 검정 (src/metrics/statistical_analysis.py)

주요 기능:
1. Paired t-test (RQ1, RQ2)
2. Multiple comparison correction (Holm-Bonferroni)
3. Bootstrap confidence intervals
4. 논문용 형식 출력
"""

class StatisticalAnalysis:
    @staticmethod
    def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict:
        """Paired t-test with Cohen's d effect size"""
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        diff = np.array(scores_a) - np.array(scores_b)
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "significant_at_0.001": p_value < 0.001,
            "effect_size_cohens_d": float(effect_size),
        }

    @staticmethod
    def apply_multiple_correction(
        p_values: List[float],
        method: str = "holm"
    ) -> Dict:
        """
        다중 비교 보정 (Holm-Bonferroni)

        Step-up enforcement로 adjusted p-value 계산
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        adjusted_p = np.zeros(n)
        significant = np.zeros(n, dtype=bool)

        # Step-up enforcement
        max_adjusted = 0.0
        for i, idx in enumerate(sorted_indices):
            raw_adjusted = sorted_p[i] * (n - i)
            max_adjusted = max(max_adjusted, raw_adjusted)
            adjusted_p[idx] = min(max_adjusted, 1.0)

        return {
            "adjusted_p_values": adjusted_p.tolist(),
            "significant": significant.tolist()
        }
```

### 9.5 LLM4CDR 구현 차이점 문서화

```python
"""
LLM4CDR 원 논문 vs KitREC 구현 차이점
(baselines/llm4cdr/prompts.py에 문서화)
"""

LLM4CDR_IMPLEMENTATION_NOTES = """
## KitREC vs Original LLM4CDR Implementation

### Key Differences:

1. **Candidate Set Size**:
   - Original: ~30 items (3 GT + 20-30 negatives)
   - KitREC: 100 items (1 GT + 99 negatives)

2. **Target History**:
   - Original: Not used
   - KitREC: Included for fair comparison

3. **Stage 1 Caching**:
   - Both: Domain gap analysis is cached per domain pair

### Paper Citation Note:
When citing results, note that LLM4CDR was re-evaluated
using KitREC's more challenging evaluation protocol.
"""
```

---

## 10. 논문 작성용 Quick Reference

### 10.1 RQ별 메트릭 매핑

| RQ | 연구 질문 | 주요 메트릭 | 통계 검정 |
|----|---------|-----------|----------|
| RQ1 | Ablation Study | Hit@10, NDCG@10 | Paired t-test |
| RQ2 | Baseline 비교 | Hit@1/5/10, MRR, NDCG@5/10 | Paired t-test + Holm |
| RQ3 | Cold-start | Core Level별 Hit@10 | User Type 분리 |
| RQ4 | Explainability (**KitREC만**) | MAE, RMSE, GPT Score | Stratified 50개/모델 |

### 10.2 논문 테이블 형식 예시

```latex
% RQ2: Baseline Comparison Table
\begin{table}[h]
\centering
\caption{Performance comparison on KitREC test set}
\begin{tabular}{lcccccc}
\toprule
Model & Hit@1 & Hit@5 & Hit@10 & MRR & NDCG@5 & NDCG@10 \\
\midrule
CoNet & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx \\
DTCDR & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx \\
LLM4CDR & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx \\
\midrule
KitREC & \textbf{0.xxx} & \textbf{0.xxx}** & \textbf{0.xxx}*** & ... \\
\bottomrule
\multicolumn{7}{l}{\footnotesize * p<0.05, ** p<0.01, *** p<0.001 (paired t-test)} \\
\end{tabular}
\end{table}
```

### 10.3 핵심 인용 문구

```
# Baseline 비교 방법론
"All baseline models were evaluated on the identical candidate set
(1 ground truth + 99 negatives) to ensure fair comparison."

# LLM4CDR 재평가
"LLM4CDR was re-implemented and evaluated using our evaluation
protocol for fair comparison. The original paper uses a smaller
candidate set (3 GT + 20-30 negatives)."

# Cold-start 분석
"Unlike existing CDR methods that only evaluate on users with
5+ target interactions, we specifically analyze performance on
extreme cold-start users (1-4 target interactions)."

# GPT-4.1 평가
"Rationale quality was evaluated on a 10% random sample using
GPT-4.1 API, scoring logic, specificity, cross-domain connection,
and preference alignment on a 1-10 scale."
```

---

**작성 완료: 2025-12-06**
**최종 수정: 2025-12-07 (구현 완료 반영)**
