
# Meta data Unknown" 메타데이터의 근본 원인. ->movie 의 평가 결과가 낮게 나올 가능성 높음

  Movies 아이템 메타데이터 커버리지 문제:
  | 도메인    | 총 아이템   | Title 있음 | Title 없음 | 누락 비율 |
  |--------|---------|----------|----------|-------|
  | Books  | 352,672 | 352,672  | 0        | 0%    |
  | Movies | 468,347 | 265,364  | 202,991  | 43.3% |
  | Music  | 339,980 | 339,980  | 14       | 0%    |
  ---
  원인 분석

  Movies 도메인에서 약 43%의 아이템이 메타데이터(title, category)가 없습니다.

  이것은 원본 Amazon Review 2023 데이터셋의 특성입니다:

  1. 데이터 소스 불일치: Amazon Review 데이터셋에서 review 데이터와 metadata 데이터가 완전히 매칭되지 않음
  2. item_id 존재하지만 metadata 없음: 사용자가 리뷰/평점을 남긴 아이템이지만, 해당 아이템의 상세 정보(title, category)가 메타데이터 파일에 없는 경우
  3. Movies 도메인 특이성: Books/Music은 거의 100% 커버되지만, Movies & TV 카테고리는 메타데이터 누락이 심함

  ---
  Unknown이 발생하는 위치

  1. User Interaction History (주로 여기서 발생)
    - 사용자가 과거에 시청한 Movies 아이템 중 metadata 없는 것들
    - 형식: [Item: B00E9E5EEK] | Unknown | Rating: 2.0
  2. Candidate Items (일부)
    - Candidate 풀에 포함된 Movies 아이템 중 metadata 없는 것들

  ---
  이것이 실험에 미치는 영향

  | 영향       | 수준  | 설명                                           |
  |----------|-----|----------------------------------------------|
  | 모델 학습    | 중간  | LLM이 Unknown 아이템에서 cross-domain 패턴을 학습하기 어려움 |
  | 평가 공정성   | 낮음  | 모든 샘플에 동일하게 적용되므로 비교 실험에는 문제 없음              |
  | RQ1-4 분석 | 낮음  | 평가 메트릭 계산에는 영향 없음 (GT item_id 기준)            |

  ---
  결론

  - 원인: Amazon Review 2023 데이터셋의 Movies & TV 카테고리에서 43%의 아이템이 metadata 누락
  - 영향: User History에서 일부 아이템이 [Item: ID] | Unknown 형태로 표시됨
  - 권장: 현재 상태로 실험 진행 가능 (모든 샘플에 동일 조건), 추후 metadata 보강 고려


# Train data 와 Test/val  Template 의  차이가 있어 test/val data 사용 시 template을 변경해야 함

파일: /Volumes/ygmac_external/Projects_e/KitREC/finetuning/scripts/evaluate.py

  수정 내용 (lines 345-353):
  # Extract prompts and ground truth
  # NOTE: Val/Test data uses 'input' field for full prompt (with candidates),
  # while 'instruction' contains only a short description.
  # Training data uses 'instruction' for full prompt.
  # This handles both cases for compatibility.
  prompts = [
      sample["input"] if sample.get("input") else sample["instruction"]
      for sample in test_dataset
  ]

  2. 문서화 완료

  파일: /Volumes/ygmac_external/Projects_e/KitREC/finetuning/finetuning_detail_task.md

  추가된 섹션: ⚠️ Critical Evaluation Reference (필수 참조 사항)

  포함 내용:
  - Template Schema Difference 설명
  - 수정 전/후 코드 비교
  - 평가 시 주의사항 3가지
  - 데이터 스키마 예시 (Training vs Val/Test)
  - 영향 범위 체크리스트

  핵심 포인트

  | 데이터 유형   | 프롬프트 필드     |
  |----------|-------------|
  | Training | instruction |
  | Val/Test | input       |

  evaluate.py는 이제 두 경우 모두 자동으로 올바른 필드를 선택합니다

### Template Schema Difference Between Training and Val/Test Data
**발견일**: 2025-11-30 (Expert Agent Verification)

| 데이터 유형 | `instruction` 필드 | `input` 필드 |
|-------------|-------------------|--------------|
| **Training** | 전체 프롬프트 (User History + Candidates) | 빈 문자열 |
| **Val/Test** | 짧은 설명 문구 | 전체 프롬프트 (User History + Candidates) |

### 문제 상황
- Training 데이터: `instruction` 필드에 전체 추천 프롬프트 포함
- Val/Test 데이터: `input` 필드에 전체 프롬프트, `instruction`은 짧은 설명만 포함

### 수정 내용 (evaluate.py)

**수정 전** (line 346):
```python
prompts = test_dataset["instruction"]
```

**수정 후** (lines 346-353):
```python
# NOTE: Val/Test data uses 'input' field for full prompt (with candidates),
# while 'instruction' contains only a short description.
# Training data uses 'instruction' for full prompt.
# This handles both cases for compatibility.
prompts = [
    sample["input"] if sample.get("input") else sample["instruction"]
    for sample in test_dataset
]
```

### 평가 시 주의사항

1. **HuggingFace Hub 데이터 사용 시**: `evaluate.py`가 자동으로 올바른 필드 선택
2. **로컬 JSONL 직접 사용 시**: `input` 필드를 프롬프트로 사용해야 함
3. **커스텀 평가 코드 작성 시**: 아래 패턴 사용

```python
# 올바른 프롬프트 추출 패턴
prompt = sample["input"] if sample.get("input") else sample["instruction"]
```

### 데이터 스키마 예시

**Training Data (instruction 사용):**
```json
{
  "instruction": "# Expert Cross-Domain Recommendation System\nYou are a specialized...\n\n## User's Reading History...\n\n## Candidate Items...",
  "output": "<think>...</think>\n```json\n{...}\n```",
  "metadata": {...}
}
```

**Val/Test Data (input 사용):**
```json
{
  "instruction": "Expert Cross-Domain Recommendation System task",
  "input": "# Expert Cross-Domain Recommendation System\nYou are a specialized...\n\n## User's Reading History...\n\n## Candidate Items...",
  "ground_truth": {"item_id": "...", "title": "...", "rating": 4.0},
  "metadata": {...}
}
```