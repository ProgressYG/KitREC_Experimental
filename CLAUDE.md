# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KitREC (Knowledge-Instruction Transfer for Recommendation) is a cross-domain recommendation research project. This directory (`Experimental_test/`) is the **evaluation workspace** for running model inference and computing metrics on test datasets.

**이 폴더의 역할:**
- 학습된 모델(HuggingFace Hub)을 로드하여 Test Set 평가 수행
- Hit@K, MRR, NDCG 등 추천 성능 지표 계산
- User Type별 stratified 분석 및 Baseline 비교 실험
- vLLM 기반 추론 환경 (Nvidia 5090, 36GB VRAM)

**Core Research Focus:**
- Cross-domain recommendation: Books (source) → Movies/Music (target)
- Cold-start problem mitigation using LLM-based knowledge transfer
- 4 fine-tuned models: DualFT-Movies, DualFT-Music, SingleFT-Movies, SingleFT-Music

---

## Evaluation Metrics

### 추천 아이템 성능 평가 (Ranking Metrics)

| Metric | 설명 | 범위 |
|--------|------|------|
| **Hit@1** | 정확히 1위로 예측 | 0~100% |
| **Hit@5** | Top-5 안에 정답 포함 | 0~100% |
| **Hit@10** | Top-10 안에 정답 포함 (있으면 1, 없으면 0) | 0~100% |
| **MRR** | Mean Reciprocal Rank (1위=1.0, 2위=0.5, 3위=0.33, 10위=0.1) | 0~1 |
| **NDCG@5** | Top-5 랭킹 품질 (DCG/IDCG) | 0~1 |
| **NDCG@10** | Top-10 랭킹 품질 (논문 표준, 1위=1.0, 2위=0.631, 10위=0.289) | 0~1 |

### 추천 설명력 평가 (Explainability Metrics)

| Metric | 대상 | 설명 |
|--------|------|------|
| **MAE, RMSE** | `confidence_score` | 예측 신뢰도와 실제 Rating 비교 |
| **Perplexity (PPL)** | `rationale` | 추천 설명의 언어적 품질 평가 |

**⚠️ Perplexity 계산 범위:**
- **rationale 필드만** 계산 (prompt, `<think>` 블록 제외)
- Fine-tuned 모델로 PPL 측정 (모델의 자기 생성 확신도)
- 낮을수록 품질 좋음 (모델이 확신을 가지고 생성)

**⚠️ Confidence Score 정규화 필수:**
- Test 데이터: Rating 5점 만점 (1~5)
- Model 출력: confidence_score **1~10** float (Template 명시)
- 평가 시 스케일 정규화 필요: `normalized = confidence / 2`
- ⚠️ confidence_score = 0 은 파싱 오류로 처리

---

## User Type & Core Level 평가

### Core Level 기준 (Target Domain 상호작용 수)

| Target Core | User Type | Count/Domain | 설명 | Training Model |
|-------------|-----------|--------------|------|----------------|
| **1-core** | source_only | 3,000 | 극한 Cold-start (target=1) | SingleFT |
| **2-core** | cold_start_2core | 3,000 | 심각한 Cold-start (target=2) | DualFT |
| **3-core** | cold_start_3core | 3,000 | 중간 Cold-start (target=3) | DualFT |
| **4-core** | cold_start_4core | 3,000 | 경미한 Cold-start (target=4) | DualFT |
| **5-core** | overlapping (target 5~9) | 필터링 필요 | Warm 시작 | DualFT |
| **10-core** | overlapping (target≥10) | 필터링 필요 | 풍부한 Target | DualFT |

### Model-User Type 매핑

| Model | User Types | Samples | 특징 |
|-------|------------|---------|------|
| **DualFT-Movies** | overlapping + cold_start_2/3/4core | 12,000 | Books+Movies 이력 활용 |
| **DualFT-Music** | overlapping + cold_start_2/3/4core | 12,000 | Books+Music 이력 활용 |
| **SingleFT-Movies** | source_only_movies | 3,000 | Books 이력만 (극한 Cold-start) |
| **SingleFT-Music** | source_only_music | 3,000 | Books 이력만 (극한 Cold-start) |

---
## User Type별 파인튜닝 모델 매핑 테이블

### 1. User Type Definition & Model Assignment Table
| User Type | Condition | Target Count | Domain Exclusivity | **Fine-tuning Model** | **Role & Rationale** |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Overlapping (Books+Movies)** | books≥5 AND movies≥5 | 3,000 | N/A | **DualFT-Movies** | **Warm-start**: 풍부한 타겟 정보로 양방향 지식 전이 극대화 |
| **Overlapping (Books+Music)** | books≥5 AND music≥5 | 3,000 | N/A | **DualFT-Music** | **Warm-start**: 풍부한 타겟 정보로 양방향 지식 전이 극대화 |
| **Source-only (Movies)** | books≥5 AND movies=1 | 3,000 | ✅ music=0 | **SingleFT-Movies** | **Extreme Cold-start**: 타겟 정보 1개. Source 의존도 최상 (Overfitting 방지) |
| **Source-only (Music)** | books≥5 AND music=1 | 3,000 | ✅ movies=0 | **SingleFT-Music** | **Extreme Cold-start**: 타겟 정보 1개. Source 의존도 최상 (Overfitting 방지) |
| **Cold-start 2-core (Movies)** | books≥5 AND movies=2 | 3,000 | ✅ music=0 | **DualFT-Movies** | **Cold-start**: 최소한의 타겟 패턴(2개) 존재. Cross-domain 패턴 학습 가능 |
| **Cold-start 2-core (Music)** | books≥5 AND music=2 | 3,000 | ✅ movies=0 | **DualFT-Music** | **Cold-start**: 최소한의 타겟 패턴(2개) 존재. Cross-domain 패턴 학습 가능 |
| **Cold-start 3-core (Movies)** | books≥5 AND movies=3 | 3,000 | ✅ music=0 | **DualFT-Movies** | **Cold-start**: 점진적 타겟 정보 증가. 추천 정확도 상승 구간 |
| **Cold-start 3-core (Music)** | books≥5 AND music=3 | 3,000 | ✅ movies=0 | **DualFT-Music** | **Cold-start**: 점진적 타겟 정보 증가. 추천 정확도 상승 구간 |
| **Cold-start 4-core (Movies)** | books≥5 AND movies=4 | 3,000 | ✅ music=0 | **DualFT-Movies** | **Mild Cold-start**: 5-core(Warm) 진입 직전 단계 |
| **Cold-start 4-core (Music)** | books≥5 AND music=4 | 3,000 | ✅ movies=0 | **DualFT-Music** | **Mild Cold-start**: 5-core(Warm) 진입 직전 단계 |

#### 2. Model Selection Logic

#### A. DualFT Models (DualFT-Movies / DualFT-Music)
* **Target Group:** `Overlapping` (5-core+) 및 `Cold-start` (2, 3, 4-core)
* **학습 데이터:** 12,000 Samples (각 도메인별)
* **선정 논리:** 타겟 도메인 아이템이 2개 이상 존재할 경우, Source와 Target 간의 연결 고리(Cross-domain Pattern)를 발견할 최소한의 단서가 있다고 판단하여 DualFT 모델을 적용합니다.

#### B. SingleFT Models (SingleFT-Movies / SingleFT-Music)
* **Target Group:** `Source-only` (1-core)
* **학습 데이터:** 3,000 Samples (각 도메인별)
* **선정 논리:** 타겟 아이템이 단 1개인 경우, 모델이 해당 아이템 하나에만 과적합(Overfitting)되거나 패턴을 찾지 못할 위험이 큽니다. 따라서 Source 도메인의 지식을 Target으로 일방향 전이하는 데 특화된 별도의 튜닝 모델(SingleFT)을 적용합니다.

---

## Dataset Structure (HuggingFace Hub)

```
Training (8 repositories):
  Younggooo/kitrec-dualft_movies-seta     # 12,000 samples
  Younggooo/kitrec-dualft_movies-setb     # 12,000 samples
  Younggooo/kitrec-dualft_music-seta      # 12,000 samples
  Younggooo/kitrec-dualft_music-setb      # 12,000 samples
  Younggooo/kitrec-singleft_movies-seta   # 3,000 samples
  Younggooo/kitrec-singleft_movies-setb   # 3,000 samples
  Younggooo/kitrec-singleft_music-seta    # 3,000 samples
  Younggooo/kitrec-singleft_music-setb    # 3,000 samples

Validation (DPO/GRPO용):
  Younggooo/kitrec-val-seta               # 12,000 samples
  Younggooo/kitrec-val-setb               # 12,000 samples

Test:
  Younggooo/kitrec-test-seta              # 30,000 samples
  Younggooo/kitrec-test-setb              # 30,000 samples
```

- **Set A** = Hybrid Candidates (Hard Negatives) - 난이도 높음
- **Set B** = Random Candidates - 공정한 Baseline 비교용

---

## Critical Implementation Notes

### 1. Template Schema Difference (필수 확인)

| Data Type | Prompt 위치 | `instruction` 필드 | `input` 필드 |
|-----------|-------------|-------------------|--------------|
| **Training** | `instruction` | 전체 프롬프트 (History + Candidates) | 빈 문자열 |
| **Val/Test** | `input` | 짧은 설명 문구만 | 전체 프롬프트 |

**평가 코드 작성 시 필수 패턴:**
```python
# 올바른 프롬프트 추출
prompt = sample["input"] if sample.get("input") else sample["instruction"]
```

### 1.1 Val/Test Data: ground_truth 필드 구조

| Field | Type | Description | 용도 |
|-------|------|-------------|------|
| `item_id` | string | GT 아이템 ASIN | 랭킹 평가 (Hit@K, MRR, NDCG) |
| `title` | string | 아이템 제목 | 검증용 |
| `rating` | float | 사용자 실제 평점 (1~5) | MAE/RMSE 계산 |

**예시:**
```json
{
  "ground_truth": {
    "item_id": "B07FLGJWKB",
    "title": "Blood Red Roses",
    "rating": 4.0
  }
}
```

**⚠️ 파싱 주의:** ground_truth가 JSON string인 경우도 있으므로 양쪽 처리 필요:
```python
gt = sample.get("ground_truth", {})
if isinstance(gt, str):
    gt = json.loads(gt)
```

### 2. Movies Metadata 누락 문제

| 도메인 | 총 아이템 | Title 있음 | 누락 비율 |
|--------|----------|-----------|----------|
| Books | 352,672 | 352,672 | 0% |
| **Movies** | 468,347 | 265,364 | **43.3%** |
| Music | 339,980 | 339,966 | 0% |

**영향:**
- User History에서 일부 아이템이 `[Item: ID] | Unknown` 형태로 표시
- 평가 메트릭 계산에는 영향 없음 (GT item_id 기준)
- Movies 도메인 성능이 상대적으로 낮게 나올 가능성 있음

**📊 Sub-group Analysis 필수 (Movies Domain):**

KitREC은 텍스트(Metadata)를 보고 추론하는 모델이므로, Unknown 아이템 성능이 낮은 것은 당연합니다. **메타데이터가 온전할 때 성능이 압도적으로 높다**는 것을 보여주면 모델의 효용성을 더 강력하게 증명할 수 있습니다.

| Group | 조건 | 분석 목적 |
|-------|------|----------|
| **Group A** | Target Items with Metadata (Title/Category 존재) | KitREC의 실제 성능 |
| **Group B** | Target Items without Metadata (Unknown) | 메타데이터 의존도 측정 |

**예상 결과:** Group A에서의 성능 향상폭이 Group B보다 훨씬 커야 함

### 3. Output Parsing 주의사항

모델 출력 형식:
```
<think>
[Chain-of-Thought reasoning]
</think>
```json
{"rank": 1, "item_id": "...", "title": "...", "confidence_score": 9.5, "rationale": "..."}
```

**파싱 시 고려:**
- `<think>...</think>` 블록과 JSON 블록 분리 필요
- trailing comma 제거 처리
- JSON 파싱 실패 시 robust fallback 구현
- item_id 출력 시 candidate list 에 없는 id 의 경우 평가에서 제외하고 얼마나 이런 오류가 발생하는지 출력이 필요함
  (	후보군 밖 item 출력 시 → 자동 fail 처리 (rank = ∞))

### 4. base line User History
 - 체크 포인트: 딥러닝 베이스라인(CoNet 등)은 텍스트(History Summary)를 입력받지 못하고 ID 시퀀스만 입력받습니다.
 - "Baseline models (CoNet, DTCDR 등등) must use the same User History sequences (Item IDs) as defined in the KitREC test set, converted to their specific input format (e.g., ID matrix)." (즉, KitREC에 들어가는 History와 베이스라인에 들어가는 History가 동일한 시점의 데이터여야 함을 명시)

---

## Inference Prompt Template (Evaluation용)

Test Set 평가 시 사용하는 프롬프트 템플릿입니다. Zero-shot baseline 및 Fine-tuned 모델 평가에 동일하게 적용됩니다.

```
# Expert Cross-Domain Recommendation System

You are a specialized recommendation system with expertise in cross-domain knowledge transfer.
Your task is to leverage comprehensive user interaction patterns from source and target domains to rank the **Top 10** most suitable items from the candidate list.

## Input Parameters
- Source Domain: {source_domain}
- Target Domain: {target_domain}
- Task: Rank the top 10 items based on user preference alignment.

## User Interaction History
The user's past interactions contain the Title, Categories, User Rating, and a Summary of the item description.

### User's {source_domain} History:
{source_history_list}
(Format: - {title} | {categories} | Rating: {rating:.1f} | {description_summary})

### User's {target_domain} History:
{target_history_list}
(Format: - {title} | {categories} | Rating: {rating:.1f} | {description_summary})

## List of Available Candidate Items (Total 100):
The candidate items contain the Title, Categories, Average Rating, and a Summary.
[
  (ID: {item_id_1}) {title} | {categories} | Rating: {avg_rating:.1f} | {description_summary}
  (ID: {item_id_2}) {title} | {categories} | Rating: {avg_rating:.1f} | {description_summary}
  ...
  (ID: {item_id_100}) {title} | {categories} | Rating: {avg_rating:.1f} | {description_summary}
]

## Reasoning Guidelines (Thinking Process)
Before generating the final JSON output, you must engage in a deep reasoning process.
Think step-by-step using the following phases:

### Phase 1: Pattern Recognition (Source Domain Analysis)
- Analyze the user's `{source_domain}` history to identify core preference signals.
- Extract key genres, thematic interests, content complexity, and stylistic preferences.
- Identify high-rated items (Rating > 4.0) to understand what the user truly values.

### Phase 2: Cross-Domain Knowledge Transfer
- Apply domain knowledge to map preferences from `{source_domain}` to `{target_domain}`.
- Example: If a user likes "Dark Fantasy Novels" (Source), infer a preference for "Dark/Gothic Atmosphere Movies" (Target).
- Consider semantic connections, author/director styles, and emotional tone.

### Phase 3: Candidate Evaluation & Selection
- Evaluate the 100 candidate items against the transferred profile.
- Select the Top 10 items that best match the inferred preferences.
- Ensure diversity in the selection while maintaining high relevance.
- Formulate a rationale for each selected item.

## Output Format
After your reasoning process, provide results **ONLY** as a JSON array containing the **Top-10** recommended items.
Ensure the **"item_id"** matches the ID provided in the candidate list exactly.

```json
[
   { "rank": 1, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..." },
   { "rank": 2, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..." },
   ...
   { "rank": 10, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..." }
]
```
```

### Template Variables 설명

| Variable | 설명 | 예시 |
|----------|------|------|
| `{source_domain}` | 소스 도메인 | `Books` |
| `{target_domain}` | 타겟 도메인 | `Movies & TV` 또는 `Music` |
| `{source_history_list}` | 사용자의 소스 도메인 이력 | Books 읽은 목록 |
| `{target_history_list}` | 사용자의 타겟 도메인 이력 | Movies/Music 시청/청취 목록 |
| `{item_id_N}` | 후보 아이템 ID (ASIN) | `B07FLGJWKB` |

### 3-Phase Reasoning 구조

| Phase | 목적 | 핵심 작업 |
|-------|------|----------|
| **Phase 1** | Pattern Recognition | Source 도메인에서 선호 패턴 추출 (장르, 테마, 스타일) |
| **Phase 2** | Cross-Domain Transfer | Source→Target 도메인 지식 전이 (의미적 연결) |
| **Phase 3** | Candidate Evaluation | 100개 후보 중 Top-10 선정 및 rationale 생성 |

---

## Baseline Models & Evaluation Protocols

> **🚨 실험 공정성 필수 조건 (Crucial):**
>
> All baseline models must perform ranking on the **exact same candidate list (1 GT + 99 Negatives)** provided in the KitREC test dataset (`candidate_set`). **Do not use random sampling for baselines during the test phase.**
>
> CoNet이나 DTCDR 같은 전통적 모델의 오픈소스 구현체들은 보통 학습 시 Negative Sampling을 랜덤으로 수행합니다. KitREC은 어려운(Hard) 후보를 풀고, 베이스라인은 쉬운(Random) 후보를 푼다면 **비교가 불가능**합니다.

### 딥러닝 기반 CDR

| Model | 출처 | Candidate 구성 | 평가 방식 | 핵심 지표 |
|-------|------|---------------|----------|----------|
| **CoNet** | CIKM 2018 | test 1 + negative 99 | LOO | Hit@10, NDCG@10 |
| **DTCDR** | CIKM 2019 | test 1 + negative 99 | LOO, Top-N Ranking | HR@10, NDCG@10 |

### LLM 기반

| Model | 출처 | Candidate 구성 | 평가 방식 | 특징 |
|-------|------|---------------|----------|------|
| **LLM4CDR** | RecSys/WWW 2025 | test 3 + negative 20~30 | Prompt-based Re-ranking | 3단계 추론 파이프라인 |
| **Vanilla Zero-shot** | NIR Paradigm | 직접 생성 | Zero-shot Generation | LLM 하한선 (Lower Bound) |

> **🚨 LLM4CDR Candidate Set 정렬 필수 (Critical):**
>
> LLM4CDR 원 논문은 **3 GT + 20~30 Negatives** (총 23~33개)를 사용하지만, KitREC은 **1 GT + 99 Negatives** (총 100개)를 사용합니다.
>
> **공정한 비교를 위해 LLM4CDR도 KitREC 프로토콜(1+99)로 재평가해야 합니다.**
>
> | 프로토콜 | LLM4CDR 원 논문 | KitREC (본 연구) |
> |---------|----------------|-----------------|
> | Positive | 3개 | 1개 |
> | Negative | 20~30개 | 99개 |
> | 총 후보 수 | 23~33개 | 100개 |
> | 난이도 | 쉬움 | 어려움 |
>
> 논문에서는 "LLM4CDR를 KitREC 평가 프로토콜로 재구현하여 비교" 명시 필요.

### KitREC vs Baseline 비교 포인트

| 비교 항목 | LLM4CDR | KitREC |
|----------|---------|--------|
| Context Window | 토큰 제약 (히스토리 20~40개) | 파인튜닝으로 긴 문맥 학습 |
| Hallucination | 존재하지 않는 아이템 추천 위험 | GT 기반 튜닝으로 환각 억제 |
| 형식 준수 | JSON 형식 미준수 가능 | 결과 형태 제어 능력 우수 |

### 📊 통계적 유의성 검정 (Statistical Significance Testing)

**논문 발표를 위해 모든 Baseline 비교에서 통계적 유의성 보고 필수:**

```python
from scipy import stats
import numpy as np

def paired_t_test(kitrec_scores: list, baseline_scores: list) -> dict:
    """Paired t-test for per-sample metric comparison"""
    t_stat, p_value = stats.ttest_rel(kitrec_scores, baseline_scores)
    effect_size = (np.mean(kitrec_scores) - np.mean(baseline_scores)) / \
                  np.std(np.concatenate([kitrec_scores, baseline_scores]))
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "effect_size_cohens_d": effect_size
    }
```

**적용 범위:**
- RQ1: KitREC-Full vs 3개 Ablation 모델
- RQ2: KitREC vs 모든 Baseline (CoNet, DTCDR, LLM4CDR, Vanilla)
- RQ3: User Type별 (1-core ~ 10-core) 성능 차이

### 🔒 Baseline Train/Test 데이터 분리 (Data Leakage 방지)

> **⚠️ 중요:** KitREC 데이터셋은 이미 Train/Test가 엄격히 분리되어 있습니다. Baseline 모델 학습 시 반드시 아래 데이터셋 매핑을 따라야 합니다.

**Baseline 학습용 데이터 (Training):**

| Baseline | Target Domain | Training Dataset | Samples |
|----------|---------------|------------------|---------|
| CoNet/DTCDR | Movies | `Younggooo/kitrec-dualft_movies-seta` | 12,000 |
| CoNet/DTCDR | Movies | `Younggooo/kitrec-dualft_movies-setb` | 12,000 |
| CoNet/DTCDR | Music | `Younggooo/kitrec-dualft_music-seta` | 12,000 |
| CoNet/DTCDR | Music | `Younggooo/kitrec-dualft_music-setb` | 12,000 |

**Baseline 평가용 데이터 (Evaluation):**

| Baseline | Target Domain | Test Dataset | Samples |
|----------|---------------|--------------|---------|
| All Baselines | Movies/Music | `Younggooo/kitrec-test-seta` | 30,000 |
| All Baselines | Movies/Music | `Younggooo/kitrec-test-setb` | 30,000 |

**올바른 Baseline 실험 프로세스:**

```python
# Step 1: Baseline 학습 (Train 데이터 사용)
train_loader = DataLoader("Younggooo/kitrec-dualft_movies-seta", hf_token=args.hf_token)
train_data = train_loader.load_test_data()
trainer.train(train_data)
trainer.save_checkpoint("checkpoints/conet_movies_seta.pt")

# Step 2: Baseline 평가 (Test 데이터 사용 - 별도 실행)
test_loader = DataLoader("Younggooo/kitrec-test-seta", hf_token=args.hf_token)
test_data = test_loader.load_test_data()
evaluator.evaluate(test_data)  # 반드시 Test 데이터만 사용
```

> **❌ 금지:** Test 데이터를 분할하여 학습에 사용하는 것은 Data Leakage 입니다.
> ```python
> # 잘못된 예시 (Data Leakage)
> train_samples = test_data[:-1000]  # 절대 금지!
> val_samples = test_data[-1000:]
> ```

---

## Research Questions

| RQ | 연구 질문 | KitREC 모델 | 비교 대상 |
|----|---------|------------|----------|
| **RQ1** | KitREC 구조의 효과성 검증 (Ablation Study) | 2×2 교차 검증 (아래 참조) | - |
| **RQ2** | CDR 방식의 효과성 검증 | DualFT-Movies/Music, SingleFT-Movies/Music | CoNet, DTCDR, LLM4CDR, Vanilla NIR |
| **RQ3** | Cold-start/Sparse 문제 해결 | DualFT Movies/Music (2/3/4-core), SingleFT Movies/Music(1 core)| LLCoNet, DTCDR, LLM4CDR, Vanilla NIRM4CDR |
| **RQ4** | Confidence/Rationale 검증 | **KitREC 전체 모델만** (Baseline 제외) | Confidence = MAE, RMSE / Rationale = PPL + GPT-4.1 (50개/모델) |

### RQ1: 2×2 Ablation Study Design

**실험 목적:** KitREC의 성능 향상이 단순한 파인튜닝 덕분인지, 아니면 설계된 Thinking Process(CoT) 덕분인지를 검증

**[학습 여부] × [추론 방식] 교차 검증:**

|  | Thinking (CoT) 적용 | Non-Thinking (Direct) 적용 |
|--|---------------------|---------------------------|
| **Fine-tuned (KitREC)** | ① KitREC-Full (제안 모델) | ② KitREC-Direct (Ablation) |
| **Base Model (Untuned)** | ③ Base-CoT (Strong Baseline) | ④ Base-Direct (Weak Baseline) |

### RQ1 비교 모델 상세 정의

**① KitREC-Full (Proposed Method)**
- Knowledge-Instruction Transfer 방식으로 파인튜닝
- 추론 시 `<think>` 태그를 통해 명시적인 추론 과정 후 추천 결과 생성
- 학습: Reasoning 데이터 포함 / 추론: Reasoning 생성 허용
- 추론 프롬프트 템플릿: Inference Prompt Template (Evaluation용) 사용
- **역할:** 본 연구에서 제안하는 최종 모델

**② KitREC-Direct (Ablation Model)**
- KitREC-Full과 동일하게 파인튜닝되었으나, Thinking 과정 제거
- 구현 방법:
  - (A) 학습 데이터에서 `<think>` 부분을 제거하고 학습한 별도 모델 (권장)
  - (B) 학습은 동일하되 추론 시 프롬프트로 Thinking 생성 억제
- 추론 프롬프트 템플릿: Inference Prompt Template (Evaluation용) 사용
- **역할:** "파인튜닝은 했지만 추론 과정이 없을 때"의 성능 측정 → Thinking Process의 기여도 증명

**③ Base-CoT (Zero-shot Chain-of-Thought)**
- 파인튜닝 되지 않은 Qwen3-14B-Instruct (Original) 모델
- 추론 프롬프트 템플릿: Inference Prompt Template (Evaluation용) 사용
- **역할:** 파인튜닝 없이 LLM 본연의 추론 능력만으로 어디까지 가능한지 측정 → Tuning의 효용성 증명

**④ Base-Direct (Vanilla Zero-shot)**
- 파인튜닝 되지 않은 Qwen3-14B-Instruct (Original) 모델
- 추론 프롬프트 템플릿: Inference Prompt Template (Evaluation용) 사용
- Reasoning 과정 없이 곧바로 추천 목록 생성
- **역할:** 가장 기본적인 Baseline (Lower Bound)

※ Note for Direct Models (②, ④): For KitREC-Direct and Base-Direct, use the same template but REMOVE the ## Reasoning Guidelines section entirely. The model should output the JSON directly without the <think> block.

---

## Model Architecture

- **Base Model**: Qwen/Qwen3-14B
- **Fine-tuning**: PEFT QLoRA (4-bit NF4 quantization)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Hyperparameters

| Parameter | DualFT (12K samples) | SingleFT (3K samples) |
|-----------|---------------------|----------------------|
| LoRA r | 32 | 24 |
| LoRA alpha | 64 | 48 |
| LoRA dropout | 0.08 | 0.15 |
| Learning Rate | 2e-4 | 6e-5 |
| **warmup_ratio** | **0.05** | **0.1** |
| Epochs | 3 | 6 |
| Batch Size | 4 (effective 32) | 2 (effective 32) |
| NEFTune Alpha | 5.0 | 3.0 |
| Weight Decay | 0.02 | 0.05 |
| Label Smoothing | 0.05 | 0.1 |

---

## Environment

- **Training**: RunPod A100/H100 80GB
- **Inference/Evaluation**: vLLM 기반, Nvidia 5090 (36GB VRAM)
- **Framework**: HuggingFace Transformers + PEFT + bitsandbytes
- **Python Packages**: torch>=2.2.0, transformers==4.57.3, peft==0.13.0

---

## Implementation Status (2025-12-07 Updated)

### ✅ Phase 1: Critical Issues (완료)

| 항목 | 상태 | 파일 | 설명 |
|------|------|------|------|
| Confidence Score 1-10 범위 | ✅ | `baselines/*/evaluator.py` | `sigmoid * 9 + 1` 정규화 |
| Train/Test 분리 | ✅ | `scripts/run_baseline_eval.py` | Data Leakage 방지 |
| User Type Mapping | ✅ | `baselines/dtcdr/evaluator.py` | RQ3 Cold-start 분석 |
| Candidate Set 검증 | ✅ | `baselines/base_evaluator.py` | 100개 후보 + GT 포함 검증 |

### ✅ Phase 2: High Priority (완료)

| 항목 | 상태 | 파일 | 설명 |
|------|------|------|------|
| Device mismatch | ✅ | `baselines/*/model.py` | `model_device` 사용 |
| Gradient Clipping | ✅ | `baselines/*/trainer.py` | `max_norm=1.0` |
| Statistical Significance | ✅ | `src/metrics/statistical_analysis.py` | Paired t-test, Holm correction |
| DIRECT_TEMPLATE 개선 | ✅ | `src/data/prompt_builder.py` | Confidence 가이드 추가 |

### ✅ Phase 3: Medium Priority (완료)

| 항목 | 상태 | 파일 | 설명 |
|------|------|------|------|
| LLM4CDR Target History | ✅ | `baselines/llm4cdr/prompts.py` | KitREC 구조 맞춤 + 차이점 문서화 |
| LR Scheduler | ✅ | `baselines/*/trainer.py` | `ReduceLROnPlateau` |
| Holm Correction | ✅ | `src/metrics/statistical_analysis.py` | Step-up enforcement 수정 |
| 무한루프 방지 | ✅ | `baselines/*/trainer.py` | Negative sampling 제한 |

### ✅ Phase 4: ExplainabilityMetrics (완료)

| 항목 | 상태 | 파일 | 설명 |
|------|------|------|------|
| MAE/RMSE 계산 | ✅ | `src/metrics/explainability_metrics.py` | Confidence vs GT Rating |
| Perplexity 계산 | ✅ | `src/metrics/explainability_metrics.py` | Rationale 품질 평가 |
| **GPT-4.1 Evaluation** | ✅ | `src/metrics/explainability_metrics.py` | User Type별 균등 추출, 모델당 50개 |

---

## GPT-4.1 Rationale Quality Evaluation (RQ4)

### 개요

User Type별 균등 추출(Stratified Sampling)로 모델당 50개 샘플을 GPT-4.1 API로 평가합니다.
- **샘플링 방식**: 10개 User Type × 5개/Type = 50개/모델
- **목적**: 비용 효율적이면서 User Type별 균형 잡힌 평가

> ⚠️ **RQ4 평가 대상**: KitREC 모델만 (Baseline 제외)
> - Baseline(CoNet, DTCDR, LLM4CDR)은 RQ4(Explainability)에서 **제외**
> - Baseline은 Confidence Score/MAE/RMSE 계산 불필요

### 평가 기준 (1-10점)

| 기준 | 영문 | 설명 |
|------|------|------|
| **논리성** | Logic | 추천 이유가 논리적인가? |
| **구체성** | Specificity | 구체적인 근거를 제시하는가? |
| **Cross-domain 연결성** | Cross-domain | Source→Target 연결이 명확한가? |
| **선호 반영** | Preference | 사용자 히스토리를 잘 반영했는가? |

### 사용법

```python
from src.metrics.explainability_metrics import GPTRationaleEvaluator

# GPT 평가기 초기화 (OPENAI_API_KEY 환경변수 필요)
# samples_per_type=5: 각 User Type당 5개 샘플 (총 50개/모델)
evaluator = GPTRationaleEvaluator(samples_per_type=5)

if evaluator.is_available():
    # User Type별 균등 추출 후 평가
    rationale_scores = evaluator.evaluate_batch(results)
    print(f"Logic: {rationale_scores['logic']:.2f}/10")
    print(f"Overall: {rationale_scores['overall']:.2f}/10")
    print(f"Sampling Stats: {rationale_scores['sampling_stats']}")
```

### 환경 설정

```bash
export OPENAI_API_KEY="your-api-key"
```

---

## User Type별 Cold-start Analysis (RQ3)

### 핵심 연구 포인트

> **⭐ KitREC의 핵심 강점:** 기존 CDR 모델들은 5-core 이상에서만 실험하지만, KitREC은 1-core (극한 Cold-start)에서도 성능이 나옵니다.

### Core Level별 평가

```python
from scripts.run_baseline_eval import build_user_type_mapping, print_user_type_metrics

# User Type 매핑 생성
user_type_mapping = build_user_type_mapping(original_samples, converter)

# Core Level별 평가
metrics_by_user_type = evaluator.evaluate_by_user_type(samples, user_type_mapping)

# 결과 출력 (1-core ~ 10+-core)
print_user_type_metrics(metrics_by_user_type, "DTCDR")
```

### Core Level 정의

| Core Level | 조건 | 특성 | 비교 의미 |
|------------|------|------|----------|
| 1-core | target=1 | 극한 Cold-start | KitREC만 실험 가능 |
| 2-core | target=2 | 심각한 Cold-start | 베이스라인 성능 급락 |
| 3-core | target=3 | 중간 Cold-start | 패턴 시작점 |
| 4-core | target=4 | 경미한 Cold-start | 5-core 진입 직전 |
| 5-9 core | target=5~9 | Warm-start | 일반적 실험 조건 |
| 10+-core | target≥10 | 풍부한 데이터 | 대부분 모델 성능 좋음 |

---

## Statistical Significance Testing

### 통계적 유의성 검정 구현

```python
from src.metrics.statistical_analysis import StatisticalAnalysis

stat = StatisticalAnalysis()

# 1. Paired t-test
result = stat.paired_t_test(kitrec_scores, baseline_scores)
print(f"p-value: {result['p_value']:.4f}")
print(f"Cohen's d: {result['effect_size_cohens_d']:.3f}")

# 2. 다중 비교 보정 (Holm-Bonferroni)
p_values = [result1['p_value'], result2['p_value'], result3['p_value']]
corrected = stat.apply_multiple_correction(p_values, method="holm")

# 3. 논문용 형식
formatted = stat.format_for_paper(result)  # e.g., "+0.123**"
```

### Effect Size 해석

| Cohen's d | 해석 |
|-----------|------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

---

## LLM4CDR vs KitREC 구현 차이점

### 주요 차이점 (논문 명시 필요)

| 항목 | LLM4CDR 원 논문 | KitREC 구현 |
|------|----------------|-------------|
| **Candidate Set** | 3 GT + 20~30 Neg (총 ~30개) | 1 GT + 99 Neg (총 100개) |
| **Target History** | 미사용 | 포함 (공정한 비교) |
| **Stage 1 Caching** | 동일 | 동일 |
| **3-Stage Pipeline** | 동일 | 동일 |

### 논문 작성 시 명시 사항

```
LLM4CDR was re-evaluated using the KitREC evaluation protocol
(1 GT + 99 negatives) for fair comparison. The original LLM4CDR
uses a smaller candidate set (3 GT + 20-30 negatives).
```

---

## Baseline 공통 인프라

### BaseEvaluator 클래스

모든 베이스라인 평가기가 공유하는 공통 기능:

```python
# baselines/base_evaluator.py
class BaseEvaluator:
    def validate_candidate_set(candidates, gt_id):
        """100개 후보 + GT 포함 검증"""

    def normalize_confidence(raw_score):
        """[0,1] → [1,10] 정규화: sigmoid * 9 + 1"""

    def calculate_metrics(gt_rank):
        """공통 메트릭 계산 (Hit@K, MRR, NDCG@K)"""
```

### 학습/평가 데이터 분리 체크리스트

- [x] Training: `kitrec-dualft_*` 데이터셋 사용
- [x] Evaluation: `kitrec-test-*` 데이터셋 사용
- [x] Data Leakage 방지: Test 데이터 분할하여 학습 금지
- [x] Checkpoint 저장/로드: Scheduler state 포함

---

## 버그 수정 기록 (2025-12-07)

### 수정된 이슈

| 이슈 | 파일 | 수정 내용 |
|------|------|----------|
| Path validation 괄호 | `baselines/*/trainer.py` | 연산자 우선순위 명확화 |
| 무한루프 가능성 | `baselines/*/trainer.py` | `max_attempts` 제한 추가 |
| Device mismatch | `baselines/*/model.py` | `model_device` 사용 |
| Holm correction | `statistical_analysis.py` | Step-up enforcement |
| Empty list 처리 | `baselines/dtcdr/trainer.py` | 안전한 max() 호출 |

### 추가 개선 사항 (2025-12-07 Updated)

| 이슈 ID | 파일 | 수정 내용 | 목적 |
|---------|------|----------|------|
| A-1 | `llm4cdr/evaluator.py` | BaseEvaluator 상속 추가 | 공통 인프라 활용 |
| A-1 | `llm4cdr/evaluator.py` | Candidate Set 검증 (100개 + GT 포함) | 실험 공정성 보장 |
| A-1 | `llm4cdr/evaluator.py` | Confidence Score 정규화 (1-10 범위) | 메트릭 일관성 |
| A-2 | `baselines/*/evaluator.py` | per-sample metrics 수집 | t-test 통계 검정 지원 |
| A-3 | `base_evaluator.py` | 검증 실패 시 logging 추가 | 능동적 오류 탐지 |

---

## per-sample Metrics 활용 (통계적 유의성 검정)

### 모든 Evaluator에서 per-sample 메트릭 수집

모든 베이스라인 평가기(CoNet, DTCDR, LLM4CDR)에서 개별 샘플별 메트릭을 수집합니다:

```python
# evaluator.evaluate() 반환값
result = evaluator.evaluate(samples)

# 집계된 메트릭
print(f"Hit@10: {result['hit@10']:.4f}")
print(f"NDCG@10: {result['ndcg@10']:.4f}")

# per-sample 메트릭 (통계 검정용)
per_sample = result["per_sample"]  # {"hit@10": [...], "ndcg@10": [...], "mrr": [...]}

# Paired t-test 실행
from src.metrics.statistical_analysis import StatisticalAnalysis
stat = StatisticalAnalysis()
comparison = stat.paired_t_test(kitrec_per_sample["hit@10"], baseline_per_sample["hit@10"])
print(f"p-value: {comparison['p_value']:.4f}")
```

### LLM4CDR 특수 기능

LLM4CDR 평가기는 추가로 다음 기능을 제공합니다:

```python
# 평가 통계 조회
stats = evaluator.get_statistics()
print(f"검증 실패 샘플 수: {stats['validation_failures']}")
print(f"무효 예측 수: {stats['total_invalid_predictions']}")
print(f"평가된 샘플 수: {stats['samples_evaluated']}")

# User Type별 per-sample 메트릭도 자동 수집
evaluator.per_sample_metrics  # {user_id: {metric: value}}
```
