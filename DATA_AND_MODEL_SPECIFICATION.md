# KitREC 데이터 및 모델 명세서

**작성일:** 2025-12-07
**버전:** 1.0
**목적:** KitREC 모델 및 Baseline의 학습/테스트 데이터 구조 종합 명세

---

## 1. 프로젝트 개요

KitREC (Knowledge-Instruction Transfer for Recommendation)은 Cross-domain 추천 시스템 연구 프로젝트입니다.

```
Source Domain: Books
Target Domains: Movies & TV, Music

핵심 연구 목표:
- Cold-start 문제 해결 (1-core ~ 4-core 사용자)
- LLM 기반 지식 전이 (Cross-domain Knowledge Transfer)
- 설명 가능한 추천 (Explainable Recommendation)
```

---

## 2. User Type 정의 (10개)

### 2.1 전체 User Type 테이블

| # | User Type | 조건 | Target Count | Domain Exclusivity | Core Level | 샘플 수 | 모델 | 역할 |
|:-:|-----------|------|:------------:|:------------------:|:----------:|:-------:|:----:|------|
| 1 | **Overlapping (Books+Movies)** | books≥5 AND movies≥5 | ≥5 | N/A | 5+ core | 3,000 | DualFT-Movies | Warm-start |
| 2 | **Overlapping (Books+Music)** | books≥5 AND music≥5 | ≥5 | N/A | 5+ core | 3,000 | DualFT-Music | Warm-start |
| 3 | **Source-only (Movies)** | books≥5 AND movies=1 | 1 | music=0 | 1-core | 3,000 | SingleFT-Movies | Extreme Cold-start |
| 4 | **Source-only (Music)** | books≥5 AND music=1 | 1 | movies=0 | 1-core | 3,000 | SingleFT-Music | Extreme Cold-start |
| 5 | **Cold-start 2-core (Movies)** | books≥5 AND movies=2 | 2 | music=0 | 2-core | 3,000 | DualFT-Movies | Severe Cold-start |
| 6 | **Cold-start 2-core (Music)** | books≥5 AND music=2 | 2 | movies=0 | 2-core | 3,000 | DualFT-Music | Severe Cold-start |
| 7 | **Cold-start 3-core (Movies)** | books≥5 AND movies=3 | 3 | music=0 | 3-core | 3,000 | DualFT-Movies | Moderate Cold-start |
| 8 | **Cold-start 3-core (Music)** | books≥5 AND music=3 | 3 | movies=0 | 3-core | 3,000 | DualFT-Music | Moderate Cold-start |
| 9 | **Cold-start 4-core (Movies)** | books≥5 AND movies=4 | 4 | music=0 | 4-core | 3,000 | DualFT-Movies | Mild Cold-start |
| 10 | **Cold-start 4-core (Music)** | books≥5 AND music=4 | 4 | movies=0 | 4-core | 3,000 | DualFT-Music | Mild Cold-start |

### 2.2 Core Level 요약

| Core Level | Target 상호작용 수 | 특성 | 연구 의의 |
|:----------:|:-----------------:|------|----------|
| **1-core** | 1개 | 극한 Cold-start | KitREC만 실험 가능 (기존 연구 미수행) |
| **2-core** | 2개 | 심각한 Cold-start | Baseline 성능 급락 구간 |
| **3-core** | 3개 | 중간 Cold-start | Cross-domain 패턴 시작 |
| **4-core** | 4개 | 경미한 Cold-start | 5-core 진입 직전 |
| **5+ core** | 5개 이상 | Warm-start | 일반적 실험 조건 |

### 2.3 모델 선택 로직

```
┌─────────────────────────────────────────────────────────────┐
│  User Type 판별 → 적합한 KitREC 모델 자동 선택              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Target 상호작용 수 = 1 (1-core)                            │
│  └─→ SingleFT 모델                                          │
│      • 극한 Cold-start 전용                                 │
│      • Source 도메인 의존도 최상                            │
│      • Overfitting 방지 설계                                │
│                                                             │
│  Target 상호작용 수 = 2~4 (2-4 core)                        │
│  └─→ DualFT 모델                                            │
│      • Cold-start 사용자 대응                               │
│      • Cross-domain 패턴 학습 가능                          │
│                                                             │
│  Target 상호작용 수 ≥ 5 (5+ core)                           │
│  └─→ DualFT 모델                                            │
│      • Warm-start 사용자                                    │
│      • 양방향 지식 전이 극대화                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. KitREC 모델 (8개)

### 3.1 모델 구성

| # | 모델명 | Target Domain | User Type | 학습 샘플 | 특징 |
|:-:|--------|:-------------:|-----------|:---------:|------|
| 1 | **DualFT-Movies-SetA** | Movies | Overlapping + Cold-start 2/3/4 | 12,000 | Hard Negatives |
| 2 | **DualFT-Movies-SetB** | Movies | Overlapping + Cold-start 2/3/4 | 12,000 | Random Negatives |
| 3 | **DualFT-Music-SetA** | Music | Overlapping + Cold-start 2/3/4 | 12,000 | Hard Negatives |
| 4 | **DualFT-Music-SetB** | Music | Overlapping + Cold-start 2/3/4 | 12,000 | Random Negatives |
| 5 | **SingleFT-Movies-SetA** | Movies | Source-only (1-core) | 3,000 | Hard Negatives |
| 6 | **SingleFT-Movies-SetB** | Movies | Source-only (1-core) | 3,000 | Random Negatives |
| 7 | **SingleFT-Music-SetA** | Music | Source-only (1-core) | 3,000 | Hard Negatives |
| 8 | **SingleFT-Music-SetB** | Music | Source-only (1-core) | 3,000 | Random Negatives |

### 3.2 DualFT vs SingleFT 비교

| 항목 | DualFT | SingleFT |
|------|--------|----------|
| **대상 User Type** | Overlapping + Cold-start (2-4 core) | Source-only (1-core) |
| **학습 샘플 수** | 12,000 | 3,000 |
| **Target History 사용** | O (2개 이상) | X (1개만, 과적합 방지) |
| **지식 전이 방향** | 양방향 (Dual) | 단방향 (Single) |
| **설계 목적** | Cross-domain 패턴 학습 | Source 의존 극대화 |

---

## 4. 데이터셋 구조 (HuggingFace Hub)

### 4.1 전체 데이터셋 목록 (12개)

#### Training Datasets (8개)

| # | Dataset Name | 샘플 수 | Target Domain | Candidate Set | 용도 |
|:-:|--------------|:-------:|:-------------:|:-------------:|------|
| 1 | `Younggooo/kitrec-dualft_movies-seta` | 12,000 | Movies | Set A (Hard) | DualFT-Movies 학습 |
| 2 | `Younggooo/kitrec-dualft_movies-setb` | 12,000 | Movies | Set B (Random) | DualFT-Movies 학습 |
| 3 | `Younggooo/kitrec-dualft_music-seta` | 12,000 | Music | Set A (Hard) | DualFT-Music 학습 |
| 4 | `Younggooo/kitrec-dualft_music-setb` | 12,000 | Music | Set B (Random) | DualFT-Music 학습 |
| 5 | `Younggooo/kitrec-singleft_movies-seta` | 3,000 | Movies | Set A (Hard) | SingleFT-Movies 학습 |
| 6 | `Younggooo/kitrec-singleft_movies-setb` | 3,000 | Movies | Set B (Random) | SingleFT-Movies 학습 |
| 7 | `Younggooo/kitrec-singleft_music-seta` | 3,000 | Music | Set A (Hard) | SingleFT-Music 학습 |
| 8 | `Younggooo/kitrec-singleft_music-setb` | 3,000 | Music | Set B (Random) | SingleFT-Music 학습 |

#### Validation Datasets (2개)

| # | Dataset Name | 샘플 수 | 용도 |
|:-:|--------------|:-------:|------|
| 9 | `Younggooo/kitrec-val-seta` | 12,000 | DPO/GRPO 검증 (Hard) |
| 10 | `Younggooo/kitrec-val-setb` | 12,000 | DPO/GRPO 검증 (Random) |

#### Test Datasets (2개)

| # | Dataset Name | 샘플 수 | 용도 |
|:-:|--------------|:-------:|------|
| 11 | `Younggooo/kitrec-test-seta` | 30,000 | 최종 평가 (Hard Negatives) |
| 12 | `Younggooo/kitrec-test-setb` | 30,000 | 최종 평가 (Random Negatives) |

### 4.2 Set A vs Set B 비교

| 항목 | Set A | Set B |
|------|-------|-------|
| **Candidate 구성** | Hard Negatives (유사 아이템) | Random Negatives |
| **난이도** | 높음 | 보통 |
| **용도** | KitREC 성능 한계 테스트 | 공정한 Baseline 비교 |
| **Negative 샘플링** | 의미적으로 유사한 아이템 | 완전 무작위 |

### 4.3 데이터셋 통계

| 카테고리 | 데이터셋 수 | 총 샘플 수 |
|----------|:-----------:|:----------:|
| Training | 8 | 60,000 |
| Validation | 2 | 24,000 |
| Test | 2 | 60,000 |
| **합계** | **12** | **144,000** |

---

## 5. Baseline 모델 (4개)

### 5.1 Baseline 개요

| # | Baseline | 출처 | 유형 | 학습 필요 | 입력 형식 |
|:-:|----------|------|------|:--------:|----------|
| 1 | **CoNet** | CIKM 2018 | Deep Learning CDR | **필요** | ID 시퀀스 |
| 2 | **DTCDR** | CIKM 2019 | Deep Learning CDR | **필요** | ID 시퀀스 |
| 3 | **LLM4CDR** | RecSys/WWW 2025 | LLM 3-Stage | 불필요 | 텍스트 프롬프트 |
| 4 | **Vanilla Zero-shot** | - | LLM Single-Stage | 불필요 | 텍스트 프롬프트 |

### 5.2 Baseline 상세 비교

| 항목 | CoNet | DTCDR | LLM4CDR | Vanilla |
|------|-------|-------|---------|---------|
| **학습 데이터** | `dualft_*` (12K) | `dualft_*` (12K) | 없음 | 없음 |
| **테스트 데이터** | `test-set*` (30K) | `test-set*` (30K) | `test-set*` (30K) | `test-set*` (30K) |
| **Candidate Set** | 100개 (1 GT + 99 Neg) | 100개 (1 GT + 99 Neg) | 100개 (1 GT + 99 Neg) | 100개 (1 GT + 99 Neg) |
| **User History** | ID 시퀀스 변환 | ID 시퀀스 변환 | 텍스트 유지 | 텍스트 유지 |
| **핵심 메커니즘** | Cross-Stitch Units | Orthogonal Mapping | 3-Stage Pipeline | Single Prompt |

### 5.3 LLM4CDR 3-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM4CDR 3-Stage Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: Domain Gap Analysis (캐싱 가능)                   │
│  └─ "Books와 Movies 도메인 간 의미적 연결 분석"             │
│                         ↓                                   │
│  Stage 2: User Interest Reasoning                           │
│  └─ "사용자의 Source 도메인 선호도 → Target 선호 추론"      │
│                         ↓                                   │
│  Stage 3: Candidate Re-ranking                              │
│  └─ "100개 후보 중 Top-10 선정 및 Rationale 생성"           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

vs Vanilla Zero-shot:
┌─────────────────────────────────────────────────────────────┐
│  Single Prompt: "User History 기반으로 Top-10 추천"         │
│  └─ Domain Analysis 없음, User Profile 생성 없음            │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 LLM4CDR 원본 논문 vs KitREC 구현 차이

| 항목 | 원본 논문 | KitREC 구현 |
|------|----------|-------------|
| **Candidate Set** | 3 GT + 20-30 Neg (~30개) | 1 GT + 99 Neg (100개) |
| **Target History** | 미사용 | 사용 (공정 비교) |
| **난이도** | 쉬움 | 어려움 |
| **논문 표기** | "LLM4CDR was re-evaluated using KitREC protocol" |

---

## 6. 학습/테스트 데이터 매핑

### 6.1 KitREC 모델 매핑

```
┌─────────────────────────────────────────────────────────────┐
│                   KitREC 모델 데이터 흐름                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DualFT-Movies-SetA:                                        │
│  ├─ 학습: kitrec-dualft_movies-seta (12,000)               │
│  └─ 평가: kitrec-test-seta (30,000 중 Movies 필터링)       │
│                                                             │
│  DualFT-Movies-SetB:                                        │
│  ├─ 학습: kitrec-dualft_movies-setb (12,000)               │
│  └─ 평가: kitrec-test-setb (30,000 중 Movies 필터링)       │
│                                                             │
│  SingleFT-Movies-SetA:                                      │
│  ├─ 학습: kitrec-singleft_movies-seta (3,000)              │
│  └─ 평가: kitrec-test-seta (30,000 중 1-core 필터링)       │
│                                                             │
│  (Music 모델도 동일한 패턴)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Baseline 모델 매핑

```
┌─────────────────────────────────────────────────────────────┐
│                  Baseline 모델 데이터 흐름                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CoNet / DTCDR (학습 필요):                                 │
│  ├─ 학습: kitrec-dualft_{domain}-{set} (12,000)            │
│  │        └─ ID 시퀀스로 변환 후 학습                       │
│  └─ 평가: kitrec-test-{set} (30,000)                       │
│           └─ 동일 ID 매핑으로 평가                          │
│                                                             │
│  LLM4CDR / Vanilla (학습 불필요):                           │
│  ├─ 학습: 없음 (Zero-shot)                                 │
│  └─ 평가: kitrec-test-{set} (30,000)                       │
│           └─ 텍스트 프롬프트 직접 사용                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Data Leakage 방지 (핵심)

```python
# ✅ 올바른 방법: Train/Test 완전 분리
train_data = DataLoader("Younggooo/kitrec-dualft_movies-seta").load()  # 학습용
test_data = DataLoader("Younggooo/kitrec-test-seta").load()            # 평가용

# ❌ 금지: Test 데이터를 분할하여 학습에 사용
test_data = DataLoader("Younggooo/kitrec-test-seta").load()
train_samples = test_data[:-1000]  # Data Leakage!
```

---

## 7. 평가 메트릭

### 7.1 Ranking Metrics

| Metric | 범위 | 계산 방식 | 해석 |
|--------|:----:|----------|------|
| **Hit@1** | 0~1 | GT가 1위인지 | 정확히 1위 예측 |
| **Hit@5** | 0~1 | GT가 Top-5 내 | 상위 5개 추천 품질 |
| **Hit@10** | 0~1 | GT가 Top-10 내 | 논문 표준 지표 |
| **MRR** | 0~1 | 1/rank | 1위=1.0, 2위=0.5, 10위=0.1 |
| **NDCG@5** | 0~1 | DCG@5/IDCG@5 | 위치 가중 Top-5 품질 |
| **NDCG@10** | 0~1 | DCG@10/IDCG@10 | 위치 가중 Top-10 품질 |

### 7.2 Explainability Metrics (RQ4)

> ⚠️ **RQ4 평가 대상**: KitREC 모델만 (Baseline 제외)
> - Baseline(CoNet, DTCDR, LLM4CDR)은 Confidence Score/MAE/RMSE 계산 불필요

| Metric | 범위 | 계산 방식 | 용도 |
|--------|:----:|----------|------|
| **MAE** | 0~5 | mean(\|pred - actual\|) | Confidence vs GT Rating |
| **RMSE** | 0~5 | sqrt(mean((pred-actual)²)) | 큰 오차에 민감 |
| **Perplexity** | 1~∞ | exp(avg_loss) | Rationale 언어 품질 |
| **GPT-4.1 평가** | 1~10 | Stratified Sampling (50개/모델) | Rationale 의미적 품질 |

**GPT-4.1 샘플링 전략:**
- 10개 User Type × 5개/Type = 50개/모델
- User Type별 균등 추출로 비용 효율적이면서 균형 잡힌 평가

### 7.3 Statistical Analysis

| 검정 방법 | 용도 | 출력 |
|----------|------|------|
| **Paired t-test** | 두 모델 per-sample 비교 | t-stat, p-value, Cohen's d |
| **Holm Correction** | 다중 비교 보정 | adjusted p-values |
| **Bootstrap CI** | 신뢰구간 추정 | mean, ci_lower, ci_upper |

### 7.4 논문용 표기

```
* p < 0.05   (95% 신뢰도)
** p < 0.01  (99% 신뢰도)
*** p < 0.001 (99.9% 신뢰도)

Cohen's d 해석:
|d| < 0.2:  negligible
0.2 ≤ |d| < 0.5: small
0.5 ≤ |d| < 0.8: medium
|d| ≥ 0.8: large
```

---

## 8. Confidence Score 정규화

### 8.1 KitREC 모델

```
모델 출력: confidence_score 1-10 (Template 명시)
GT Rating: 1-5

MAE/RMSE 계산 시:
normalized_confidence = confidence_score / 2
→ 1-10 → 0.5-5.0 범위로 변환
```

### 8.2 Baseline 모델

```
모델 출력: raw_score (범위 무제한)

정규화:
sigmoid = 1 / (1 + exp(-raw_score))
confidence = sigmoid * 9 + 1
→ (-∞, +∞) → (0, 1) → (1, 10) 범위로 변환
```

---

## 9. 실험 실행 명령어

### 9.1 KitREC 평가

```bash
# 전체 평가
python scripts/run_kitrec_eval.py \
    --model_name dualft_movies_seta \
    --dataset Younggooo/kitrec-test-seta \
    --output_dir results/kitrec

# 테스트 (샘플 제한)
python scripts/run_kitrec_eval.py \
    --model_name dualft_movies_seta \
    --max_samples 100
```

### 9.2 Baseline 평가

```bash
# CoNet 학습 + 평가
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain movies \
    --candidate_set seta \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_movies-seta

# LLM4CDR 평가 (학습 불필요)
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain movies \
    --candidate_set seta
```

### 9.3 Ablation Study (RQ1)

```bash
python scripts/run_ablation_study.py \
    --target_domain movies \
    --candidate_set seta \
    --output_dir results/ablation
```

---

## 10. RQ별 실험 매핑

| RQ | 연구 질문 | 비교 대상 | 주요 메트릭 |
|----|----------|----------|------------|
| **RQ1** | Ablation Study (2×2) | KitREC-Full vs Direct vs Base-CoT vs Base-Direct | Hit@10, NDCG@10 |
| **RQ2** | Baseline 비교 | KitREC vs CoNet vs DTCDR vs LLM4CDR vs Vanilla | Hit@1/5/10, MRR, NDCG |
| **RQ3** | Cold-start 분석 | 1-core ~ 10+-core별 성능 | User Type별 Hit@10 |
| **RQ4** | Explainability (**KitREC만**) | Confidence vs GT Rating, Rationale 품질 | MAE, RMSE, GPT-4.1 (50개/모델) |

---

## 11. 핵심 파일 위치

| 카테고리 | 파일 | 경로 |
|----------|------|------|
| **문서** | 프로젝트 가이드 | `CLAUDE.md` |
| | 작업 계획서 | `detail_task_plan.md` |
| | 본 문서 | `DATA_AND_MODEL_SPECIFICATION.md` |
| **데이터** | 데이터 로더 | `src/data/data_loader.py` |
| | 프롬프트 빌더 | `src/data/prompt_builder.py` |
| **메트릭** | Ranking | `src/metrics/ranking_metrics.py` |
| | Explainability | `src/metrics/explainability_metrics.py` |
| | Statistical | `src/metrics/statistical_analysis.py` |
| **Baseline** | 공통 인프라 | `baselines/base_evaluator.py` |
| | CoNet | `baselines/conet/` |
| | DTCDR | `baselines/dtcdr/` |
| | LLM4CDR | `baselines/llm4cdr/` |
| **실행** | KitREC 평가 | `scripts/run_kitrec_eval.py` |
| | Baseline 평가 | `scripts/run_baseline_eval.py` |
| | Ablation | `scripts/run_ablation_study.py` |

---

## 12. 요약 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KitREC 실험 구조 요약                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [10 User Types]                    [8 KitREC Models]                   │
│  ├─ Overlapping (Movies/Music)  →   DualFT-Movies/Music (×2 Set)       │
│  ├─ Source-only (Movies/Music)  →   SingleFT-Movies/Music (×2 Set)     │
│  └─ Cold-start 2/3/4-core       →   DualFT (재사용)                     │
│                                                                         │
│  [12 Datasets]                                                          │
│  ├─ Training: 8개 (dualft×4 + singleft×4) = 60,000 samples             │
│  ├─ Validation: 2개 (DPO/GRPO) = 24,000 samples                        │
│  └─ Test: 2개 (Set A + Set B) = 60,000 samples                         │
│                                                                         │
│  [4 Baselines]                                                          │
│  ├─ CoNet (학습 필요) ─────────┐                                        │
│  ├─ DTCDR (학습 필요) ─────────┤→ 동일 Candidate Set (100개)            │
│  ├─ LLM4CDR (Zero-shot) ───────┤→ 동일 Test Data (30,000)              │
│  └─ Vanilla (Zero-shot) ───────┘                                       │
│                                                                         │
│  [4 Research Questions]                                                 │
│  ├─ RQ1: Ablation (2×2 설계)                                           │
│  ├─ RQ2: Baseline 비교 (통계적 유의성)                                  │
│  ├─ RQ3: Cold-start 분석 (1-core ~ 10+-core)                           │
│  └─ RQ4: Explainability (KitREC만, GPT-4.1 50개/모델)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**작성 완료: 2025-12-07**
