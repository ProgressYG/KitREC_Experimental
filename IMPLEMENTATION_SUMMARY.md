# KitREC Implementation Summary

**최종 업데이트:** 2025-12-07
**목적:** 논문 작성 및 실험 재현을 위한 구현 요약

---

## 1. 프로젝트 구조

```
Experimental_test/
├── CLAUDE.md                    # 프로젝트 가이드 (상세)
├── detail_task_plan.md          # 작업 계획서 (상세)
├── IMPLEMENTATION_SUMMARY.md    # 이 문서 (요약)
│
├── src/
│   ├── data/                    # 데이터 로딩 및 프롬프트 생성
│   ├── inference/               # vLLM 추론 및 출력 파싱
│   └── metrics/                 # 평가 지표 (Ranking, Explainability, Stats)
│
├── baselines/
│   ├── base_evaluator.py        # 공통 평가 인프라
│   ├── conet/                   # CoNet (CIKM 2018)
│   ├── dtcdr/                   # DTCDR (CIKM 2019)
│   └── llm4cdr/                 # LLM4CDR (RecSys/WWW 2025)
│
└── scripts/                     # 실행 스크립트
```

---

## 2. 확정 베이스라인

| Model | 출처 | 유형 | 구현 위치 |
|-------|------|------|----------|
| **CoNet** | CIKM 2018 | Deep Learning CDR | `baselines/conet/` |
| **DTCDR** | CIKM 2019 | Deep Learning CDR | `baselines/dtcdr/` |
| **LLM4CDR** | RecSys/WWW 2025 | LLM 3-Stage | `baselines/llm4cdr/` |
| **Vanilla Zero-shot** | - | LLM Lower Bound | Base model direct |

> ❌ SSCDR은 베이스라인 아님 (제외)

---

## 3. 핵심 구현 사항

### 3.1 Confidence Score 정규화

```python
# KitREC 출력: 1-10
# Baseline 출력: sigmoid(raw_score) ∈ [0,1]
# 변환: sigmoid * 9 + 1 → [1,10]
confidence = (1 / (1 + np.exp(-raw_score))) * 9 + 1
```

### 3.2 Train/Test 데이터 분리

| 용도 | 데이터셋 | 샘플 수 |
|------|---------|---------|
| Training | `kitrec-dualft_*` | 12,000 |
| Evaluation | `kitrec-test-*` | 30,000 |

> ⚠️ Test 데이터 분할하여 학습 금지 (Data Leakage)

### 3.3 Candidate Set 검증

```python
# 모든 모델 동일 조건
assert len(candidates) == 100  # 1 GT + 99 Negatives
assert gt_id in candidates     # GT 반드시 포함
```

### 3.4 Device Mismatch 방지

```python
# 모델의 실제 device 사용
model_device = next(self.parameters()).device
user_ids = torch.tensor([user_id], device=model_device)
```

---

## 4. RQ별 구현

### RQ1: Ablation Study (2×2)

| 모델 | 학습 | 추론 |
|------|------|------|
| KitREC-Full | Fine-tuned | Thinking |
| KitREC-Direct | Fine-tuned | No Thinking |
| Base-CoT | Untuned | Thinking |
| Base-Direct | Untuned | No Thinking |

### RQ2: Baseline 비교

```python
from src.metrics.statistical_analysis import StatisticalAnalysis

# Paired t-test
result = stat.paired_t_test(kitrec_scores, baseline_scores)

# 다중 비교 보정 (Holm-Bonferroni)
corrected = stat.apply_multiple_correction(p_values, method="holm")
```

### RQ3: Cold-start Analysis

```python
# User Type 매핑
user_type_mapping = build_user_type_mapping(samples, converter)

# Core Level별 평가
metrics = evaluator.evaluate_by_user_type(samples, user_type_mapping)
```

| Core Level | 특성 | 의미 |
|------------|------|------|
| 1-core | 극한 Cold-start | KitREC 강점 |
| 2-4 core | Cold-start | 베이스라인 성능 급락 |
| 5+ core | Warm-start | 일반 조건 |

### RQ4: Explainability

```python
# MAE/RMSE: Confidence vs GT Rating
explainability = ExplainabilityMetrics()
mae = explainability.mae(predictions, gt_ratings)

# GPT-4.1 Rationale 평가 (10% 샘플)
gpt_eval = GPTRationaleEvaluator(sample_ratio=0.1)
scores = gpt_eval.evaluate_batch(results)
```

**GPT 평가 기준:**
1. Logic (논리성)
2. Specificity (구체성)
3. Cross-domain (연결성)
4. Preference (선호 반영)

---

## 5. LLM4CDR 구현 차이점

| 항목 | 원 논문 | KitREC 구현 |
|------|---------|-------------|
| Candidate Set | 3 GT + 20-30 Neg | 1 GT + 99 Neg |
| Target History | 미사용 | 포함 |
| 난이도 | 쉬움 | 어려움 |

> 논문에 명시: "LLM4CDR was re-evaluated using KitREC protocol"

---

## 6. 통계적 유의성

### 검정 방법

```python
# Paired t-test
t_stat, p_value = stats.ttest_rel(a, b)

# Effect Size (Cohen's d)
effect_size = np.mean(diff) / np.std(diff)
```

### Effect Size 해석

| Cohen's d | 해석 |
|-----------|------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### 논문 표기

```
* p < 0.05
** p < 0.01
*** p < 0.001
```

---

## 7. 실행 명령어

### Baseline 학습

```bash
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain movies \
    --candidate_set seta \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_movies-seta
```

### Baseline 평가

```bash
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain movies \
    --candidate_set seta \
    --baseline_checkpoint checkpoints/conet_movies_seta.pt
```

### KitREC 평가

```bash
python scripts/run_kitrec_eval.py \
    --model_name kitrec-dualft-movies-seta \
    --set seta
```

---

## 8. 파일 체크리스트

### 필수 구현 파일

- [x] `baselines/base_evaluator.py` - 공통 인프라
- [x] `baselines/conet/model.py` - CoNet 모델
- [x] `baselines/conet/trainer.py` - CoNet 학습 (LR Scheduler 포함)
- [x] `baselines/conet/evaluator.py` - CoNet 평가
- [x] `baselines/dtcdr/model.py` - DTCDR 모델
- [x] `baselines/dtcdr/trainer.py` - DTCDR 학습 (LR Scheduler 포함)
- [x] `baselines/dtcdr/evaluator.py` - DTCDR 평가 (User Type 분석 포함)
- [x] `baselines/llm4cdr/prompts.py` - 3-Stage 프롬프트
- [x] `baselines/llm4cdr/evaluator.py` - LLM4CDR 평가
- [x] `src/metrics/explainability_metrics.py` - MAE/RMSE/Perplexity/GPT
- [x] `src/metrics/statistical_analysis.py` - t-test/Holm correction

### 버그 수정 완료

- [x] Path validation 연산자 우선순위
- [x] Negative sampling 무한루프 방지
- [x] Device mismatch 수정
- [x] Holm correction step-up enforcement
- [x] Empty list 안전 처리

### 추가 개선 완료 (2025-12-07 Updated)

- [x] **A-1**: LLM4CDR evaluator → BaseEvaluator 상속 (공통 인프라 활용)
- [x] **A-1**: LLM4CDR Candidate Set 검증 추가 (100개 + GT 포함)
- [x] **A-1**: LLM4CDR Confidence Score 정규화 (1-10 범위)
- [x] **A-2**: 모든 evaluator에 per-sample metrics 수집 추가 (t-test 지원)
- [x] **A-3**: base_evaluator.py 검증 실패 시 logging 추가 (능동적 오류 탐지)

---

## 9. 환경 설정

### 필수 환경변수

```bash
export HF_TOKEN="your-huggingface-token"
export OPENAI_API_KEY="your-openai-key"  # GPT-4.1 평가용
```

### Python 패키지

```
torch>=2.2.0
transformers==4.57.3
vllm
openai
scipy
tqdm
```

---

## 10. Quick Reference (논문 작성용)

### 평가 메트릭

| Metric | 범위 | 설명 |
|--------|------|------|
| Hit@K | 0-1 | Top-K에 GT 포함 여부 |
| MRR | 0-1 | 1/rank 평균 |
| NDCG@K | 0-1 | 1/log2(rank+1) 정규화 |
| MAE | 0-5 | 절대 오차 평균 |
| RMSE | 0-5 | 제곱근 평균 오차 |

### 핵심 주장 (Paper Claims)

1. **Fair Comparison**: 모든 모델 동일 Candidate Set (100개)
2. **Cold-start Strength**: 1-core에서도 성능 발휘
3. **Explainability**: GPT-4.1 평가로 rationale 품질 검증
4. **Statistical Rigor**: Paired t-test + Holm correction

---

**작성 완료: 2025-12-07**
