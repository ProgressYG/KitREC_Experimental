# KitREC 석사 논문 실험평가 전문가 검증 리포트

**검증일:** 2025-12-07  
**검증자:** AI Research Expert  
**버전:** 1.0

---

## 📋 종합 평가 요약

| 영역 | 평가 | 점수 |
|------|------|------|
| **연구 설계** | 우수 | ⭐⭐⭐⭐⭐ |
| **실험 방법론** | 우수 | ⭐⭐⭐⭐⭐ |
| **통계적 엄밀성** | 우수 | ⭐⭐⭐⭐⭐ |
| **코드 품질** | 양호 | ⭐⭐⭐⭐☆ |
| **재현 가능성** | 양호 | ⭐⭐⭐⭐☆ |
| **문서화** | 우수 | ⭐⭐⭐⭐⭐ |

### 총평
> KitREC 프로젝트는 **학술적으로 충분히 엄밀한 실험 설계**를 갖추고 있습니다. Research Questions이 명확하고, Baseline 비교가 공정하며, 통계적 검정이 적절합니다. 몇 가지 개선 권고사항을 반영하면 **논문 게재 수준의 완성도**에 도달할 수 있습니다.

---

## ✅ 1. 연구 설계 (Research Design)

### 1.1 Research Questions 평가

| RQ | 질문 | 평가 | 설명 |
|----|------|------|------|
| **RQ1** | Ablation Study (2×2) | ⭐⭐⭐⭐⭐ | 체계적인 2×2 factorial design으로 Thinking과 Fine-tuning의 독립적 기여도 분리 |
| **RQ2** | Baseline 비교 | ⭐⭐⭐⭐⭐ | 3개 대표적 CDR 모델 + Vanilla 포함. 공정한 비교 조건 명시 |
| **RQ3** | Cold-start 분석 | ⭐⭐⭐⭐⭐ | 1-core부터 10+-core까지 세분화. **기존 연구(5-core+)보다 확장된 범위** |
| **RQ4** | Explainability | ⭐⭐⭐⭐☆ | MAE/RMSE + GPT-4.1 평가. Stratified sampling으로 비용 효율적 |

**강점:**
- RQ1의 2×2 설계는 ML 연구에서 권장되는 ablation 방법론
- RQ3에서 기존 연구들이 5-core 이상만 다룬 것을 1-core까지 확장한 것은 **논문의 주요 contribution**
- RQ4에서 GPT-4.1을 Human evaluation의 대리(proxy)로 사용한 것은 최신 트렌드 반영

**권고사항:**
- ⚠️ RQ4에서 Human evaluation과의 상관관계 검증을 추가하면 더 강력한 주장 가능

### 1.2 실험 변수 통제

```
✅ 통제된 변수:
- Candidate Set: 100개 (1 GT + 99 Negatives) - 모든 모델 동일
- User History: 동일 시점 데이터
- Train/Test 분리: HuggingFace Hub에서 사전 분리
- Random Seed: 42 (재현성 보장)

✅ 독립 변수:
- Fine-tuning 여부 (Tuned vs Untuned)
- Thinking Process 여부 (CoT vs Direct)
- User Type (1-core ~ 10+-core)
- Domain (Movies vs Music)

✅ 종속 변수:
- Ranking Metrics: Hit@K, MRR, NDCG@K
- Explainability: MAE, RMSE, PPL, GPT Score
```

---

## ✅ 2. 실험 방법론 (Methodology)

### 2.1 Baseline 비교 공정성

| 체크 항목 | 상태 | 구현 위치 |
|----------|------|----------|
| 동일 Candidate Set (100개) | ✅ | `BaseEvaluator.validate_candidate_set()` |
| GT 포함 검증 | ✅ | `gt_id in candidates` 체크 |
| 동일 User History | ✅ | CLAUDE.md 명시 |
| Train/Test 분리 | ✅ | 별도 HuggingFace repo |
| LLM4CDR 프로토콜 정렬 | ✅ | 논문 명시 문구 준비됨 |

**매우 우수함**: 
> "All baseline models were evaluated on the identical candidate set (1 GT + 99 negatives)"라는 문구와 함께 `BaseEvaluator` 클래스에서 강제 검증 구현

### 2.2 데이터 Leakage 방지

```python
# ✅ 올바른 구현 확인됨 (CLAUDE.md)
# Training: kitrec-dualft_* (별도 repo)
# Test: kitrec-test-* (별도 repo)

# ✅ 코드에서 강제
train_dataset = "Younggooo/kitrec-dualft_movies-seta"  # Training용
test_dataset = "Younggooo/kitrec-test-seta"           # Evaluation용
```

### 2.3 Confidence Score 처리

| 항목 | 구현 상태 | 평가 |
|------|----------|------|
| KitREC 정규화 (÷2) | ✅ `ExplainabilityMetrics.normalize_confidence()` | 정확함 |
| Baseline 정규화 (sigmoid*9+1) | ✅ `BaseEvaluator.normalize_confidence()` | 정확함 |
| 0 값 처리 | ⚠️ 명시적 처리 없음 | 권고: 파싱 오류로 처리 로직 추가 |

**발견된 이슈:**
- `confidence_score = 0`인 경우 파싱 오류로 처리해야 하나, 현재 코드에서는 기본값 5.0으로 대체됨

```python
# 현재 코드 (explainability_metrics.py:249)
confidences = [
    float(p.get("confidence_score", 5.0))  # 기본값 5.0
    for p in predictions
]

# 권고: 0 값 명시적 처리
confidences = [
    float(p.get("confidence_score", 5.0)) if p.get("confidence_score", 5.0) > 0 else None
    for p in predictions
]
```

---

## ✅ 3. 통계적 유의성 검정

### 3.1 구현 상태

| 검정 방법 | 구현 상태 | 사용 목적 |
|----------|----------|----------|
| Paired t-test | ✅ 완료 | RQ1, RQ2 비교 |
| Cohen's d (Effect Size) | ✅ 완료 | 실질적 유의성 |
| Holm-Bonferroni | ✅ 완료 | 다중 비교 보정 |
| Bonferroni | ✅ 완료 | 보수적 보정 |
| FDR (Benjamini-Hochberg) | ✅ 완료 | 탐색적 연구용 |
| Bootstrap BCa CI | ✅ 완료 | 신뢰구간 |
| Wilcoxon (비모수) | ✅ 완료 | 정규성 위배 시 |
| Shapiro-Wilk (정규성) | ✅ 완료 | 검정 선택 기준 |

**매우 우수함:**
> `StatisticalAnalysis` 클래스가 포괄적인 통계 검정을 지원하며, `robust_paired_test()`에서 정규성에 따라 자동으로 적절한 검정 선택

### 3.2 다중 비교 보정 검증

```python
# Holm correction 구현 검증 (statistical_analysis.py:373-429)
# Step-up enforcement가 올바르게 구현됨

max_adjusted = 0.0
for i, idx in enumerate(sorted_indices):
    raw_adjusted = sorted_p[i] * (n - i)
    max_adjusted = max(max_adjusted, raw_adjusted)  # ✅ Step-up 강제
    adjusted_p[idx] = min(max_adjusted, 1.0)
```

**권고사항:**
- ⚠️ 논문에 사용한 보정 방법을 명시적으로 기술 (권장: Holm-Bonferroni)
- 민감도 분석: Bonferroni, Holm, FDR 결과 모두 보고하면 robustness 입증

### 3.3 Effect Size 해석

| Cohen's d | 해석 | 논문 기준 |
|-----------|------|----------|
| |d| < 0.2 | Negligible | 무시 가능 |
| 0.2 ≤ |d| < 0.5 | Small | 작은 효과 |
| 0.5 ≤ |d| < 0.8 | Medium | 중간 효과 |
| |d| ≥ 0.8 | Large | 큰 효과 |

> ✅ `interpret_effect_size()` 함수에서 올바르게 구현됨

---

## ⚠️ 4. 발견된 잠재적 이슈

### 4.1 Critical Issues (반드시 수정)

| ID | 이슈 | 위치 | 권고 조치 |
|----|------|------|----------|
| **C-1** | `confidence_score = 0` 처리 없음 | `explainability_metrics.py` | 파싱 오류로 처리, 통계에서 제외 |
| **C-2** | per-sample metrics 수집 누락 가능성 | 일부 evaluator | 모든 evaluator에서 확인 필요 |

### 4.2 High Priority Issues (권장 수정)

| ID | 이슈 | 위치 | 권고 조치 |
|----|------|------|----------|
| **H-1** | Movies metadata 43.3% 누락 Sub-group 분석 | 미구현 | Group A/B 분리 평가 코드 추가 |
| **H-2** | GPT-4.1 vs Human evaluation 상관관계 | 미검증 | 소규모 human study 추가 권장 |
| **H-3** | Perplexity 계산 시 tokenization 차이 | `explainability_metrics.py` | 동일 tokenizer 사용 명시 |

### 4.3 Medium Priority Issues (선택적 개선)

| ID | 이슈 | 설명 |
|----|------|------|
| **M-1** | Test set 크기 justification | 30,000 샘플이 통계적으로 충분한지 power analysis |
| **M-2** | Hyperparameter sensitivity | LoRA rank, alpha 변경 시 성능 변화 분석 |
| **M-3** | Cross-validation | Hold-out 대신 k-fold 적용 가능성 |

---

## ✅ 5. 코드 품질 검토

### 5.1 강점

```
✅ 모듈화: src/, baselines/, scripts/ 분리 우수
✅ 문서화: CLAUDE.md, detail_task_plan.md 상세함
✅ 타입 힌트: 대부분의 함수에 type hints 적용
✅ 에러 처리: OutputParser, ErrorStatistics 클래스 구현
✅ 로깅: logging 모듈 활용, EvaluationLogger 구현
✅ 재현성: set_seed() 함수, RANDOM_SEED 상수 정의
```

### 5.2 개선 권고

```python
# 1. Docstring 표준화 (Google style 권장)
def calculate_metrics(self, gt_rank: int) -> Dict[str, float]:
    """
    공통 메트릭 계산.
    
    Args:
        gt_rank: Ground truth 아이템의 순위 (1-indexed)
    
    Returns:
        Dict containing hit@1, hit@5, hit@10, mrr, ndcg@5, ndcg@10
        
    Raises:
        ValueError: gt_rank가 0 이하인 경우
    """

# 2. Unit Test 추가 필요 (tests/ 폴더 생성 권장)
def test_ranking_metrics_hit_at_k():
    predictions = [{"item_id": "A"}, {"item_id": "B"}]
    assert RankingMetrics.hit_at_k(predictions, "A", 1) == 1.0
    assert RankingMetrics.hit_at_k(predictions, "B", 1) == 0.0
```

### 5.3 코드 커버리지 권고

| 모듈 | 테스트 필요성 | 우선순위 |
|------|-------------|----------|
| `ranking_metrics.py` | 필수 | 🔴 높음 |
| `output_parser.py` | 필수 | 🔴 높음 |
| `statistical_analysis.py` | 필수 | 🔴 높음 |
| `explainability_metrics.py` | 권장 | 🟡 중간 |
| `base_evaluator.py` | 권장 | 🟡 중간 |

---

## ✅ 6. 논문 작성 권고사항

### 6.1 Method 섹션 필수 포함 사항

```
□ Candidate Set 구성 명시 (1 GT + 99 Negatives)
□ Train/Test 분리 방법 명시 (별도 HuggingFace repo)
□ LLM4CDR 재평가 프로토콜 명시
□ 통계 검정 방법 명시 (Paired t-test + Holm correction)
□ Effect size 해석 기준 명시 (Cohen's d)
□ Random seed (42) 명시
```

### 6.2 Results 섹션 필수 포함 사항

```
□ 유의성 표기: * p<0.05, ** p<0.01, *** p<0.001
□ 95% 신뢰구간 (최소 주요 메트릭)
□ Cohen's d 값 (최소 주요 비교)
□ User Type별 분석 결과 (RQ3)
□ Error rate 보고 (parse failure, invalid item)
```

### 6.3 Discussion 섹션 권고

```
1. Limitation 명시:
   - Movies metadata 43.3% 누락의 영향
   - GPT-4.1 평가의 한계 (human evaluation과의 차이)
   - 특정 도메인(Books→Movies/Music)에 한정

2. Future Work:
   - 다른 도메인 쌍으로 확장
   - DPO/RLHF 적용
   - Real-time 추천 시나리오 적용
```

---

## ✅ 7. 최종 체크리스트

### 7.1 논문 제출 전 필수 확인

| 항목 | 상태 | 비고 |
|------|------|------|
| 모든 실험 완료 | ⬜ 미완료 | 실행 대기 |
| 통계 검정 결과 확보 | ⬜ 미완료 | 실행 후 확인 |
| Error rate < 5% | ⬜ 미확인 | 실행 후 확인 |
| 코드 재현성 테스트 | ⬜ 미완료 | 다른 환경에서 테스트 |
| 결과 파일 백업 | ⬜ 미완료 | 중요! |

### 7.2 권장 개선사항 우선순위

| 우선순위 | 항목 | 예상 소요 |
|----------|------|----------|
| 1 | C-1: confidence=0 처리 | 30분 |
| 2 | H-1: Movies metadata sub-group | 2시간 |
| 3 | Unit tests 추가 | 4시간 |
| 4 | M-1: Power analysis | 1시간 |

---

## 📚 참고 문헌

검토 과정에서 참고한 학술적 기준:

1. **Statistical Testing**: 
   - Demšar, J. (2006). "Statistical comparisons of classifiers over multiple data sets." JMLR.

2. **Effect Size**:
   - Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences."

3. **Multiple Comparison**:
   - Holm, S. (1979). "A simple sequentially rejective multiple test procedure." Scandinavian Journal of Statistics.

4. **Recommendation Evaluation**:
   - He et al. (2017). "Neural Collaborative Filtering." WWW.
   - Rendle et al. (2020). "Neural Collaborative Filtering vs. Matrix Factorization Revisited." RecSys.

---

## 결론

KitREC 프로젝트는 **석사 논문으로서 충분한 학술적 엄밀성**을 갖추고 있습니다. 특히:

1. **연구 설계**가 체계적이며, RQ가 명확함
2. **Baseline 비교**가 공정하고, 동일 조건 강제
3. **통계 검정**이 포괄적이며, 다중 비교 보정 적용
4. **코드 품질**이 양호하며, 문서화 우수

**권고사항을 반영하면 Top-tier 학회/저널 투고 수준의 완성도에 도달할 수 있습니다.**

---

*검증 완료: 2025-12-07*

