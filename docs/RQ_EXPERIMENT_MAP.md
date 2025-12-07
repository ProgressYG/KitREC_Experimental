# KitREC 연구 질문별 실험 매핑

**마지막 업데이트:** 2025-12-07  
**버전:** 1.0

---

## 개요

이 문서는 KitREC 논문의 4개 Research Questions(RQ)에 대한 실험 설계, 비교 대상, 메트릭, 예상 결과를 상세히 기술합니다.

```mermaid
mindmap
  root((KitREC<br/>연구 질문))
    RQ1[RQ1: Ablation Study]
      2×2 교차 검증
      Thinking Process 효과
      Fine-tuning 효과
    RQ2[RQ2: CDR 비교]
      Deep Learning CDR
      LLM-based CDR
      통계적 유의성
    RQ3[RQ3: Cold-start]
      1-core ~ 10-core
      User Type별 분석
      KitREC 강점 입증
    RQ4[RQ4: Explainability]
      Confidence Score
      Rationale Quality
      GPT-4.1 평가
```

---

## RQ1: KitREC 구조의 효과성 검증 (Ablation Study)

### 연구 질문
> "KitREC의 성능 향상이 단순한 Fine-tuning 덕분인지, 아니면 설계된 Thinking Process(CoT) 덕분인지?"

### 실험 설계: 2×2 교차 검증

```mermaid
quadrantChart
    title 2×2 Ablation Study
    x-axis "No Thinking" --> "Thinking (CoT)"
    y-axis "Untuned (Base)" --> "Fine-tuned"
    quadrant-1 "① KitREC-Full"
    quadrant-2 "② KitREC-Direct"
    quadrant-3 "④ Base-Direct"
    quadrant-4 "③ Base-CoT"
```

### 비교 모델 상세

| 모델 | 학습 | 추론 | 설명 |
|------|------|------|------|
| **① KitREC-Full** | Fine-tuned | Thinking | 제안 모델 (Full) |
| **② KitREC-Direct** | Fine-tuned | No Thinking | `<think>` 제거 학습 |
| **③ Base-CoT** | Untuned | Thinking | Zero-shot CoT |
| **④ Base-Direct** | Untuned | No Thinking | Vanilla Zero-shot |

### 프롬프트 차이

```mermaid
flowchart LR
    subgraph "Thinking 모델 (①, ③)"
        T1["## Reasoning Guidelines<br/>Phase 1: Pattern Recognition<br/>Phase 2: Cross-Domain Transfer<br/>Phase 3: Candidate Evaluation"]
    end

    subgraph "Direct 모델 (②, ④)"
        T2["## Output Format<br/>Provide results directly<br/>(Reasoning 섹션 제거)"]
    end

    style T1 fill:#c8e6c9
    style T2 fill:#fff3e0
```

### KitREC-Direct 생성 방법 (Option A 권장)

```python
# Training 데이터에서 <think> 블록 제거
def remove_thinking_block(output: str) -> str:
    pattern = r'<think>[\s\S]*?</think>\s*'
    return re.sub(pattern, '', output).strip()

# 별도 학습 데이터 생성 후 모델 학습
```

### 평가 메트릭

| 메트릭 | 범위 | 핵심 의미 |
|--------|------|----------|
| **Hit@10** | 0-1 | Top-10 포함 여부 |
| **NDCG@10** | 0-1 | 랭킹 품질 (논문 표준) |
| **MRR** | 0-1 | 평균 역순위 |

### 예상 결과 패턴

```mermaid
graph TB
    subgraph "예상 성능 순위"
        BEST["① KitREC-Full<br/>(최고 성능)"]
        SECOND["② KitREC-Direct<br/>또는<br/>③ Base-CoT"]
        WORST["④ Base-Direct<br/>(최저 성능)"]
        
        BEST --> |"Fine-tuning +<br/>Thinking 효과"| SECOND
        SECOND --> WORST
    end

    subgraph "핵심 비교"
        C1["① vs ② : Thinking 효과"]
        C2["① vs ③ : Fine-tuning 효과"]
        C3["② vs ③ : 어느 것이 더 중요?"]
    end

    style BEST fill:#c8e6c9
    style WORST fill:#ffcdd2
```

### 통계 검정

```python
# RQ1 통계 검정
comparisons = [
    ("KitREC-Full", "KitREC-Direct"),   # Thinking 효과
    ("KitREC-Full", "Base-CoT"),         # Fine-tuning 효과
    ("KitREC-Direct", "Base-Direct"),    # Fine-tuning 효과 (No Thinking)
    ("Base-CoT", "Base-Direct"),         # Thinking 효과 (Untuned)
]

for a, b in comparisons:
    result = paired_t_test(scores[a], scores[b])
    print(f"{a} vs {b}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f}")
```

---

## RQ2: CDR 방식의 효과성 검증 (Baseline 비교)

### 연구 질문
> "KitREC이 기존 Cross-Domain Recommendation 방법들보다 효과적인가?"

### 비교 Baseline

```mermaid
graph TB
    subgraph "Deep Learning CDR"
        CONET["CoNet (CIKM 2018)<br/>• Cross-Network Transfer<br/>• ID Sequence Input"]
        DTCDR["DTCDR (CIKM 2019)<br/>• Dual Transfer Learning<br/>• MLP Mapping"]
    end

    subgraph "LLM-based CDR"
        LLM4CDR["LLM4CDR (2025)<br/>• 3-Stage Pipeline<br/>• Domain Gap Analysis"]
        VANILLA["Vanilla Zero-shot<br/>• Base Model Direct<br/>• Lower Bound"]
    end

    subgraph "KitREC (Proposed)"
        KITREC["KitREC-Full<br/>• Knowledge Transfer<br/>• Thinking Process"]
    end

    style KITREC fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
```

### 공정한 비교 조건

```mermaid
flowchart TB
    subgraph "모든 모델 공통 조건"
        CAND["Candidate Set<br/>1 GT + 99 Negatives<br/>(100개 동일)"]
        HIST["User History<br/>동일 시점 데이터"]
        TEST["Test Dataset<br/>kitrec-test-seta/setb"]
    end

    CAND --> CONET
    CAND --> DTCDR
    CAND --> LLM4CDR
    CAND --> KITREC

    style CAND fill:#fff3e0
```

### LLM4CDR 구현 차이점 (중요!)

| 항목 | 원 논문 | KitREC 구현 |
|------|---------|-------------|
| **Positive** | 3개 | 1개 |
| **Negative** | 20-30개 | 99개 |
| **총 후보** | ~30개 | 100개 |
| **난이도** | 쉬움 | 어려움 |
| **Target History** | 미사용 | 포함 |

> **논문 명시 필요**: "LLM4CDR was re-evaluated using KitREC protocol (1 GT + 99 negatives) for fair comparison."

### 평가 메트릭

| 메트릭 | 중요도 | 설명 |
|--------|--------|------|
| **Hit@1** | ⭐⭐⭐ | 정확히 1위 예측 |
| **Hit@5** | ⭐⭐ | Top-5 포함 |
| **Hit@10** | ⭐⭐⭐ | Top-10 포함 (핵심) |
| **MRR** | ⭐⭐⭐ | 랭킹 품질 |
| **NDCG@5** | ⭐⭐ | Top-5 품질 |
| **NDCG@10** | ⭐⭐⭐ | Top-10 품질 (논문 표준) |

### 통계적 유의성 검정

```mermaid
flowchart TB
    subgraph "Step 1: Paired t-test"
        T1["KitREC vs CoNet"]
        T2["KitREC vs DTCDR"]
        T3["KitREC vs LLM4CDR"]
        T4["KitREC vs Vanilla"]
    end

    subgraph "Step 2: 다중 비교 보정"
        PVALS["p-values 수집"]
        HOLM["Holm-Bonferroni 보정"]
    end

    subgraph "Step 3: 결과 해석"
        SIG["유의성 표기<br/>* p<0.05<br/>** p<0.01<br/>*** p<0.001"]
        EFF["Effect Size<br/>Cohen's d"]
    end

    T1 --> PVALS
    T2 --> PVALS
    T3 --> PVALS
    T4 --> PVALS
    PVALS --> HOLM
    HOLM --> SIG
    T1 --> EFF
```

### 논문 테이블 형식

```latex
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
Vanilla & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx & 0.xxx \\
\midrule
\textbf{KitREC} & \textbf{0.xxx}*** & \textbf{0.xxx}** & \textbf{0.xxx}*** & ... \\
\bottomrule
\end{tabular}
\end{table}
```

---

## RQ3: Cold-start/Sparse 문제 해결 (User Type별 분석)

### 연구 질문
> "KitREC이 다양한 Cold-start 수준에서 효과적인가? 특히 극한 Cold-start(1-core)에서도 성능이 나오는가?"

### Core Level 정의

```mermaid
graph LR
    subgraph "Core Level 스펙트럼"
        C1["1-core<br/>(극한 Cold-start)"]
        C2["2-core<br/>(심각)"]
        C3["3-core<br/>(중간)"]
        C4["4-core<br/>(경미)"]
        C5["5-9 core<br/>(Warm-start)"]
        C10["10+ core<br/>(풍부)"]
        
        C1 --> C2 --> C3 --> C4 --> C5 --> C10
    end

    subgraph "기존 연구 범위"
        EXIST["기존 CDR: 5-core+ 만 실험"]
    end

    subgraph "KitREC 범위"
        KITREC["KitREC: 1-core부터 실험<br/>(핵심 강점!)"]
    end

    C5 --> EXIST
    C10 --> EXIST
    C1 --> KITREC
    
    style C1 fill:#ffcdd2,stroke:#c62828
    style KITREC fill:#c8e6c9,stroke:#2e7d32
```

### User Type → Model 매핑

| User Type | Core Level | Training Model | 샘플 수 |
|-----------|------------|----------------|---------|
| source_only_movies | 1-core | SingleFT-Movies | 3,000 |
| source_only_music | 1-core | SingleFT-Music | 3,000 |
| cold_start_2core_movies | 2-core | DualFT-Movies | 3,000 |
| cold_start_2core_music | 2-core | DualFT-Music | 3,000 |
| cold_start_3core_movies | 3-core | DualFT-Movies | 3,000 |
| cold_start_3core_music | 3-core | DualFT-Music | 3,000 |
| cold_start_4core_movies | 4-core | DualFT-Movies | 3,000 |
| cold_start_4core_music | 4-core | DualFT-Music | 3,000 |
| overlapping_books_movies | 5+-core | DualFT-Movies | 3,000 |
| overlapping_books_music | 5+-core | DualFT-Music | 3,000 |

### 분석 방법

```python
def evaluate_by_user_type(samples, user_type_mapping):
    grouped = defaultdict(list)
    
    for sample in samples:
        user_type = user_type_mapping.get(sample.user_id, "unknown")
        metrics = evaluate_sample(sample)
        grouped[user_type].append(metrics)
    
    # Core Level별 집계
    core_level_results = {}
    for user_type, metrics_list in grouped.items():
        core = USER_TYPE_MAPPING[user_type]["core"]
        aggregated = aggregate_metrics(metrics_list)
        core_level_results[f"{core}-core"] = aggregated
    
    return core_level_results
```

### 예상 결과 시각화

```mermaid
xychart-beta
    title "Core Level별 Hit@10 성능 (예상)"
    x-axis ["1-core", "2-core", "3-core", "4-core", "5-9 core", "10+ core"]
    y-axis "Hit@10" 0 --> 0.8
    bar [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    line [0.05, 0.10, 0.20, 0.35, 0.50, 0.60]
```

> **범례**: Bar = KitREC, Line = Best Baseline  
> **핵심 주장**: 1-4 core에서 KitREC과 Baseline 간 격차가 가장 큼

### 핵심 논문 주장

```
"Unlike existing CDR methods that only evaluate on users with 
5+ target interactions (warm-start), KitREC demonstrates strong 
performance even on extreme cold-start users with only 1 target 
interaction."
```

---

## RQ4: Confidence/Rationale 검증 (Explainability)

### 연구 질문
> "KitREC이 생성하는 Confidence Score와 Rationale이 신뢰할 수 있는가?"

### ⚠️ 중요: 평가 대상

```mermaid
flowchart TB
    subgraph "RQ4 평가 대상"
        KITREC["✅ KitREC 모델만"]
    end

    subgraph "제외"
        CONET["❌ CoNet"]
        DTCDR["❌ DTCDR"]
        LLM4CDR["❌ LLM4CDR"]
        VANILLA["❌ Vanilla"]
    end

    style KITREC fill:#c8e6c9
    style CONET fill:#ffcdd2
    style DTCDR fill:#ffcdd2
    style LLM4CDR fill:#ffcdd2
    style VANILLA fill:#ffcdd2
```

> **이유**: Baseline 모델들은 Confidence Score와 Rationale을 생성하지 않거나, 형식이 다름

### 평가 메트릭 체계

```mermaid
graph TB
    subgraph "Confidence Score 평가"
        CONF["confidence_score (1-10)"]
        CONF --> NORM["÷ 2 정규화 (0.5-5)"]
        NORM --> MAE["MAE: 평균 절대 오차"]
        NORM --> RMSE["RMSE: 평균 제곱근 오차"]
        
        GT["Ground Truth Rating (1-5)"]
        GT --> MAE
        GT --> RMSE
    end

    subgraph "Rationale 평가"
        RAT["rationale 텍스트"]
        RAT --> PPL["Perplexity (PPL)<br/>언어 모델 확신도"]
        RAT --> GPT["GPT-4.1 평가<br/>품질 점수 (1-10)"]
    end

    style CONF fill:#e1f5fe
    style RAT fill:#fff3e0
```

### GPT-4.1 평가 설계

#### Stratified Sampling (균등 추출)

```mermaid
flowchart LR
    RESULTS["전체 결과"] --> GROUP["10개 User Type으로<br/>그룹화"]
    GROUP --> SAMPLE["각 타입에서<br/>5개 랜덤 샘플"]
    SAMPLE --> TOTAL["총 50개/모델"]
    TOTAL --> GPT["GPT-4.1 API 호출"]

    style SAMPLE fill:#fff3e0
```

#### 평가 기준 (4가지)

| 기준 | 영문 | 설명 | 점수 범위 |
|------|------|------|----------|
| **논리성** | Logic | 추천 이유가 논리적인가? | 1-10 |
| **구체성** | Specificity | 구체적인 근거를 제시하는가? | 1-10 |
| **Cross-domain 연결성** | Cross-domain | Source→Target 연결이 명확한가? | 1-10 |
| **선호 반영** | Preference | 사용자 히스토리를 잘 반영했는가? | 1-10 |

#### GPT-4.1 프롬프트

```python
EVALUATION_PROMPT = '''You are an expert evaluator for recommendation system explanations.

Evaluate the following rationale based on:
1. Logic (1-10): Is the reasoning logically coherent?
2. Specificity (1-10): Are specific items/preferences referenced?
3. Cross-domain (1-10): Is the connection between source and target domains clear?
4. Preference (1-10): Does it reflect the user's actual preferences?

User History:
{user_history}

Recommended Item:
{recommended_item}

Rationale to evaluate:
{rationale}

Respond ONLY with a JSON object:
{"logic": <1-10>, "specificity": <1-10>, "cross_domain": <1-10>, 
 "preference": <1-10>, "overall": <1-10>}
'''
```

### 결과 테이블 예시

```markdown
| Model | MAE ↓ | RMSE ↓ | PPL ↓ | Logic | Specificity | Cross-domain | Preference | Overall |
|-------|-------|--------|-------|-------|-------------|--------------|------------|---------|
| DualFT-Movies (A) | 0.xx | 0.xx | xx.x | x.x | x.x | x.x | x.x | x.x |
| DualFT-Movies (B) | 0.xx | 0.xx | xx.x | x.x | x.x | x.x | x.x | x.x |
| DualFT-Music (A) | 0.xx | 0.xx | xx.x | x.x | x.x | x.x | x.x | x.x |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
```

---

## 실험 실행 순서

```mermaid
gantt
    title KitREC 실험 로드맵
    dateFormat  YYYY-MM-DD
    section Phase 1
    환경 설정           :p1, 2025-01-01, 1d
    데이터 검증         :p2, after p1, 1d
    section Phase 2
    KitREC 평가 (8모델) :p3, after p2, 4d
    section Phase 3
    RQ1 Ablation       :p4, after p3, 2d
    section Phase 4
    Baseline 학습      :p5, after p4, 3d
    Baseline 평가      :p6, after p5, 2d
    section Phase 5
    Stratified 분석    :p7, after p6, 2d
    GPT-4.1 평가       :p8, after p7, 1d
    section Phase 6
    리포트 생성        :p9, after p8, 2d
```

---

## 실행 명령어 요약

```bash
# RQ1: Ablation Study
python scripts/run_ablation_study.py --config configs/eval_config.yaml

# RQ2: Baseline 평가
python scripts/run_baseline_eval.py --baseline conet --target_domain movies
python scripts/run_baseline_eval.py --baseline dtcdr --target_domain movies
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain movies

# RQ3: Cold-start 분석 (KitREC 결과에서 User Type별 집계)
python scripts/run_kitrec_eval.py --model_name dualft_movies_seta --analyze_user_types

# RQ4: Explainability (KitREC만)
python scripts/run_explainability_eval.py --model_name dualft_movies_seta --gpt_eval
```

---

## 핵심 논문 인용 문구

### RQ1
> "The 2×2 ablation study demonstrates that both the fine-tuning process and the thinking mechanism contribute to KitREC's superior performance, with the combination achieving the best results."

### RQ2
> "KitREC significantly outperforms all baselines across all metrics (p < 0.01, paired t-test with Holm-Bonferroni correction)."

### RQ3
> "Most importantly, KitREC maintains competitive performance even in extreme cold-start scenarios (1-core), where traditional CDR methods fail to provide meaningful recommendations."

### RQ4
> "The generated rationales receive high scores (>7/10) on logic, specificity, and cross-domain connection according to GPT-4.1 evaluation, demonstrating KitREC's strong explainability."

