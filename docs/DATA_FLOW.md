# KitREC 데이터 흐름 다이어그램

**마지막 업데이트:** 2025-12-07  
**버전:** 1.0

---

## 1. HuggingFace 데이터셋 구조

```mermaid
graph TB
    subgraph "HuggingFace Hub (Younggooo)"
        subgraph "Training Data (8 repos)"
            DUAL_M_A["kitrec-dualft_movies-seta<br/>12,000 samples"]
            DUAL_M_B["kitrec-dualft_movies-setb<br/>12,000 samples"]
            DUAL_S_A["kitrec-dualft_music-seta<br/>12,000 samples"]
            DUAL_S_B["kitrec-dualft_music-setb<br/>12,000 samples"]
            SING_M_A["kitrec-singleft_movies-seta<br/>3,000 samples"]
            SING_M_B["kitrec-singleft_movies-setb<br/>3,000 samples"]
            SING_S_A["kitrec-singleft_music-seta<br/>3,000 samples"]
            SING_S_B["kitrec-singleft_music-setb<br/>3,000 samples"]
        end

        subgraph "Validation Data (2 repos)"
            VAL_A["kitrec-val-seta<br/>12,000 samples<br/>(DPO/GRPO용)"]
            VAL_B["kitrec-val-setb<br/>12,000 samples<br/>(DPO/GRPO용)"]
        end

        subgraph "Test Data (2 repos)"
            TEST_A["kitrec-test-seta<br/>30,000 samples<br/>(Hard Negatives)"]
            TEST_B["kitrec-test-setb<br/>30,000 samples<br/>(Random Negatives)"]
        end
    end

    style DUAL_M_A fill:#c8e6c9
    style DUAL_S_A fill:#c8e6c9
    style TEST_A fill:#ffecb3
    style TEST_B fill:#ffecb3
```

---

## 2. 데이터 스키마

### Training Data Schema
```mermaid
classDiagram
    class TrainingSample {
        +string instruction
        +string input
        +string output
        +Metadata metadata
    }
    
    class Metadata {
        +string user_id
        +string user_type
        +string user_category
        +string target_domain
        +string source_domain
        +int target_core
        +int books_core
        +string candidate_set
        +string gt_item_id
        +int thinking_length
        +float confidence_score
        +float generation_time_sec
    }
    
    TrainingSample --> Metadata

    note for TrainingSample "instruction: 전체 프롬프트\ninput: 빈 문자열\noutput: <think>...JSON"
```

### Val/Test Data Schema
```mermaid
classDiagram
    class ValTestSample {
        +string instruction
        +string input
        +string output
        +GroundTruth ground_truth
        +Metadata metadata
    }
    
    class GroundTruth {
        +string item_id
        +string title
        +float rating
    }
    
    ValTestSample --> GroundTruth

    note for ValTestSample "instruction: 짧은 설명\ninput: 전체 프롬프트\noutput: 빈 문자열"
```

---

## 3. 프롬프트 추출 흐름

```mermaid
flowchart TD
    START["Sample 입력"]
    
    START --> CHECK{"sample.input<br/>존재하는가?"}
    
    CHECK --> |"Yes (Val/Test)"| INPUT["prompt = sample['input']"]
    CHECK --> |"No (Training)"| INST["prompt = sample['instruction']"]
    
    INPUT --> RESULT["프롬프트 반환"]
    INST --> RESULT
    
    RESULT --> THINKING{"Thinking 모드?"}
    
    THINKING --> |"KitREC-Full<br/>Base-CoT"| FULL["Reasoning Guidelines 포함"]
    THINKING --> |"KitREC-Direct<br/>Base-Direct"| DIRECT["Reasoning Guidelines 제거"]
    
    FULL --> OUTPUT["모델 입력"]
    DIRECT --> OUTPUT

    style CHECK fill:#ffecb3
    style THINKING fill:#e1f5fe
```

---

## 4. 모델 출력 파싱 흐름

```mermaid
flowchart TD
    RAW["Raw Model Output"]
    
    RAW --> THINK["1. <think> 블록 추출"]
    THINK --> JSON["2. JSON 블록 추출"]
    JSON --> TRAIL["3. Trailing Comma 제거"]
    TRAIL --> PARSE["4. JSON 파싱"]
    
    PARSE --> VALID{"파싱 성공?"}
    
    VALID --> |"No"| ERROR["ParseError 기록<br/>predictions = []"]
    VALID --> |"Yes"| ITEMS["예측 아이템 리스트"]
    
    ITEMS --> LOOP["각 item_id 검증"]
    
    LOOP --> CAND{"item_id in<br/>candidates?"}
    
    CAND --> |"Yes"| KEEP["유효 예측 추가"]
    CAND --> |"No"| INVALID["무효 item_id 기록<br/>rank = ∞"]
    
    KEEP --> RESULT["ParseResult 반환"]
    INVALID --> RESULT
    ERROR --> RESULT

    style VALID fill:#ffecb3
    style CAND fill:#ffecb3
    style INVALID fill:#ffcdd2
```

---

## 5. Candidate Set 처리

```mermaid
flowchart LR
    subgraph "프롬프트에서 추출"
        PROMPT["## List of Available<br/>Candidate Items"]
        PROMPT --> REGEX["정규표현식<br/>(ID: [A-Z0-9]+)"]
        REGEX --> IDS["100개 item_id<br/>리스트"]
    end

    subgraph "검증"
        IDS --> CHECK1{"len == 100?"}
        CHECK1 --> |"Yes"| CHECK2{"GT in candidates?"}
        CHECK1 --> |"No"| WARN1["⚠️ Warning"]
        CHECK2 --> |"Yes"| VALID["✅ Valid"]
        CHECK2 --> |"No"| WARN2["⚠️ Warning"]
    end

    subgraph "예측 검증"
        VALID --> PRED["모델 예측 item_id"]
        PRED --> MATCH{"candidates에<br/>포함?"}
        MATCH --> |"Yes"| RANK["정상 rank 계산"]
        MATCH --> |"No"| INF["rank = ∞<br/>(fail 처리)"]
    end

    style CHECK1 fill:#fff3e0
    style CHECK2 fill:#fff3e0
    style MATCH fill:#fff3e0
    style INF fill:#ffcdd2
```

---

## 6. Confidence Score 정규화

```mermaid
flowchart TB
    subgraph "KitREC 모델"
        K_OUT["모델 출력<br/>confidence_score: 1-10"]
        K_OUT --> K_NORM["÷ 2"]
        K_NORM --> K_FINAL["정규화: 0.5-5"]
    end

    subgraph "Baseline 모델"
        B_OUT["Raw Score<br/>(unbounded)"]
        B_OUT --> B_SIG["sigmoid(x)"]
        B_SIG --> B_SCALE["× 9 + 1"]
        B_SCALE --> B_FINAL["정규화: 1-10"]
    end

    subgraph "Ground Truth"
        GT["Rating: 1-5"]
    end

    subgraph "평가"
        K_FINAL --> MAE["MAE 계산"]
        K_FINAL --> RMSE["RMSE 계산"]
        GT --> MAE
        GT --> RMSE
    end

    style K_OUT fill:#c8e6c9
    style B_OUT fill:#e1f5fe
    style GT fill:#fff3e0
```

---

## 7. User Type별 데이터 분포

```mermaid
pie title User Type Distribution (30,000 users)
    "overlapping_books_movies" : 3000
    "overlapping_books_music" : 3000
    "source_only_movies" : 3000
    "source_only_music" : 3000
    "cold_start_2core_movies" : 3000
    "cold_start_2core_music" : 3000
    "cold_start_3core_movies" : 3000
    "cold_start_3core_music" : 3000
    "cold_start_4core_movies" : 3000
    "cold_start_4core_music" : 3000
```

---

## 8. Train/Test 분리 검증

```mermaid
flowchart TB
    subgraph "✅ 올바른 방법"
        TRAIN_D["Training Dataset<br/>kitrec-dualft_*"]
        TRAIN_D --> MODEL["Baseline 학습"]
        MODEL --> SAVE["체크포인트 저장"]
        
        TEST_D["Test Dataset<br/>kitrec-test-*"]
        TEST_D --> EVAL["평가 실행"]
        SAVE --> EVAL
    end

    subgraph "❌ 금지 (Data Leakage)"
        BAD_TEST["Test Data"]
        BAD_TEST --> |"분할"| BAD_TRAIN["Train (80%)"]
        BAD_TEST --> |"분할"| BAD_VAL["Val (20%)"]
        BAD_TRAIN --> |"학습"| BAD_MODEL["모델"]
        
        style BAD_TRAIN fill:#ffcdd2
        style BAD_VAL fill:#ffcdd2
    end

    style TRAIN_D fill:#c8e6c9
    style TEST_D fill:#fff3e0
```

---

## 9. 결과 저장 구조

```mermaid
graph TB
    subgraph "results/"
        subgraph "kitrec/"
            K_DUAL_M_A["dualft_movies_seta/"]
            K_DUAL_M_B["dualft_movies_setb/"]
            K_DUAL_S_A["dualft_music_seta/"]
            K_DUAL_S_B["dualft_music_setb/"]
            K_SING_M_A["singleft_movies_seta/"]
            K_SING_M_B["singleft_movies_setb/"]
            K_SING_S_A["singleft_music_seta/"]
            K_SING_S_B["singleft_music_setb/"]
        end
        
        subgraph "ablation/"
            ABL_FULL["kitrec_full/"]
            ABL_DIR["kitrec_direct/"]
            ABL_COT["base_cot/"]
            ABL_BDIR["base_direct/"]
        end
        
        subgraph "baselines/"
            BL_CON["conet/"]
            BL_DTC["dtcdr/"]
            BL_LLM["llm4cdr/"]
            BL_VAN["vanilla_zeroshot/"]
        end
        
        subgraph "reports/"
            REP["rq1_ablation_report.md<br/>rq2_cdr_comparison.md<br/>rq3_coldstart_analysis.md<br/>rq4_explainability.md<br/>final_paper_tables.md"]
        end
    end

    subgraph "각 실험 폴더 내용"
        FILES["predictions.jsonl<br/>metrics_summary.json<br/>error_statistics.json"]
    end

    K_DUAL_M_A --> FILES
    BL_CON --> FILES

    style K_DUAL_M_A fill:#c8e6c9
    style ABL_FULL fill:#e3f2fd
    style BL_CON fill:#f3e5f5
    style REP fill:#fff3e0
```

---

## 10. 메트릭 계산 흐름

```mermaid
flowchart TB
    subgraph "Input"
        PRED["predictions: List[Dict]"]
        GT["ground_truth_id: str"]
    end

    subgraph "Ranking Metrics"
        PRED --> HIT["Hit@K"]
        GT --> HIT
        
        PRED --> MRR["MRR"]
        GT --> MRR
        
        PRED --> NDCG["NDCG@K"]
        GT --> NDCG
    end

    subgraph "Hit@K 계산"
        HIT --> H1{"GT in<br/>top_1?"}
        HIT --> H5{"GT in<br/>top_5?"}
        HIT --> H10{"GT in<br/>top_10?"}
        
        H1 --> |"Yes"| HIT1["hit@1 = 1"]
        H1 --> |"No"| HIT1_0["hit@1 = 0"]
    end

    subgraph "MRR 계산"
        MRR --> FIND["GT 위치 찾기"]
        FIND --> |"rank = i"| RR["1 / (i + 1)"]
        FIND --> |"Not found"| RR0["0"]
    end

    subgraph "NDCG@K 계산"
        NDCG --> DCG["DCG = 1 / log2(rank + 1)"]
        DCG --> IDCG["IDCG = 1 / log2(2) = 1"]
        DCG --> NORM["NDCG = DCG / IDCG"]
    end

    style PRED fill:#e1f5fe
    style GT fill:#fff3e0
```

---

## 11. 통계 분석 흐름

```mermaid
flowchart TB
    subgraph "Input"
        KITREC_S["KitREC per-sample scores"]
        BASE_S["Baseline per-sample scores"]
    end

    subgraph "Paired t-test"
        KITREC_S --> DIFF["diff = kitrec - baseline"]
        BASE_S --> DIFF
        DIFF --> TSTAT["t_stat, p_value = ttest_rel()"]
        DIFF --> COHEN["Cohen's d = mean(diff) / std(diff)"]
    end

    subgraph "다중 비교 보정"
        PVALS["여러 p-values"]
        PVALS --> SORT["정렬 (오름차순)"]
        SORT --> HOLM["Holm-Bonferroni<br/>adjusted_p = p × (n - i)"]
        HOLM --> STEPUP["Step-up enforcement<br/>max(current, previous)"]
    end

    subgraph "결과 해석"
        TSTAT --> SIG{"p < 0.05?"}
        SIG --> |"Yes"| STAR["유의함 (*, **, ***)"]
        SIG --> |"No"| NS["유의하지 않음"]
        
        COHEN --> EFF{"Cohen's d"}
        EFF --> SMALL["0.2: Small"]
        EFF --> MED["0.5: Medium"]
        EFF --> LARGE["0.8: Large"]
    end

    style DIFF fill:#e1f5fe
    style STEPUP fill:#fff3e0
```

---

## 12. GPT-4.1 평가 흐름 (RQ4)

```mermaid
flowchart TB
    subgraph "Stratified Sampling"
        RESULTS["KitREC 결과"]
        RESULTS --> GROUP["User Type별 그룹화<br/>(10개 타입)"]
        GROUP --> SAMPLE["각 타입에서 5개 샘플<br/>총 50개/모델"]
    end

    subgraph "GPT-4.1 API 호출"
        SAMPLE --> PROMPT["평가 프롬프트 생성"]
        PROMPT --> API["OpenAI API 호출"]
        API --> RESP["JSON 응답"]
    end

    subgraph "평가 기준 (1-10점)"
        RESP --> LOGIC["Logic: 논리성"]
        RESP --> SPEC["Specificity: 구체성"]
        RESP --> CROSS["Cross-domain: 연결성"]
        RESP --> PREF["Preference: 선호 반영"]
        RESP --> OVERALL["Overall: 종합"]
    end

    subgraph "집계"
        LOGIC --> MEAN["평균 점수 계산"]
        SPEC --> MEAN
        CROSS --> MEAN
        PREF --> MEAN
        OVERALL --> MEAN
        MEAN --> FINAL["최종 GPT Score"]
    end

    style SAMPLE fill:#fff3e0
    style API fill:#e1f5fe
    style FINAL fill:#c8e6c9
```

---

## 요약: 핵심 데이터 흐름 체크포인트

| # | 체크포인트 | 검증 방법 |
|---|-----------|----------|
| 1 | `input` 필드에서 프롬프트 추출 (Val/Test) | 단위 테스트 |
| 2 | Candidate Set = 100개 | `len(candidates) == 100` |
| 3 | GT가 Candidate에 포함 | `gt_id in candidates` |
| 4 | 후보군 외 item_id → rank = ∞ | 오류율 로깅 |
| 5 | Confidence ÷ 2 정규화 | MAE/RMSE 범위 검증 |
| 6 | Train/Test 완전 분리 | 별도 데이터셋 사용 |
| 7 | per-sample 메트릭 수집 | t-test 입력 준비 |

