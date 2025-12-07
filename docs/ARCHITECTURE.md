# KitREC 시스템 아키텍처

**마지막 업데이트:** 2025-12-07  
**버전:** 1.0

---

## 1. 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "📊 Data Layer"
        HF[("HuggingFace Hub<br/>Datasets")]
        HF --> TRAIN["Training Data<br/>kitrec-dualft_*<br/>kitrec-singleft_*"]
        HF --> VAL["Validation Data<br/>kitrec-val-*"]
        HF --> TEST["Test Data<br/>kitrec-test-*<br/>30,000 samples/set"]
    end

    subgraph "🤖 Model Layer"
        BASE["Qwen3-14B<br/>(Base Model)"]
        BASE --> DUALFT["DualFT Models<br/>Movies/Music<br/>12K samples"]
        BASE --> SINGLEFT["SingleFT Models<br/>Movies/Music<br/>3K samples"]
        
        DUALFT --> FULL["KitREC-Full<br/>(Thinking)"]
        DUALFT --> DIRECT["KitREC-Direct<br/>(No Thinking)"]
        
        BASE --> COT["Base-CoT<br/>(Zero-shot Thinking)"]
        BASE --> BDIRECT["Base-Direct<br/>(Vanilla Zero-shot)"]
    end

    subgraph "⚙️ Inference Layer"
        VLLM["vLLM Engine<br/>Nvidia 5090 36GB"]
        PARSER["Output Parser<br/>JSON + Think Block"]
        VLLM --> PARSER
    end

    subgraph "📈 Evaluation Layer"
        RANK["Ranking Metrics<br/>Hit@K, MRR, NDCG"]
        EXPLAIN["Explainability<br/>MAE, RMSE, PPL"]
        STATS["Statistical Analysis<br/>t-test, Holm"]
        GPT["GPT-4.1 Eval<br/>Rationale Quality"]
    end

    subgraph "🔬 Baseline Layer"
        CONET["CoNet<br/>(CIKM 2018)"]
        DTCDR["DTCDR<br/>(CIKM 2019)"]
        LLM4CDR["LLM4CDR<br/>(3-Stage Pipeline)"]
    end

    TEST --> VLLM
    FULL --> VLLM
    DIRECT --> VLLM
    COT --> VLLM
    BDIRECT --> VLLM
    
    PARSER --> RANK
    PARSER --> EXPLAIN
    RANK --> STATS
    EXPLAIN --> GPT

    TRAIN --> CONET
    TRAIN --> DTCDR
    TEST --> LLM4CDR
    
    style HF fill:#e1f5fe
    style VLLM fill:#fff3e0
    style RANK fill:#e8f5e9
    style GPT fill:#fce4ec
```

---

## 2. 프로젝트 폴더 구조

```mermaid
graph LR
    subgraph "Experimental_test/"
        ROOT["/"]
        
        ROOT --> SRC["src/"]
        SRC --> DATA["data/<br/>• data_loader.py<br/>• prompt_builder.py<br/>• candidate_handler.py"]
        SRC --> INF["inference/<br/>• vllm_inference.py<br/>• output_parser.py<br/>• batch_inference.py"]
        SRC --> MET["metrics/<br/>• ranking_metrics.py<br/>• explainability_metrics.py<br/>• statistical_analysis.py<br/>• stratified_analysis.py"]
        SRC --> MOD["models/<br/>• kitrec_model.py<br/>• base_model.py"]
        SRC --> UTL["utils/<br/>• logger.py<br/>• io_utils.py<br/>• visualization.py"]
        
        ROOT --> BASE["baselines/"]
        BASE --> BEVAL["base_evaluator.py"]
        BASE --> CONET2["conet/<br/>model, trainer, evaluator"]
        BASE --> DTCDR2["dtcdr/<br/>model, trainer, evaluator"]
        BASE --> LLM["llm4cdr/<br/>prompts, evaluator"]
        
        ROOT --> SCRIPTS["scripts/<br/>• run_kitrec_eval.py<br/>• run_ablation_study.py<br/>• run_baseline_eval.py"]
        ROOT --> RESULTS["results/<br/>kitrec/, ablation/, baselines/"]
        ROOT --> CONFIGS["configs/<br/>*.yaml"]
    end

    style ROOT fill:#fff9c4
    style SRC fill:#e3f2fd
    style BASE fill:#f3e5f5
    style SCRIPTS fill:#e8f5e9
```

---

## 3. 모델 학습 파이프라인

```mermaid
flowchart TB
    subgraph "Phase 1: Base Model"
        QWEN["Qwen/Qwen3-14B"]
    end

    subgraph "Phase 2: PEFT QLoRA Training"
        QWEN --> |"12K samples<br/>3 epochs"| DUALM["DualFT-Movies"]
        QWEN --> |"12K samples<br/>3 epochs"| DUALS["DualFT-Music"]
        QWEN --> |"3K samples<br/>6 epochs"| SINGM["SingleFT-Movies"]
        QWEN --> |"3K samples<br/>6 epochs"| SINGS["SingleFT-Music"]
    end

    subgraph "Phase 3: Ablation Models"
        DUALM --> FULL_M["KitREC-Full-Movies"]
        DUALS --> FULL_S["KitREC-Full-Music"]
        
        QWEN --> |"No Training"| BASE_COT["Base-CoT"]
        QWEN --> |"No Training"| BASE_DIR["Base-Direct"]
        
        DUALM --> |"학습 데이터에서<br/>think 제거"| DIRECT_M["KitREC-Direct-Movies"]
        DUALS --> |"학습 데이터에서<br/>think 제거"| DIRECT_S["KitREC-Direct-Music"]
    end

    subgraph "LoRA Config"
        DUAL_CFG["DualFT Config<br/>r=32, alpha=64<br/>dropout=0.08<br/>LR=2e-4"]
        SING_CFG["SingleFT Config<br/>r=24, alpha=48<br/>dropout=0.15<br/>LR=6e-5"]
    end

    DUAL_CFG -.-> DUALM
    DUAL_CFG -.-> DUALS
    SING_CFG -.-> SINGM
    SING_CFG -.-> SINGS

    style QWEN fill:#bbdefb
    style DUALM fill:#c8e6c9
    style DUALS fill:#c8e6c9
    style SINGM fill:#fff9c4
    style SINGS fill:#fff9c4
```

---

## 4. 추론 및 평가 파이프라인

```mermaid
sequenceDiagram
    participant HF as HuggingFace Hub
    participant DL as DataLoader
    participant PB as PromptBuilder
    participant VLLM as vLLM Engine
    participant OP as OutputParser
    participant RM as RankingMetrics
    participant EM as ExplainabilityMetrics
    participant SA as StatisticalAnalysis

    HF->>DL: load_test_data()
    DL->>DL: extract_prompt(sample)
    Note over DL: input > instruction<br/>Template Schema 적용

    DL->>PB: build_thinking_prompt() / build_direct_prompt()
    PB->>VLLM: generate(prompt)
    
    VLLM->>OP: parse(raw_output, candidate_ids)
    Note over OP: 1. <think> 블록 분리<br/>2. JSON 추출<br/>3. trailing comma 제거<br/>4. item_id 검증

    OP->>RM: calculate_all(predictions, gt_id)
    Note over RM: Hit@1, Hit@5, Hit@10<br/>MRR, NDCG@5, NDCG@10

    OP->>EM: mae(), rmse(), perplexity()
    Note over EM: Confidence Score ÷ 2<br/>정규화 적용

    RM->>SA: paired_t_test(kitrec, baseline)
    SA->>SA: apply_multiple_correction(p_values)
    Note over SA: Holm-Bonferroni<br/>Step-up enforcement
```

---

## 5. Baseline 비교 아키텍처

```mermaid
graph TB
    subgraph "공통 평가 인프라"
        BEVAL["BaseEvaluator<br/>• validate_candidate_set()<br/>• normalize_confidence()<br/>• calculate_metrics()"]
    end

    subgraph "Deep Learning CDR"
        CONET["CoNet (CIKM 2018)<br/>• Cross-Network Transfer<br/>• ID Sequence Input"]
        DTCDR["DTCDR (CIKM 2019)<br/>• Dual Transfer Learning<br/>• MLP Mapping"]
    end

    subgraph "LLM-based CDR"
        LLM4CDR["LLM4CDR (2025)<br/>• 3-Stage Pipeline<br/>• Domain Gap Analysis<br/>• User Interest Reasoning<br/>• Candidate Reranking"]
        VANILLA["Vanilla Zero-shot<br/>• Base Model Direct<br/>• Lower Bound"]
    end

    subgraph "KitREC (제안 모델)"
        KITREC["KitREC-Full<br/>• Cross-Domain Thinking<br/>• Knowledge Transfer<br/>• Confidence + Rationale"]
    end

    BEVAL --> CONET
    BEVAL --> DTCDR
    BEVAL --> LLM4CDR
    BEVAL --> VANILLA
    BEVAL --> KITREC

    subgraph "Candidate Set (동일 조건)"
        CAND["100 candidates<br/>1 GT + 99 Negatives"]
    end

    CAND --> CONET
    CAND --> DTCDR
    CAND --> LLM4CDR
    CAND --> VANILLA
    CAND --> KITREC

    style KITREC fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style BEVAL fill:#e1f5fe
    style CAND fill:#fff3e0
```

---

## 6. User Type 및 Core Level 매핑

```mermaid
graph LR
    subgraph "User Distribution (30,000 total)"
        MOVIES["Target: Movies<br/>15,000 users"]
        MUSIC["Target: Music<br/>15,000 users"]
    end

    subgraph "Movies User Types"
        OVL_M["overlapping_books_movies<br/>3,000 (5+ core)"]
        CS2_M["cold_start_2core_movies<br/>3,000"]
        CS3_M["cold_start_3core_movies<br/>3,000"]
        CS4_M["cold_start_4core_movies<br/>3,000"]
        SO_M["source_only_movies<br/>3,000 (1-core)"]
    end

    subgraph "Music User Types"
        OVL_S["overlapping_books_music<br/>3,000 (5+ core)"]
        CS2_S["cold_start_2core_music<br/>3,000"]
        CS3_S["cold_start_3core_music<br/>3,000"]
        CS4_S["cold_start_4core_music<br/>3,000"]
        SO_S["source_only_music<br/>3,000 (1-core)"]
    end

    subgraph "Training Models"
        DUALFT_M["DualFT-Movies<br/>12,000 samples"]
        SINGLEFT_M["SingleFT-Movies<br/>3,000 samples"]
        DUALFT_S["DualFT-Music<br/>12,000 samples"]
        SINGLEFT_S["SingleFT-Music<br/>3,000 samples"]
    end

    OVL_M --> DUALFT_M
    CS2_M --> DUALFT_M
    CS3_M --> DUALFT_M
    CS4_M --> DUALFT_M
    SO_M --> SINGLEFT_M

    OVL_S --> DUALFT_S
    CS2_S --> DUALFT_S
    CS3_S --> DUALFT_S
    CS4_S --> DUALFT_S
    SO_S --> SINGLEFT_S

    style SO_M fill:#ffcdd2
    style SO_S fill:#ffcdd2
    style OVL_M fill:#c8e6c9
    style OVL_S fill:#c8e6c9
```

---

## 7. 평가 지표 체계

```mermaid
graph TB
    subgraph "Ranking Metrics (모든 모델)"
        HIT1["Hit@1<br/>정확히 1위 예측"]
        HIT5["Hit@5<br/>Top-5 포함"]
        HIT10["Hit@10<br/>Top-10 포함"]
        MRR["MRR<br/>1/rank 평균"]
        NDCG5["NDCG@5<br/>Top-5 랭킹 품질"]
        NDCG10["NDCG@10<br/>Top-10 랭킹 품질<br/>(논문 표준)"]
    end

    subgraph "Explainability Metrics (KitREC만)"
        MAE["MAE<br/>|confidence/2 - rating|"]
        RMSE["RMSE<br/>√(confidence/2 - rating)²"]
        PPL["Perplexity<br/>rationale 품질"]
        GPT["GPT-4.1 Score<br/>• Logic<br/>• Specificity<br/>• Cross-domain<br/>• Preference"]
    end

    subgraph "Statistical Testing"
        TTEST["Paired t-test<br/>per-sample 비교"]
        HOLM["Holm Correction<br/>다중 비교 보정"]
        COHEN["Cohen's d<br/>Effect Size"]
    end

    HIT10 --> TTEST
    NDCG10 --> TTEST
    MRR --> TTEST
    TTEST --> HOLM
    TTEST --> COHEN

    style NDCG10 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style GPT fill:#fce4ec
    style HOLM fill:#e1f5fe
```

---

## 8. Research Questions 실험 매핑

```mermaid
graph TB
    subgraph "RQ1: Ablation Study (2×2)"
        RQ1["KitREC 구조 효과 검증"]
        RQ1 --> FULL["① KitREC-Full<br/>(Fine-tuned + Thinking)"]
        RQ1 --> DIRECT["② KitREC-Direct<br/>(Fine-tuned + No Thinking)"]
        RQ1 --> COT["③ Base-CoT<br/>(Untuned + Thinking)"]
        RQ1 --> BDIR["④ Base-Direct<br/>(Untuned + No Thinking)"]
    end

    subgraph "RQ2: CDR 비교"
        RQ2["CDR 방식 효과 검증"]
        RQ2 --> VS_CONET["vs CoNet"]
        RQ2 --> VS_DTCDR["vs DTCDR"]
        RQ2 --> VS_LLM4["vs LLM4CDR"]
        RQ2 --> VS_VAN["vs Vanilla"]
    end

    subgraph "RQ3: Cold-start"
        RQ3["Cold-start 해결 검증"]
        RQ3 --> C1["1-core<br/>(극한 Cold-start)"]
        RQ3 --> C2["2-core"]
        RQ3 --> C3["3-core"]
        RQ3 --> C4["4-core"]
        RQ3 --> C5["5+-core<br/>(Warm-start)"]
    end

    subgraph "RQ4: Explainability"
        RQ4["설명력 검증<br/>(KitREC만)"]
        RQ4 --> CONF["Confidence Score<br/>MAE, RMSE"]
        RQ4 --> RAT["Rationale Quality<br/>PPL, GPT-4.1"]
    end

    style RQ1 fill:#e3f2fd
    style RQ2 fill:#f3e5f5
    style RQ3 fill:#fff3e0
    style RQ4 fill:#fce4ec
```

---

## 9. 데이터 흐름 다이어그램

```mermaid
flowchart LR
    subgraph "Input"
        HF["HuggingFace Hub"]
        HF --> |"Training"| TRAIN["kitrec-dualft_*<br/>kitrec-singleft_*"]
        HF --> |"Test"| TEST["kitrec-test-seta<br/>kitrec-test-setb"]
    end

    subgraph "Processing"
        TRAIN --> |"90:10 split"| TRVAL["Train/Val<br/>Stratified by user_type"]
        TEST --> |"extract_prompt()"| PROMPT["Prompt<br/>(input field)"]
        TEST --> |"extract_ground_truth()"| GT["Ground Truth<br/>{item_id, title, rating}"]
        TEST --> |"extract_candidate_ids()"| CAND["Candidate IDs<br/>100 items"]
    end

    subgraph "Inference"
        PROMPT --> VLLM["vLLM Engine"]
        VLLM --> RAW["Raw Output<br/><think>...JSON"]
        RAW --> PARSE["Parsed Result<br/>• thinking<br/>• predictions[]<br/>• errors[]"]
    end

    subgraph "Validation"
        PARSE --> VALID{"item_id in<br/>candidates?"}
        VALID --> |"Yes"| METRICS["Metrics Calculation"]
        VALID --> |"No"| FAIL["rank = ∞<br/>(fail)"]
    end

    subgraph "Output"
        METRICS --> RESULTS["results/<br/>• predictions.jsonl<br/>• metrics_summary.json<br/>• error_statistics.json"]
        FAIL --> STATS["Error Statistics<br/>• parse_failure_rate<br/>• invalid_item_rate"]
    end

    style HF fill:#e1f5fe
    style VLLM fill:#fff3e0
    style VALID fill:#ffecb3
    style RESULTS fill:#c8e6c9
```

---

## 10. 환경 구성

```mermaid
graph TB
    subgraph "Hardware"
        GPU["Nvidia 5090<br/>36GB VRAM"]
        CPU["Host System"]
    end

    subgraph "Software Stack"
        VENV["Python venv"]
        VENV --> TORCH["PyTorch 2.2+"]
        VENV --> TRANS["Transformers 4.57.3"]
        VENV --> PEFT["PEFT 0.13.0"]
        VENV --> VLLM["vLLM"]
        VENV --> BNB["bitsandbytes"]
    end

    subgraph "External Services"
        HF["HuggingFace Hub<br/>• Datasets<br/>• Models"]
        OPENAI["OpenAI API<br/>GPT-4.1 (RQ4)"]
    end

    subgraph "Environment Variables"
        ENV["HF_TOKEN<br/>OPENAI_API_KEY"]
    end

    GPU --> VLLM
    TORCH --> TRANS
    ENV --> HF
    ENV --> OPENAI

    style GPU fill:#ffecb3
    style VLLM fill:#e1f5fe
    style HF fill:#f3e5f5
```

---

## 참조 문서

| 문서 | 설명 |
|------|------|
| `CLAUDE.md` | 프로젝트 상세 가이드 |
| `detail_task_plan.md` | 작업 계획서 |
| `IMPLEMENTATION_SUMMARY.md` | 구현 요약 |
| `DATA_FLOW.md` | 데이터 흐름 상세 |
| `RQ_EXPERIMENT_MAP.md` | RQ별 실험 매핑 |

