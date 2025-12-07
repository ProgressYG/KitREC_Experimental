# Experimental_test 평가 진행과정 설명

## huggingface 의 모델괴 데이터를 읽어들여 kitec 평가를 진행함

## KITREC 의 베이스라인 모델
 - @baselinemodel.md 파일 참조
 - CoNet, DTCRD, LLM4CDR 

## 평가지표 :
### KITREC 의 추천 아이템 성능 평가 및 베이스라인(모델 비교) 평가 기준 
 - Hit@1 : 정확히 1위로 예측 / 0~100%
 - Hit@5 : Top-5 안에 정답 포함 / 0~100%
 - Hit@10 : Top-10 안에 정답 포함 / 0~100%  -> Hit@10:  "Top-10에 있나요?"  →  있으면 1, 없으면 0
 - MRR (Mean Reciprocal Rank): 예측 리스트에서 몇 번째에 나타나는지 평가
    순위 | Reciprocal Rank
    1위 | 1.000
    2위 | 0.500
    3위 | 0.333
    5위 | 0.200
    10위 | 0.100
    특징: 상위 순위에 가중치를 부여 (1위 예측이 매우 중요)
 - NDCG@5  : Top-5 에 랭킹품질
 - NDCG@10 : Top-10 에 랭킹품질(논문 표준) -> "Top-10에서 순위 품질은?"  →  순위에 따라 0.29~1.0
    순위 | NDCG@10
    1위  | 1.000
    2위  | 0.631
    3위  | 0.500
    5위  | 0.387
    10위 | 0.289
    특징: 로그 할인으로 순위가 낮아질수록 감소폭이 줄어듦

### USER TYPE 별 추천아이템 성능평가
#### CORE LEVEL 평가
| Target Core | User Type | Count per Domain | 설명 |
|-------------|-----------|------------------|------|
| **1-core** | source_only | 3,000 | 극한 Cold-start (target=1) |
| **2-core** | cold_start_2core | 3,000 | 심각한 Cold-start (target=2) |
| **3-core** | cold_start_3core | 3,000 | 중간 Cold-start (target=3) |
| **4-core** | cold_start_4core | 3,000 | 경미한 Cold-start (target=4) |
| **5-core** | overlapping (target=5~9) | 필터링 필요 | Warm 시작 (target≥5) |
| **10-core** | overlapping (target≥10) | 필터링 필요 | 풍부한 Target (target≥10) |

#### 데이터 설명
```
Younggooo/kitrec-dualft_movies-seta     # 12,000 samples
Younggooo/kitrec-dualft_movies-setb     # 12,000 samples
Younggooo/kitrec-dualft_music-seta      # 12,000 samples
Younggooo/kitrec-dualft_music-setb      # 12,000 samples
Younggooo/kitrec-singleft_movies-seta   # 3,000 samples
Younggooo/kitrec-singleft_movies-setb   # 3,000 samples
Younggooo/kitrec-singleft_music-seta    # 3,000 samples
Younggooo/kitrec-singleft_music-setb    # 3,000 samples
```
**Validation Data (2 repositories):** (추후 DPO, RLVR 에 활용할 예정)
```
Younggooo/kitrec-val-seta               # 12,000 samples
Younggooo/kitrec-val-setb               # 12,000 samples
```

**Test Data (2 repositories):**
```
Younggooo/kitrec-test-seta              # 30,000 samples
Younggooo/kitrec-test-setb              # 30,000 samples
```
aining Datasets (8 repositories):*
- https://huggingface.co/datasets/Younggooo/kitrec-dualft_movies-seta
- https://huggingface.co/datasets/Younggooo/kitrec-dualft_movies-setb
- https://huggingface.co/datasets/Younggooo/kitrec-dualft_music-seta
- https://huggingface.co/datasets/Younggooo/kitrec-dualft_music-setb
- https://huggingface.co/datasets/Younggooo/kitrec-singleft_movies-seta
- https://huggingface.co/datasets/Younggooo/kitrec-singleft_movies-setb
- https://huggingface.co/datasets/Younggooo/kitrec-singleft_music-seta
- https://huggingface.co/datasets/Younggooo/kitrec-singleft_music-setb


*Validation Datasets (2 repositories):*(추후 DPO, RLVR 에 활용할 예정)
- https://huggingface.co/datasets/Younggooo/kitrec-val-seta
- https://huggingface.co/datasets/Younggooo/kitrec-val-setb


*Test Datasets (2 repositories):*
- https://huggingface.co/datasets/Younggooo/kitrec-test-seta
- https://huggingface.co/datasets/Younggooo/kitrec-test-setb


### KIREC 의 추천 설명력 평가 
#### [ "confidence": 9.5 ] 와 같은 추천 신뢰 점수 평가
 - 정답 아이템의 Rating 과 평균계열 평가 지표로 비교
 - TEST데이터 상에서는 5점 만점, Prediction confidence float 형태의 0~10 점에 대한 정규화가 필요함
 - Rating Prediction based :  MAE, RMSE

#### [ "rationale": "역사 소설 선호도와 연계된..." ] 와 같은 추천 설명력 평가 
 - Perplexity(PPL) : 추천 설명에 대한 평가지표로 사용
 - 퍼플렉시티는 **"LLM이 얼마나 확신을 가지고 다음 토큰을 예측하는가"**를 측정하는 지표이므로 rationale 에 평가지표로 사용함


### 실험 평가 진행 환경
 - vllm 기반
 - Nvida 5090 vram 36GB 환경
  - VENV 환경으로 Kitec 평가실험을 진행할 예정임



# (부가설명) KitREC Fine-tuning PRD (Product Requirements Document)
## Key Decisions (Finalized)

| 항목 | 결정 사항 |
|------|----------|
| **Train/Val Split** | 코드 내부 90/10 stratified split (user_type 기준) |
| **SingleFT 초기화** | Base Model(Qwen3-14B) 독립 학습 (DualFT 체크포인트 미사용) |
| **Thinking 학습** | `<think>...</think>` 포함 전체 output 학습 (Chain-of-Thought) |
| **데이터 제공** | HuggingFace Hub 업로드 → RunPod에서 다운로드 |
| **구현 범위** | 전체 파이프라인 (train, evaluate, upload_to_hub) |

---

## 1. Overview

### 1.1 Project Summary

KitREC (Knowledge-Instruction Transfer for Recommendation) Fine-tuning 프로젝트는 Cross-Domain Recommendation을 위해 Qwen3-14B 모델을 PEFT QLoRA 방식으로 학습합니다.

**핵심 목표:**
- Source Domain(Books)의 풍부한 이력으로 Target Domain(Movies/Music)의 Cold-start 문제 해결
- 4개 Fine-tuning 모델을 통한 체계적인 Cross-Domain Transfer 학습
- Set A(Hard Negatives) vs Set B(Random) A/B Testing으로 실험 공정성 확보

### 1.2 Hardware & Infrastructure

| 항목 | 사양 |
|------|------|
| **GPU** | RunPod A100 80GB VRAM 또는 H100 80GB |
| **Training Framework** | Hugging Face Transformers + PEFT |
| **Data Storage** | Hugging Face Hub |
| **Model Storage** | Hugging Face Hub (Fine-tuned models) |
| **Working Directory** | `./finetuning/` |

---

## 2. Instruction Data Structure

### 2.1 Directory Structure (Set-Centric View)

Fine-tuning에 사용할 **Set 중심 데이터 구조**입니다. Set A/B 하위에 모델별 Train 데이터가 정리되어 있습니다.

```
./dataprocess/data/instruction/
│
├── setA/                                      # ⭐ Set A: Hybrid Candidates (Hard Negatives)
│   │
│   ├── train.jsonl                            # 30,000 samples (717MB) - 전체 Train
│   │
│   ├── by_model/                              # 📁 모델별 Train 데이터 (SFT 학습용)
│   │   ├── dualft_movies.jsonl                # 12,000 samples - Movies & TV 추천
│   │   ├── dualft_music.jsonl                 # 12,000 samples - Music 추천
│   │   ├── singleft_movies.jsonl              # 3,000 samples - Movies 극한 Cold-start
│   │   └── singleft_music.jsonl               # 3,000 samples - Music 극한 Cold-start
│   │
│   ├── val.jsonl                              # 12,000 samples (142MB) - DPO/GRPO용
│   └── test.jsonl                             # 30,000 samples (351MB) - 실험 결과용
│
└── setB/                                      # ⭐ Set B: Random Candidates (Fair Baseline)
    │
    ├── train.jsonl                            # 30,000 samples (716MB) - 전체 Train
    │
    ├── by_model/                              # 📁 모델별 Train 데이터 (SFT 학습용)
    │   ├── dualft_movies.jsonl                # 12,000 samples - Movies & TV 추천
    │   ├── dualft_music.jsonl                 # 12,000 samples - Music 추천
    │   ├── singleft_movies.jsonl              # 3,000 samples - Movies 극한 Cold-start
    │   └── singleft_music.jsonl               # 3,000 samples - Music 극한 Cold-start
    │
    ├── val.jsonl                              # 12,000 samples (146MB) - DPO/GRPO용
    └── test.jsonl                             # 30,000 samples (362MB) - 실험 결과용
```

### 2.2 Data File Summary

| Set | 파일 | Samples | Size | 용도 |
|-----|------|---------|------|------|
| **Set A** | `setA/by_model/dualft_movies.jsonl` | 12,000 | - | SFT 학습 (DualFT-Movies) |
| **Set A** | `setA/by_model/dualft_music.jsonl` | 12,000 | - | SFT 학습 (DualFT-Music) |
| **Set A** | `setA/by_model/singleft_movies.jsonl` | 3,000 | - | SFT 학습 (SingleFT-Movies) |
| **Set A** | `setA/by_model/singleft_music.jsonl` | 3,000 | - | SFT 학습 (SingleFT-Music) |
| **Set A** | `setA/val.jsonl` | 12,000 | 142MB | DPO/GRPO (추후) |
| **Set A** | `setA/test.jsonl` | 30,000 | 351MB | 실험 결과 평가 |
| **Set B** | `setB/by_model/dualft_movies.jsonl` | 12,000 | - | SFT 학습 (DualFT-Movies) |
| **Set B** | `setB/by_model/dualft_music.jsonl` | 12,000 | - | SFT 학습 (DualFT-Music) |
| **Set B** | `setB/by_model/singleft_movies.jsonl` | 3,000 | - | SFT 학습 (SingleFT-Movies) |
| **Set B** | `setB/by_model/singleft_music.jsonl` | 3,000 | - | SFT 학습 (SingleFT-Music) |
| **Set B** | `setB/val.jsonl` | 12,000 | 146MB | DPO/GRPO (추후) |
| **Set B** | `setB/test.jsonl` | 30,000 | 362MB | 실험 결과 평가 |

### 2.3 Model-Data Mapping

| Fine-tuning Model | Target Domain | User Types | Samples | Set A Path | Set B Path |
|-------------------|---------------|------------|---------|------------|------------|
| **DualFT-Movies** | Movies & TV | overlapping_books_movies (3K) + cold_start_*_movies (9K) | **12,000** | `setA/by_model/dualft_movies.jsonl` | `setB/by_model/dualft_movies.jsonl` |
| **DualFT-Music** | Music | overlapping_books_music (3K) + cold_start_*_music (9K) | **12,000** | `setA/by_model/dualft_music.jsonl` | `setB/by_model/dualft_music.jsonl` |
| **SingleFT-Movies** | Movies & TV | source_only_movies (3K) | **3,000** | `setA/by_model/singleft_movies.jsonl` | `setB/by_model/singleft_movies.jsonl` |
| **SingleFT-Music** | Music | source_only_music (3K) | **3,000** | `setA/by_model/singleft_music.jsonl` | `setB/by_model/singleft_music.jsonl` |

### 2.4 Data Usage by Training Stage

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KitREC Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: SFT (Supervised Fine-Tuning)                                  │
│  ├── Data: setA/by_model/*.jsonl or setB/by_model/*.jsonl               │
│  ├── Format: instruction + output (with <think> reasoning)              │
│  └── Purpose: 기본 추천 능력 학습                                        │
│                                                                         │
│  Stage 2: DPO/GRPO (Preference Optimization) - 추후 진행                 │
│  ├── Data: setA/val.jsonl or setB/val.jsonl                             │
│  ├── Format: instruction + input + ground_truth (output 없음)           │
│  └── Purpose: 선호도 최적화, 랭킹 품질 향상                               │
│                                                                         │
│  Stage 3: Evaluation (실험 결과)                                         │
│  ├── Data: setA/test.jsonl or setB/test.jsonl                           │
│  ├── Format: instruction + input + ground_truth (output 없음)           │
│  └── Purpose: Hit@K, MRR, NDCG@10 등 최종 성능 평가                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 User Type → Model Mapping Detail

```
User Type Distribution (30,000 users total)
│
├── Target: Movies & TV (15,000 users)
│   │
│   ├── DualFT-Movies (12,000 users) ─────────────────────────┐
│   │   ├── overlapping_books_movies    3,000   (Warm users) │
│   │   ├── cold_start_2core_movies     3,000   (Cold-start) │
│   │   ├── cold_start_3core_movies     3,000   (Cold-start) │
│   │   └── cold_start_4core_movies     3,000   (Cold-start) │
│   │                                                         │
│   └── SingleFT-Movies (3,000 users) ────────────────────────┤
│       └── source_only_movies          3,000   (Extreme CS)  │
│                                                             │
└── Target: Music (15,000 users)                              │
    │                                                         │
    ├── DualFT-Music (12,000 users) ──────────────────────────┤
    │   ├── overlapping_books_music     3,000   (Warm users) │
    │   ├── cold_start_2core_music      3,000   (Cold-start) │
    │   ├── cold_start_3core_music      3,000   (Cold-start) │
    │   └── cold_start_4core_music      3,000   (Cold-start) │
    │                                                         │
    └── SingleFT-Music (3,000 users) ─────────────────────────┘
        └── source_only_music           3,000   (Extreme CS)
```

### 2.6 Fine-tuning Working Directory

```
./finetuning/                                  # Fine-tuning 작업 폴더
├── PRD_FINETUNING.md                          # 본 문서
├── finetuning_detail_task.md                  # Task tracking
├── CLAUDE.md                                  # Claude Code guidance
├── configs/                                   # 학습 설정 파일
│   ├── base_config.yaml
│   ├── dualft_movies.yaml
│   ├── dualft_music.yaml
│   ├── singleft_movies.yaml
│   └── singleft_music.yaml
├── docs/                                      # 📁 Documentation (NEW)
│   ├── HYPERPARAMETER_TUNING_GUIDE.md         # 하이퍼파라미터 튜닝 가이드
│   └── RUNPOD_TRAINING_GUIDE.md               # RunPod 학습 실행 가이드
├── scripts/                                   # 학습 스크립트
│   ├── train.py                               # Training (with monitoring)
│   ├── evaluate.py                            # Evaluation (robust parsing)
│   ├── upload_to_hub.py                       # Training data upload
│   ├── upload_val_to_hub.py                   # Validation data upload
│   ├── upload_test_to_hub.py                  # Test data upload
│   └── upload_model_to_hub.py                 # Trained model upload
├── src/                                       # Source utilities
│   ├── data_utils.py                          # Data loading, tokenization
│   ├── model_utils.py                         # QLoRA model setup
│   └── metrics.py                             # Evaluation metrics
├── results/                                   # 체크포인트 및 결과
│   ├── dualft_movies/
│   │   ├── setA/                              # Set A로 학습한 모델
│   │   └── setB/                              # Set B로 학습한 모델
│   ├── dualft_music/
│   │   ├── setA/
│   │   └── setB/
│   ├── singleft_movies/
│   │   ├── setA/
│   │   └── setB/
│   └── singleft_music/
│       ├── setA/
│       └── setB/
└── logs/                                      # 학습 로그 (WandB + local)
```

### 2.7 Data Format

**Training Data (train.jsonl)**
```json
{
  "instruction": "[Student Prompt - GT 미포함]",
  "input": "",
  "output": "<think>\n[Teacher가 생성한 GT 기반 reasoning]\n</think>\n```json\n{GT item JSON}\n```",
  "metadata": {
    "user_id": "string",
    "user_type": "string",
    "user_category": "overlapping | source_only | cold_start",
    "target_domain": "Movies & TV | Music",
    "source_domain": "Books",
    "target_core": "integer",
    "books_core": "integer",
    "candidate_set": "A | B",
    "gt_item_id": "string",
    "thinking_length": "integer",
    "confidence_score": "float",
    "generation_time_sec": "float"
  }
}
```

**Validation/Test Data (val.jsonl, test.jsonl)**
```json
{
  "instruction": "[System Prompt]",
  "input": "[User Prompt with candidates]",
  "output": "",
  "ground_truth": "[{\"item_id\": \"xxx\", \"title\": \"...\", ...}]",
  "metadata": { ... }
}
```

---

## 3. User Type Distribution

### 3.1 10 User Types (30,000 Users Total)

| User Type | Count | Target Domain | Training Model |
|-----------|-------|---------------|----------------|
| overlapping_books_movies | 3,000 | Movies & TV | DualFT-Movies |
| overlapping_books_music | 3,000 | Music | DualFT-Music |
| source_only_movies | 3,000 | Movies & TV | SingleFT-Movies |
| source_only_music | 3,000 | Music | SingleFT-Music |
| cold_start_2core_movies | 3,000 | Movies & TV | DualFT-Movies |
| cold_start_2core_music | 3,000 | Music | DualFT-Music |
| cold_start_3core_movies | 3,000 | Movies & TV | DualFT-Movies |
| cold_start_3core_music | 3,000 | Music | DualFT-Music |
| cold_start_4core_movies | 3,000 | Movies & TV | DualFT-Movies |
| cold_start_4core_music | 3,000 | Music | DualFT-Music |

### 3.2 4 Fine-tuning Models

| Model | Training Users | Sample Count | Purpose |
|-------|----------------|--------------|---------|
| **DualFT-Movies** | overlapping_books_movies + cold_start_*_movies | 12,000 | Movies & TV 추천 (Warm + Cold) |
| **DualFT-Music** | overlapping_books_music + cold_start_*_music | 12,000 | Music 추천 (Warm + Cold) |
| **SingleFT-Movies** | source_only_movies | 3,000 | Movies & TV 극한 Cold-start |
| **SingleFT-Music** | source_only_music | 3,000 | Music 극한 Cold-start |

---

## 4. Train/Val Split Strategy

### 4.1 Stratified Split by User Type (9:1)

모든 10개 user_type에서 동일한 비율로 Train/Val을 분할하여 **정확한 train loss와 val loss** 측정을 보장합니다.

```python
# Stratified split implementation
SPLIT_CONFIG = {
    "method": "stratified_by_user_type",
    "train_ratio": 0.9,
    "val_ratio": 0.1,
    "stratify_by": ["user_type"],
    "seed": 42,
    "ensure_same_users_across_sets": True  # Set A와 Set B 동일 사용자 분할
}
```

### 4.2 Split Counts per Model

**DualFT Models (12,000 samples each)**
| User Type | Total | Train (90%) | Val (10%) |
|-----------|-------|-------------|-----------|
| overlapping_books_movies | 3,000 | 2,700 | 300 |
| cold_start_2core_movies | 3,000 | 2,700 | 300 |
| cold_start_3core_movies | 3,000 | 2,700 | 300 |
| cold_start_4core_movies | 3,000 | 2,700 | 300 |
| **DualFT-Movies Total** | **12,000** | **10,800** | **1,200** |

| User Type | Total | Train (90%) | Val (10%) |
|-----------|-------|-------------|-----------|
| overlapping_books_music | 3,000 | 2,700 | 300 |
| cold_start_2core_music | 3,000 | 2,700 | 300 |
| cold_start_3core_music | 3,000 | 2,700 | 300 |
| cold_start_4core_music | 3,000 | 2,700 | 300 |
| **DualFT-Music Total** | **12,000** | **10,800** | **1,200** |

**SingleFT Models (3,000 samples each)**
| User Type | Total | Train (90%) | Val (10%) |
|-----------|-------|-------------|-----------|
| source_only_movies | 3,000 | 2,700 | 300 |
| **SingleFT-Movies Total** | **3,000** | **2,700** | **300** |

| User Type | Total | Train (90%) | Val (10%) |
|-----------|-------|-------------|-----------|
| source_only_music | 3,000 | 2,700 | 300 |
| **SingleFT-Music Total** | **3,000** | **2,700** | **300** |

### 4.3 Set A/B Consistency

**중요**: Set A와 Set B는 **동일한 사용자 분할**을 사용합니다.

```python
# 동일한 user_id를 Train/Val에 할당
# Set A train users == Set B train users
# Set A val users == Set B val users

def create_consistent_split(users_df, val_ratio=0.1, seed=42):
    """User-level split을 먼저 수행하고, Set A/B 모두에 적용"""
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=seed
    )

    train_idx, val_idx = next(splitter.split(
        users_df,
        users_df['user_type']
    ))

    train_users = set(users_df.iloc[train_idx]['user_id'])
    val_users = set(users_df.iloc[val_idx]['user_id'])

    return train_users, val_users
```

---

## 5. PEFT QLoRA Configuration

### 5.1 Model Configuration

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType
import torch

# =============================================================================
# BASE MODEL: Qwen3-14B
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-14B"

# =============================================================================
# 4-bit QUANTIZATION (QLoRA)
# =============================================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normal Float 4-bit (optimal for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16,  # A100/H100 native support
    bnb_4bit_use_double_quant=True,       # ~1GB VRAM 절약
)

# =============================================================================
# LoRA CONFIGURATION - DualFT (12K samples, cross-domain transfer)
# =============================================================================
lora_config_dualft = LoraConfig(
    # Rank: 32 (Expert-Verified: increased for better cross-domain reasoning)
    r=32,

    # Alpha: 64 (Maintain alpha/r = 2 ratio)
    lora_alpha=64,

    # Target Modules: Qwen3 Attention + MLP
    target_modules=[
        "q_proj",      # Query projection
        "k_proj",      # Key projection
        "v_proj",      # Value projection
        "o_proj",      # Output projection
        "gate_proj",   # MLP gating (SwiGLU)
        "up_proj",     # MLP up projection
        "down_proj",   # MLP down projection
    ],

    # Dropout: 0.08 (Expert-Verified: explicit for DualFT regularization)
    lora_dropout=0.08,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# =============================================================================
# LoRA CONFIGURATION - SingleFT (3K samples, overfitting prevention)
# =============================================================================
lora_config_singleft = LoraConfig(
    # Rank: 24 (Expert-Verified: slightly less than DualFT)
    r=24,

    # Alpha: 48 (Maintain alpha/r = 2 ratio)
    lora_alpha=48,

    # Target Modules: Qwen3 Attention + MLP
    target_modules=[
        "q_proj",      # Query projection
        "k_proj",      # Key projection
        "v_proj",      # Value projection
        "o_proj",      # Output projection
        "gate_proj",   # MLP gating (SwiGLU)
        "up_proj",     # MLP up projection
        "down_proj",   # MLP down projection
    ],

    # Dropout: 0.15 (Expert-Verified: aggressive overfitting prevention)
    lora_dropout=0.15,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### 5.2 Training Arguments (DualFT - 12K samples)

```python
from transformers import TrainingArguments

# =============================================================================
# TRAINING ARGUMENTS - DualFT Models (12,000 samples)
# =============================================================================
training_args_dualft = TrainingArguments(
    output_dir="./results/dualft_movies",  # 또는 dualft_music

    # ----- Batch Size -----
    # Effective batch = 4 × 8 = 32
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,

    # ----- Learning Rate -----
    # 2e-4: QLoRA 표준 (frozen base weights로 높은 LR 가능)
    learning_rate=2e-4,

    # ----- Epochs -----
    # 3 epochs: 12K × 3 = 36K effective samples
    # Early stopping으로 과적합 방지
    num_train_epochs=3,

    # ----- Evaluation Strategy -----
    # 200 steps마다 검증 (epoch당 ~5-6회)
    eval_strategy="steps",
    eval_steps=200,

    # ----- Save Strategy -----
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,  # 최근 3개 체크포인트만 유지

    # ----- Best Model Selection -----
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ----- Learning Rate Schedule -----
    warmup_ratio=0.05,           # 5% warmup
    lr_scheduler_type="cosine",  # Cosine annealing

    # ----- Precision -----
    bf16=True,                   # A100/H100 native BF16
    fp16=False,

    # ----- Memory Optimization -----
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # ----- Regularization (Expert-Verified: Cross-Domain Overfitting Prevention) -----
    weight_decay=0.02,           # Cross-domain overfitting prevention
    label_smoothing_factor=0.05, # Confidence calibration for cross-domain

    # ----- NEFTune (Instruction Tuning 성능 향상) -----
    # Expert-Verified: DualFT 5.0 (standard), SingleFT 3.0 (smaller dataset)
    neftune_noise_alpha=5.0,  # DualFT: 5.0 explicit

    # ----- Optimizer -----
    optim="adamw_torch_fused",   # A100/H100 최적화
    max_grad_norm=1.0,           # Gradient clipping

    # ----- Data Loading -----
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    group_by_length=True,        # 유사 길이 샘플 그룹화

    # ----- Logging -----
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    report_to=["wandb"],  # TensorBoard → WandB (with error handling in train.py)

    # ----- Reproducibility -----
    seed=42,
    data_seed=42,
)
```

### 5.3 Training Arguments (SingleFT - 3K samples)

```python
# =============================================================================
# TRAINING ARGUMENTS - SingleFT Models (3,000 samples)
# =============================================================================
# SingleFT는 Base Model(Qwen3-14B)에서 독립적으로 학습 (DualFT 체크포인트 미사용)

training_args_singleft = TrainingArguments(
    output_dir="./results/singleft_movies",  # 또는 singleft_music

    # ----- Batch Size -----
    # 작은 데이터셋이므로 배치 크기 축소
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,  # Effective batch = 32 유지

    # ----- Learning Rate -----
    # Expert-Verified: 더 낮은 LR로 과적합 방지
    learning_rate=6e-5,  # 1e-4 → 6e-5 (aggressive overfitting prevention)

    # ----- Epochs -----
    # 6 epochs: 3K × 6 = 18K effective samples
    # DualFT (36K)와 유사한 수준의 학습량
    num_train_epochs=6,

    # ----- Evaluation Strategy -----
    # 더 자주 검증 (과적합 모니터링) - Expert-Verified: 100 → 50
    eval_strategy="steps",
    eval_steps=50,  # 100 → 50 for faster early stopping detection

    # ----- Save Strategy -----
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    # ----- Best Model Selection -----
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ----- Learning Rate Schedule -----
    warmup_ratio=0.1,            # 10% warmup (작은 데이터셋)
    lr_scheduler_type="cosine",

    # ----- Regularization (Expert-Verified: Aggressive Overfitting Prevention) -----
    weight_decay=0.05,           # 0.01 → 0.05 (stronger L2 regularization)
    label_smoothing_factor=0.1,  # Prevent overconfidence, improve generalization

    # ----- Precision & Memory -----
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # ----- NEFTune -----
    # Expert-Verified: SingleFT uses 3.0 (smaller dataset needs less noise)
    neftune_noise_alpha=3.0,  # 5.0 → 3.0 for SingleFT

    # ----- Optimizer -----
    optim="adamw_torch_fused",
    max_grad_norm=1.0,

    # ----- Data Loading -----
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    group_by_length=True,

    # ----- Logging -----
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=25,
    report_to=["wandb"],  # TensorBoard → WandB (with error handling in train.py)

    # ----- Reproducibility -----
    seed=42,
    data_seed=42,
)
```

### 5.4 Early Stopping Configuration

```python
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,      # 3회 연속 개선 없으면 중단
    early_stopping_threshold=0.001,  # 최소 0.1% 개선 필요
)
```

---

## 6. Training Pipeline

### 6.1 Training Hierarchy (Independent Training)

```
Training Pipeline (All models start from Base Model):

                    Qwen3-14B (Base Model)
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     ┌───────────┐   ┌───────────┐   ┌───────────┐
     │  DualFT   │   │  DualFT   │   │ SingleFT  │
     │  Movies   │   │  Music    │   │  Movies   │
     │   12K     │   │   12K     │   │    3K     │
     │ 3 epochs  │   │ 3 epochs  │   │ 6 epochs  │
     └───────────┘   └───────────┘   └───────────┘
                                           │
                                     ┌─────┘
                                     ▼
                               ┌───────────┐
                               │ SingleFT  │
                               │  Music    │
                               │    3K     │
                               │ 6 epochs  │
                               └───────────┘

Note: SingleFT 모델은 DualFT 체크포인트가 아닌 Base Model에서 독립 학습
```

### 6.2 Training Order

1. **DualFT-Movies** (Set A) → **DualFT-Movies** (Set B)
2. **DualFT-Music** (Set A) → **DualFT-Music** (Set B)
3. **SingleFT-Movies** (Set A) → **SingleFT-Movies** (Set B)
4. **SingleFT-Music** (Set A) → **SingleFT-Music** (Set B)

### 6.3 Estimated Training Time (A100 80GB)

| Model | Samples | Epochs | Est. Time |
|-------|---------|--------|-----------|
| DualFT-Movies (Set A) | 10,800 train | 3 | ~2-3 hours |
| DualFT-Movies (Set B) | 10,800 train | 3 | ~2-3 hours |
| DualFT-Music (Set A) | 10,800 train | 3 | ~2-3 hours |
| DualFT-Music (Set B) | 10,800 train | 3 | ~2-3 hours |
| SingleFT-Movies (Set A) | 2,700 train | 6 | ~1.5-2 hours |
| SingleFT-Movies (Set B) | 2,700 train | 6 | ~1.5-2 hours |
| SingleFT-Music (Set A) | 2,700 train | 6 | ~1.5-2 hours |
| SingleFT-Music (Set B) | 2,700 train | 6 | ~1.5-2 hours |
| **Total** | - | - | **~16-20 hours** |

---

## 7. VRAM Usage Estimation

### 7.1 Memory Breakdown (A100/H100 80GB)

```
=============================================================================
VRAM BREAKDOWN (Qwen3-14B QLoRA)
=============================================================================

Component                          | VRAM Usage
-----------------------------------|-------------
Qwen3-14B (4-bit quantized)        | ~8 GB
LoRA adapters (r=16, all modules)  | ~50 MB
Optimizer states (AdamW fused)     | ~100 MB
Gradients (with checkpointing)     | ~4 GB
Activations (batch=4, seq=8192)    | ~35 GB
KV Cache (during eval)             | ~2 GB
CUDA kernels + overhead            | ~3 GB
-----------------------------------|-------------
TOTAL (Training)                   | ~52 GB
TOTAL (Inference/Eval)             | ~15 GB

Headroom on 80GB                   | ~28 GB (SAFE)
```

### 7.2 Memory Optimization (OOM 발생 시)

```python
# Option 1: Batch size 축소
per_device_train_batch_size=2  # 4 → 2
gradient_accumulation_steps=16  # 8 → 16 (effective batch 유지)

# Option 2: Sequence length 제한 (참고: 기본값은 8192)
max_length=4096  # 8192 → 4096 (if OOM)

# Option 3: 8-bit Optimizer
optim="adamw_8bit"  # bitsandbytes 8-bit optimizer
```

---

## 8. Evaluation Metrics

### 8.1 Primary Metrics (Mandatory)

| Metric | Formula | Purpose |
|--------|---------|---------|
| **eval_loss** | Cross-entropy loss | 학습 진행 모니터링 |
| **Hit@1** | 1 if GT in top-1 | 정확도 (Exact match) |
| **Hit@5** | 1 if GT in top-5 | Top-K 성능 |
| **MRR** | 1/rank of GT | 랭킹 품질 |
| **NDCG@10** | DCG@10 / IDCG@10 | Position-weighted 성능 |

### 8.2 Secondary Metrics

| Metric | Purpose |
|--------|---------|
| **Confidence MAE** | 예측 신뢰도 보정 |
| **Thinking Length** | 추론 깊이 |
| **Cross-Domain Refs** | Source→Target 참조 횟수 |

### 8.3 Stratified Evaluation

모든 평가는 다음 기준으로 층화(stratified) 분석:

```python
STRATIFIED_EVALUATION = {
    "by_user_type": [
        "overlapping", "source_only",
        "cold_start_2core", "cold_start_3core", "cold_start_4core"
    ],
    "by_target_domain": ["Movies & TV", "Music"],
    "by_target_core": [1, 2, 3, 4, "5+", "10+"],
    "by_candidate_set": ["Set A (Hybrid)", "Set B (Random)"]
}
```

---

## 9. Hugging Face Integration

### 9.1 Data Upload to Hub (Per-Model Repositories)

**변경 사항 (2025-11-30):**
- 모델별 별도 리포지토리로 업로드 (8개 리포지토리)
- test 데이터 제외 (로컬 평가)
- 각 리포지토리에 README.md (데이터셋 카드) 포함

```bash
# Upload single model dataset with README.md
python scripts/upload_to_hub.py --model_type dualft_movies --set A

# Upload all datasets for Set A
python scripts/upload_to_hub.py --set A --all

# Upload all datasets (both sets, 8 repositories)
python scripts/upload_to_hub.py --all
```

**Dataset Repositories (8개):**
```
Younggooo/kitrec-dualft-movies-seta     # 12,000 samples
Younggooo/kitrec-dualft-movies-setb     # 12,000 samples
Younggooo/kitrec-dualft-music-seta      # 12,000 samples
Younggooo/kitrec-dualft-music-setb      # 12,000 samples
Younggooo/kitrec-singleft-movies-seta   # 3,000 samples
Younggooo/kitrec-singleft-movies-setb   # 3,000 samples
Younggooo/kitrec-singleft-music-seta    # 3,000 samples
Younggooo/kitrec-singleft-music-setb    # 3,000 samples
```

### 9.2 Model Upload to Hub

```python
from peft import PeftModel

# After training
def save_and_upload_model(model, tokenizer, output_dir, hub_name):
    # Save locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push to Hub
    model.push_to_hub(
        hub_name,
        private=True,
        token="hf_xxx"
    )
    tokenizer.push_to_hub(
        hub_name,
        private=True,
        token="hf_xxx"
    )

# Usage
save_and_upload_model(
    model=trainer.model,
    tokenizer=tokenizer,
    output_dir="./results/dualft_movies/best",
    hub_name="your-username/kitrec-dualft-movies-setA"
)
```

### 9.3 Hub Repository Structure (Updated 2025-11-30)

```
Hugging Face Hub:
├── Training Datasets (모델별 별도 리포지토리)
│   ├── kitrec-dualft_movies-seta    # DualFT Movies Set A (12K)
│   ├── kitrec-dualft_movies-setb    # DualFT Movies Set B (12K)
│   ├── kitrec-dualft_music-seta     # DualFT Music Set A (12K)
│   ├── kitrec-dualft_music-setb     # DualFT Music Set B (12K)
│   ├── kitrec-singleft_movies-seta  # SingleFT Movies Set A (3K)
│   ├── kitrec-singleft_movies-setb  # SingleFT Movies Set B (3K)
│   ├── kitrec-singleft_music-seta   # SingleFT Music Set A (3K)
│   └── kitrec-singleft_music-setb   # SingleFT Music Set B (3K)
│
├── Validation Datasets (Set별)
│   ├── kitrec-val-seta              # Validation Set A (12K)
│   └── kitrec-val-setb              # Validation Set B (12K)
│
├── Test Datasets (Set별)
│   ├── kitrec-test-seta             # Test Set A (30K)
│   └── kitrec-test-setb             # Test Set B (30K)
│
└── Models (학습 완료 후 업로드)
    ├── kitrec-dualft-movies-setA-model
    ├── kitrec-dualft-movies-setB-model
    ├── kitrec-dualft-music-setA-model
    ├── kitrec-dualft-music-setB-model
    ├── kitrec-singleft-movies-setA-model
    ├── kitrec-singleft-movies-setB-model
    ├── kitrec-singleft-music-setA-model
    └── kitrec-singleft-music-setB-model
```

---

## 10. Implementation Checklist

### 10.1 Pre-Training

- [ ] RunPod A100/H100 80GB 인스턴스 준비
- [x] Hugging Face Hub 데이터셋 업로드 (12 repositories - train/val/test)
- [ ] Flash Attention 2 설치: `pip install flash-attn --no-build-isolation`
- [ ] bitsandbytes 설치: `pip install bitsandbytes`
- [ ] PEFT 설치: `pip install peft`
- [ ] Train/Val stratified split 스크립트 작성
- [ ] 설정 파일 (YAML) 작성

### 10.2 Training

- [ ] DualFT-Movies (Set A) 학습
- [ ] DualFT-Movies (Set B) 학습
- [ ] DualFT-Music (Set A) 학습
- [ ] DualFT-Music (Set B) 학습
- [ ] SingleFT-Movies (Set A) 학습 (Base Model 독립 학습)
- [ ] SingleFT-Movies (Set B) 학습 (Base Model 독립 학습)
- [ ] SingleFT-Music (Set A) 학습 (Base Model 독립 학습)
- [ ] SingleFT-Music (Set B) 학습 (Base Model 독립 학습)

### 10.3 Evaluation

- [ ] Val set eval_loss 수렴 확인
- [ ] Test set Hit@1, Hit@5, MRR, NDCG@10 계산
- [ ] User type별 stratified 성능 분석
- [ ] Set A vs Set B 비교 분석

### 10.4 Post-Training

- [ ] Best 체크포인트 Hugging Face Hub 업로드
- [ ] Training loss/val loss 그래프 저장
- [ ] 최종 성능 리포트 작성

---

## 11. Appendix

### A. Complete Training Script

```python
#!/usr/bin/env python3
"""
KitREC Fine-tuning Script
PEFT QLoRA on Qwen3-14B for Cross-Domain Recommendation
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import argparse

def main(args):
    # Model configuration
    MODEL_NAME = "Qwen/Qwen3-14B"

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset from Hub
    print("Loading dataset...")
    dataset = load_dataset(f"your-username/kitrec-instruction-{args.candidate_set.lower()}")

    # Filter by model type
    if args.model_type.startswith("dualft"):
        domain = "Movies & TV" if "movies" in args.model_type else "Music"
        categories = ["overlapping", "cold_start"]
        dataset = dataset.filter(
            lambda x: x["metadata"]["target_domain"] == domain and
                     x["metadata"]["user_category"] in categories
        )
    else:  # singleft
        domain = "Movies & TV" if "movies" in args.model_type else "Music"
        dataset = dataset.filter(
            lambda x: x["metadata"]["target_domain"] == domain and
                     x["metadata"]["user_category"] == "source_only"
        )

    # Tokenization
    def tokenize_function(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=8192,  # Expert-Verified: 4096→8192 (99.26% coverage)
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    val_dataset = dataset["validation"].map(tokenize_function, batched=True)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        max_length=8192,  # Expert-Verified: 4096→8192 (99.26% coverage)
        label_pad_token_id=-100,
    )

    # Training arguments
    is_singleft = args.model_type.startswith("singleft")
    training_args = TrainingArguments(
        output_dir=f"./results/{args.model_type}_{args.candidate_set}",
        per_device_train_batch_size=2 if is_singleft else 4,
        per_device_eval_batch_size=4 if is_singleft else 8,
        gradient_accumulation_steps=16 if is_singleft else 8,
        learning_rate=1e-4 if is_singleft else 2e-4,
        num_train_epochs=6 if is_singleft else 3,
        eval_strategy="steps",
        eval_steps=100 if is_singleft else 200,
        save_strategy="steps",
        save_steps=100 if is_singleft else 200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.1 if is_singleft else 0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=5.0,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        logging_steps=25 if is_singleft else 50,
        report_to=["tensorboard"],
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save best model
    print("Saving model...")
    trainer.save_model(f"./results/{args.model_type}_{args.candidate_set}/best")
    tokenizer.save_pretrained(f"./results/{args.model_type}_{args.candidate_set}/best")

    # Push to Hub
    if args.push_to_hub:
        print("Pushing to Hub...")
        trainer.model.push_to_hub(
            f"your-username/kitrec-{args.model_type}-{args.candidate_set.lower()}",
            private=True
        )

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["dualft_movies", "dualft_music",
                               "singleft_movies", "singleft_music"])
    parser.add_argument("--candidate_set", type=str, required=True,
                       choices=["setA", "setB"])
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    main(args)
```

### B. Data Preparation Script

```python
#!/usr/bin/env python3
"""
Prepare KitREC data: Stratified Train/Val split by user_type
"""

import json
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import random

def stratified_split(data, val_ratio=0.1, seed=42):
    """Stratified split by user_type"""
    random.seed(seed)

    # Group by user_type
    type_groups = defaultdict(list)
    for item in data:
        user_type = item["metadata"]["user_type"]
        type_groups[user_type].append(item)

    train_data, val_data = [], []

    for user_type, items in type_groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * (1 - val_ratio))
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])

        print(f"{user_type}: Train {split_idx}, Val {len(items) - split_idx}")

    return train_data, val_data

def main():
    for candidate_set in ["setA", "setB"]:
        print(f"\n=== Processing {candidate_set} ===")

        # Load full train data
        filepath = f"../data/instruction/{candidate_set}/train.jsonl"
        with open(filepath, 'r') as f:
            data = [json.loads(line) for line in f]

        print(f"Total samples: {len(data)}")

        # Stratified split
        train_data, val_data = stratified_split(data, val_ratio=0.1, seed=42)

        # Save
        train_out = f"../data/instruction/{candidate_set}/train_split.jsonl"
        val_out = f"../data/instruction/{candidate_set}/val_split.jsonl"

        with open(train_out, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(val_out, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Train: {len(train_data)} -> {train_out}")
        print(f"Val: {len(val_data)} -> {val_out}")

if __name__ == "__main__":
    main()
```

---

---

**END OF DOCUMENT**
