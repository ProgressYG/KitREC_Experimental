# KitREC RunPod 실험 가이드 (Music Domain)

## 실험 개요

| 실험 | 목적 | 모델 수 |
|------|------|---------|
| KitREC 평가 | RQ2, RQ3 | 4개 (DualFT/SingleFT × SetA/SetB) |
| LLM Baseline (Qwen) | RQ2 비교 | 2개 (LLM4CDR, Vanilla) |
| LLM Baseline (GPT-4.1-mini) | RQ2 비교 | 2개 (LLM4CDR, Vanilla) |
| DL Baseline | RQ2 비교 | 2개 (CoNet, DTCDR) |
| Ablation Study | RQ1 | 4개 변형 |

---

## Phase 0: 환경 설정 (5분)

```bash
# 1. 프로젝트 디렉토리로 이동
cd /workspace/Experimental_test

# 2. 의존성 설치
pip install -r requirements.txt

# 3. .env 파일 확인 (HF_TOKEN, OPENAI_API_KEY)
cat .env

# 4. 환경 검증
python scripts/verify_local.py

# 5. 도메인 상태 확인
python scripts/run_kitrec_eval.py --show_status
```

**필요한 환경변수 (.env 파일):**
```
HF_TOKEN=hf_xxxxxxxxxxxxx           # 필수
OPENAI_API_KEY=sk-xxxxxxxxxxxxx     # GPT-4.1-mini 사용 시 필요
```

---

## Phase 1: Sanity Check (10분)

전체 실험 전 빠른 검증:

```bash
# KitREC 모델 테스트 (10 샘플)
python scripts/run_kitrec_eval.py \
    --model_name dualft_music_seta \
    --max_samples 10

# LLM4CDR Qwen 테스트 (10 샘플)
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --llm_backend qwen \
    --max_samples 10

# LLM4CDR GPT-4.1-mini 테스트 (10 샘플)
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --llm_backend gpt4-mini \
    --max_samples 10
```

---

## Phase 2: KitREC 모델 평가 (2-4시간)

### 2.1 개별 모델 실행

```bash
# DualFT Music Set A
python scripts/run_kitrec_eval.py --model_name dualft_music_seta

# DualFT Music Set B
python scripts/run_kitrec_eval.py --model_name dualft_music_setb

# SingleFT Music Set A
python scripts/run_kitrec_eval.py --model_name singleft_music_seta

# SingleFT Music Set B
python scripts/run_kitrec_eval.py --model_name singleft_music_setb
```

### 2.2 전체 Music 모델 한번에 실행

```bash
python scripts/run_kitrec_eval.py --target_domain music --run_all
```

### 결과 저장 위치
```
results/kitrec/
├── dualft_music_seta/{timestamp}/
│   ├── metrics_summary.json      # Hit@K, MRR, NDCG 등
│   ├── stratified_analysis.json  # User Type별 성능 (RQ3)
│   ├── predictions.json          # 전체 예측 결과
│   └── logs/
├── dualft_music_setb/{timestamp}/
├── singleft_music_seta/{timestamp}/
└── singleft_music_setb/{timestamp}/
```

---

## Phase 3: LLM Baseline 평가 - Qwen3-14B (2-4시간)

```bash
# LLM4CDR (3-stage pipeline) - Set A
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --candidate_set seta \
    --llm_backend qwen

# LLM4CDR - Set B
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --candidate_set setb \
    --llm_backend qwen

# Vanilla Zero-shot - Set A
python scripts/run_baseline_eval.py \
    --baseline vanilla \
    --target_domain music \
    --candidate_set seta \
    --llm_backend qwen

# Vanilla Zero-shot - Set B
python scripts/run_baseline_eval.py \
    --baseline vanilla \
    --target_domain music \
    --candidate_set setb \
    --llm_backend qwen
```

### 결과 저장 위치
```
results/baselines/
├── llm4cdr/qwen/music/{timestamp}/
│   ├── metrics_summary.json
│   ├── predictions.json
│   └── user_type_analysis.json
└── vanilla/qwen/music/{timestamp}/
```

---

## Phase 4: LLM Baseline 평가 - GPT-4.1-mini (2-4시간)

**주의**: OpenAI API 비용 발생

```bash
# LLM4CDR (3-stage pipeline) - Set A
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --candidate_set seta \
    --llm_backend gpt4-mini

# LLM4CDR - Set B
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --candidate_set setb \
    --llm_backend gpt4-mini

# Vanilla Zero-shot - Set A
python scripts/run_baseline_eval.py \
    --baseline vanilla \
    --target_domain music \
    --candidate_set seta \
    --llm_backend gpt4-mini

# Vanilla Zero-shot - Set B
python scripts/run_baseline_eval.py \
    --baseline vanilla \
    --target_domain music \
    --candidate_set setb \
    --llm_backend gpt4-mini
```

### 결과 저장 위치
```
results/baselines/
├── llm4cdr/gpt4-mini/music/{timestamp}/
└── vanilla/gpt4-mini/music/{timestamp}/
```

---

## Phase 5: Deep Learning Baseline 평가 (4-8시간)

**주의**: 학습 후 평가 (train_baseline 필수)

### 5.1 CoNet

```bash
# CoNet 학습 + 평가 - Set A
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain music \
    --candidate_set seta \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_music-seta

# CoNet 학습 + 평가 - Set B
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain music \
    --candidate_set setb \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_music-setb
```

### 5.2 DTCDR

```bash
# DTCDR 학습 + 평가 - Set A
python scripts/run_baseline_eval.py \
    --baseline dtcdr \
    --target_domain music \
    --candidate_set seta \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_music-seta

# DTCDR 학습 + 평가 - Set B
python scripts/run_baseline_eval.py \
    --baseline dtcdr \
    --target_domain music \
    --candidate_set setb \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_music-setb
```

### 결과 저장 위치
```
results/baselines/
├── conet/music/{timestamp}/
│   ├── metrics_summary.json
│   ├── checkpoint_best.pt        # 학습된 모델
│   ├── training_log.json
│   └── user_type_analysis.json
└── dtcdr/music/{timestamp}/
```

---

## Phase 6: Ablation Study - RQ1 (2-4시간)

```bash
# Music Set A
python scripts/run_ablation_study.py \
    --target_domain music \
    --candidate_set seta

# Music Set B
python scripts/run_ablation_study.py \
    --target_domain music \
    --candidate_set setb
```

### 비교 모델 4개
| 모델 | Fine-tuned | Thinking |
|------|------------|----------|
| KitREC-Full | Yes | Yes |
| KitREC-Direct | Yes | No |
| Base-CoT | No | Yes |
| Base-Direct | No | No |

### 결과 저장 위치
```
results/ablation/{timestamp}/
├── ablation_results.json     # 4개 모델 메트릭
├── ablation_table.md         # 마크다운 테이블
└── statistical_tests.json    # 통계적 유의성 검정
```

---

## 전체 결과 디렉토리 구조

```
results/
├── kitrec/                              # Phase 2
│   ├── dualft_music_seta/{timestamp}/
│   ├── dualft_music_setb/{timestamp}/
│   ├── singleft_music_seta/{timestamp}/
│   └── singleft_music_setb/{timestamp}/
│
├── baselines/                           # Phase 3, 4, 5
│   ├── llm4cdr/
│   │   ├── qwen/music/{timestamp}/      # Qwen3-14B
│   │   └── gpt4-mini/music/{timestamp}/ # GPT-4.1-mini
│   ├── vanilla/
│   │   ├── qwen/music/{timestamp}/
│   │   └── gpt4-mini/music/{timestamp}/
│   ├── conet/music/{timestamp}/
│   └── dtcdr/music/{timestamp}/
│
└── ablation/{timestamp}/                # Phase 6
```

---

## 전체 실행 스크립트 (순차 실행)

```bash
#!/bin/bash
# run_all_experiments.sh

echo "=========================================="
echo "KitREC Music Domain Full Experiment"
echo "=========================================="

# Phase 0: Setup
pip install -r requirements.txt
python scripts/run_kitrec_eval.py --show_status

# Phase 1: Sanity Check
echo "[Phase 1] Sanity Check..."
python scripts/run_kitrec_eval.py --model_name dualft_music_seta --max_samples 10

# Phase 2: KitREC
echo "[Phase 2] KitREC Evaluation..."
python scripts/run_kitrec_eval.py --target_domain music --run_all

# Phase 3: LLM Baselines (Qwen)
echo "[Phase 3] LLM Baselines (Qwen)..."
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set seta --llm_backend qwen
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set setb --llm_backend qwen
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set seta --llm_backend qwen
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set setb --llm_backend qwen

# Phase 4: LLM Baselines (GPT-4.1-mini)
echo "[Phase 4] LLM Baselines (GPT-4.1-mini)..."
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set seta --llm_backend gpt4-mini
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set setb --llm_backend gpt4-mini
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set seta --llm_backend gpt4-mini
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set setb --llm_backend gpt4-mini

# Phase 5: DL Baselines
echo "[Phase 5] DL Baselines..."
python scripts/run_baseline_eval.py --baseline conet --target_domain music --candidate_set seta --train_baseline --train_dataset Younggooo/kitrec-dualft_music-seta
python scripts/run_baseline_eval.py --baseline conet --target_domain music --candidate_set setb --train_baseline --train_dataset Younggooo/kitrec-dualft_music-setb
python scripts/run_baseline_eval.py --baseline dtcdr --target_domain music --candidate_set seta --train_baseline --train_dataset Younggooo/kitrec-dualft_music-seta
python scripts/run_baseline_eval.py --baseline dtcdr --target_domain music --candidate_set setb --train_baseline --train_dataset Younggooo/kitrec-dualft_music-setb

# Phase 6: Ablation Study
echo "[Phase 6] Ablation Study..."
python scripts/run_ablation_study.py --target_domain music --candidate_set seta
python scripts/run_ablation_study.py --target_domain music --candidate_set setb

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: results/"
echo "=========================================="
```

---

## 예상 소요 시간

| Phase | 작업 | 예상 시간 (A100) |
|-------|------|-----------------|
| 0 | 환경 설정 | 5분 |
| 1 | Sanity Check | 10분 |
| 2 | KitREC (4 모델) | 2-4시간 |
| 3 | LLM Baseline Qwen (4 실험) | 2-4시간 |
| 4 | LLM Baseline GPT (4 실험) | 2-4시간 |
| 5 | DL Baseline (4 실험) | 4-8시간 |
| 6 | Ablation Study (2 실험) | 2-4시간 |
| **Total** | | **12-24시간** |

---

## 결과 확인

```bash
# 모든 결과 파일 확인
find results/ -name "metrics_summary.json" | head -20

# KitREC 결과 요약
cat results/kitrec/dualft_music_seta/*/metrics_summary.json | jq '.aggregated_metrics'

# Baseline 비교
cat results/baselines/llm4cdr/qwen/music/*/metrics_summary.json
cat results/baselines/llm4cdr/gpt4-mini/music/*/metrics_summary.json
```

---

## Troubleshooting

| 에러 | 원인 | 해결 |
|------|------|------|
| `Repository not found` | HF_TOKEN 문제 | `.env` 확인 |
| `CUDA out of memory` | GPU 메모리 부족 | `--batch_size 4` 추가 |
| `OpenAI API key not found` | API 키 미설정 | `.env`에 OPENAI_API_KEY 추가 |
| `Domain 'movies' is not yet available` | Movies 미지원 | `--target_domain music` 사용 |
