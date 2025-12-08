# KitREC RunPod Experiment Guide (Music Domain)

## Prerequisites

- RunPod GPU Server (A100/H100 80GB recommended, or RTX 5090 36GB)
- HuggingFace Token (in `.env` file)
- Python 3.10+

---

## Step 0: Environment Setup

```bash
# 1. Clone or upload project to RunPod
cd /workspace
# (Upload Experimental_test folder)

# 2. Install dependencies
cd /workspace/Experimental_test
pip install -r requirements.txt

# 3. Verify .env file exists with HF_TOKEN
cat .env | grep HF_TOKEN
# Should show: HF_TOKEN=hf_xxxxx

# 4. Verify environment
python scripts/verify_local.py

# 5. Check domain status
python scripts/run_kitrec_eval.py --show_status
```

---

## Step 1: Quick Sanity Check (10 samples)

Before running full experiments, verify everything works:

```bash
# Test KitREC model loading and inference
python scripts/run_kitrec_eval.py \
    --model_name dualft_music_seta \
    --max_samples 10

# Test Baseline (LLM4CDR)
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --max_samples 10
```

---

## Step 2: KitREC Model Evaluation (RQ2, RQ3)

### 2.1 DualFT Models (Overlapping + Cold-start users)

```bash
# Set A (Hard Negatives)
python scripts/run_kitrec_eval.py --model_name dualft_music_seta

# Set B (Random Negatives)
python scripts/run_kitrec_eval.py --model_name dualft_music_setb
```

### 2.2 SingleFT Models (Source-only users - Extreme Cold-start)

```bash
# Set A
python scripts/run_kitrec_eval.py --model_name singleft_music_seta

# Set B
python scripts/run_kitrec_eval.py --model_name singleft_music_setb
```

### 2.3 Run All Music Models at Once

```bash
python scripts/run_kitrec_eval.py --target_domain music --run_all
```

**Expected Output**: `results/kitrec/{model_name}/{timestamp}/`
- `metrics_summary.json`: Hit@K, MRR, NDCG
- `stratified_analysis.json`: User Type별 성능
- `predictions.json`: 전체 예측 결과

---

## Step 3: Baseline Model Evaluation (RQ2)

### 3.1 LLM4CDR (3-Stage Pipeline)

```bash
# Set A
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --candidate_set seta

# Set B
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain music \
    --candidate_set setb
```

### 3.2 Vanilla Zero-shot (NIR Lower Bound)

```bash
python scripts/run_baseline_eval.py \
    --baseline vanilla \
    --target_domain music \
    --candidate_set seta

python scripts/run_baseline_eval.py \
    --baseline vanilla \
    --target_domain music \
    --candidate_set setb
```

### 3.3 CoNet (Deep Learning Baseline)

**Note**: CoNet requires training first!

```bash
# Train CoNet on Music training data
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain music \
    --candidate_set seta \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_music-seta

# Evaluate (after training)
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain music \
    --candidate_set seta \
    --baseline_checkpoint results/baselines/conet/music/*/checkpoint_best.pt
```

### 3.4 DTCDR (Deep Learning Baseline)

```bash
# Train DTCDR
python scripts/run_baseline_eval.py \
    --baseline dtcdr \
    --target_domain music \
    --candidate_set seta \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_music-seta

# Evaluate
python scripts/run_baseline_eval.py \
    --baseline dtcdr \
    --target_domain music \
    --candidate_set seta \
    --baseline_checkpoint results/baselines/dtcdr/music/*/checkpoint_best.pt
```

---

## Step 4: Ablation Study (RQ1)

```bash
# Music domain ablation (4 model variants)
python scripts/run_ablation_study.py \
    --target_domain music \
    --candidate_set seta

python scripts/run_ablation_study.py \
    --target_domain music \
    --candidate_set setb
```

**4 Models Compared**:
1. KitREC-Full (Fine-tuned + Thinking)
2. KitREC-Direct (Fine-tuned + No Thinking)
3. Base-CoT (Untuned + Thinking)
4. Base-Direct (Untuned + No Thinking)

---

## Step 5: Explainability Evaluation (RQ4) - Optional

**Note**: Requires OPENAI_API_KEY in .env for GPT-4.1 evaluation

```bash
# Add to .env if needed
echo "OPENAI_API_KEY=sk-xxxxx" >> .env

# Run with explainability metrics
# (This is integrated into the main evaluation scripts)
```

---

## Complete Experiment Execution Order

```bash
# ============================================================
# PHASE 1: Environment Setup (5 min)
# ============================================================
pip install -r requirements.txt
python scripts/verify_local.py
python scripts/run_kitrec_eval.py --show_status

# ============================================================
# PHASE 2: Sanity Check (10 min)
# ============================================================
python scripts/run_kitrec_eval.py --model_name dualft_music_seta --max_samples 10
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --max_samples 10

# ============================================================
# PHASE 3: KitREC Full Evaluation (2-4 hours)
# ============================================================
python scripts/run_kitrec_eval.py --target_domain music --run_all

# ============================================================
# PHASE 4: LLM Baselines (2-4 hours)
# ============================================================
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set seta
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set setb
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set seta
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set setb

# ============================================================
# PHASE 5: Deep Learning Baselines (4-8 hours)
# ============================================================
# CoNet
python scripts/run_baseline_eval.py --baseline conet --target_domain music --candidate_set seta --train_baseline --train_dataset Younggooo/kitrec-dualft_music-seta
python scripts/run_baseline_eval.py --baseline conet --target_domain music --candidate_set setb --train_baseline --train_dataset Younggooo/kitrec-dualft_music-setb

# DTCDR
python scripts/run_baseline_eval.py --baseline dtcdr --target_domain music --candidate_set seta --train_baseline --train_dataset Younggooo/kitrec-dualft_music-seta
python scripts/run_baseline_eval.py --baseline dtcdr --target_domain music --candidate_set setb --train_baseline --train_dataset Younggooo/kitrec-dualft_music-setb

# ============================================================
# PHASE 6: Ablation Study (2-4 hours)
# ============================================================
python scripts/run_ablation_study.py --target_domain music --candidate_set seta
python scripts/run_ablation_study.py --target_domain music --candidate_set setb

# ============================================================
# PHASE 7: Collect Results
# ============================================================
ls -la results/kitrec/
ls -la results/baselines/
ls -la results/ablation/
```

---

## Results Directory Structure

```
results/
├── kitrec/
│   ├── dualft_music_seta/{timestamp}/
│   │   ├── metrics_summary.json
│   │   ├── stratified_analysis.json
│   │   └── predictions.json
│   ├── dualft_music_setb/{timestamp}/
│   ├── singleft_music_seta/{timestamp}/
│   └── singleft_music_setb/{timestamp}/
├── baselines/
│   ├── llm4cdr/music/{timestamp}/
│   ├── vanilla/music/{timestamp}/
│   ├── conet/music/{timestamp}/
│   └── dtcdr/music/{timestamp}/
└── ablation/
    └── {timestamp}/
        ├── ablation_results.json
        ├── ablation_table.md
        └── statistical_tests.json
```

---

## Troubleshooting

### Error: "Repository not found"
```bash
# Check HF_TOKEN
cat .env | grep HF_TOKEN
# Verify token is valid at huggingface.co
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size
python scripts/run_kitrec_eval.py --model_name dualft_music_seta --batch_size 4
```

### Error: "Domain 'movies' is not yet available"
```
Movies domain is pending. Use --target_domain music
```

---

## Estimated Time (A100 80GB)

| Phase | Task | Time |
|-------|------|------|
| Setup | Environment + Verify | 5 min |
| Sanity | Quick test | 10 min |
| KitREC | 4 models × 30K samples | 2-4 hours |
| LLM Baselines | LLM4CDR + Vanilla | 2-4 hours |
| DL Baselines | CoNet + DTCDR (train+eval) | 4-8 hours |
| Ablation | 4 variants | 2-4 hours |
| **Total** | | **10-20 hours** |

---

## Quick Reference Commands

```bash
# Status check
python scripts/run_kitrec_eval.py --show_status
python scripts/run_baseline_eval.py --show_status

# Help
python scripts/run_kitrec_eval.py --help
python scripts/run_baseline_eval.py --help
python scripts/run_ablation_study.py --help
```
