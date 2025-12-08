#!/bin/bash
# run_all_experiments.sh
# KitREC RunPod 실험 일괄 실행 스크립트 (Music Domain)

set -e  # 에러 발생 시 즉시 중단

echo "=========================================="
echo "KitREC Music Domain Full Experiment"
echo "=========================================="

# Phase 0: Setup & Verification
echo "[Phase 0] Environment Setup..."
pip install -r requirements.txt
python scripts/verify_local.py
python scripts/run_kitrec_eval.py --show_status

# Phase 1: Sanity Check
echo "[Phase 1] Sanity Check (10 samples)..."
python scripts/run_kitrec_eval.py --model_name dualft_music_seta --max_samples 10
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set seta --llm_backend qwen --max_samples 10

# Phase 2: KitREC Evaluation
echo "[Phase 2] KitREC Evaluation (All 4 models)..."
# Music 도메인 전체 모델 일괄 실행
python scripts/run_kitrec_eval.py --target_domain music --run_all

# Phase 3: LLM Baselines (Qwen3-14B)
echo "[Phase 3] LLM Baselines (Qwen3-14B)..."
# LLM4CDR
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set seta --llm_backend qwen
python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set setb --llm_backend qwen
# Vanilla Zero-shot (NIR)
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set seta --llm_backend qwen
python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set setb --llm_backend qwen

# Phase 4: LLM Baselines (GPT-4.1-mini) - Optional
# .env에 OPENAI_API_KEY가 있는지 확인
if grep -q "OPENAI_API_KEY" .env; then
    echo "[Phase 4] LLM Baselines (GPT-4.1-mini)..."
    python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set seta --llm_backend gpt4-mini
    python scripts/run_baseline_eval.py --baseline llm4cdr --target_domain music --candidate_set setb --llm_backend gpt4-mini
    python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set seta --llm_backend gpt4-mini
    python scripts/run_baseline_eval.py --baseline vanilla --target_domain music --candidate_set setb --llm_backend gpt4-mini
else
    echo "[Skip] Phase 4 (GPT-4.1-mini): OPENAI_API_KEY not found in .env"
fi

# Phase 5: DL Baselines (CoNet, DTCDR)
echo "[Phase 5] DL Baselines (CoNet, DTCDR)..."
# CoNet - Train & Eval
python scripts/run_baseline_eval.py --baseline conet --target_domain music --candidate_set seta --train_baseline --train_dataset Younggooo/kitrec-dualft_music-seta
python scripts/run_baseline_eval.py --baseline conet --target_domain music --candidate_set setb --train_baseline --train_dataset Younggooo/kitrec-dualft_music-setb
# DTCDR - Train & Eval
python scripts/run_baseline_eval.py --baseline dtcdr --target_domain music --candidate_set seta --train_baseline --train_dataset Younggooo/kitrec-dualft_music-seta
python scripts/run_baseline_eval.py --baseline dtcdr --target_domain music --candidate_set setb --train_baseline --train_dataset Younggooo/kitrec-dualft_music-setb

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: results/"
echo "=========================================="
