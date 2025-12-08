#!/usr/bin/env python3
"""
Baseline 모델 평가 실행 스크립트

RQ2: CDR 방식의 효과성 검증
- CoNet
- DTCDR
- LLM4CDR
- Vanilla Zero-shot
"""

import argparse
import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file automatically
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[INFO] Loaded environment from {env_path}")
except ImportError:
    print("[WARN] python-dotenv not installed. Install with: pip install python-dotenv")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

from src.data.data_loader import DataLoader
from src.inference.vllm_inference import VLLMInference
from src.inference.output_parser import OutputParser
from src.metrics.ranking_metrics import RankingMetrics
from src.metrics.statistical_analysis import StatisticalAnalysis
from src.utils.io_utils import save_json, load_json

from tqdm import tqdm


def build_user_type_mapping(original_samples, converter):
    """
    원본 샘플에서 user_type 추출하여 매핑 생성 (RQ3: Cold-start analysis)

    ⭐ 핵심 연구 결과:
    - KitREC은 5-core 이하에서도 성능이 나옴
    - 다른 베이스라인은 실험조차 하지 않는 조건
    - 극단적 상황 비교가 핵심

    Args:
        original_samples: 원본 테스트 샘플 (user_type 필드 포함)
        converter: 데이터 변환기 (user_vocab 포함)

    Returns:
        {user_int_id: user_type} 매핑
    """
    user_type_mapping = {}
    for sample in original_samples:
        user_str = sample.get("user_id", "")
        user_int_id = converter.user_vocab.get(user_str, 0)
        user_type_mapping[user_int_id] = sample.get("user_type", "unknown")
    return user_type_mapping


def print_user_type_metrics(user_type_metrics, baseline_name):
    """
    User Type별 메트릭 출력 (Core level별)

    Args:
        user_type_metrics: {user_type: metrics}
        baseline_name: 베이스라인 이름
    """
    print(f"\n{'='*60}")
    print(f"User Type Analysis ({baseline_name.upper()}) - RQ3: Cold-start")
    print(f"{'='*60}")

    # Core level 순서 정렬
    core_order = ["1-core", "2-core", "3-core", "4-core", "5-core",
                  "6-core", "7-core", "8-core", "9-core", "10+-core",
                  "source_only", "overlapping", "unknown"]

    print(f"{'User Type':<15} | {'Samples':<8} | {'Hit@10':<8} | {'NDCG@10':<8} | {'MRR':<8}")
    print("-" * 60)

    for user_type in core_order:
        if user_type in user_type_metrics:
            m = user_type_metrics[user_type]
            print(f"{user_type:<15} | {m.get('sample_count', 0):<8} | "
                  f"{m.get('hit@10', 0):.4f}   | {m.get('ndcg@10', 0):.4f}   | "
                  f"{m.get('mrr', 0):.4f}")

    # 나머지 user_type 출력
    for user_type, m in user_type_metrics.items():
        if user_type not in core_order:
            print(f"{user_type:<15} | {m.get('sample_count', 0):<8} | "
                  f"{m.get('hit@10', 0):.4f}   | {m.get('ndcg@10', 0):.4f}   | "
                  f"{m.get('mrr', 0):.4f}")


# Domain availability status (sync with KitRECModel.AVAILABLE_DOMAINS)
AVAILABLE_DOMAINS = {
    "music": True,   # Ready for evaluation
    "movies": False  # Pending - models not yet trained
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Domain Status:
  Music:  READY   - Can run baseline evaluation
  Movies: PENDING - Models not yet available

Examples:
  # Evaluate CoNet on Music domain
  python run_baseline_eval.py --baseline conet --target_domain music

  # Evaluate with training
  python run_baseline_eval.py --baseline dtcdr --target_domain music \\
      --train_baseline --train_dataset Younggooo/kitrec-dualft_music-seta
        """
    )

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["conet", "dtcdr", "llm4cdr", "vanilla"],
        help="Baseline model to evaluate"
    )

    parser.add_argument(
        "--target_domain",
        type=str,
        choices=["movies", "music"],
        required=True,
        help="Target domain (currently only 'music' is available)"
    )

    parser.add_argument(
        "--show_status",
        action="store_true",
        help="Show domain availability status and exit"
    )

    parser.add_argument(
        "--candidate_set",
        type=str,
        choices=["seta", "setb"],
        default="seta",
        help="Candidate set"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baselines",
        help="Output directory"
    )

    parser.add_argument(
        "--kitrec_results",
        type=str,
        default=None,
        help="Path to KitREC results for comparison"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples for testing"
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token"
    )

    # LLM Backend selection (for llm4cdr, vanilla baselines)
    parser.add_argument(
        "--llm_backend",
        type=str,
        choices=["qwen", "gpt4-mini"],
        default="qwen",
        help="LLM backend for LLM4CDR/Vanilla: 'qwen' (Qwen3-14B via vLLM) or 'gpt4-mini' (GPT-4.1-mini via OpenAI API)"
    )

    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (for gpt4-mini backend, or set OPENAI_API_KEY env)"
    )

    # CoNet/DTCDR specific
    parser.add_argument(
        "--train_baseline",
        action="store_true",
        help="Train baseline model (CoNet/DTCDR)"
    )

    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default=None,
        help="Pre-trained baseline checkpoint"
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="Training dataset name (e.g., Younggooo/kitrec-dualft_movies-seta). "
             "Required when --train_baseline is used. "
             "⚠️ Must use separate train dataset, NOT test data!"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def evaluate_conet(args, test_data):
    """CoNet 평가

    ⚠️ Data Leakage Prevention:
    - Training: Uses separate train dataset (--train_dataset)
    - Evaluation: Uses test dataset only (test_data parameter)
    - NEVER split test data for training!
    """
    from baselines.conet import CoNet, CoNetDataConverter, CoNetTrainer, CoNetEvaluator

    print("Initializing CoNet...")

    # Data conversion for TEST data (evaluation only)
    converter = CoNetDataConverter()
    test_samples = list(test_data)
    converter.build_vocabulary(test_samples)
    converted_test_samples = converter.convert_dataset(test_samples)

    vocab_sizes = converter.get_vocab_sizes()
    print(f"  Vocab sizes: {vocab_sizes}")

    # Initialize model
    model = CoNet(
        num_users=vocab_sizes["num_users"],
        num_items_source=vocab_sizes["num_source_items"],
        num_items_target=vocab_sizes["num_target_items"],
        embedding_dim=128,
        hidden_dims=[256, 128, 64],
    )

    # Train or load checkpoint
    if args.train_baseline:
        # ⚠️ CRITICAL: Load SEPARATE training dataset (NOT test data!)
        if not args.train_dataset:
            raise ValueError(
                "⚠️ Data Leakage Prevention: --train_dataset is required when --train_baseline is used.\n"
                "Use a training dataset like: Younggooo/kitrec-dualft_movies-seta\n"
                "Do NOT use test data for training!"
            )

        print(f"Loading TRAINING data from: {args.train_dataset}")
        train_loader = DataLoader(args.train_dataset, hf_token=args.hf_token)
        train_data_raw = list(train_loader.load_test_data())

        # Build vocab on combined data for consistent IDs
        all_samples = test_samples + train_data_raw
        converter_train = CoNetDataConverter()
        converter_train.build_vocabulary(all_samples)
        converted_train_samples = converter_train.convert_dataset(train_data_raw)

        # Split training data for train/val (NOT test data!)
        split_idx = int(len(converted_train_samples) * 0.9)
        train_samples = converted_train_samples[:split_idx]
        val_samples = converted_train_samples[split_idx:]

        print(f"Training CoNet on {len(train_samples)} samples (val: {len(val_samples)})...")
        trainer = CoNetTrainer(model)
        trainer.train(train_samples, val_samples)

        # Update converter with training vocab
        converter = converter_train
        converted_test_samples = converter.convert_dataset(test_samples)

    elif args.baseline_checkpoint:
        print(f"Loading checkpoint: {args.baseline_checkpoint}")
        trainer = CoNetTrainer(model)
        trainer.load_checkpoint(args.baseline_checkpoint)
    else:
        print("⚠️ Warning: No training or checkpoint specified. Using random initialization.")

    # Evaluate on TEST data only
    print(f"Evaluating on {len(converted_test_samples)} test samples...")
    evaluator = CoNetEvaluator(model, converter)
    metrics = evaluator.evaluate(converted_test_samples)

    # Return converter and original samples for user type mapping (RQ3)
    return metrics, evaluator, converted_test_samples, converter, test_samples


def evaluate_dtcdr(args, test_data):
    """DTCDR 평가

    ⚠️ Data Leakage Prevention:
    - Training: Uses separate train dataset (--train_dataset)
    - Evaluation: Uses test dataset only (test_data parameter)
    - NEVER split test data for training!
    """
    from baselines.dtcdr import DTCDR, DTCDRDataConverter, DTCDRTrainer, DTCDREvaluator

    print("Initializing DTCDR...")

    # Data conversion for TEST data (evaluation only)
    converter = DTCDRDataConverter()
    test_samples = list(test_data)
    converter.build_vocabulary(test_samples)
    converted_test_samples = converter.convert_dataset(test_samples)

    vocab_sizes = converter.get_vocab_sizes()
    print(f"  Vocab sizes: {vocab_sizes}")

    model = DTCDR(
        num_users=vocab_sizes["num_users"],
        num_items_source=vocab_sizes["num_source_items"],
        num_items_target=vocab_sizes["num_target_items"],
        embedding_dim=128,
        mlp_layers=[256, 128],
        use_mapping=True,
    )

    if args.train_baseline:
        # ⚠️ CRITICAL: Load SEPARATE training dataset (NOT test data!)
        if not args.train_dataset:
            raise ValueError(
                "⚠️ Data Leakage Prevention: --train_dataset is required when --train_baseline is used.\n"
                "Use a training dataset like: Younggooo/kitrec-dualft_movies-seta\n"
                "Do NOT use test data for training!"
            )

        print(f"Loading TRAINING data from: {args.train_dataset}")
        train_loader = DataLoader(args.train_dataset, hf_token=args.hf_token)
        train_data_raw = list(train_loader.load_test_data())

        # Build vocab on combined data for consistent IDs
        all_samples = test_samples + train_data_raw
        converter_train = DTCDRDataConverter()
        converter_train.build_vocabulary(all_samples)
        converted_train_samples = converter_train.convert_dataset(train_data_raw)

        # Split training data for train/val (NOT test data!)
        split_idx = int(len(converted_train_samples) * 0.9)
        train_samples = converted_train_samples[:split_idx]
        val_samples = converted_train_samples[split_idx:]

        print(f"Training DTCDR on {len(train_samples)} samples (val: {len(val_samples)})...")
        trainer = DTCDRTrainer(model)
        trainer.train(train_samples, val_samples)

        # Update converter with training vocab
        converter = converter_train
        converted_test_samples = converter.convert_dataset(test_samples)

    elif args.baseline_checkpoint:
        print(f"Loading checkpoint: {args.baseline_checkpoint}")
        trainer = DTCDRTrainer(model)
        trainer.load_checkpoint(args.baseline_checkpoint)
    else:
        print("⚠️ Warning: No training or checkpoint specified. Using random initialization.")

    # Evaluate on TEST data only
    print(f"Evaluating on {len(converted_test_samples)} test samples...")
    evaluator = DTCDREvaluator(model, converter)
    metrics = evaluator.evaluate(converted_test_samples)

    # Return converter and original samples for user type mapping (RQ3)
    return metrics, evaluator, converted_test_samples, converter, test_samples


def create_llm_inference_engine(args):
    """
    LLM 추론 엔진 생성 (Qwen or GPT-4.1-mini)

    Args:
        args: 명령줄 인자 (llm_backend, hf_token, openai_api_key)

    Returns:
        추론 엔진 인스턴스
    """
    if args.llm_backend == "qwen":
        print(f"  LLM Backend: Qwen3-14B (vLLM)")
        inference_engine = VLLMInference(
            model_name="Qwen/Qwen3-14B",
            hf_token=args.hf_token,
            enable_lora=False,
        )
        inference_engine.initialize()
        return inference_engine

    elif args.llm_backend == "gpt4-mini":
        print(f"  LLM Backend: GPT-4.1-mini (OpenAI API)")
        from src.inference.openai_inference import OpenAIInference

        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for gpt4-mini backend.\n"
                "Set OPENAI_API_KEY in .env or pass --openai_api_key"
            )

        inference_engine = OpenAIInference(
            model_name="gpt-4.1-mini",
            api_key=api_key,
        )
        inference_engine.initialize()
        return inference_engine

    else:
        raise ValueError(f"Unknown LLM backend: {args.llm_backend}")


def evaluate_llm4cdr(args, test_data):
    """LLM4CDR 평가 (3-stage pipeline)"""
    from baselines.llm4cdr import LLM4CDREvaluator

    print("Initializing LLM4CDR (3-stage)...")

    # Initialize inference engine based on backend selection
    inference_engine = create_llm_inference_engine(args)

    evaluator = LLM4CDREvaluator(
        inference_engine=inference_engine,
        use_3stage=True,  # 3-stage pipeline
        cache_domain_analysis=True,
    )

    samples = list(test_data)
    if args.max_samples:
        samples = samples[:args.max_samples]

    metrics = evaluator.evaluate(
        samples,
        source_domain="Books",
        target_domain=args.target_domain.capitalize()
    )

    # Print LLM statistics if available
    if hasattr(inference_engine, 'get_statistics'):
        stats = inference_engine.get_statistics()
        print(f"\n  LLM Statistics: {stats}")

    # LLM-based: return None for converter (no embedding conversion)
    return metrics, evaluator, samples, None, samples


def evaluate_vanilla(args, test_data):
    """Vanilla Zero-shot 평가"""
    from baselines.llm4cdr import LLM4CDREvaluator

    print("Initializing Vanilla Zero-shot...")

    # Initialize inference engine based on backend selection
    inference_engine = create_llm_inference_engine(args)

    evaluator = LLM4CDREvaluator(
        inference_engine=inference_engine,
        use_3stage=False,  # Vanilla single prompt
        cache_domain_analysis=False,
    )

    samples = list(test_data)
    if args.max_samples:
        samples = samples[:args.max_samples]

    metrics = evaluator.evaluate(
        samples,
        source_domain="Books",
        target_domain=args.target_domain.capitalize()
    )

    # Print LLM statistics if available
    if hasattr(inference_engine, 'get_statistics'):
        stats = inference_engine.get_statistics()
        print(f"\n  LLM Statistics: {stats}")

    # LLM-based: return None for converter (no embedding conversion)
    return metrics, evaluator, samples, None, samples


def show_domain_status():
    """도메인 상태 출력"""
    print("=" * 60)
    print("Baseline Evaluation - Domain Availability Status")
    print("=" * 60)
    for domain, available in AVAILABLE_DOMAINS.items():
        status = "READY" if available else "PENDING"
        print(f"  [{domain.upper()}] - {status}")
    print("\nCurrently available: music")
    print("Pending (models not trained): movies")
    print("=" * 60)


def validate_domain(domain: str):
    """도메인 사용 가능 여부 검증"""
    if not AVAILABLE_DOMAINS.get(domain, False):
        print(f"\n{'='*60}")
        print(f"ERROR: Domain '{domain}' is not yet available!")
        print(f"{'='*60}")
        print(f"\nThe {domain.upper()} domain models are still in training.")
        print(f"\nCurrently available domains:")
        for d, available in AVAILABLE_DOMAINS.items():
            if available:
                print(f"  - {d}")
        print(f"\nPlease use --target_domain music instead.")
        sys.exit(1)


def main():
    args = parse_args()

    # Show status and exit if requested
    if args.show_status:
        show_domain_status()
        return

    # Validate domain availability
    validate_domain(args.target_domain)

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include LLM backend in output path for LLM-based baselines
    if args.baseline in ["llm4cdr", "vanilla"]:
        output_dir = os.path.join(
            args.output_dir, args.baseline, args.llm_backend, args.target_domain, timestamp
        )
    else:
        output_dir = os.path.join(
            args.output_dir, args.baseline, args.target_domain, timestamp
        )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Baseline Evaluation: {args.baseline.upper()}")
    print(f"Target Domain: {args.target_domain}")
    print(f"Candidate Set: {args.candidate_set}")
    if args.baseline in ["llm4cdr", "vanilla"]:
        backend_info = "Qwen3-14B (vLLM)" if args.llm_backend == "qwen" else "GPT-4.1-mini (OpenAI)"
        print(f"LLM Backend: {backend_info}")
    print("=" * 60)

    # Load test data
    dataset_name = f"Younggooo/kitrec-test-{args.candidate_set}"
    data_loader = DataLoader(dataset_name, hf_token=args.hf_token)
    test_data = data_loader.load_test_data()

    # Filter by domain
    test_data = [
        s for s in test_data
        if s.get("target_domain", "").lower() == args.target_domain
    ]

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"\nLoaded {len(test_data)} samples")

    # Evaluate baseline
    # Returns: metrics, evaluator, converted_samples, converter, original_samples
    if args.baseline == "conet":
        metrics, evaluator, samples, converter, original_samples = evaluate_conet(args, test_data)
    elif args.baseline == "dtcdr":
        metrics, evaluator, samples, converter, original_samples = evaluate_dtcdr(args, test_data)
    elif args.baseline == "llm4cdr":
        metrics, evaluator, samples, converter, original_samples = evaluate_llm4cdr(args, test_data)
    elif args.baseline == "vanilla":
        metrics, evaluator, samples, converter, original_samples = evaluate_vanilla(args, test_data)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    # User Type Analysis (RQ3: Cold-start)
    user_type_metrics = None
    if args.baseline in ["conet", "dtcdr"] and converter is not None:
        # Build user type mapping for embedding-based models
        user_type_mapping = build_user_type_mapping(original_samples, converter)
        user_type_metrics = evaluator.evaluate_by_user_type(samples, user_type_mapping)
        print_user_type_metrics(user_type_metrics, args.baseline)
    elif args.baseline in ["llm4cdr", "vanilla"]:
        # For LLM-based models, compute user type metrics directly from samples
        from collections import defaultdict
        grouped = defaultdict(list)
        for sample in original_samples:
            user_type = sample.get("user_type", "unknown")
            # LLM evaluators store per-sample metrics differently
            if hasattr(evaluator, 'per_sample_metrics') and sample.get("user_id") in evaluator.per_sample_metrics:
                grouped[user_type].append(evaluator.per_sample_metrics[sample["user_id"]])
        if grouped:
            user_type_metrics = {}
            for ut, metrics_list in grouped.items():
                if metrics_list:
                    import numpy as np
                    aggregated = {}
                    for key in metrics_list[0].keys():
                        aggregated[key] = np.mean([m[key] for m in metrics_list])
                    aggregated["sample_count"] = len(metrics_list)
                    user_type_metrics[ut] = aggregated
            if user_type_metrics:
                print_user_type_metrics(user_type_metrics, args.baseline)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    # Compare with KitREC if results provided
    if args.kitrec_results:
        print("\n" + "=" * 60)
        print("Comparison with KitREC")
        print("=" * 60)

        kitrec_results = load_json(args.kitrec_results)
        kitrec_metrics = kitrec_results.get("aggregated_metrics", {})
        kitrec_per_sample = kitrec_results.get("per_sample_metrics", {})

        print(f"{'Metric':<12} | {'KitREC':<10} | {args.baseline.upper():<10} | {'Diff':<10}")
        print("-" * 50)

        for metric in ["hit@10", "ndcg@10", "mrr"]:
            kitrec_val = kitrec_metrics.get(metric, 0)
            baseline_val = metrics.get(metric, 0)
            diff = kitrec_val - baseline_val
            sign = "+" if diff > 0 else ""
            print(f"{metric:<12} | {kitrec_val:.4f}     | {baseline_val:.4f}     | {sign}{diff:.4f}")

        # Statistical significance testing
        if kitrec_per_sample and metrics.get("per_sample"):
            print("\n" + "=" * 60)
            print("Statistical Significance Testing (Paired t-test)")
            print("=" * 60)

            stat_results = {}
            for metric in ["hit@10", "ndcg@10", "mrr"]:
                kitrec_scores = kitrec_per_sample.get(metric, [])
                baseline_scores = metrics.get("per_sample", {}).get(metric, [])

                if kitrec_scores and baseline_scores and len(kitrec_scores) == len(baseline_scores):
                    test_result = StatisticalAnalysis.paired_t_test(
                        kitrec_scores, baseline_scores
                    )
                    stat_results[metric] = test_result

                    # Format significance stars
                    sig = "***" if test_result["significant_at_0.001"] else \
                          "**" if test_result["significant_at_0.01"] else \
                          "*" if test_result["significant_at_0.05"] else ""

                    print(f"  {metric}: diff={test_result['mean_diff']:+.4f}{sig}, "
                          f"p={test_result['p_value']:.4f}, "
                          f"Cohen's d={test_result['effect_size_cohens_d']:.3f}")
                else:
                    print(f"  {metric}: Cannot compute (sample mismatch or missing data)")

            # Save statistical results
            save_json(stat_results, os.path.join(output_dir, "statistical_tests.json"))
            print("  (* p<0.05, ** p<0.01, *** p<0.001)")

    # Save results
    results_dict = {
        "baseline": args.baseline,
        "target_domain": args.target_domain,
        "candidate_set": args.candidate_set,
        "total_samples": len(samples),
        "metrics": metrics,
    }

    # Include user type metrics for RQ3 analysis
    if user_type_metrics:
        results_dict["user_type_metrics"] = user_type_metrics

    save_json(results_dict, os.path.join(output_dir, "results.json"))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
