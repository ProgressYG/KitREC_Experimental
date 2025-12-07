#!/usr/bin/env python3
"""
KitREC 모델 평가 실행 스크립트

8개 모델 × 30,000 샘플 평가
- DualFT-Movies (Set A/B)
- DualFT-Music (Set A/B)
- SingleFT-Movies (Set A/B)
- SingleFT-Music (Set A/B)
"""

import argparse
import os
import sys
import random
from datetime import datetime

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

from src.data.data_loader import DataLoader
from src.data.prompt_builder import PromptBuilder
from src.models.kitrec_model import KitRECModel
from src.inference.vllm_inference import VLLMInference, GenerationConfig
from src.inference.output_parser import OutputParser, ErrorStatistics
from src.inference.batch_inference import BatchInference
from src.metrics.ranking_metrics import RankingMetrics
from src.metrics.explainability_metrics import ExplainabilityMetrics
from src.metrics.stratified_analysis import StratifiedAnalysis
from src.utils.io_utils import save_results, save_json
from src.utils.logger import EvaluationLogger


def parse_args():
    parser = argparse.ArgumentParser(description="KitREC Model Evaluation")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "dualft_movies_seta", "dualft_movies_setb",
            "dualft_music_seta", "dualft_music_setb",
            "singleft_movies_seta", "singleft_movies_setb",
            "singleft_music_seta", "singleft_music_setb",
        ],
        help="KitREC model name"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Younggooo/kitrec-test-seta",
        help="HuggingFace test dataset name"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/kitrec",
        help="Output directory for results"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for testing)"
    )

    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env)"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Checkpoint save interval"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, args.model_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    logger = EvaluationLogger(
        name=f"kitrec.{args.model_name}",
        log_dir=os.path.join(output_dir, "logs")
    )

    print(f"=" * 60)
    print(f"KitREC Evaluation: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    # Load data
    print("\n[1/5] Loading test data...")
    data_loader = DataLoader(args.dataset, hf_token=args.hf_token)
    test_data = data_loader.load_test_data()

    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))

    print(f"  Loaded {len(test_data)} samples")
    logger.log_start(args.model_name, args.dataset, len(test_data))

    # Load model
    print("\n[2/5] Loading KitREC model...")
    model = KitRECModel.load(args.model_name, hf_token=args.hf_token)
    print(f"  Model: {model.config.lora_path}")

    # Initialize inference engine
    print("\n[3/5] Initializing inference engine...")
    inference_engine = VLLMInference(
        model_name=model.config.base_model_name,
        lora_path=model.config.lora_path,
        hf_token=args.hf_token,
        enable_lora=True,
    )

    output_parser = OutputParser()

    # Run evaluation
    print("\n[4/5] Running evaluation...")
    batch_inference = BatchInference(
        inference_engine=inference_engine,
        output_parser=output_parser,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=os.path.join(output_dir, "checkpoints")
    )

    config = GenerationConfig(
        max_new_tokens=2048,
        temperature=0.0  # Greedy decoding
    )

    results = batch_inference.run(
        samples=list(test_data),
        extract_prompt_fn=lambda s: data_loader.extract_prompt(s),
        extract_candidates_fn=lambda s: data_loader.extract_candidate_ids(s),
        extract_gt_fn=lambda s: {
            "item_id": data_loader.extract_ground_truth(s).item_id,
            "rating": data_loader.extract_ground_truth(s).rating
        },
        extract_metadata_fn=lambda s: data_loader.extract_metadata(s),
        config=config,
        resume_from_checkpoint=args.resume,
    )

    # Calculate metrics
    print("\n[5/5] Calculating metrics...")

    # Ranking metrics
    all_metrics = []
    for result in results:
        metrics = RankingMetrics.calculate_all(
            result.parse_result.predictions,
            result.ground_truth["item_id"]
        )
        all_metrics.append(metrics)

    aggregated = RankingMetrics.aggregate_metrics(all_metrics)

    # Stratified analysis
    stratified = StratifiedAnalysis()
    results_for_analysis = [
        {
            "metrics": m,
            "metadata": r.metadata,
            "ground_truth": r.ground_truth
        }
        for m, r in zip(all_metrics, results)
    ]

    user_type_analysis = stratified.analyze_by_user_type(results_for_analysis)
    core_level_analysis = stratified.analyze_by_core_level(results_for_analysis)

    # Save results
    print("\n[6/6] Saving results...")

    # Predictions
    batch_inference.save_final_results(output_dir)

    # Metrics summary
    save_json({
        "model_name": args.model_name,
        "dataset": args.dataset,
        "total_samples": len(results),
        "aggregated_metrics": aggregated,
        "error_statistics": batch_inference.get_error_statistics(),
        "timing_statistics": batch_inference.get_timing_statistics(),
    }, os.path.join(output_dir, "metrics_summary.json"))

    # Stratified analysis
    save_json({
        "by_user_type": user_type_analysis,
        "by_core_level": core_level_analysis,
    }, os.path.join(output_dir, "stratified_analysis.json"))

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nAggregated Metrics:")
    for metric, value in aggregated.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    print(f"\nError Statistics:")
    error_stats = batch_inference.get_error_statistics()
    print(f"  Parse failure rate: {error_stats.get('parse_failure_rate', 0):.2%}")
    print(f"  Invalid item rate: {error_stats.get('invalid_item_rate', 0):.2%}")

    print(f"\nResults saved to: {output_dir}")

    logger.log_metrics(aggregated)
    logger.log_finish(
        batch_inference.get_timing_statistics().get("total_time_seconds", 0),
        error_stats.get("parse_failure_rate", 0)
    )


if __name__ == "__main__":
    main()
