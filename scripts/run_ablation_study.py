#!/usr/bin/env python3
"""
RQ1: 2×2 Ablation Study

4개 모델 비교:
① KitREC-Full (제안 모델) - Thinking + Fine-tuned
② KitREC-Direct (Ablation) - No Thinking + Fine-tuned (Option A: 별도 학습)
③ Base-CoT (Strong Baseline) - Thinking + Untuned
④ Base-Direct (Weak Baseline) - No Thinking + Untuned
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
from src.data.prompt_builder import PromptBuilder
from src.models.kitrec_model import KitRECModel
from src.models.base_model import BaseModel
from src.inference.vllm_inference import VLLMInference, GenerationConfig
from src.inference.output_parser import OutputParser, ErrorStatistics
from src.metrics.ranking_metrics import RankingMetrics
from src.metrics.statistical_analysis import StatisticalAnalysis
from src.utils.io_utils import save_json
from src.utils.visualization import generate_ablation_table

from tqdm import tqdm


# Domain availability status (sync with KitRECModel.AVAILABLE_DOMAINS)
AVAILABLE_DOMAINS = {
    "music": True,   # Ready for evaluation
    "movies": False  # Pending - models not yet trained
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ1 Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Domain Status:
  Music:  READY   - Can run ablation study
  Movies: PENDING - Models not yet available

Examples:
  # Run ablation study for Music domain
  python run_ablation_study.py --target_domain music

  # Quick test with 100 samples
  python run_ablation_study.py --target_domain music --max_samples 100
        """
    )

    parser.add_argument(
        "--target_domain",
        type=str,
        choices=["movies", "music"],
        required=True,
        help="Target domain for evaluation (currently only 'music' is available)"
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
        help="Candidate set (A: hard negatives, B: random)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablation",
        help="Output directory"
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def evaluate_model(
    model_name: str,
    model,
    prompt_builder: PromptBuilder,
    use_thinking: bool,
    samples: list,
    output_parser: OutputParser,
) -> dict:
    """단일 모델 평가"""
    print(f"\n  Evaluating {model_name}...")

    all_metrics = []
    per_sample_metrics = {
        "hit@10": [],
        "ndcg@10": [],
        "mrr": []
    }

    for sample in tqdm(samples, desc=f"  {model_name}"):
        # Build prompt based on model type
        if use_thinking:
            prompt = prompt_builder.build_thinking_prompt(sample)
        else:
            prompt = prompt_builder.build_direct_prompt(sample)

        # Get candidate IDs using static methods (no dataset loading needed)
        from src.data.data_loader import DataLoader
        candidate_ids = DataLoader._extract_candidate_ids_static(sample)
        gt = DataLoader._extract_ground_truth_static(sample)

        # Generate
        try:
            output = model.generate(prompt)
            parse_result = output_parser.parse(output, candidate_ids)
            predictions = parse_result.predictions
        except Exception as e:
            print(f"    Error: {e}")
            predictions = []

        # Calculate metrics
        metrics = RankingMetrics.calculate_all(predictions, gt.item_id)
        all_metrics.append(metrics)

        # Store per-sample for statistical testing
        for key in per_sample_metrics:
            per_sample_metrics[key].append(metrics.get(key, 0.0))

    aggregated = RankingMetrics.aggregate_metrics(all_metrics)
    aggregated["per_sample"] = per_sample_metrics

    return aggregated


def show_domain_status():
    """도메인 상태 출력"""
    print("=" * 60)
    print("Ablation Study - Domain Availability Status")
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
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("RQ1: 2×2 Ablation Study")
    print(f"Target Domain: {args.target_domain}")
    print(f"Candidate Set: {args.candidate_set}")
    print("=" * 60)

    # Load test data
    dataset_name = f"Younggooo/kitrec-test-{args.candidate_set}"
    data_loader = DataLoader(dataset_name, hf_token=args.hf_token)
    test_data = list(data_loader.load_test_data())

    # Filter by target domain
    test_data = [
        s for s in test_data
        if s.get("target_domain", "").lower() == args.target_domain
    ]

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"\nLoaded {len(test_data)} samples for {args.target_domain}")

    # Initialize components
    prompt_builder = PromptBuilder()
    output_parser = OutputParser()

    # Model configurations
    model_configs = {
        "kitrec_full": {
            "type": "kitrec",
            "name": f"dualft_{args.target_domain}_{args.candidate_set}",
            "use_thinking": True,
            "description": "Fine-tuned + Thinking"
        },
        "kitrec_direct": {
            "type": "kitrec",
            "name": f"direct_dualft_{args.target_domain}_{args.candidate_set}",
            "use_thinking": False,
            "description": "Fine-tuned + Direct"
        },
        "base_cot": {
            "type": "base",
            "use_thinking": True,
            "description": "Untuned + Thinking"
        },
        "base_direct": {
            "type": "base",
            "use_thinking": False,
            "description": "Untuned + Direct"
        }
    }

    results = {}

    for model_key, config in model_configs.items():
        print(f"\n[{model_key}] {config['description']}")

        try:
            if config["type"] == "kitrec":
                model = KitRECModel.load(config["name"], hf_token=args.hf_token)
            else:
                if config["use_thinking"]:
                    model = BaseModel.load_cot(hf_token=args.hf_token)
                else:
                    model = BaseModel.load_direct(hf_token=args.hf_token)

            model.initialize()

            metrics = evaluate_model(
                model_key, model, prompt_builder,
                config["use_thinking"], test_data, output_parser
            )
            results[model_key] = metrics

            print(f"    Hit@10: {metrics.get('hit@10', 0):.4f}")
            print(f"    NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
            print(f"    MRR: {metrics.get('mrr', 0):.4f}")

        except Exception as e:
            print(f"    Error loading model: {e}")
            results[model_key] = {"error": str(e)}

    # Statistical significance testing
    print("\n" + "=" * 60)
    print("Statistical Significance Testing")
    print("=" * 60)

    if "kitrec_full" in results and "per_sample" in results["kitrec_full"]:
        kitrec_scores = results["kitrec_full"]["per_sample"]

        comparisons = {}
        for model_key in ["kitrec_direct", "base_cot", "base_direct"]:
            if model_key in results and "per_sample" in results[model_key]:
                model_scores = results[model_key]["per_sample"]
                comparisons[model_key] = {}

                for metric in ["hit@10", "ndcg@10", "mrr"]:
                    test_result = StatisticalAnalysis.paired_t_test(
                        kitrec_scores[metric],
                        model_scores[metric]
                    )
                    comparisons[model_key][metric] = test_result

                    sig = "***" if test_result["significant_at_0.001"] else \
                          "**" if test_result["significant_at_0.01"] else \
                          "*" if test_result["significant_at_0.05"] else ""

                    print(f"  {model_key} vs kitrec_full ({metric}): "
                          f"diff={test_result['mean_diff']:+.4f}{sig}")

        # Save comparisons
        save_json(comparisons, os.path.join(output_dir, "statistical_tests.json"))

    # Generate ablation table
    ablation_results = {k: v for k, v in results.items() if "error" not in v}
    table_md = generate_ablation_table(ablation_results)

    with open(os.path.join(output_dir, "ablation_table.md"), "w") as f:
        f.write("# RQ1: 2×2 Ablation Study Results\n\n")
        f.write(f"Domain: {args.target_domain}\n")
        f.write(f"Candidate Set: {args.candidate_set}\n\n")
        f.write(table_md)

    # Save all results
    save_json(results, os.path.join(output_dir, "ablation_results.json"))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
