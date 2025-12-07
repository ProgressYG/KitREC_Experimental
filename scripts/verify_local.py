#!/usr/bin/env python3
"""
Local Verification Script (No GPU Required)

RunPod 배포 전 로컬에서 검증할 수 있는 모든 항목 체크:
1. Python 문법 검사 (모든 .py 파일)
2. Import 검증 (GPU 없이)
3. 프로젝트 구조 검증
4. Baseline 모델 인스턴스화 (CPU 모드)
5. 데이터 로더 기본 기능
6. 설정 파일 검증

Usage:
    python scripts/verify_local.py
    python scripts/verify_local.py --verbose
    python scripts/verify_local.py --skip-imports  # 빠른 검사
"""

import os
import sys
import ast
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """Terminal colors"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_success(text: str):
    print(f"  {Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    print(f"  {Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str):
    print(f"  {Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    print(f"  {Colors.BLUE}ℹ️  {text}{Colors.RESET}")


# =============================================================================
# 1. Python Syntax Check
# =============================================================================

def check_python_syntax(verbose: bool = False) -> Tuple[int, int, List[str]]:
    """모든 .py 파일의 문법 검사"""
    print_header("[1/6] Python Syntax Check")

    py_files = list(PROJECT_ROOT.rglob("*.py"))
    # Exclude __pycache__ and .git
    py_files = [f for f in py_files if "__pycache__" not in str(f) and ".git" not in str(f)]

    passed = 0
    failed = 0
    errors = []

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source)
            passed += 1
            if verbose:
                print_success(f"{py_file.relative_to(PROJECT_ROOT)}")
        except SyntaxError as e:
            failed += 1
            error_msg = f"{py_file.relative_to(PROJECT_ROOT)}:{e.lineno} - {e.msg}"
            errors.append(error_msg)
            print_error(error_msg)

    print(f"\n  Total: {len(py_files)} files, {Colors.GREEN}{passed} passed{Colors.RESET}, {Colors.RED}{failed} failed{Colors.RESET}")
    return passed, failed, errors


# =============================================================================
# 2. Project Structure Check
# =============================================================================

def check_project_structure() -> Tuple[int, int, List[str]]:
    """필수 파일/디렉토리 존재 확인"""
    print_header("[2/6] Project Structure Check")

    required_structure = {
        "directories": [
            "src",
            "src/data",
            "src/inference",
            "src/metrics",
            "baselines",
            "baselines/conet",
            "baselines/dtcdr",
            "baselines/llm4cdr",
            "scripts",
        ],
        "files": [
            "src/__init__.py",
            "src/data/__init__.py",
            "src/data/data_loader.py",
            "src/data/prompt_builder.py",
            "src/inference/__init__.py",
            "src/inference/vllm_inference.py",
            "src/inference/output_parser.py",
            "src/metrics/__init__.py",
            "src/metrics/ranking_metrics.py",
            "src/metrics/explainability_metrics.py",
            "src/metrics/statistical_analysis.py",
            "baselines/__init__.py",
            "baselines/base_evaluator.py",
            "baselines/conet/__init__.py",
            "baselines/conet/model.py",
            "baselines/conet/trainer.py",
            "baselines/conet/evaluator.py",
            "baselines/dtcdr/__init__.py",
            "baselines/dtcdr/model.py",
            "baselines/dtcdr/trainer.py",
            "baselines/dtcdr/evaluator.py",
            "baselines/llm4cdr/__init__.py",
            "baselines/llm4cdr/prompts.py",
            "baselines/llm4cdr/evaluator.py",
            "scripts/run_baseline_eval.py",
            "scripts/run_kitrec_eval.py",
        ],
    }

    passed = 0
    failed = 0
    missing = []

    # Check directories
    for dir_path in required_structure["directories"]:
        full_path = PROJECT_ROOT / dir_path
        if full_path.is_dir():
            passed += 1
            print_success(f"Directory: {dir_path}/")
        else:
            failed += 1
            missing.append(f"Directory: {dir_path}/")
            print_error(f"Missing directory: {dir_path}/")

    # Check files
    for file_path in required_structure["files"]:
        full_path = PROJECT_ROOT / file_path
        if full_path.is_file():
            passed += 1
            print_success(f"File: {file_path}")
        else:
            failed += 1
            missing.append(f"File: {file_path}")
            print_error(f"Missing file: {file_path}")

    print(f"\n  Total: {passed + failed} items, {Colors.GREEN}{passed} found{Colors.RESET}, {Colors.RED}{failed} missing{Colors.RESET}")
    return passed, failed, missing


# =============================================================================
# 3. Import Verification (No GPU)
# =============================================================================

def check_imports(skip: bool = False) -> Tuple[int, int, List[str]]:
    """Import 검증 (GPU 의존성 제외)"""
    print_header("[3/6] Import Verification")

    if skip:
        print_warning("Skipped (--skip-imports)")
        return 0, 0, []

    # Modules to test (순서 중요 - 의존성 순)
    modules_to_test = [
        # Core modules
        ("src.data.data_loader", "DataLoader"),
        ("src.data.prompt_builder", "PromptBuilder"),
        ("src.inference.output_parser", "OutputParser"),
        ("src.metrics.ranking_metrics", "RankingMetrics"),
        ("src.metrics.explainability_metrics", "ExplainabilityMetrics"),
        ("src.metrics.statistical_analysis", "StatisticalAnalysis"),
        # Baseline modules
        ("baselines.base_evaluator", "BaseEvaluator"),
        ("baselines.conet.model", "CoNet"),
        ("baselines.conet.trainer", "CoNetTrainer"),
        ("baselines.conet.evaluator", "CoNetEvaluator"),
        ("baselines.dtcdr.model", "DTCDRModel"),
        ("baselines.dtcdr.trainer", "DTCDRTrainer"),
        ("baselines.dtcdr.evaluator", "DTCDREvaluator"),
        ("baselines.llm4cdr.prompts", "LLM4CDRPrompts"),
        ("baselines.llm4cdr.evaluator", "LLM4CDREvaluator"),
    ]

    passed = 0
    failed = 0
    errors = []

    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                passed += 1
                print_success(f"{module_name}.{class_name}")
            else:
                failed += 1
                error_msg = f"{module_name} - class {class_name} not found"
                errors.append(error_msg)
                print_error(error_msg)
        except ImportError as e:
            failed += 1
            error_msg = f"{module_name} - ImportError: {e}"
            errors.append(error_msg)
            print_error(error_msg)
        except Exception as e:
            failed += 1
            error_msg = f"{module_name} - {type(e).__name__}: {e}"
            errors.append(error_msg)
            print_error(error_msg)

    print(f"\n  Total: {len(modules_to_test)} modules, {Colors.GREEN}{passed} passed{Colors.RESET}, {Colors.RED}{failed} failed{Colors.RESET}")
    return passed, failed, errors


# =============================================================================
# 4. Baseline Model Instantiation (CPU)
# =============================================================================

def check_baseline_models() -> Tuple[int, int, List[str]]:
    """Baseline 모델 인스턴스화 테스트 (CPU 모드)"""
    print_header("[4/6] Baseline Model Instantiation (CPU)")

    passed = 0
    failed = 0
    errors = []

    # Test CoNet
    try:
        from baselines.conet.model import CoNet
        model = CoNet(
            num_users=100,
            num_source_items=100,
            num_target_items=100,
            embedding_dim=64,
            hidden_dims=[128, 64]
        )
        # Move to CPU explicitly
        model = model.cpu()
        print_success("CoNet instantiation (CPU)")
        passed += 1
    except Exception as e:
        error_msg = f"CoNet - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Test DTCDR
    try:
        from baselines.dtcdr.model import DTCDRModel
        model = DTCDRModel(
            num_users=100,
            num_source_items=100,
            num_target_items=100,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        model = model.cpu()
        print_success("DTCDRModel instantiation (CPU)")
        passed += 1
    except Exception as e:
        error_msg = f"DTCDRModel - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Test Evaluators (don't need inference engine for instantiation check)
    try:
        from baselines.base_evaluator import BaseEvaluator
        # BaseEvaluator is abstract, just check it's importable
        print_success("BaseEvaluator import")
        passed += 1
    except Exception as e:
        error_msg = f"BaseEvaluator - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    print(f"\n  Total: {passed + failed} tests, {Colors.GREEN}{passed} passed{Colors.RESET}, {Colors.RED}{failed} failed{Colors.RESET}")
    return passed, failed, errors


# =============================================================================
# 5. Data Loader Check
# =============================================================================

def check_data_loader() -> Tuple[int, int, List[str]]:
    """Data Loader 기본 기능 검증"""
    print_header("[5/6] Data Loader Check")

    passed = 0
    failed = 0
    errors = []

    # Check DataLoader class
    try:
        from src.data.data_loader import DataLoader
        print_success("DataLoader class import")
        passed += 1
    except Exception as e:
        error_msg = f"DataLoader import - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1
        return passed, failed, errors

    # Check DataLoader instantiation (without loading data)
    try:
        loader = DataLoader("Younggooo/kitrec-test-seta")
        print_success("DataLoader instantiation")
        passed += 1
    except Exception as e:
        error_msg = f"DataLoader instantiation - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Check helper methods exist
    try:
        from src.data.data_loader import DataLoader
        loader = DataLoader("Younggooo/kitrec-test-seta")

        methods_to_check = [
            "load_test_data",
            "extract_candidate_ids",
            "extract_ground_truth",
        ]

        for method in methods_to_check:
            if hasattr(loader, method) and callable(getattr(loader, method)):
                print_success(f"DataLoader.{method}() method exists")
                passed += 1
            else:
                error_msg = f"DataLoader.{method}() method not found"
                errors.append(error_msg)
                print_error(error_msg)
                failed += 1
    except Exception as e:
        error_msg = f"DataLoader method check - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Note: Actual data loading requires network/HF token
    print_info("Note: Actual data loading requires HF_TOKEN and network access")

    print(f"\n  Total: {passed + failed} checks, {Colors.GREEN}{passed} passed{Colors.RESET}, {Colors.RED}{failed} failed{Colors.RESET}")
    return passed, failed, errors


# =============================================================================
# 6. Metrics Classes Check (Non-PyTorch)
# =============================================================================

def check_metrics_classes() -> Tuple[int, int, List[str]]:
    """Metrics 클래스 검증 (PyTorch 없이 테스트 가능한 부분만)"""
    print_header("[6/6] Metrics Classes Check (Non-PyTorch)")

    passed = 0
    failed = 0
    errors = []

    # Test RankingMetrics (uses only numpy, should work without torch)
    try:
        from src.metrics.ranking_metrics import RankingMetrics

        # Create dummy predictions
        dummy_predictions = [
            {"item_id": "A", "confidence_score": 9.0},
            {"item_id": "B", "confidence_score": 8.0},
            {"item_id": "C", "confidence_score": 7.0},
            {"item_id": "D", "confidence_score": 6.0},
            {"item_id": "E", "confidence_score": 5.0},
        ]
        gt_id = "C"

        # Test hit_at_k (static method)
        result = RankingMetrics.hit_at_k(dummy_predictions, gt_id, k=5)
        assert result == 1.0, f"Expected 1.0 (C in top 5), got {result}"

        result = RankingMetrics.hit_at_k(dummy_predictions, gt_id, k=2)
        assert result == 0.0, f"Expected 0.0 (C not in top 2), got {result}"

        print_success("RankingMetrics.hit_at_k()")
        passed += 1
    except Exception as e:
        error_msg = f"RankingMetrics.hit_at_k - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Test NDCG calculation
    try:
        from src.metrics.ranking_metrics import RankingMetrics

        dummy_predictions = [
            {"item_id": "GT_ITEM", "confidence_score": 9.0},  # rank 1
            {"item_id": "B", "confidence_score": 8.0},
        ]

        result = RankingMetrics.ndcg_at_k(dummy_predictions, "GT_ITEM", k=10)
        assert abs(result - 1.0) < 0.01, f"Expected ~1.0 for rank 1, got {result}"

        print_success("RankingMetrics.ndcg_at_k()")
        passed += 1
    except Exception as e:
        error_msg = f"RankingMetrics.ndcg_at_k - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Test MRR calculation
    try:
        from src.metrics.ranking_metrics import RankingMetrics

        dummy_predictions = [
            {"item_id": "A", "confidence_score": 9.0},
            {"item_id": "GT_ITEM", "confidence_score": 8.0},  # rank 2
        ]

        result = RankingMetrics.mrr(dummy_predictions, "GT_ITEM")
        assert abs(result - 0.5) < 0.01, f"Expected 0.5 for rank 2, got {result}"

        print_success("RankingMetrics.mrr()")
        passed += 1
    except Exception as e:
        error_msg = f"RankingMetrics.mrr - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Test calculate_all
    try:
        from src.metrics.ranking_metrics import RankingMetrics

        dummy_predictions = [
            {"item_id": "A", "confidence_score": 9.0},
            {"item_id": "B", "confidence_score": 8.0},
            {"item_id": "GT", "confidence_score": 7.0},  # rank 3
        ]

        result = RankingMetrics.calculate_all(dummy_predictions, "GT")
        assert "hit@1" in result
        assert "hit@5" in result
        assert "mrr" in result
        assert "ndcg@10" in result

        print_success("RankingMetrics.calculate_all()")
        passed += 1
    except Exception as e:
        error_msg = f"RankingMetrics.calculate_all - {type(e).__name__}: {e}"
        errors.append(error_msg)
        print_error(error_msg)
        failed += 1

    # Note about torch-dependent metrics
    print_info("StatisticalAnalysis and ExplainabilityMetrics require PyTorch (tested on RunPod)")

    print(f"\n  Total: {passed + failed} tests, {Colors.GREEN}{passed} passed{Colors.RESET}, {Colors.RED}{failed} failed{Colors.RESET}")
    return passed, failed, errors


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KitREC Local Verification Script")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-imports", action="store_true", help="Skip import checks (faster)")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  KitREC Local Verification (No GPU Required){Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"\nProject Root: {PROJECT_ROOT}")

    all_results: Dict[str, Tuple[int, int, List[str]]] = {}

    # Run all checks
    all_results["syntax"] = check_python_syntax(args.verbose)
    all_results["structure"] = check_project_structure()
    all_results["imports"] = check_imports(args.skip_imports)
    all_results["models"] = check_baseline_models()
    all_results["data_loader"] = check_data_loader()
    all_results["metrics"] = check_metrics_classes()

    # Summary
    print_header("VERIFICATION SUMMARY")

    total_passed = 0
    total_failed = 0
    all_errors = []
    torch_errors = []
    critical_errors = []

    for name, (passed, failed, errors) in all_results.items():
        total_passed += passed
        total_failed += failed

        for error in errors:
            if "No module named 'torch'" in error or "ModuleNotFoundError" in error:
                torch_errors.append(error)
            else:
                critical_errors.append(error)
                all_errors.append(error)

        status = f"{Colors.GREEN}PASS{Colors.RESET}" if failed == 0 else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {name:15} : {status} ({passed}/{passed+failed})")

    print(f"\n  {'='*40}")
    print(f"  Total: {Colors.GREEN}{total_passed} passed{Colors.RESET}, {Colors.RED}{total_failed} failed{Colors.RESET}")

    # Separate torch errors from critical errors
    if torch_errors:
        print(f"\n{Colors.YELLOW}  ⚠️  PyTorch-dependent checks skipped (expected on local Mac):{Colors.RESET}")
        print(f"      {len(torch_errors)} tests require PyTorch (will work on RunPod)")

    if critical_errors:
        print(f"\n{Colors.RED}  ❌ Critical errors that need fixing:{Colors.RESET}")
        for error in critical_errors[:10]:
            print(f"    - {error}")
        if len(critical_errors) > 10:
            print(f"    ... and {len(critical_errors) - 10} more errors")
        sys.exit(1)
    else:
        # Check if core checks passed
        syntax_passed = all_results["syntax"][1] == 0
        structure_passed = all_results["structure"][1] == 0
        data_loader_passed = all_results["data_loader"][1] == 0
        metrics_passed = all_results["metrics"][1] == 0

        if syntax_passed and structure_passed and data_loader_passed and metrics_passed:
            print(f"\n{Colors.GREEN}  ✅ All critical checks passed!{Colors.RESET}")
            print(f"  {Colors.GREEN}  Ready for RunPod deployment.{Colors.RESET}")
            print(f"\n  {Colors.BLUE}Note: PyTorch-dependent modules will be tested on RunPod.{Colors.RESET}")
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}  ⚠️  Some critical checks failed. Please fix before RunPod deployment.{Colors.RESET}")
            sys.exit(1)


if __name__ == "__main__":
    main()
