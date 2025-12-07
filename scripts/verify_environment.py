#!/usr/bin/env python3
"""
KitREC 환경 검증 스크립트

RunPod 환경 설정 후 실행하여 모든 의존성이 올바르게 설치되었는지 확인
"""

import sys
import os

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_import(module_name, min_version=None):
    """모듈 import 및 버전 확인"""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        status = "✅"

        if min_version and version != "unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                status = "⚠️ "

        print(f"  {status} {module_name}: {version}")
        return True
    except ImportError as e:
        print(f"  ❌ {module_name}: NOT INSTALLED ({e})")
        return False

def check_cuda():
    """CUDA 및 GPU 확인"""
    print_header("GPU & CUDA Check")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_available}")

        if cuda_available:
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")

            gpu_count = torch.cuda.device_count()
            print(f"  GPU Count: {gpu_count}")

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")

            # Memory test
            print(f"\n  Memory Test:")
            try:
                x = torch.zeros(1024, 1024, 1024, device='cuda')  # ~4GB
                del x
                torch.cuda.empty_cache()
                print(f"    ✅ Successfully allocated 4GB tensor")
            except Exception as e:
                print(f"    ⚠️  Memory test failed: {e}")

            return True
        else:
            print("  ❌ CUDA not available!")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_vllm():
    """vLLM 설치 확인"""
    print_header("vLLM Check")

    try:
        import vllm
        print(f"  ✅ vLLM Version: {vllm.__version__}")

        # Check if vLLM can see GPU
        from vllm import LLM
        print(f"  ✅ vLLM LLM class accessible")

        return True
    except ImportError as e:
        print(f"  ❌ vLLM not installed: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  vLLM installed but error: {e}")
        return True

def check_huggingface():
    """HuggingFace 토큰 및 접근 확인"""
    print_header("HuggingFace Check")

    # Check token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        masked = hf_token[:8] + "..." + hf_token[-4:]
        print(f"  ✅ HF_TOKEN set: {masked}")
    else:
        print(f"  ⚠️  HF_TOKEN not set (needed for private models)")

    # Check huggingface_hub
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"  ✅ huggingface_hub accessible")

        # Try to access a public model
        try:
            api.model_info("Qwen/Qwen3-14B")
            print(f"  ✅ Can access Qwen/Qwen3-14B model info")
        except Exception as e:
            print(f"  ⚠️  Cannot access model info: {e}")

        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_project_structure():
    """프로젝트 구조 확인"""
    print_header("Project Structure Check")

    required_dirs = [
        "configs",
        "src/data",
        "src/models",
        "src/inference",
        "src/metrics",
        "src/utils",
        "baselines/conet",
        "baselines/dtcdr",
        "baselines/llm4cdr",
        "scripts",
        "results",
    ]

    required_files = [
        "configs/eval_config.yaml",
        "configs/model_paths.yaml",
        "src/data/data_loader.py",
        "src/inference/vllm_inference.py",
        "src/metrics/ranking_metrics.py",
        "scripts/run_kitrec_eval.py",
    ]

    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print(f"  Project root: {project_root}")

    all_ok = True

    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(project_root, dir_path)
        if os.path.isdir(full_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ NOT FOUND")
            all_ok = False

    # Check key files
    print()
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.isfile(full_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} NOT FOUND")
            all_ok = False

    return all_ok

def run_quick_inference_test():
    """간단한 추론 테스트 (선택적)"""
    print_header("Quick Inference Test (Optional)")

    response = input("  Run quick inference test? (y/N): ").strip().lower()
    if response != 'y':
        print("  Skipped.")
        return True

    try:
        print("  Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
        print("  ✅ Tokenizer loaded successfully")

        # Simple tokenization test
        test_text = "This is a test recommendation prompt."
        tokens = tokenizer.encode(test_text)
        print(f"  ✅ Tokenization works: {len(tokens)} tokens")

        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print(" KitREC Environment Verification")
    print("="*60)

    results = {}

    # 1. Core packages
    print_header("Core Packages")
    packages = [
        ("torch", "2.2.0"),
        ("transformers", "4.50.0"),
        ("accelerate", "1.0.0"),
        ("peft", "0.10.0"),
        ("datasets", "2.16.0"),
        ("numpy", None),
        ("scipy", None),
        ("pandas", None),
        ("tqdm", None),
        ("yaml", None),
    ]

    pkg_ok = all(check_import(pkg, ver) for pkg, ver in packages)
    results["packages"] = pkg_ok

    # 2. CUDA
    results["cuda"] = check_cuda()

    # 3. vLLM
    results["vllm"] = check_vllm()

    # 4. HuggingFace
    results["huggingface"] = check_huggingface()

    # 5. Project structure
    results["project"] = check_project_structure()

    # Summary
    print_header("Summary")

    all_ok = all(results.values())

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")

    print()
    if all_ok:
        print("  🎉 All checks passed! Environment is ready.")
        print()
        print("  Next steps:")
        print("  1. python scripts/run_kitrec_eval.py --help")
        print("  2. python scripts/run_kitrec_eval.py --model_name dualft_music_seta --max_samples 10")
    else:
        print("  ⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
