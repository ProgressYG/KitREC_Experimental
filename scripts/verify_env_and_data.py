#!/usr/bin/env python3
"""
Phase 1: Environment & Data Verification Script
Verifies:
1. vLLM installation
2. GPU availability and properties
3. HuggingFace Token
4. Data Loader functionality
5. Candidate Set integrity (1 GT + 99 Neg)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy import wrapper for DataLoader
DataLoader = None

def lazy_import_dataloader():
    global DataLoader
    try:
        from src.data.data_loader import DataLoader as DL
        DataLoader = DL
        return True
    except ImportError as e:
        print(f"  ❌ Could not import DataLoader: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error importing DataLoader: {e}")
        return False

def check_vllm():
    print("\n[1/5] Checking vLLM installation...")
    try:
        import vllm
        print(f"  ✅ vLLM installed version: {vllm.__version__}")
        return True
    except ImportError:
        print("  ❌ vLLM NOT installed.")
        return False
    except Exception as e:
        print(f"  ❌ vLLM import error: {e}")
        return False

def check_gpu():
    print("\n[2/5] Checking GPU availability...")
    try:
        import torch
    except ImportError:
        print("  ❌ 'torch' module NOT found.")
        return False

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"  ✅ CUDA Available: {count} device(s)")
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("  ⚠️ CUDA NOT available. (Running on CPU or Metal?)")
        # Check for MPS (Mac)
        if torch.backends.mps.is_available():
             print("  ✅ MacOS MPS (Metal Performance Shaders) is available.")
        return False

def check_hf_token():
    print("\n[3/5] Checking HuggingFace Token...")
    token = os.environ.get("HF_TOKEN")
    
    # Try loading from .env if not in env vars
    if not token and os.path.exists(".env"):
        print("  ℹ️ .env file found, attempting to read...")
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except:
            pass

    if token:
        print("  ✅ HF_TOKEN found.")
        print(f"  Token: {token[:4]}...{token[-4:]}")
        # Export for this process
        os.environ["HF_TOKEN"] = token
        return True
    else:
        print("  ⚠️ HF_TOKEN NOT found in environment variables or .env")
        return False

def check_data_loader():
    print("\n[4/5] Checking Data Loading (Test Set A)...")
    
    # Try to import necessary libraries for data loading
    try:
        import datasets
    except ImportError:
        print("  ❌ 'datasets' library NOT installed.")
        return False, None

    if not lazy_import_dataloader():
        return False, None

    try:
        loader = DataLoader("Younggooo/kitrec-test-seta")
        # Try loading just the first few samples to avoid full download if streaming possible
        print("  Loading dataset... (this may take a moment)")
        dataset = loader.load_test_data() 
        print(f"  ✅ Dataset loaded. Total samples: {len(dataset)}")
        
        sample = dataset[0]
        print(f"  Sample ID: {sample.get('user_id')}")
        print(f"  Fields: {list(sample.keys())}")
        return True, dataset
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False, None

def check_candidate_integrity(dataset):
    print("\n[5/5] Verifying Candidate Set Integrity...")
    if not dataset:
        print("  ❌ Skipping (Dataset not loaded)")
        return
        
    if not DataLoader:
        return

    # Create temporary loader instance
    loader = DataLoader("Younggooo/kitrec-test-seta")
    valid_count = 0
    total_checked = 100 # Check first 100 samples
    
    print(f"  Checking first {total_checked} samples...")
    
    for i in range(total_checked):
        sample = dataset[i]
        
        try:
            # 1. Extract Candidates
            candidates = loader.extract_candidate_ids(sample)
            
            # 2. Extract GT
            gt = loader.extract_ground_truth(sample)
            
            # 3. Verify Size
            if len(candidates) != 100:
                print(f"  ❌ Sample {i} Invalid Size: {len(candidates)}")
                continue
                
            # 4. Verify GT Inclusion
            if gt.item_id not in candidates:
                print(f"  ❌ Sample {i} GT Missing: {gt.item_id} not in candidates")
                continue
                
            valid_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing sample {i}: {e}")
    
    if valid_count == total_checked:
        print(f"  ✅ All {total_checked} samples passed integrity check (100 items + GT included).")
    else:
        print(f"  ⚠️ Only {valid_count}/{total_checked} samples passed.")

def main():
    print("=== KitREC Environment Verification ===")
    
    vllm_ok = check_vllm()
    # On Mac/Local environment for agent, GPU might be absent or different. 
    # Logic: Just report.
    check_gpu()
    check_hf_token()
    
    data_ok, dataset = check_data_loader()
    
    if data_ok:
        check_candidate_integrity(dataset)
    else:
        print("  ❌ Skipping candidate check due to load failure.")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    main()
