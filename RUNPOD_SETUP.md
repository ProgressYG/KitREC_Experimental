# KitREC RunPod 환경 설정 가이드

**최종 업데이트:** 2025-12-07
**목적:** KitREC 실험을 위한 RunPod 환경 구축 및 실행 가이드

---

## 목차

1. [사전 준비 (로컬)](#1-사전-준비-로컬)
2. [RunPod 계정 및 결제 설정](#2-runpod-계정-및-결제-설정)
3. [Pod 생성](#3-pod-생성)
4. [환경 설정](#4-환경-설정)
5. [프로젝트 업로드](#5-프로젝트-업로드)
6. [환경 검증](#6-환경-검증)
7. [실험 실행](#7-실험-실행)
8. [문제 해결](#8-문제-해결)
9. [비용 최적화 팁](#9-비용-최적화-팁)

---

## 1. 사전 준비 (로컬)

### 1.1 필수 준비물

| 항목 | 설명 | 확인 방법 |
|------|------|----------|
| **HuggingFace Token** | Private 모델 접근용 | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **RunPod 계정** | GPU 서버 대여 | [runpod.io](https://runpod.io) |
| **결제 수단** | 크레딧 카드 또는 암호화폐 | RunPod Billing |
| **프로젝트 코드** | 로컬 검증 완료된 코드 | `scripts/verify_local.py` 실행 |

### 1.2 HuggingFace Token 발급

```bash
# 1. https://huggingface.co/settings/tokens 접속
# 2. "New token" 클릭
# 3. Name: "KitREC-RunPod"
# 4. Type: "Read" (읽기 전용이면 충분)
# 5. 토큰 복사 후 안전한 곳에 저장

# 토큰 형식 예시
hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
```

### 1.3 로컬 코드 검증 (필수!)

```bash
# RunPod 배포 전 로컬에서 반드시 실행
cd /path/to/Experimental_test
python scripts/verify_local.py

# 모든 체크가 통과해야 함
# ✅ Python syntax check
# ✅ Import verification
# ✅ Baseline model instantiation
# ✅ Data loader test
```

---

## 2. RunPod 계정 및 결제 설정

### 2.1 계정 생성

1. [runpod.io](https://runpod.io) 접속
2. "Sign Up" → GitHub/Google 계정으로 가입
3. 이메일 인증 완료

### 2.2 결제 설정

```
Settings → Billing → Add Payment Method
- Credit Card (권장)
- Crypto (Bitcoin, Ethereum 등)

권장 초기 충전: $50-100 (테스트 + 초기 실험용)
```

### 2.3 예상 비용

| GPU | 시간당 비용 | 24시간 비용 | 용도 |
|-----|-----------|------------|------|
| RTX 4090 (24GB) | $0.44 | $10.56 | 테스트/소규모 실험 |
| A100 40GB | $1.29 | $30.96 | 중규모 실험 |
| A100 80GB | $1.99 | $47.76 | 대규모 배치 |
| H100 80GB | $3.49 | $83.76 | 최대 속도 |

**KitREC 전체 실험 예상 비용:** $30-50 (RTX 4090 기준 약 3일)

---

## 3. Pod 생성

### 3.1 권장 스펙

| 항목 | 최소 사양 | 권장 사양 | 최적 사양 |
|------|----------|----------|----------|
| **GPU** | RTX 4090 (24GB) | A100 40GB | A100 80GB / H100 |
| **vRAM** | 24GB | 40GB | 80GB |
| **RAM** | 32GB | 64GB | 128GB |
| **Storage** | 50GB | 100GB | 200GB |
| **vCPU** | 8 | 16 | 32 |

### 3.2 Pod 생성 단계

```
1. RunPod Console → "Pods" → "+ Deploy"

2. GPU Selection:
   - Community Cloud (저렴) 또는 Secure Cloud (안정)
   - GPU: NVIDIA RTX 4090 또는 A100

3. Container Configuration:
   - Template: "RunPod PyTorch 2.2.0"
   - 또는 Custom Image: runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04

4. Volume:
   - Container Disk: 20GB
   - Volume Disk: 100GB (모델 캐시용)
   - Volume Mount Path: /workspace

5. Environment Variables:
   - HF_TOKEN: (HuggingFace 토큰)
   - HF_HOME: /workspace/.cache/huggingface

6. "Deploy On-Demand Pod" 클릭
```

### 3.3 권장 Template 설정

```yaml
# Pod Configuration
GPU: RTX 4090 또는 A100
Container Image: runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04
Volume: 100GB at /workspace
Expose Ports: 8888 (Jupyter), 22 (SSH)

# Environment Variables
HF_TOKEN=hf_your_token_here
HF_HOME=/workspace/.cache/huggingface
TRANSFORMERS_CACHE=/workspace/.cache/huggingface
CUDA_VISIBLE_DEVICES=0
```

---

## 4. 환경 설정

### 4.1 Pod 접속

```bash
# 방법 1: Web Terminal (RunPod Console에서 "Connect" → "Web Terminal")

# 방법 2: SSH (권장)
ssh root@{POD_IP} -p {SSH_PORT} -i ~/.ssh/your_key
```

### 4.2 자동 설정 스크립트 (권장)

```bash
# 1. 프로젝트 디렉토리로 이동
cd /workspace/Experimental_test

# 2. 설정 스크립트 실행
chmod +x scripts/setup_runpod.sh
./scripts/setup_runpod.sh
```

### 4.3 수동 설정 (문제 발생 시)

```bash
# 1. 환경 변수 확인
echo $HF_TOKEN
# 비어있으면 설정
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/workspace/.cache/huggingface"

# 2. pip 업그레이드
pip install --upgrade pip

# 3. PyTorch 확인 (이미 설치되어 있어야 함)
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 4. 필수 패키지 설치
pip install transformers==4.57.3 accelerate==1.12.0 peft==0.13.0
pip install bitsandbytes==0.46.1
pip install datasets huggingface-hub
pip install numpy scipy pandas scikit-learn tqdm pyyaml
pip install openai  # GPT-4.1 평가용

# 5. vLLM 설치 (가장 중요!)
pip install vllm>=0.6.0

# 6. 설치 확인
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
```

---

## 5. 프로젝트 업로드

### 5.1 방법 1: Git Clone (권장)

```bash
cd /workspace

# GitHub에서 직접 clone (private repo면 token 필요)
git clone https://github.com/your-username/KitREC.git
cd KitREC/Experimental_test
```

### 5.2 방법 2: SCP 업로드

```bash
# 로컬에서 실행
scp -P {SSH_PORT} -r ./Experimental_test root@{POD_IP}:/workspace/
```

### 5.3 방법 3: RunPod File Browser

```
1. RunPod Console → Pod → "Connect" → "File Browser"
2. Navigate to /workspace
3. Upload ZIP file
4. Unzip: unzip Experimental_test.zip
```

### 5.4 업로드 후 권한 설정

```bash
cd /workspace/Experimental_test
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

---

## 6. 환경 검증

### 6.1 전체 검증

```bash
cd /workspace/Experimental_test
python scripts/verify_environment.py
```

### 6.2 예상 출력

```
============================================================
 KitREC Environment Verification
============================================================

 Core Packages
  ✅ torch: 2.2.0+cu121
  ✅ transformers: 4.57.3
  ✅ accelerate: 1.12.0
  ✅ peft: 0.13.0
  ✅ datasets: 2.21.0
  ✅ numpy: 1.26.4
  ✅ scipy: 1.14.0

 GPU & CUDA Check
  CUDA Available: True
  CUDA Version: 12.1
  GPU Count: 1
  GPU 0: NVIDIA GeForce RTX 4090 (24.0 GB)
  ✅ Successfully allocated 4GB tensor

 vLLM Check
  ✅ vLLM Version: 0.6.3
  ✅ vLLM LLM class accessible

 HuggingFace Check
  ✅ HF_TOKEN set: hf_aBcD...6789
  ✅ huggingface_hub accessible
  ✅ Can access Qwen/Qwen3-14B model info

 Summary
  packages: ✅ PASS
  cuda: ✅ PASS
  vllm: ✅ PASS
  huggingface: ✅ PASS
  project: ✅ PASS

  🎉 All checks passed! Environment is ready.
```

### 6.3 데이터 로딩 테스트

```bash
python scripts/verify_env_and_data.py
```

---

## 7. 실험 실행

### 7.1 실행 순서 (권장)

```bash
# Phase 1: 소규모 테스트 (10 샘플)
python scripts/run_kitrec_eval.py \
    --model_name dualft_music_seta \
    --max_samples 10 \
    --output_dir results/test

# Phase 2: 중규모 테스트 (100 샘플)
python scripts/run_kitrec_eval.py \
    --model_name dualft_music_seta \
    --max_samples 100 \
    --output_dir results/test_100

# Phase 3: 전체 평가 (권장: tmux/screen 사용)
tmux new -s kitrec
python scripts/run_kitrec_eval.py \
    --model_name dualft_music_seta \
    --dataset Younggooo/kitrec-test-seta \
    --output_dir results/kitrec \
    --batch_size 8
# Ctrl+B, D 로 detach
```

### 7.2 전체 실험 배치 실행

```bash
# 8개 모델 순차 실행 스크립트
./scripts/run_all_evaluations.sh
```

### 7.3 Baseline 모델 실행

```bash
# CoNet 학습 + 평가
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain movies \
    --candidate_set seta \
    --train_baseline \
    --train_dataset Younggooo/kitrec-dualft_movies-seta

# DTCDR 학습 + 평가
python scripts/run_baseline_eval.py \
    --baseline dtcdr \
    --target_domain movies \
    --candidate_set seta \
    --train_baseline

# LLM4CDR 평가 (학습 불필요)
python scripts/run_baseline_eval.py \
    --baseline llm4cdr \
    --target_domain movies \
    --candidate_set seta
```

---

## 8. 문제 해결

### 8.1 CUDA Out of Memory

```bash
# 증상: torch.cuda.OutOfMemoryError

# 해결 1: batch_size 줄이기
--batch_size 4  # 기본 8 → 4

# 해결 2: GPU 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"

# 해결 3: 다른 프로세스 확인
nvidia-smi
kill -9 {PID}  # 불필요한 프로세스 종료
```

### 8.2 vLLM 설치 실패

```bash
# 증상: vLLM import error

# 해결 1: 재설치
pip uninstall vllm -y
pip install vllm --no-cache-dir

# 해결 2: CUDA 버전 확인
nvcc --version
python -c "import torch; print(torch.version.cuda)"
# 불일치 시 PyTorch 재설치
```

### 8.3 HuggingFace 인증 오류

```bash
# 증상: 401 Unauthorized

# 해결 1: 토큰 확인
echo $HF_TOKEN

# 해결 2: CLI 로그인
pip install huggingface-hub
huggingface-cli login
# 토큰 입력

# 해결 3: 환경 변수 재설정
export HF_TOKEN="hf_your_new_token"
```

### 8.4 모델 다운로드 느림

```bash
# 해결: 캐시 경로 확인 및 Volume 사용
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# 모델 사전 다운로드
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-14B')"
```

### 8.5 Pod 연결 끊김

```bash
# 해결: tmux 또는 screen 사용
tmux new -s experiment
# 실험 실행
# Ctrl+B, D 로 detach

# 재접속 시
tmux attach -t experiment
```

---

## 9. 비용 최적화 팁

### 9.1 Spot Instance 활용

```
- Community Cloud의 Spot Instance 사용 (최대 50% 저렴)
- 단, 언제든 중단될 수 있으므로 checkpoint 저장 필수
```

### 9.2 사용하지 않을 때 Pod 중지

```bash
# RunPod Console에서 "Stop" 클릭
# Volume 데이터는 유지됨
# 재시작 시 환경 재설정 불필요
```

### 9.3 효율적인 실험 순서

```
1. 작은 샘플로 테스트 (10-100개) → 코드 검증
2. 중간 샘플로 성능 추정 (1,000개) → 예상 결과 확인
3. 전체 실험은 야간/주말에 실행 → Spot 가격 낮음
```

### 9.4 모델 캐시 활용

```bash
# 첫 실행 시 모델 다운로드 (시간 소요)
# 이후 캐시에서 로드 (빠름)
# Volume에 캐시 저장하면 Pod 재시작 후에도 유지
```

---

## 10. 체크리스트

### 10.1 RunPod 배포 전 (로컬)

- [ ] HuggingFace Token 발급 완료
- [ ] 로컬 코드 검증 통과 (`python scripts/verify_local.py`)
- [ ] RunPod 계정 생성 및 결제 설정
- [ ] 프로젝트 코드 최신 상태 확인

### 10.2 Pod 생성 후

- [ ] SSH 또는 Web Terminal 접속 확인
- [ ] 환경 변수 설정 (HF_TOKEN)
- [ ] 패키지 설치 완료
- [ ] `verify_environment.py` 통과
- [ ] 데이터 로딩 테스트 통과

### 10.3 실험 실행 전

- [ ] 소규모 테스트 (10 샘플) 성공
- [ ] tmux/screen 세션 생성
- [ ] 결과 저장 경로 확인

---

## 부록: 유용한 명령어

```bash
# GPU 상태 모니터링
watch -n 1 nvidia-smi

# 디스크 사용량 확인
df -h

# 프로세스 확인
htop

# 로그 실시간 확인
tail -f results/experiment.log

# tmux 세션 관리
tmux ls                    # 세션 목록
tmux attach -t kitrec      # 세션 연결
tmux kill-session -t name  # 세션 종료
```

---

**작성 완료: 2025-12-07**
