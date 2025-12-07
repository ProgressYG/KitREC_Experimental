# KitREC 학습된 모델 사용 가이드

**마지막 업데이트**: 2025-12-06

## 개요

KitREC 모델은 **LoRA 어댑터**만 저장됩니다. Base Model(Qwen3-14B)은 포함되지 않으므로, 사용 시 별도로 다운로드해야 합니다.

| 항목 | 크기 | 저장 위치 |
|------|------|----------|
| **LoRA 어댑터** | ~500MB | HuggingFace Hub (Private) |
| **Base Model** (Qwen3-14B) | ~28GB | HuggingFace Hub (Public) |

---

## 1. 환경 설정

### 필수 패키지 설치

```bash
pip install torch>=2.2.0
pip install transformers==4.57.3
pip install accelerate==1.12.0
pip install peft==0.13.0
pip install bitsandbytes==0.46.1
```

### 환경 변수 설정

```bash
# HuggingFace Token (Private 모델 접근용)
export HF_TOKEN="hf_your_token_here"
```

---

## 2. 모델 로드 방법

### 방법 1: 4-bit 양자화 (권장, 메모리 절약)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Base Model 로드 (Qwen3-14B)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

# LoRA 어댑터 로드 (Private 저장소)
model = PeftModel.from_pretrained(
    base_model,
    "Younggooo/kitrec-dualft-music-seta-model",  # 실제 저장소 이름으로 변경
    token="hf_your_token_here",  # Private 접근 시 필수
)

print("모델 로드 완료!")
```

### 방법 2: Full Precision (고성능, 메모리 많이 필요)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base Model 로드 (Full Precision)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    "Younggooo/kitrec-dualft-music-seta-model",
    token="hf_your_token_here",
)
```

---

## 3. 추론 (Inference)

### 단일 추천 생성

```python
def generate_recommendation(model, tokenizer, prompt: str) -> str:
    """추천 결과 생성"""
    
    # Chat 형식으로 변환
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 토큰화
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).to(model.device)
    
    # 생성
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 디코딩 (입력 부분 제외)
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    
    return result


# 사용 예시
prompt = """# Expert Cross-Domain Recommendation System
You are a specialized recommendation engine...
(프롬프트 내용)
"""

result = generate_recommendation(model, tokenizer, prompt)
print(result)
```

### 출력 파싱

```python
import json
import re

def parse_recommendation(output: str) -> dict:
    """모델 출력에서 추천 JSON 추출"""
    
    # JSON 블록 찾기
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, output, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        # trailing comma 제거
        json_str = re.sub(r",\s*}", "}", json_str)
        return json.loads(json_str)
    
    return None


# 사용 예시
result = generate_recommendation(model, tokenizer, prompt)
recommendation = parse_recommendation(result)

if recommendation:
    print(f"추천 아이템: {recommendation['item_id']}")
    print(f"순위: {recommendation['rank']}")
```

---

## 4. 배치 추론

```python
from tqdm import tqdm

def batch_inference(model, tokenizer, prompts: list, batch_size: int = 4) -> list:
    """배치 추론"""
    
    model.eval()
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        
        # Chat 형식 변환
        formatted = []
        for prompt in batch:
            messages = [{"role": "user", "content": prompt}]
            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        
        # 토큰화
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)
        
        # 생성
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 디코딩
        for j, output in enumerate(outputs):
            input_len = inputs["attention_mask"][j].sum().item()
            generated = output[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text)
    
    return results
```

---

## 5. 메모리 요구사항

| 로드 방식 | GPU 메모리 | 권장 GPU |
|----------|-----------|----------|
| 4-bit 양자화 | ~15GB | RTX 3090, A100 |
| Full Precision (bfloat16) | ~28GB | A100 80GB |

---

## 6. 모델 저장소 목록

학습 완료 후 업로드되는 모델들:

| 모델 | 저장소 이름 | 용도 |
|------|------------|------|
| DualFT-Movies Set A | `kitrec-dualft-movies-seta-model` | Movies 추천 (Hard Negatives) |
| DualFT-Movies Set B | `kitrec-dualft-movies-setb-model` | Movies 추천 (Random) |
| DualFT-Music Set A | `kitrec-dualft-music-seta-model` | Music 추천 (Hard Negatives) |
| DualFT-Music Set B | `kitrec-dualft-music-setb-model` | Music 추천 (Random) |
| SingleFT-Movies Set A | `kitrec-singleft-movies-seta-model` | Source-only Movies |
| SingleFT-Movies Set B | `kitrec-singleft-movies-setb-model` | Source-only Movies |
| SingleFT-Music Set A | `kitrec-singleft-music-seta-model` | Source-only Music |
| SingleFT-Music Set B | `kitrec-singleft-music-setb-model` | Source-only Music |

---

## 7. 문제 해결

### 인증 오류

```
OSError: Unauthorized access to repository
```

**해결**: HF_TOKEN 환경변수 확인 또는 `token` 파라미터 전달

```python
model = PeftModel.from_pretrained(
    base_model,
    "Younggooo/kitrec-model",
    token="hf_..."  # 토큰 명시
)
```

### 메모리 부족

```
OutOfMemoryError: CUDA out of memory
```

**해결**: 4-bit 양자화 사용 또는 `device_map="auto"` 확인

### Base Model 다운로드 느림

```bash
# 다운로드 진행 확인
watch -n 5 'du -sh ~/.cache/huggingface/hub/'
```

---

## 8. vLLM 기반 추론 (Production Inference)

KitREC 평가 환경은 vLLM 기반입니다 (Nvidia 5090, 36GB VRAM).

### 설치

```bash
pip install vllm>=0.4.0
```

### vLLM LoRA 로딩

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# vLLM 초기화 (LoRA 지원 활성화)
llm = LLM(
    model="Qwen/Qwen3-14B",
    enable_lora=True,
    max_lora_rank=64,  # LoRA rank (32) 이상 설정
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

# LoRA 어댑터 요청 생성
lora_request = LoRARequest(
    lora_name="kitrec-dualft-music",
    lora_int_id=1,
    lora_local_path="Younggooo/kitrec-dualft-music-seta-model"  # HuggingFace Hub 경로
)

# Sampling 설정 (Greedy decoding)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
)

# 추론 실행
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=lora_request
)

# 결과 추출
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
```

### 배치 추론 with Multiple LoRAs

```python
# 도메인별 다른 LoRA 어댑터
lora_movies = LoRARequest("movies", 1, "Younggooo/kitrec-dualft-movies-seta-model")
lora_music = LoRARequest("music", 2, "Younggooo/kitrec-dualft-music-seta-model")

# 배치 추론 (각 프롬프트에 맞는 LoRA 적용)
outputs = llm.generate(
    [prompt_movies, prompt_music],
    sampling_params,
    lora_request=[lora_movies, lora_music]
)
```

### vLLM vs Transformers 비교

| 항목 | vLLM | Transformers + PEFT |
|------|------|---------------------|
| **처리량** | 높음 (연속 배칭) | 중간 |
| **메모리 효율** | 높음 (PagedAttention) | 중간 |
| **LoRA 동적 로딩** | 지원 | 제한적 |
| **권장 용도** | 대규모 평가 (30K 샘플) | 개발/디버깅 |

---

## 참고

- [PEFT 공식 문서](https://huggingface.co/docs/peft)
- [Qwen3 모델 카드](https://huggingface.co/Qwen/Qwen3-14B)
- [BitsAndBytes 문서](https://github.com/TimDettmers/bitsandbytes)
- [vLLM 공식 문서](https://docs.vllm.ai/)
- [vLLM LoRA 가이드](https://docs.vllm.ai/en/latest/models/lora.html)

