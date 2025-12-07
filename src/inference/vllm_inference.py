"""
vLLM 기반 LLM 추론 엔진

Model_Usage_Guide.md Section 8: vLLM 기반 추론
- KitREC 평가 환경: vLLM 기반 (Nvidia 5090, 36GB VRAM)
- LoRA 동적 로딩 지원
"""

import os
import tempfile
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


# Global random seed for reproducibility
RANDOM_SEED = 42


@dataclass
class GenerationConfig:
    """생성 설정"""
    max_new_tokens: int = 2048
    temperature: float = 0.0  # Greedy decoding
    top_p: float = 1.0
    top_k: int = -1
    enable_thinking: bool = True  # Qwen3 thinking mode


def _download_lora_to_local(lora_path: str, hf_token: Optional[str] = None) -> str:
    """
    HuggingFace Hub에서 LoRA 어댑터를 로컬로 다운로드

    vLLM의 LoRARequest는 로컬 경로만 지원하므로,
    HuggingFace Hub 경로인 경우 먼저 다운로드 필요

    Args:
        lora_path: HuggingFace Hub 경로 (e.g., "Younggooo/kitrec-model")
                   또는 로컬 경로
        hf_token: HuggingFace 토큰

    Returns:
        로컬 파일 경로
    """
    # 이미 로컬 경로인 경우 그대로 반환
    if os.path.exists(lora_path):
        return lora_path

    # HuggingFace Hub에서 다운로드
    try:
        from huggingface_hub import snapshot_download

        # 캐시 디렉토리 설정 (환경 변수 또는 기본값)
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        lora_cache_dir = os.path.join(cache_dir, "lora_adapters")
        os.makedirs(lora_cache_dir, exist_ok=True)

        # repo_id에서 안전한 디렉토리 이름 생성
        safe_name = lora_path.replace("/", "_")
        local_dir = os.path.join(lora_cache_dir, safe_name)

        # 이미 다운로드된 경우 스킵
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"  Using cached LoRA adapter: {local_dir}")
            return local_dir

        print(f"  Downloading LoRA adapter from {lora_path}...")
        local_dir = snapshot_download(
            repo_id=lora_path,
            local_dir=local_dir,
            token=hf_token,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"]
        )
        print(f"  Downloaded to: {local_dir}")
        return local_dir

    except Exception as e:
        raise RuntimeError(
            f"Failed to download LoRA adapter from {lora_path}: {e}\n"
            f"If this is a local path, ensure it exists."
        )


class VLLMInference:
    """vLLM 기반 추론 엔진"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        lora_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        enable_lora: bool = True,
        max_lora_rank: int = 64,
        gpu_memory_utilization: float = 0.85,  # 0.9 -> 0.85 for OOM prevention
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,  # vLLM optimization
    ):
        """
        vLLM 추론 엔진 초기화

        Args:
            model_name: Base model name (e.g., "Qwen/Qwen3-14B")
            lora_path: LoRA adapter path (HuggingFace Hub or local)
            hf_token: HuggingFace token for private models
            enable_lora: Enable LoRA loading
            max_lora_rank: Maximum LoRA rank (should be >= actual rank, e.g., 64 for r=32)
            gpu_memory_utilization: GPU memory utilization ratio (default 0.85 for OOM prevention)
            max_model_len: Maximum model context length
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enable_prefix_caching: Enable prefix caching for 20-30% speedup
        """
        self.model_name = model_name
        self.lora_path = lora_path
        self.hf_token = hf_token
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_prefix_caching = enable_prefix_caching

        self._llm = None
        self._lora_request = None
        self._tokenizer = None
        self._lora_id_counter = 1  # For unique LoRA IDs

    def initialize(self):
        """vLLM 엔진 초기화"""
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm>=0.4.0"
            )

        # LLM 초기화
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "trust_remote_code": True,
            "dtype": "bfloat16",  # Qwen3 optimized dtype
        }

        # Prefix caching for speedup (vLLM 0.4.0+)
        if self.enable_prefix_caching:
            llm_kwargs["enable_prefix_caching"] = True

        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = self.max_lora_rank

        self._llm = LLM(**llm_kwargs)

        # LoRA 어댑터 설정 (있는 경우)
        if self.lora_path and self.enable_lora:
            # HuggingFace Hub 경로인 경우 로컬로 다운로드
            local_lora_path = _download_lora_to_local(self.lora_path, self.hf_token)

            self._lora_request = LoRARequest(
                lora_name="kitrec-lora",
                lora_int_id=self._lora_id_counter,
                lora_local_path=local_lora_path
            )
            self._lora_id_counter += 1

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        단일 프롬프트 생성

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Generated text
        """
        if self._llm is None:
            self.initialize()

        config = config or GenerationConfig()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            max_tokens=config.max_new_tokens,
        )

        outputs = self._llm.generate(
            [prompt],
            sampling_params,
            lora_request=self._lora_request
        )

        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        배치 프롬프트 생성

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of generated texts
        """
        if self._llm is None:
            self.initialize()

        config = config or GenerationConfig()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            max_tokens=config.max_new_tokens,
        )

        outputs = self._llm.generate(
            prompts,
            sampling_params,
            lora_request=self._lora_request
        )

        return [output.outputs[0].text for output in outputs]

    def generate_with_multiple_loras(
        self,
        prompts: List[str],
        lora_requests: List[Any],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        다중 LoRA로 배치 생성 (각 프롬프트에 다른 LoRA 적용)

        Model_Usage_Guide.md: "배치 추론 with Multiple LoRAs"
        """
        if self._llm is None:
            self.initialize()

        config = config or GenerationConfig()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )

        outputs = self._llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_requests
        )

        return [output.outputs[0].text for output in outputs]

    def switch_lora(self, lora_path: str, lora_name: str = "kitrec-lora"):
        """
        LoRA 어댑터 전환 (메모리 안전)

        Args:
            lora_path: HuggingFace Hub 경로 또는 로컬 경로
            lora_name: LoRA 이름 (식별용)
        """
        import torch
        from vllm.lora.request import LoRARequest

        # 기존 LoRA 해제 및 GPU 캐시 정리 (메모리 누수 방지)
        self._lora_request = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # HuggingFace Hub 경로인 경우 로컬로 다운로드
        local_lora_path = _download_lora_to_local(lora_path, self.hf_token)

        self._lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=self._lora_id_counter,
            lora_local_path=local_lora_path
        )
        self._lora_id_counter += 1

    def disable_lora(self):
        """LoRA 비활성화 (Base model only) with memory cleanup"""
        import torch

        self._lora_request = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_tokenizer(self):
        """Tokenizer 반환"""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer


class TransformersInference:
    """
    Transformers + PEFT 기반 추론 (개발/디버깅용)

    Model_Usage_Guide.md: 개발/디버깅에 적합
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        lora_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_4bit: bool = True,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.lora_path = lora_path
        self.hf_token = hf_token
        self.use_4bit = use_4bit
        self.device_map = device_map

        self._model = None
        self._tokenizer = None

    def initialize(self):
        """모델 초기화"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # 4-bit 양자화 설정
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Base Model 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device_map,
            trust_remote_code=True,
            token=self.hf_token,
        )

        # Tokenizer 로드
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        # LoRA 어댑터 로드 (있는 경우)
        if self.lora_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                token=self.hf_token,
            )
        else:
            self._model = base_model

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """단일 프롬프트 생성"""
        import torch

        if self._model is None:
            self.initialize()

        config = config or GenerationConfig()

        # Chat 형식 변환
        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 토큰화
        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self._model.device)

        # 생성
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=config.temperature if config.temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # 디코딩 (입력 부분 제외)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        result = self._tokenizer.decode(generated, skip_special_tokens=True)

        return result

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        배치 프롬프트 생성

        Args:
            prompts: 프롬프트 리스트
            config: 생성 설정

        Returns:
            생성된 텍스트 리스트
        """
        import torch

        if self._model is None:
            self.initialize()

        config = config or GenerationConfig()
        results = []

        # TransformersInference는 배치 처리가 비효율적이므로 순차 처리
        # (vLLM을 권장)
        for prompt in prompts:
            try:
                result = self.generate(prompt, config)
                results.append(result)
            except Exception as e:
                print(f"    Warning: Generation failed for prompt: {e}")
                results.append("")

            # 메모리 관리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results
