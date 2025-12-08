"""
OpenAI API based Inference Engine

GPT-4.1-mini 등 OpenAI 모델을 사용한 추론
LLM baseline (LLM4CDR, Vanilla) 실험용
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenAIGenerationConfig:
    """OpenAI 생성 설정"""
    max_tokens: int = 2048
    temperature: float = 0.0  # Greedy decoding
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OpenAIInference:
    """
    OpenAI API 기반 추론 엔진

    LLM4CDR, Vanilla baseline 실험에서 GPT-4.1-mini 사용 시 활용
    """

    # Supported models
    SUPPORTED_MODELS = {
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    }

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            model_name: OpenAI 모델 이름
            api_key: OpenAI API 키 (없으면 환경변수에서 로드)
            max_retries: 재시도 횟수
            retry_delay: 재시도 간 대기 시간 (초)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = None
        self._initialized = False

        # Statistics
        self._total_tokens = 0
        self._total_requests = 0
        self._failed_requests = 0

    def initialize(self):
        """OpenAI 클라이언트 초기화"""
        if self._initialized:
            return

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"OpenAI client initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

    def generate(
        self,
        prompt: str,
        config: Optional[OpenAIGenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        단일 프롬프트 생성

        Args:
            prompt: 입력 프롬프트
            config: 생성 설정
            system_prompt: 시스템 프롬프트 (선택)

        Returns:
            생성된 텍스트
        """
        if not self._initialized:
            self.initialize()

        config = config or OpenAIGenerationConfig()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                )

                self._total_requests += 1
                if response.usage:
                    self._total_tokens += response.usage.total_tokens

                return response.choices[0].message.content

            except Exception as e:
                self._failed_requests += 1
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"OpenAI API failed after {self.max_retries} attempts")
                    return ""

        return ""

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[OpenAIGenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        배치 프롬프트 생성 (순차 처리)

        Note: OpenAI API는 batch endpoint가 있지만,
              실시간 응답을 위해 순차 처리
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, config, system_prompt)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """사용 통계 반환"""
        return {
            "model": self.model_name,
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "total_tokens": self._total_tokens,
            "success_rate": (
                (self._total_requests - self._failed_requests) / self._total_requests
                if self._total_requests > 0 else 0
            ),
        }

    def is_available(self) -> bool:
        """API 사용 가능 여부 확인"""
        return bool(self.api_key)


def create_inference_engine(
    backend: str = "qwen",
    hf_token: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """
    추론 엔진 팩토리 함수

    Args:
        backend: "qwen" (vLLM) or "gpt4-mini" (OpenAI)
        hf_token: HuggingFace 토큰 (qwen용)
        openai_api_key: OpenAI API 키 (gpt4-mini용)
        model_name: 모델 이름 오버라이드

    Returns:
        추론 엔진 인스턴스
    """
    if backend == "qwen":
        from .vllm_inference import VLLMInference
        return VLLMInference(
            model_name=model_name or "Qwen/Qwen3-14B",
            hf_token=hf_token,
            enable_lora=False,
        )
    elif backend in ["gpt4-mini", "gpt-4.1-mini", "openai"]:
        return OpenAIInference(
            model_name=model_name or "gpt-4.1-mini",
            api_key=openai_api_key,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'qwen' or 'gpt4-mini'")
