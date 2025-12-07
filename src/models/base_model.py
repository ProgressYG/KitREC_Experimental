"""
Base Model 로더

Untuned Qwen3-14B 모델 (Zero-shot / Base-CoT / Base-Direct)
RQ1 Ablation Study용
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class BaseModelConfig:
    """Base Model 설정"""
    model_name: str = "Qwen/Qwen3-14B"
    use_thinking: bool = True  # True: Base-CoT, False: Base-Direct
    use_quantization: bool = True
    quantization_bits: int = 4


class BaseModel:
    """
    Base Model 래퍼 (Untuned)

    RQ1 Ablation Study:
    - Base-CoT: Thinking 포함 프롬프트 + Untuned
    - Base-Direct: Direct 프롬프트 + Untuned
    """

    def __init__(
        self,
        config: Optional[BaseModelConfig] = None,
        hf_token: Optional[str] = None,
        use_vllm: bool = True
    ):
        """
        Args:
            config: 모델 설정
            hf_token: HuggingFace token
            use_vllm: vLLM 사용 여부
        """
        self.config = config or BaseModelConfig()
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.use_vllm = use_vllm

        self._inference_engine = None

    @classmethod
    def load_cot(cls, hf_token: str = None, use_vllm: bool = True) -> "BaseModel":
        """
        Base-CoT 모델 로드 (Thinking 포함)

        Args:
            hf_token: HuggingFace token
            use_vllm: vLLM 사용 여부

        Returns:
            BaseModel 인스턴스
        """
        config = BaseModelConfig(use_thinking=True)
        return cls(config, hf_token, use_vllm)

    @classmethod
    def load_direct(cls, hf_token: str = None, use_vllm: bool = True) -> "BaseModel":
        """
        Base-Direct 모델 로드 (Thinking 없음)

        Args:
            hf_token: HuggingFace token
            use_vllm: vLLM 사용 여부

        Returns:
            BaseModel 인스턴스
        """
        config = BaseModelConfig(use_thinking=False)
        return cls(config, hf_token, use_vllm)

    def initialize(self):
        """추론 엔진 초기화"""
        if self.use_vllm:
            from ..inference.vllm_inference import VLLMInference

            self._inference_engine = VLLMInference(
                model_name=self.config.model_name,
                lora_path=None,  # No LoRA for base model
                hf_token=self.hf_token,
                enable_lora=False,
            )
        else:
            from ..inference.vllm_inference import TransformersInference

            self._inference_engine = TransformersInference(
                model_name=self.config.model_name,
                lora_path=None,
                hf_token=self.hf_token,
                use_4bit=self.config.use_quantization,
            )

        self._inference_engine.initialize()

    def generate(self, prompt: str, **kwargs) -> str:
        """
        추천 생성

        Args:
            prompt: 입력 프롬프트
            **kwargs: 추가 생성 파라미터

        Returns:
            모델 출력
        """
        if self._inference_engine is None:
            self.initialize()

        return self._inference_engine.generate(prompt, **kwargs)

    def generate_batch(self, prompts: list, **kwargs) -> list:
        """배치 생성"""
        if self._inference_engine is None:
            self.initialize()

        return self._inference_engine.generate_batch(prompts, **kwargs)

    def get_model_type(self) -> str:
        """모델 타입 반환"""
        return "base_cot" if self.config.use_thinking else "base_direct"

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.config.model_name,
            "model_type": self.get_model_type(),
            "use_thinking": self.config.use_thinking,
            "use_quantization": self.config.use_quantization,
            "use_vllm": self.use_vllm,
        }


def load_base_model(
    model_type: str = "cot",
    hf_token: str = None,
    use_vllm: bool = True
) -> BaseModel:
    """
    Base Model 로드 헬퍼 함수

    Args:
        model_type: "cot" or "direct"
        hf_token: HuggingFace token
        use_vllm: vLLM 사용 여부

    Returns:
        초기화된 BaseModel
    """
    if model_type == "cot":
        model = BaseModel.load_cot(hf_token, use_vllm)
    elif model_type == "direct":
        model = BaseModel.load_direct(hf_token, use_vllm)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'cot' or 'direct'")

    model.initialize()
    return model
