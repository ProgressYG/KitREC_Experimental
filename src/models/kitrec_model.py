"""
KitREC 모델 로더

Fine-tuned KitREC 모델 (LoRA 어댑터) 로딩 및 관리
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class KitRECConfig:
    """KitREC 모델 설정"""
    base_model_name: str = "Qwen/Qwen3-14B"
    lora_path: str = ""
    model_type: str = "full"  # "full" or "direct"
    target_domain: str = "Movies"  # "Movies" or "Music"
    training_type: str = "DualFT"  # "DualFT" or "SingleFT"
    candidate_set: str = "A"  # "A" or "B"


class KitRECModel:
    """KitREC 모델 래퍼"""

    # 모델 경로 매핑
    MODEL_PATHS = {
        # DualFT Models
        "dualft_movies_seta": "Younggooo/kitrec-dualft-movies-seta-model",
        "dualft_movies_setb": "Younggooo/kitrec-dualft-movies-setb-model",
        "dualft_music_seta": "Younggooo/kitrec-dualft-music-seta-model",
        "dualft_music_setb": "Younggooo/kitrec-dualft-music-setb-model",

        # SingleFT Models
        "singleft_movies_seta": "Younggooo/kitrec-singleft-movies-seta-model",
        "singleft_movies_setb": "Younggooo/kitrec-singleft-movies-setb-model",
        "singleft_music_seta": "Younggooo/kitrec-singleft-music-seta-model",
        "singleft_music_setb": "Younggooo/kitrec-singleft-music-setb-model",

        # Direct Models (Option A: 별도 학습)
        "direct_dualft_movies_seta": "Younggooo/kitrec-direct-dualft-movies-seta-model",
        "direct_dualft_movies_setb": "Younggooo/kitrec-direct-dualft-movies-setb-model",
        "direct_dualft_music_seta": "Younggooo/kitrec-direct-dualft-music-seta-model",
        "direct_dualft_music_setb": "Younggooo/kitrec-direct-dualft-music-setb-model",
    }

    def __init__(
        self,
        config: KitRECConfig,
        hf_token: Optional[str] = None,
        use_vllm: bool = True
    ):
        """
        Args:
            config: 모델 설정
            hf_token: HuggingFace token (Private 모델용)
            use_vllm: vLLM 사용 여부 (False면 Transformers 사용)
        """
        self.config = config
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.use_vllm = use_vllm

        self._inference_engine = None

    @classmethod
    def load(
        cls,
        model_name: str,
        hf_token: Optional[str] = None,
        use_vllm: bool = True
    ) -> "KitRECModel":
        """
        모델 이름으로 로드

        Args:
            model_name: 모델 이름 (e.g., "dualft_music_seta")
            hf_token: HuggingFace token
            use_vllm: vLLM 사용 여부

        Returns:
            KitRECModel 인스턴스
        """
        if model_name not in cls.MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODEL_PATHS.keys())}")

        lora_path = cls.MODEL_PATHS[model_name]

        # 모델 타입 파싱
        is_direct = model_name.startswith("direct_")
        parts = model_name.replace("direct_", "").split("_")

        training_type = parts[0].upper()  # "DualFT" or "SingleFT"
        target_domain = parts[1].capitalize()  # "Movies" or "Music"
        candidate_set = parts[2].replace("set", "").upper()  # "A" or "B"

        config = KitRECConfig(
            lora_path=lora_path,
            model_type="direct" if is_direct else "full",
            target_domain=target_domain,
            training_type=training_type,
            candidate_set=candidate_set
        )

        return cls(config, hf_token, use_vllm)

    def initialize(self):
        """추론 엔진 초기화"""
        if self.use_vllm:
            from ..inference.vllm_inference import VLLMInference

            self._inference_engine = VLLMInference(
                model_name=self.config.base_model_name,
                lora_path=self.config.lora_path,
                hf_token=self.hf_token,
                enable_lora=True,
                max_lora_rank=64,
            )
        else:
            from ..inference.vllm_inference import TransformersInference

            self._inference_engine = TransformersInference(
                model_name=self.config.base_model_name,
                lora_path=self.config.lora_path,
                hf_token=self.hf_token,
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

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.config.lora_path,
            "base_model": self.config.base_model_name,
            "model_type": self.config.model_type,
            "target_domain": self.config.target_domain,
            "training_type": self.config.training_type,
            "candidate_set": self.config.candidate_set,
            "use_vllm": self.use_vllm,
        }

    @classmethod
    def list_available_models(cls) -> list:
        """사용 가능한 모델 목록 반환"""
        return list(cls.MODEL_PATHS.keys())


def load_kitrec_model(model_name: str, hf_token: str = None) -> KitRECModel:
    """
    KitREC 모델 로드 헬퍼 함수

    Args:
        model_name: 모델 이름
        hf_token: HuggingFace token

    Returns:
        초기화된 KitRECModel
    """
    model = KitRECModel.load(model_name, hf_token)
    model.initialize()
    return model
