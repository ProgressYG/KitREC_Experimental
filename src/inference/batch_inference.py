"""
배치 추론 관리

대규모 평가 (30K 샘플)를 위한 배치 처리 및 체크포인트 관리
"""

import os
import json
import time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
from datetime import datetime

from .vllm_inference import VLLMInference, GenerationConfig
from .output_parser import OutputParser, ParseResult, ErrorStatistics


@dataclass
class BatchResult:
    """배치 결과 구조체"""
    sample_id: str
    prompt: str
    raw_output: str
    parse_result: ParseResult
    ground_truth: Dict
    metadata: Dict
    inference_time: float


class BatchInference:
    """배치 추론 관리자"""

    def __init__(
        self,
        inference_engine: VLLMInference,
        output_parser: OutputParser,
        batch_size: int = 8,
        checkpoint_interval: int = 100,
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Args:
            inference_engine: 추론 엔진 (VLLMInference)
            output_parser: 출력 파서
            batch_size: 배치 크기
            checkpoint_interval: 체크포인트 저장 간격 (샘플 수)
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.inference_engine = inference_engine
        self.output_parser = output_parser
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir

        self.error_stats = ErrorStatistics()
        self._results = []
        self._processed_ids = set()

    def run(
        self,
        samples: List[Dict],
        extract_prompt_fn: Callable[[Dict], str],
        extract_candidates_fn: Callable[[Dict], List[str]],
        extract_gt_fn: Callable[[Dict], Dict],
        extract_metadata_fn: Callable[[Dict], Dict],
        config: Optional[GenerationConfig] = None,
        resume_from_checkpoint: bool = True,
    ) -> List[BatchResult]:
        """
        배치 추론 실행

        Args:
            samples: 테스트 샘플 리스트
            extract_prompt_fn: 프롬프트 추출 함수
            extract_candidates_fn: candidate_ids 추출 함수
            extract_gt_fn: ground_truth 추출 함수
            extract_metadata_fn: metadata 추출 함수
            config: 생성 설정
            resume_from_checkpoint: 체크포인트에서 재개 여부

        Returns:
            BatchResult 리스트
        """
        # 체크포인트 로드 (있는 경우)
        if resume_from_checkpoint:
            self._load_checkpoint()

        # 처리되지 않은 샘플만 필터링
        remaining_samples = [
            s for s in samples
            if s.get("user_id", str(id(s))) not in self._processed_ids
        ]

        print(f"Total samples: {len(samples)}")
        print(f"Already processed: {len(self._processed_ids)}")
        print(f"Remaining: {len(remaining_samples)}")

        config = config or GenerationConfig()

        # 배치 단위 처리
        for i in tqdm(range(0, len(remaining_samples), self.batch_size),
                     desc="Processing batches"):
            batch = remaining_samples[i:i + self.batch_size]

            # 프롬프트 추출
            prompts = [extract_prompt_fn(s) for s in batch]
            candidate_ids_list = [extract_candidates_fn(s) for s in batch]

            # 배치 추론
            start_time = time.time()
            outputs = self.inference_engine.generate_batch(prompts, config)
            batch_time = time.time() - start_time
            per_sample_time = batch_time / len(batch)

            # 결과 파싱 및 저장
            for j, (sample, output) in enumerate(zip(batch, outputs)):
                sample_id = sample.get("user_id", str(id(sample)))
                candidate_ids = candidate_ids_list[j]

                # 파싱
                parse_result = self.output_parser.parse(output, candidate_ids)
                self.error_stats.update(parse_result)

                # 결과 저장
                result = BatchResult(
                    sample_id=sample_id,
                    prompt=prompts[j],
                    raw_output=output,
                    parse_result=parse_result,
                    ground_truth=extract_gt_fn(sample),
                    metadata=extract_metadata_fn(sample),
                    inference_time=per_sample_time
                )
                self._results.append(result)
                self._processed_ids.add(sample_id)

            # 체크포인트 저장
            if len(self._results) % self.checkpoint_interval == 0:
                self._save_checkpoint()

        # 최종 체크포인트 저장
        self._save_checkpoint()

        return self._results

    def _save_checkpoint(self):
        """체크포인트 저장"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_count": len(self._results),
            "processed_ids": list(self._processed_ids),
            "error_stats": self.error_stats.get_summary(),
        }

        # 메타데이터 저장
        meta_path = os.path.join(self.checkpoint_dir, "checkpoint_meta.json")
        with open(meta_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        # 결과 저장 (JSONL)
        results_path = os.path.join(self.checkpoint_dir, "results.jsonl")
        with open(results_path, "w") as f:
            for result in self._results:
                result_dict = {
                    "sample_id": result.sample_id,
                    "raw_output": result.raw_output,
                    "predictions": result.parse_result.predictions,
                    "thinking": result.parse_result.thinking,
                    "parse_errors": result.parse_result.parse_errors,
                    "ground_truth": result.ground_truth,
                    "metadata": result.metadata,
                    "inference_time": result.inference_time,
                }
                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

        print(f"Checkpoint saved: {len(self._results)} samples")

    def _load_checkpoint(self):
        """체크포인트 로드"""
        meta_path = os.path.join(self.checkpoint_dir, "checkpoint_meta.json")
        results_path = os.path.join(self.checkpoint_dir, "results.jsonl")

        if not os.path.exists(meta_path) or not os.path.exists(results_path):
            print("No checkpoint found, starting fresh")
            return

        # 메타데이터 로드
        with open(meta_path, "r") as f:
            checkpoint_data = json.load(f)

        self._processed_ids = set(checkpoint_data.get("processed_ids", []))

        # 결과 로드
        with open(results_path, "r") as f:
            for line in f:
                result_dict = json.loads(line)
                # BatchResult 재구성 (간략화)
                parse_result = ParseResult(
                    thinking=result_dict.get("thinking"),
                    predictions=result_dict.get("predictions", []),
                    parse_errors=result_dict.get("parse_errors", []),
                    raw_output=result_dict.get("raw_output", ""),
                )
                result = BatchResult(
                    sample_id=result_dict["sample_id"],
                    prompt="",  # 체크포인트에서 복원하지 않음
                    raw_output=result_dict.get("raw_output", ""),
                    parse_result=parse_result,
                    ground_truth=result_dict.get("ground_truth", {}),
                    metadata=result_dict.get("metadata", {}),
                    inference_time=result_dict.get("inference_time", 0.0),
                )
                self._results.append(result)

        print(f"Checkpoint loaded: {len(self._results)} samples")

    def get_error_statistics(self) -> Dict:
        """에러 통계 반환"""
        return self.error_stats.get_summary()

    def get_timing_statistics(self) -> Dict:
        """시간 통계 반환"""
        if not self._results:
            return {}

        times = [r.inference_time for r in self._results]
        return {
            "total_samples": len(times),
            "total_time_seconds": sum(times),
            "avg_time_per_sample": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }

    def save_final_results(self, output_dir: str):
        """최종 결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # Predictions 저장
        predictions_path = os.path.join(output_dir, "predictions.jsonl")
        with open(predictions_path, "w") as f:
            for result in self._results:
                f.write(json.dumps({
                    "sample_id": result.sample_id,
                    "predictions": result.parse_result.predictions,
                    "ground_truth": result.ground_truth,
                    "metadata": result.metadata,
                }, ensure_ascii=False) + "\n")

        # Error statistics 저장
        error_path = os.path.join(output_dir, "error_statistics.json")
        with open(error_path, "w") as f:
            json.dump(self.error_stats.get_summary(), f, indent=2)

        # Timing statistics 저장
        timing_path = os.path.join(output_dir, "timing_statistics.json")
        with open(timing_path, "w") as f:
            json.dump(self.get_timing_statistics(), f, indent=2)

        print(f"Results saved to: {output_dir}")

    def reset(self):
        """상태 초기화"""
        self._results = []
        self._processed_ids = set()
        self.error_stats.reset()
