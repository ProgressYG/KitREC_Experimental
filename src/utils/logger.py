"""
로깅 설정

평가 실행 로그 관리
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "kitrec",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    로거 설정

    Args:
        name: 로거 이름
        log_dir: 로그 파일 디렉토리
        level: 로그 레벨
        console_output: 콘솔 출력 여부
        file_output: 파일 출력 여부

    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 기존 핸들러 제거
    logger.handlers = []

    # 포맷 설정
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 핸들러
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 파일 핸들러
    if file_output:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{timestamp}_eval.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "kitrec") -> logging.Logger:
    """
    기존 로거 가져오기

    Args:
        name: 로거 이름

    Returns:
        로거 (없으면 기본 설정으로 생성)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # 기본 설정 적용
        return setup_logger(name)

    return logger


class EvaluationLogger:
    """평가 전용 로거"""

    def __init__(self, name: str = "kitrec.eval", log_dir: str = "logs"):
        self.logger = setup_logger(name, log_dir)
        self.metrics_log = []

    def log_start(self, model_name: str, dataset_name: str, total_samples: int):
        """평가 시작 로그"""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting evaluation: {model_name}")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info("=" * 60)

    def log_progress(self, current: int, total: int, metrics: Optional[dict] = None):
        """진행 상황 로그"""
        progress = (current / total) * 100
        msg = f"Progress: {current}/{total} ({progress:.1f}%)"

        if metrics:
            msg += f" | Current Hit@10: {metrics.get('hit@10', 0):.4f}"

        self.logger.info(msg)

    def log_batch_complete(self, batch_idx: int, batch_size: int, avg_time: float):
        """배치 완료 로그"""
        self.logger.debug(
            f"Batch {batch_idx} complete | Size: {batch_size} | Avg time: {avg_time:.3f}s"
        )

    def log_error(self, sample_id: str, error_type: str, error_msg: str):
        """에러 로그"""
        self.logger.warning(f"Error [{error_type}] Sample {sample_id}: {error_msg}")

    def log_metrics(self, metrics: dict):
        """메트릭 로그"""
        self.metrics_log.append(metrics)
        self.logger.info("Metrics Summary:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def log_finish(self, total_time: float, error_rate: float):
        """평가 완료 로그"""
        self.logger.info("=" * 60)
        self.logger.info("Evaluation Complete")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Error rate: {error_rate:.2%}")
        self.logger.info("=" * 60)

    def log_comparison(self, model_a: str, model_b: str, diff: dict):
        """모델 비교 로그"""
        self.logger.info(f"Comparison: {model_a} vs {model_b}")
        for metric, value in diff.items():
            sign = "+" if value > 0 else ""
            self.logger.info(f"  {metric}: {sign}{value:.4f}")
