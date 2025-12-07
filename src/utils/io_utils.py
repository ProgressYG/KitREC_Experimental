"""
파일 I/O 유틸리티

결과 저장 및 로딩
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    JSON 파일 저장

    Args:
        data: 저장할 데이터
        file_path: 파일 경로
        indent: 들여쓰기 (기본 2)
    """
    # 경로 검증 및 디렉토리 생성
    dir_path = os.path.dirname(file_path)
    if dir_path:  # 빈 문자열이 아닌 경우에만 디렉토리 생성
        os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """
    JSON 파일 로딩

    Args:
        file_path: 파일 경로

    Returns:
        로드된 데이터
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict], file_path: str):
    """
    JSONL 파일 저장 (한 줄에 하나의 JSON)

    Args:
        data: 저장할 데이터 리스트
        file_path: 파일 경로
    """
    # 경로 검증 및 디렉토리 생성
    dir_path = os.path.dirname(file_path)
    if dir_path:  # 빈 문자열이 아닌 경우에만 디렉토리 생성
        os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(file_path: str) -> List[Dict]:
    """
    JSONL 파일 로딩

    Args:
        file_path: 파일 경로

    Returns:
        데이터 리스트
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_results(
    results: List[Dict],
    output_dir: str,
    model_name: str,
    include_raw_output: bool = False
):
    """
    평가 결과 저장

    저장 파일:
    - predictions.jsonl: 예측 결과
    - metrics_summary.json: 집계 메트릭
    - metadata.json: 실행 메타데이터

    Args:
        results: 결과 리스트
        output_dir: 출력 디렉토리
        model_name: 모델 이름
        include_raw_output: 원본 출력 포함 여부
    """
    os.makedirs(output_dir, exist_ok=True)

    # Predictions 저장
    predictions = []
    for r in results:
        pred = {
            "sample_id": r.get("sample_id"),
            "predictions": r.get("predictions", []),
            "ground_truth": r.get("ground_truth"),
            "metrics": r.get("metrics"),
            "metadata": r.get("metadata"),
        }
        if include_raw_output:
            pred["raw_output"] = r.get("raw_output", "")
        predictions.append(pred)

    save_jsonl(predictions, os.path.join(output_dir, "predictions.jsonl"))

    # Metrics Summary 저장
    if results and "metrics" in results[0]:
        from ..metrics.ranking_metrics import RankingMetrics

        metrics_list = [r["metrics"] for r in results]
        summary = RankingMetrics.aggregate_metrics(metrics_list)
        summary["total_samples"] = len(results)

        save_json(summary, os.path.join(output_dir, "metrics_summary.json"))

    # Metadata 저장
    metadata = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "output_dir": output_dir,
    }
    save_json(metadata, os.path.join(output_dir, "metadata.json"))


def load_results(output_dir: str) -> Dict[str, Any]:
    """
    저장된 결과 로딩

    Args:
        output_dir: 결과 디렉토리

    Returns:
        {
            "predictions": List[Dict],
            "metrics_summary": Dict,
            "metadata": Dict
        }
    """
    result = {}

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    if os.path.exists(predictions_path):
        result["predictions"] = load_jsonl(predictions_path)

    summary_path = os.path.join(output_dir, "metrics_summary.json")
    if os.path.exists(summary_path):
        result["metrics_summary"] = load_json(summary_path)

    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        result["metadata"] = load_json(metadata_path)

    return result


def save_comparison_report(
    comparison: Dict[str, Dict],
    output_path: str,
    format: str = "markdown"
):
    """
    비교 리포트 저장

    Args:
        comparison: 비교 결과
        output_path: 출력 경로
        format: 출력 형식 (markdown, json)
    """
    if format == "json":
        save_json(comparison, output_path)
    elif format == "markdown":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            for model_name, metrics in comparison.items():
                f.write(f"## {model_name}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"| {metric} | {value:.4f} |\n")
                    else:
                        f.write(f"| {metric} | {value} |\n")

                f.write("\n")


def create_experiment_dir(
    base_dir: str = "results",
    experiment_name: str = None
) -> str:
    """
    실험 디렉토리 생성

    Args:
        base_dir: 기본 디렉토리
        experiment_name: 실험 이름 (없으면 타임스탬프)

    Returns:
        생성된 디렉토리 경로
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    return experiment_dir


def merge_results(
    result_dirs: List[str]
) -> List[Dict]:
    """
    여러 결과 디렉토리 병합

    Args:
        result_dirs: 결과 디렉토리 리스트

    Returns:
        병합된 예측 리스트
    """
    all_predictions = []

    for result_dir in result_dirs:
        predictions_path = os.path.join(result_dir, "predictions.jsonl")
        if os.path.exists(predictions_path):
            all_predictions.extend(load_jsonl(predictions_path))

    return all_predictions
