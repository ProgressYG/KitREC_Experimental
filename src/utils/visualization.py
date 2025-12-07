"""
결과 시각화

평가 결과 그래프 및 테이블 생성
- LaTeX/Markdown 테이블
- Matplotlib 그래프 (신뢰구간 포함)
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import os


def _check_matplotlib():
    """Matplotlib 가용성 확인"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_confidence_intervals(
    results: Dict[str, Dict],
    metric: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[Any]:
    """
    신뢰구간이 포함된 bar plot 생성

    Args:
        results: {model_name: {metric: {"mean": float, "ci_lower": float, "ci_upper": float}}}
        metric: 표시할 메트릭 이름
        save_path: 저장 경로 (None이면 표시만)
        title: 그래프 제목
        figsize: 그래프 크기

    Returns:
        Figure 객체 (matplotlib 없으면 None)
    """
    plt = _check_matplotlib()
    if plt is None:
        print("Warning: matplotlib not available for plotting")
        return None

    import numpy as np

    models = list(results.keys())

    # 데이터 추출
    means = []
    ci_lowers = []
    ci_uppers = []

    for model in models:
        model_data = results[model].get(metric, {})
        if isinstance(model_data, dict):
            means.append(model_data.get("mean", model_data.get(metric, 0)))
            ci_lowers.append(model_data.get("ci_lower", means[-1]))
            ci_uppers.append(model_data.get("ci_upper", means[-1]))
        else:
            means.append(model_data)
            ci_lowers.append(model_data)
            ci_uppers.append(model_data)

    means = np.array(means)
    errors = [
        means - np.array(ci_lowers),
        np.array(ci_uppers) - means
    ]

    # 플롯 생성
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=errors, capsize=5, color='steelblue', alpha=0.7)

    ax.set_ylabel(metric.upper())
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_title(title or f'{metric.upper()} with 95% Confidence Intervals')

    # 값 표시
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_cold_start_analysis(
    core_analysis: Dict[str, Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Cold-Start Performance Analysis",
    figsize: Tuple[int, int] = (12, 6)
) -> Optional[Any]:
    """
    Cold-start 분석 라인 플롯

    Args:
        core_analysis: Core level별 분석 결과
        metrics: 표시할 메트릭 (기본: hit@10, ndcg@10)
        save_path: 저장 경로
        title: 그래프 제목
        figsize: 그래프 크기

    Returns:
        Figure 객체
    """
    plt = _check_matplotlib()
    if plt is None:
        print("Warning: matplotlib not available for plotting")
        return None

    import numpy as np

    if metrics is None:
        metrics = ["hit@10", "ndcg@10"]

    # Core level 순서
    core_order = ["1-core", "2-core", "3-core", "4-core", "5-core", "6-9 core", "10+-core", "5+-core"]
    categories = [c for c in core_order if c in core_analysis]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(categories))
    width = 0.35
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    for i, metric in enumerate(metrics):
        values = [core_analysis[c].get(metric, 0) for c in categories]
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.upper(), color=colors[i % len(colors)], alpha=0.8)

        # 값 표시
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score')
    ax.set_xlabel('Core Level')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_model_comparison(
    comparison_results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6)
) -> Optional[Any]:
    """
    모델 비교 그룹 바 차트

    Args:
        comparison_results: {model_name: {metric: value}}
        metrics: 표시할 메트릭
        save_path: 저장 경로
        title: 그래프 제목
        figsize: 그래프 크기

    Returns:
        Figure 객체
    """
    plt = _check_matplotlib()
    if plt is None:
        return None

    import numpy as np

    if metrics is None:
        metrics = ["hit@10", "ndcg@10", "mrr"]

    models = list(comparison_results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        values = [comparison_results[model].get(m, 0) for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_ablation_heatmap(
    ablation_results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "2x2 Ablation Study Results",
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[Any]:
    """
    Ablation Study 히트맵 (2x2)

    Args:
        ablation_results: {model_name: {metric: value}}
        metrics: 표시할 메트릭
        save_path: 저장 경로
        title: 그래프 제목
        figsize: 그래프 크기

    Returns:
        Figure 객체
    """
    plt = _check_matplotlib()
    if plt is None:
        return None

    import numpy as np

    if metrics is None:
        metrics = ["hit@10", "ndcg@10", "mrr"]

    # 2x2 Grid: (Reasoning Type) x (Fine-tuning)
    row_labels = ["Thinking", "Direct"]
    col_labels = ["Fine-tuned", "Untuned"]

    model_grid = {
        (0, 0): "kitrec_full",    # Thinking + Fine-tuned
        (0, 1): "base_cot",       # Thinking + Untuned
        (1, 0): "kitrec_direct",  # Direct + Fine-tuned
        (1, 1): "base_direct",    # Direct + Untuned
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # 2x2 매트릭스 생성
        matrix = np.zeros((2, 2))
        for (r, c), model_name in model_grid.items():
            matrix[r, c] = ablation_results.get(model_name, {}).get(metric, 0)

        im = ax.imshow(matrix, cmap='YlGn', aspect='auto')

        # 축 설정
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # 값 표시
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{matrix[i, j]:.4f}',
                              ha="center", va="center", color="black", fontsize=11)

        ax.set_title(metric.upper())
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def generate_latex_table(
    data: Dict[str, Dict[str, float]],
    metrics: List[str],
    caption: str = "Evaluation Results",
    label: str = "tab:results"
) -> str:
    """
    LaTeX 테이블 생성

    Args:
        data: {model_name: {metric: value}}
        metrics: 포함할 메트릭 리스트
        caption: 테이블 캡션
        label: 테이블 라벨

    Returns:
        LaTeX 테이블 문자열
    """
    # 열 정의
    col_spec = "l" + "c" * len(metrics)

    # 헤더
    header = " & ".join(["Model"] + [m.upper() for m in metrics])

    # 데이터 행
    rows = []
    for model_name, model_metrics in data.items():
        values = [model_name]
        for m in metrics:
            val = model_metrics.get(m, 0)
            values.append(f"{val:.4f}")
        rows.append(" & ".join(values))

    # LaTeX 조합
    latex = f"""\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header} \\\\
\\midrule
{chr(92) + chr(92) + chr(10).join(rows)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    return latex


def generate_markdown_table(
    data: Dict[str, Dict[str, float]],
    metrics: List[str]
) -> str:
    """
    Markdown 테이블 생성

    Args:
        data: {model_name: {metric: value}}
        metrics: 포함할 메트릭 리스트

    Returns:
        Markdown 테이블 문자열
    """
    # 헤더
    header = "| Model | " + " | ".join(metrics) + " |"
    separator = "|" + "---|" * (len(metrics) + 1)

    # 데이터 행
    rows = []
    for model_name, model_metrics in data.items():
        values = [model_name]
        for m in metrics:
            val = model_metrics.get(m, 0)
            values.append(f"{val:.4f}")
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def generate_user_type_chart_data(
    analysis: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> Dict:
    """
    User Type별 차트 데이터 생성

    Args:
        analysis: User Type별 분석 결과
        metrics: 포함할 메트릭

    Returns:
        차트 데이터 (plotly/matplotlib 호환)
    """
    if metrics is None:
        metrics = ["hit@10", "ndcg@10"]

    # Core level 순서 정렬
    order = ["1-core", "2-core", "3-core", "4-core", "5+-core"]
    sorted_types = sorted(
        analysis.keys(),
        key=lambda x: order.index(x) if x in order else 999
    )

    chart_data = {
        "categories": sorted_types,
        "series": {}
    }

    for m in metrics:
        chart_data["series"][m] = [
            analysis.get(ut, {}).get(m, 0)
            for ut in sorted_types
        ]

    return chart_data


def generate_comparison_heatmap_data(
    comparison: Dict[str, Dict[str, Dict]],
    metric: str = "mean_diff"
) -> Dict:
    """
    모델 비교 히트맵 데이터 생성

    Args:
        comparison: 통계 비교 결과
        metric: 표시할 값 (mean_diff, p_value 등)

    Returns:
        히트맵 데이터
    """
    baselines = list(comparison.keys())
    metrics = list(comparison[baselines[0]].keys()) if baselines else []

    matrix = []
    for baseline in baselines:
        row = []
        for m in metrics:
            val = comparison[baseline].get(m, {}).get(metric, 0)
            row.append(val)
        matrix.append(row)

    return {
        "x_labels": metrics,
        "y_labels": baselines,
        "values": matrix
    }


def generate_ablation_table(
    ablation_results: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> str:
    """
    RQ1 Ablation Study 테이블 생성

    2×2 형태:
              | Fine-tuned | Untuned
    Thinking  | KitREC-Full| Base-CoT
    Direct    | KitREC-Dir | Base-Dir

    Args:
        ablation_results: {model_name: metrics}
        metrics: 포함할 메트릭

    Returns:
        Markdown 테이블
    """
    if metrics is None:
        metrics = ["hit@10", "ndcg@10", "mrr"]

    # 모델 매핑
    model_grid = {
        ("Thinking", "Fine-tuned"): "kitrec_full",
        ("Thinking", "Untuned"): "base_cot",
        ("Direct", "Fine-tuned"): "kitrec_direct",
        ("Direct", "Untuned"): "base_direct",
    }

    # 헤더
    header = "| Reasoning | Model Type | " + " | ".join(metrics) + " |"
    separator = "|" + "---|" * (len(metrics) + 2)

    rows = [header, separator]

    for (reasoning, model_type), model_name in model_grid.items():
        model_metrics = ablation_results.get(model_name, {})
        values = [f"{model_metrics.get(m, 0):.4f}" for m in metrics]
        row = f"| {reasoning} | {model_type} | " + " | ".join(values) + " |"
        rows.append(row)

    return "\n".join(rows)


def generate_report_summary(
    kitrec_metrics: Dict[str, float],
    baseline_comparison: Dict[str, Dict],
    ablation_results: Optional[Dict] = None,
    user_type_analysis: Optional[Dict] = None
) -> str:
    """
    최종 리포트 요약 생성

    Args:
        kitrec_metrics: KitREC 최종 메트릭
        baseline_comparison: 베이스라인 비교 결과
        ablation_results: Ablation study 결과
        user_type_analysis: User Type별 분석

    Returns:
        Markdown 리포트
    """
    report = []

    # 제목
    report.append("# KitREC Evaluation Report\n")

    # KitREC 성능 요약
    report.append("## 1. KitREC Performance Summary\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    for metric, value in kitrec_metrics.items():
        if isinstance(value, float):
            report.append(f"| {metric} | {value:.4f} |")
    report.append("")

    # Baseline 비교
    report.append("## 2. Baseline Comparison (RQ2)\n")
    report.append("Improvement over baselines:\n")
    for baseline, results in baseline_comparison.items():
        report.append(f"\n### vs {baseline}")
        for metric, data in results.items():
            if isinstance(data, dict):
                diff = data.get("mean_diff", 0)
                sig = "***" if data.get("significant_at_0.001") else \
                      "**" if data.get("significant_at_0.01") else \
                      "*" if data.get("significant_at_0.05") else ""
                report.append(f"- {metric}: {diff:+.4f}{sig}")
    report.append("")

    # Ablation Study
    if ablation_results:
        report.append("## 3. Ablation Study (RQ1)\n")
        report.append(generate_ablation_table(ablation_results))
        report.append("")

    # User Type 분석
    if user_type_analysis:
        report.append("## 4. Cold-Start Analysis (RQ3)\n")
        report.append("| Core Level | Hit@10 | NDCG@10 | Count |")
        report.append("|------------|--------|---------|-------|")
        for level, data in user_type_analysis.items():
            report.append(
                f"| {level} | {data.get('hit@10', 0):.4f} | "
                f"{data.get('ndcg@10', 0):.4f} | {data.get('sample_count', 0)} |"
            )
        report.append("")

    return "\n".join(report)
