# KitREC Documentation

**마지막 업데이트:** 2025-12-07

---

## 📚 문서 목록

| 문서 | 설명 | 주요 내용 |
|------|------|----------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 시스템 아키텍처 | 전체 구조, 모듈 관계, 평가 파이프라인 |
| [DATA_FLOW.md](./DATA_FLOW.md) | 데이터 흐름 | 데이터셋 구조, 스키마, 처리 흐름 |
| [RQ_EXPERIMENT_MAP.md](./RQ_EXPERIMENT_MAP.md) | 연구 질문 매핑 | RQ1-4 실험 설계, 메트릭, 예상 결과 |

---

## 🗂️ 프로젝트 루트 문서

| 문서 | 설명 |
|------|------|
| [CLAUDE.md](../CLAUDE.md) | 프로젝트 상세 가이드 (Claude AI용) |
| [detail_task_plan.md](../detail_task_plan.md) | 작업 계획서 |
| [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) | 구현 요약 |
| [.cursorrules](../.cursorrules) | Cursor AI 규칙 |

---

## 📊 다이어그램 가이드

모든 다이어그램은 **Mermaid** 형식으로 작성되었습니다.

### 지원 다이어그램 유형

| 유형 | 용도 | 예시 문서 |
|------|------|----------|
| `flowchart` | 프로세스 흐름 | DATA_FLOW.md |
| `graph` | 시스템 구조 | ARCHITECTURE.md |
| `sequenceDiagram` | 시퀀스 흐름 | ARCHITECTURE.md |
| `classDiagram` | 데이터 스키마 | DATA_FLOW.md |
| `mindmap` | 개념 구조 | RQ_EXPERIMENT_MAP.md |
| `xychart-beta` | 차트/그래프 | RQ_EXPERIMENT_MAP.md |
| `gantt` | 일정 계획 | RQ_EXPERIMENT_MAP.md |
| `pie` | 분포 시각화 | DATA_FLOW.md |
| `quadrantChart` | 2×2 매트릭스 | RQ_EXPERIMENT_MAP.md |

### 렌더링 방법

- **GitHub**: 자동 렌더링 지원
- **VSCode**: Markdown Preview Mermaid Support 확장 설치
- **Cursor**: 기본 지원
- **웹**: [Mermaid Live Editor](https://mermaid.live/)

---

## 🔑 핵심 개념 요약

### KitREC이란?

**K**nowledge-**I**nstruction **T**ransfer for **REC**ommendation

- Cross-Domain 추천 시스템 연구 프로젝트
- Books (Source) → Movies/Music (Target) 지식 전이
- Cold-start 문제 해결에 특화

### Research Questions

| RQ | 질문 | 핵심 비교 |
|----|------|----------|
| **RQ1** | Ablation Study | KitREC 구조 효과 검증 (2×2) |
| **RQ2** | CDR 비교 | vs CoNet, DTCDR, LLM4CDR |
| **RQ3** | Cold-start | 1-core ~ 10-core 성능 분석 |
| **RQ4** | Explainability | Confidence + Rationale 품질 |

### 핵심 모델

| 모델 | 샘플 수 | 대상 User Type |
|------|---------|----------------|
| DualFT-Movies | 12,000 | overlapping + cold_start_2/3/4core |
| DualFT-Music | 12,000 | overlapping + cold_start_2/3/4core |
| SingleFT-Movies | 3,000 | source_only (1-core) |
| SingleFT-Music | 3,000 | source_only (1-core) |

---

## 🛠️ Quick Start

```bash
# 환경 설정 확인
python scripts/verify_environment.py

# KitREC 평가 실행
python scripts/run_kitrec_eval.py \
    --model_name dualft_movies_seta \
    --dataset Younggooo/kitrec-test-seta \
    --output_dir results/kitrec

# Baseline 평가
python scripts/run_baseline_eval.py \
    --baseline conet \
    --target_domain movies

# Ablation Study
python scripts/run_ablation_study.py \
    --config configs/eval_config.yaml
```

---

## 📞 참조

- **HuggingFace**: [Younggooo](https://huggingface.co/Younggooo)
- **Base Model**: [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)

