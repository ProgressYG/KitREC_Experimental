# KitREC 연구를 위한 베이스라인 모델 정리

## 1. 딥러닝 기반 CDR 베이스라인

### 1.1 CoNet (Collaborative Cross Networks)

| 항목 | 내용 |
|------|------|
| **출처** | Hu, G., Zhang, Y., & Yang, Q. (2018). ACM CIKM |
| **핵심 메커니즘** | Cross-stitch Network를 통한 은닉층(Hidden Layer) 간 정보 공유 |
| **장점** | 구조가 직관적이며, 두 도메인 간 비선형적 상호작용 모델링 가능 |
| **한계** | 두 도메인 데이터 동시 입력 구조 가정, 전이 방향성 모호, 'Knowledge Transfer'보다 'Parameter Sharing'에 가까움. 최신 트렌드 대변에 다소 노후화 |
| **데이터 구성** | 두 도메인 간 공통 사용자 집합(U) 기준, 평점 4~5점을 Positive로 변환 |
| **평가 프로토콜** | Leave-One-Out (LOO), 테스트 1 + 음성 99 |
| **평가 지표** | Hit@10, NDCG@10 |

---

### 1.2 DTCDR (Dual-Target Cross-Domain Recommendation)

| 항목 | 내용 |
|------|------|
| **출처** | Zhu, F., et al. (2019). ACM CIKM |
| **핵심 메커니즘** | Multi-Task Learning 기반의 임베딩 공유 및 도메인 적응(Domain Adaptation) |
| **선정 이유** | KitREC의 DualFT1-Model, DualFT2-Model과 'Dual-Target' 시나리오가 일치. "임베딩 매핑(DTCDR) vs 지식 전이(KitREC)" 대결 구도 형성 가능 |
| **목표** | 소스와 타겟 도메인 모두의 추천 성능을 동시에 향상 |
| **데이터 구성** | 사용자·아이템 ≥5 상호작용 유지, 리뷰·태그·프로필 등 다중소스 정보로 텍스트 임베딩 |
| **평가 프로토콜** | Leave-One-Out (LOO), Top-N Ranking |
| **평가 지표** | HR@10, NDCG@10 |

---

## 2. LLM 기반 In-Context Learning 베이스라인

### 2.1 LLM4CDR (Uncovering Cross-Domain Recommendation Ability of LLMs)

| 항목 | 내용 |
|------|------|
| **출처** | Liu, X., et al. (2025). ACM RecSys / WWW Companion |
| **유형** | Prompt-based Re-ranking / Chain-of-Thought Reasoning |
| **선정 이유** | LLM의 교차 도메인 추천 능력을 가장 체계적으로 분석한 연구. "모델 튜닝의 비용 가치" 질문에 답하기 위함 |

**3단계 추론 파이프라인:**

| 단계 | 설명 | 예시 프롬프트 |
|------|------|-------------|
| **1. Domain Gap Analysis** | Source-Target 도메인 간 관계 분석 | "Books와 Movies 도메인 사이의 상관관계를 분석하라" |
| **2. User Interest Reasoning** | 사용자 Source 히스토리 기반 잠재적 관심사 서술 | "사용자 A는 'Sapiens'를 읽었다. 이 사용자의 성향은..." |
| **3. Candidate Re-ranking** | 도출된 정보 바탕으로 후보 아이템 순위 재조정 | 후보 목록에서 Top-K 선정 |

**KitREC 대비 실험적 통찰:**

| 비교 항목 | LLM4CDR | KitREC |
|----------|---------|--------|
| **Context Window** | 토큰 제약으로 히스토리 20~40개 제한 | 파인튜닝으로 긴 문맥/압축 표현 학습 가능 |
| **Hallucination** | 존재하지 않는 아이템 추천 위험 | Ground Truth 기반 튜닝으로 환각 억제 |
| **데이터 필터링** | 5점 평점만, 사용자 >20, 아이템 >10 | 다양한 상호작용 수준 포함 |

---

### 2.2 Vanilla Zero-shot Prompting (NIR Paradigm)

| 항목 | 내용 |
|------|------|
| **관련 연구** | Wang, L., & Lim, E. P. (2023). "Zero-Shot Next-Item Recommendation using Large Pre-trained Language Models" |
| **유형** | Direct Generation / Zero-shot |
| **선정 이유** | 베이스라인 하한선(Lower Bound) 설정. "복잡한 기법 없이 LLM에게 물어보면 얼마나 잘하는가?" 질문에 답변 |

**프롬프트 템플릿:**
```
User History in Books: [...]
Based on the history above, recommend top-10 movies from the Candidate List: [Movie 1, Movie 2,..., Movie 100].
Output Format: JSON list of titles.
```

**KitREC 대비 비교 포인트:**

| 비교 항목 | Vanilla Zero-shot | KitREC |
|----------|-------------------|--------|
| **Domain Shift 적응** | Amazon 데이터셋 편향/마이너 취향 이해 어려움 | 도메인 특수성(Domain Specificity) 포착 |
| **형식 준수** | JSON 형식 미준수, 후보군 외 아이템 추천 가능 | "결과 형태 제어" 능력 우수 |
| **비용 효율성** | 튜닝 비용 없음 | 튜닝 비용 대비 NDCG +20% 이상 향상 목표 |

---

## 3. 베이스라인 모델 종합 비교

### 3.1 데이터 구성 및 필터링 기준

| 모델 | 사용 데이터 | 전처리·필터링 기준 | 특징 |
|------|-----------|------------------|------|
| **CoNet** | Amazon (Movies&TV → Books) | 공통 사용자 집합(U) 기준, 평점 4~5점 Positive | 필터링 기준 불명확, 단순 cross-domain 구조 |
| **DTCDR** | Douban 3도메인 + MovieLens 20M | 사용자·아이템 ≥5 상호작용 | 다중소스 정보 활용, 풍부한 부가정보 |
| **LLM4CDR** | Amazon 5-core (Movies, CDs, Games, Electronics) | 5점만, 사용자 >20, 아이템 >10, 히스토리 ≥20/30/40 | LLM 토큰 제약 맞춤 시퀀스 제한 |
| **Vanilla (NIR)** | 다양한 도메인 | 최근 10개 히스토리 | LLM 사전학습 지식에 전적 의존 |

### 3.2 Candidate List 구축 및 평가 방식

| 모델 | Candidate List 구성 | Negative Sampling | 평가 프로토콜 | 핵심 평가 지표 |
|------|-------------------|-------------------|-----------|-------------|
| **CoNet** | 테스트 1 + 음성 99 | 무작위 99개 | LOO | Hit@10, NDCG@10 |
| **DTCDR** | 테스트 1 + 음성 99 | 무작위 99개 | LOO (Top-N Ranking) | HR@10, NDCG@10 |
| **LLM4CDR** | 테스트 3 + 음성 20~30 | 무작위 20~30개 | Prompt-based Re-ranking | Accuracy of Ranking |
| **Vanilla NIR** | 직접 생성 | 해당 없음 | Zero-shot Generation | NDCG, Hit Rate |

### 3.3 Train/Val/Test 분할 방식

| 모델 | 주요 목적 | 데이터 분할 기준 | 핵심 특징 |
|------|---------|---------------|---------|
| **CoNet** | 전통적 CDR 평가 | LOO: 최근 1개 테스트, 무작위 1개 검증, 나머지 훈련 | 단일 평가 샘플 중심 |
| **DTCDR** | Dual-Target CDR 평가 | LOO: 최근/무작위 1개 테스트, 나머지 훈련 | 전통적 MF 평가 기반 |
| **LLM4CDR** | LLM Zero-shot/Few-shot 추론 평가 | 프롬프트 기반 조건부 테스트 (학습 없음) | LLM 추론 중심, 평가 대상 ~100명 |
| **Vanilla NIR** |     |   |  |  |



---

## 4. KitREC Research Questions별 베이스라인 활용 계획

| RQ | 연구 질문 | KitREC 모델 | 비교 베이스라인 |
|----|---------|-----------|-------------|
| **RQ1** | KitREC 구조의 효과성 검증 | Full KitREC vs FT-only vs Zero-shot only | - |
| **RQ2** | CDR 방식의 효과성 검증 | DualFT1, DualFT2, SingleFT1,  SingleFT2| CoNet, DTCDR, LLM4CDR, Vanilla NIR |
| **RQ3** | Cold-start/Sparse 문제 해결 | DualFT1, DualFT2 | LLM4CDR |
| **RQ4** | Confidence Score/Rationale 검증 | 전체 모델 | MAE, RMSE로 실제 평점과 비교 |

---

이 정리가 연구 진행에 도움이 되시길 바랍니다. 추가적인 세부사항이나 수정이 필요하시면 말씀해 주세요.