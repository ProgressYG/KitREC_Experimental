"""
LLM4CDR 3-stage Prompting Pipeline

Stage 1: Domain Gap Analysis
Stage 2: User Interest Reasoning
Stage 3: Candidate Re-ranking

## KitREC Implementation vs Original Paper Differences

| Aspect | Original Paper | KitREC Implementation |
|--------|---------------|----------------------|
| Candidate Set | 3 GT + 20-30 Negatives | 1 GT + 99 Negatives |
| Target History | Not used | Included for fair comparison |
| Stage 1 Caching | Domain-level caching | Same implementation |
| Output Format | Varies | Standardized JSON |

Note: Target history is included in KitREC to ensure fair comparison with
KitREC model which has access to target history. This difference should be
reported in evaluation results.
"""

from typing import Dict, List, Optional
import re


# Documentation of implementation differences
LLM4CDR_IMPLEMENTATION_NOTES = """
## KitREC vs Original LLM4CDR Implementation

### Key Differences:
1. **Candidate Set Size**: Original uses ~30 items, KitREC uses 100 (1 GT + 99 negatives)
2. **Target History**: Original doesn't use target history, KitREC includes it for fairness
3. **Evaluation Protocol**: KitREC uses standardized metrics (Hit@K, NDCG@K, MRR)

### Rationale:
- Including target history ensures fair comparison with KitREC model
- Using the same candidate set size (100) ensures comparable metrics
- Standardized output format enables consistent parsing and evaluation

### Reporting:
When presenting results, note that LLM4CDR was adapted to use KitREC's
evaluation protocol which differs from the original paper's setup.
"""


class LLM4CDRPrompts:
    """LLM4CDR 3-stage 프롬프트 생성"""

    # Stage 1: Domain Gap Analysis
    STAGE1_TEMPLATE = '''You are an expert in cross-domain recommendation systems.

## Task: Analyze Domain Relationship

Analyze the relationship between the following two domains and identify key transferable patterns:

- Source Domain: {source_domain}
- Target Domain: {target_domain}

## Analysis Requirements
1. Identify semantic connections between domains
2. Describe common user preference patterns that transfer
3. Note domain-specific characteristics that may affect recommendations

## Output
Provide a concise analysis (2-3 paragraphs) of how preferences in {source_domain} can inform recommendations in {target_domain}.'''

    # Stage 2: User Interest Reasoning
    # Note: Target history added for KitREC fair comparison (differs from original paper)
    STAGE2_TEMPLATE = '''You are an expert recommendation system analyzing user preferences.

## User's {source_domain} History (Source Domain):
{user_history}

## User's {target_domain} History (Target Domain):
{target_history}

## Domain Relationship Context:
{domain_analysis}

## Task: Infer User Preferences

Based on the user's interaction history from both domains, describe:
1. Core preference patterns (genres, themes, complexity)
2. High-value signals (items rated > 4.0)
3. Cross-domain preference alignment
4. Potential interests in {target_domain} based on source patterns

## Output
Provide a detailed user preference profile that can guide {target_domain} recommendations.'''

    # Stage 3: Candidate Re-ranking
    STAGE3_TEMPLATE = '''You are an expert cross-domain recommendation system.

## User Preference Profile:
{user_profile}

## Domain Context:
{domain_analysis}

## Task: Re-rank Candidate Items

From the following {target_domain} candidate items, select and rank the **Top 10** most suitable items for this user.

## Candidate Items (100 total):
{candidate_list}

## Output Requirements
Return results as a JSON array with the top 10 items:

```json
[
   {{"rank": 1, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..."}},
   ...
   {{"rank": 10, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..."}}
]
```

Only recommend items from the candidate list above. Do not make up item IDs.'''

    def build_stage1_prompt(
        self,
        source_domain: str,
        target_domain: str
    ) -> str:
        """Stage 1: Domain Gap Analysis 프롬프트"""
        return self.STAGE1_TEMPLATE.format(
            source_domain=source_domain,
            target_domain=target_domain
        )

    def build_stage2_prompt(
        self,
        source_domain: str,
        target_domain: str,
        user_history: str,
        domain_analysis: str,
        target_history: str = ""
    ) -> str:
        """
        Stage 2: User Interest Reasoning 프롬프트

        Note: target_history added for KitREC fair comparison.
        Original LLM4CDR paper does not use target history.
        """
        return self.STAGE2_TEMPLATE.format(
            source_domain=source_domain,
            target_domain=target_domain,
            user_history=user_history,
            target_history=target_history if target_history else "(No target history available)",
            domain_analysis=domain_analysis
        )

    def build_stage3_prompt(
        self,
        target_domain: str,
        user_profile: str,
        domain_analysis: str,
        candidate_list: str
    ) -> str:
        """Stage 3: Candidate Re-ranking 프롬프트"""
        return self.STAGE3_TEMPLATE.format(
            target_domain=target_domain,
            user_profile=user_profile,
            domain_analysis=domain_analysis,
            candidate_list=candidate_list
        )

    def extract_user_history_from_kitrec_prompt(self, prompt: str) -> str:
        """KitREC 프롬프트에서 Source User History 추출"""
        pattern = r"### User's \w+ History \(Source Domain\):\s*(.*?)(?=###|## List)"
        match = re.search(pattern, prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_target_history_from_kitrec_prompt(self, prompt: str) -> str:
        """
        KitREC 프롬프트에서 Target User History 추출

        Note: Target history is included for fair comparison with KitREC model.
        Original LLM4CDR paper does not use target history.
        """
        pattern = r"### User's \w+ History \(Target Domain\):\s*(.*?)(?=###|## List)"
        match = re.search(pattern, prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_candidate_list_from_kitrec_prompt(self, prompt: str) -> str:
        """KitREC 프롬프트에서 Candidate List 추출"""
        pattern = r'## List of Available Candidate Items.*?\n(.*?)(?=## (?:Reasoning|Output))'
        match = re.search(pattern, prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def build_all_stages(
        self,
        source_domain: str,
        target_domain: str,
        user_history: str,
        candidate_list: str,
        target_history: str = ""
    ) -> Dict[str, str]:
        """
        모든 스테이지 프롬프트 생성

        Args:
            source_domain: Source domain name
            target_domain: Target domain name
            user_history: Source domain user history
            candidate_list: Candidate items to rank
            target_history: Target domain user history (KitREC extension)

        Returns:
            {"stage1": str, "stage2": str, "stage3": str}
        """
        # Stage 1은 도메인 분석만 필요 (캐싱 가능)
        stage1 = self.build_stage1_prompt(source_domain, target_domain)

        # Stage 2, 3는 placeholder (실제 실행 시 이전 단계 결과 필요)
        stage2_template = self.build_stage2_prompt(
            source_domain, target_domain, user_history, "{domain_analysis}",
            target_history=target_history
        )

        stage3_template = self.build_stage3_prompt(
            target_domain, "{user_profile}", "{domain_analysis}", candidate_list
        )

        return {
            "stage1": stage1,
            "stage2": stage2_template,
            "stage3": stage3_template
        }


class VanillaZeroShotPrompt:
    """
    Vanilla Zero-shot Prompting (NIR Paradigm)

    LLM4CDR와 달리 단일 프롬프트로 직접 생성
    """

    TEMPLATE = '''You are a recommendation system.

## User's {source_domain} History:
{user_history}

## Candidate {target_domain} Items:
{candidate_list}

## Task
Based on the user's {source_domain} history, recommend the top 10 {target_domain} items from the candidate list.

## Output Format
Return results as a JSON array:

```json
[
   {{"rank": 1, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..."}},
   ...
   {{"rank": 10, "item_id": "...", "title": "...", "confidence_score": <float 1-10>, "rationale": "..."}}
]
```'''

    def build_prompt(
        self,
        source_domain: str,
        target_domain: str,
        user_history: str,
        candidate_list: str
    ) -> str:
        return self.TEMPLATE.format(
            source_domain=source_domain,
            target_domain=target_domain,
            user_history=user_history,
            candidate_list=candidate_list
        )
