"""
Inference 프롬프트 생성

CLAUDE.md RQ1 Note: Direct 모델용 Reasoning Guidelines 제거 버전 필요
- KitREC-Full: Thinking 포함 프롬프트
- KitREC-Direct: Reasoning Guidelines 제거된 프롬프트
"""

import re
from typing import Dict, Optional


class PromptBuilder:
    """추론용 프롬프트 빌더"""

    # CLAUDE.md Inference Prompt Template (Full version with Thinking)
    # Synced with CLAUDE.md Section: Inference Prompt Template
    THINKING_TEMPLATE = '''# Expert Cross-Domain Recommendation System

You are a specialized recommendation system with expertise in cross-domain knowledge transfer.
Your task is to leverage comprehensive user interaction patterns from source and target domains to rank the **Top 10** most suitable items from the candidate list.

## Input Parameters
- Source Domain: {source_domain}
- Target Domain: {target_domain}
- Task: Rank the top 10 items based on user preference alignment.

## User Interaction History
The user's past interactions contain the Title, Categories, User Rating, and a Summary of the item description.

### User's {source_domain} History (Source Domain):
{source_history}
(Format: - {{title}} | {{categories}} | Rating: {{rating:.1f}} | {{description_summary}})

### User's {target_domain} History (Target Domain):
{target_history}
(Format: - {{title}} | {{categories}} | Rating: {{rating:.1f}} | {{description_summary}})

## List of Available Candidate Items (Total 100):
The candidate items contain the Title, Categories, Average Rating, and a Summary.
{candidate_list}

## Reasoning Guidelines (Thinking Process)
Before generating the final JSON output, you must engage in a deep reasoning process.
Think step-by-step using the following phases:

### Phase 1: Pattern Recognition (Source Domain Analysis)
- Analyze the user's `{source_domain}` history to identify core preference signals.
- Extract key genres, thematic interests, content complexity, and stylistic preferences.
- Identify high-rated items (Rating > 4.0) to understand what the user truly values.

### Phase 2: Cross-Domain Knowledge Transfer
- Apply domain knowledge to map preferences from `{source_domain}` to `{target_domain}`.
- Example: If a user likes "Dark Fantasy Novels" (Source), infer a preference for "Dark/Gothic Atmosphere Movies" (Target).
- Consider semantic connections, author/director styles, and emotional tone.

### Phase 3: Candidate Evaluation & Selection
- Evaluate the 100 candidate items against the transferred profile.
- Select the Top 10 items that best match the inferred preferences.
- Ensure diversity in the selection while maintaining high relevance.
- Formulate a rationale for each selected item.

## Confidence Score Guidelines
Rate each recommendation on a 1-10 scale:
- 1-3: Low confidence (weak pattern match, uncertain transfer)
- 4-6: Moderate confidence (some preference signals align)
- 7-9: High confidence (strong cross-domain pattern match)
- 10: Very high confidence (near-perfect preference alignment)

**Note:** Distribute scores across the range; avoid assigning all items the same score.

## Output Format
After your reasoning process, provide results **ONLY** as a JSON array containing the **Top-10** recommended items.
Ensure the **"item_id"** matches the ID provided in the candidate list exactly (format: alphanumeric like "B07FLGJWKB").

```json
[
   {{"rank": 1, "item_id": "B07FLGJWKB", "title": "Example Title", "confidence_score": 9.5, "rationale": "Strong thematic alignment..."}},
   {{"rank": 2, "item_id": "B08XYZ1234", "title": "Second Pick", "confidence_score": 8.7, "rationale": "Similar style..."}},
   ...
   {{"rank": 10, "item_id": "B09ABC5678", "title": "Final Pick", "confidence_score": 6.2, "rationale": "Moderate relevance..."}}
]
```
'''

    # Direct Template (Reasoning Guidelines 제거)
    # No <think> block, direct JSON output
    DIRECT_TEMPLATE = '''# Expert Cross-Domain Recommendation System

You are a specialized recommendation system with expertise in cross-domain knowledge transfer.
Your task is to rank the **Top 10** most suitable items from the candidate list.

## Input Parameters
- Source Domain: {source_domain}
- Target Domain: {target_domain}
- Task: Rank the top 10 items based on user preference alignment.

## User Interaction History
### User's {source_domain} History (Source Domain):
{source_history}

### User's {target_domain} History (Target Domain):
{target_history}

## List of Available Candidate Items (Total 100):
{candidate_list}

## Confidence Score Guidelines
Rate each recommendation on a 1-10 scale:
- 1-3: Low confidence (weak pattern match)
- 4-6: Moderate confidence (some signals align)
- 7-9: High confidence (strong pattern match)
- 10: Very high confidence (near-perfect alignment)

**Note:** Distribute scores across the range; avoid assigning all items the same score.

## Output Instructions
Provide results **DIRECTLY** as a JSON array containing the **Top-10** recommended items.
**DO NOT** include any reasoning, explanation, or thinking process before the JSON output.
Ensure the **"item_id"** matches the ID provided in the candidate list exactly.

```json
[
   {{"rank": 1, "item_id": "...", "title": "...", "confidence_score": <1.0-10.0>, "rationale": "Brief one-line reason"}},
   ...
   {{"rank": 10, "item_id": "...", "title": "...", "confidence_score": <1.0-10.0>, "rationale": "Brief one-line reason"}}
]
```
'''

    def build_thinking_prompt(self, sample: Dict) -> str:
        """
        KitREC-Full, Base-CoT용 (Reasoning Guidelines 포함)
        - Fine-tuned 모델 또는 Zero-shot CoT 평가에 사용
        - Val/Test 데이터는 이미 완성된 프롬프트가 input에 있음
        """
        # Val/Test 데이터는 이미 완성된 프롬프트가 input 또는 instruction에 있음
        if sample.get("input") and sample["input"].strip():
            return sample["input"]
        elif sample.get("instruction") and sample["instruction"].strip():
            return sample["instruction"]
        else:
            raise ValueError("No valid prompt field found")

    def build_direct_prompt(self, sample: Dict) -> str:
        """
        KitREC-Direct, Base-Direct용 (Reasoning Guidelines 제거)
        - Thinking Process 없이 바로 JSON 출력
        """
        original_prompt = self.build_thinking_prompt(sample)

        # "## Reasoning Guidelines" 섹션 제거 (Phase 1~3 포함)
        pattern = r'## Reasoning Guidelines.*?(?=## Output Format)'
        modified = re.sub(pattern, '', original_prompt, flags=re.DOTALL)

        # Output Format 설명 수정 (reasoning 없이 바로 출력)
        modified = modified.replace(
            "After your reasoning process, provide results",
            "Provide results directly"
        )

        # 추가 지시 삽입
        modified = modified.replace(
            "## Output Format",
            "## Output Format\nDo NOT include any reasoning or thinking process.\n"
        )

        return modified.strip()

    def build_custom_prompt(
        self,
        source_domain: str,
        target_domain: str,
        source_history: str,
        target_history: str,
        candidate_list: str,
        use_thinking: bool = True
    ) -> str:
        """
        커스텀 프롬프트 생성 (Baseline 평가 등에 사용)

        Args:
            source_domain: Source domain name (e.g., "Books")
            target_domain: Target domain name (e.g., "Movies", "Music")
            source_history: Formatted source domain history string
            target_history: Formatted target domain history string
            candidate_list: Formatted candidate list string
            use_thinking: Whether to include Reasoning Guidelines
        """
        template = self.THINKING_TEMPLATE if use_thinking else self.DIRECT_TEMPLATE

        return template.format(
            source_domain=source_domain,
            target_domain=target_domain,
            source_history=source_history,
            target_history=target_history,
            candidate_list=candidate_list
        )

    def build_custom_prompt_legacy(
        self,
        source_domain: str,
        target_domain: str,
        user_history: str,
        candidate_list: str,
        use_thinking: bool = True
    ) -> str:
        """
        레거시 커스텀 프롬프트 생성 (단일 user_history 사용)

        Note: 새 코드에서는 build_custom_prompt 사용 권장
        """
        # Split user_history into source and target if possible
        return self.build_custom_prompt(
            source_domain=source_domain,
            target_domain=target_domain,
            source_history=user_history,
            target_history="(No target domain history available)",
            candidate_list=candidate_list,
            use_thinking=use_thinking
        )

    @staticmethod
    def extract_user_history_from_prompt(prompt: str) -> Optional[str]:
        """프롬프트에서 User History 섹션 추출"""
        pattern = r'## User Interaction History\s*(.*?)(?=## List of Available)'
        match = re.search(pattern, prompt, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_candidate_list_from_prompt(prompt: str) -> Optional[str]:
        """프롬프트에서 Candidate List 섹션 추출"""
        pattern = r'## List of Available Candidate Items.*?\n(.*?)(?=## (?:Reasoning|Output))'
        match = re.search(pattern, prompt, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def get_prompt_type(prompt: str) -> str:
        """프롬프트 타입 판별 (Thinking/Direct)"""
        if "## Reasoning Guidelines" in prompt:
            return "thinking"
        else:
            return "direct"
