"""
문제은행 생성 엔진 v2
- 2-Pass 생성: 1차 문항 생성 → 2차 오답 매력도 개선
- SAT 학습목표 기반 평가문항
- Bloom's Taxonomy 수준별 문항 (지식/이해/적용/분석)
- 시나리오 기반 문항 (Application/Analysis 수준)
- Few-shot 예시로 품질 앵커링

연구 근거:
- Few-shot 예시로 비기능적 오답 비율 50% → 16% 감소 가능
- 레슨 컨텍스트 포함 시 오답의 의미적/주제적 품질 향상
- 오답별 "선택 이유" 명시 시 매력도 대폭 향상
"""
import json
from typing import Optional

import httpx

from src.utils.http_client import get_ssl_verify

from src.prompts.template_manager import get_template_manager


class QuestionGenerator:
    """Solar LLM 기반 문제은행 생성기 v2 (2-Pass + 시나리오)"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
        model: str = "solar-pro3",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def _call_llm(
        self, system_prompt: str, user_prompt: str,
        temperature: float = 0.3, max_tokens: int = 8192,
    ) -> dict:
        """LLM 호출 공통 메서드"""
        async with httpx.AsyncClient(timeout=120.0, verify=get_ssl_verify()) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)

    async def generate_questions(
        self,
        slides: list[dict],
        learning_objectives: list[str],
        rag_context: list[dict],
        bloom_distribution: Optional[dict] = None,
        num_questions: int = 20,
        few_shot_examples: list[dict] = None,
        enable_two_pass: bool = True,
    ) -> list[dict]:
        """
        2-Pass 문항 생성

        Pass 1: 문항 + 오답 + distractor_rationale 생성
        Pass 2: 비기능적 오답 자동 감지 및 교체
        """
        if bloom_distribution is None:
            bloom_distribution = {
                "Knowledge": 0.3,
                "Comprehension": 0.3,
                "Application": 0.25,
                "Analysis": 0.15,
            }

        # === Pass 1: 문항 생성 ===
        content_summary = "\n\n".join([
            f"[페이지 {s['page_number']}] {s.get('title', '')}\n{s.get('content', '')[:300]}"
            for s in slides
        ])

        # VLM 분석 결과가 있으면 포함
        vlm_summaries = []
        for s in slides:
            for analysis in s.get("vlm_analyses", []):
                vlm_summaries.append(
                    f"[시각자료 p.{s['page_number']}] {analysis.get('description', '')}"
                )
        vlm_text = "\n".join(vlm_summaries[:10]) if vlm_summaries else "(시각자료 분석 없음)"

        context_text = "\n\n".join([
            f"[참고] {ctx['text']}" for ctx in rag_context[:5]
        ])

        distribution_text = "\n".join([
            f"- {level}: {int(num_questions * ratio)}문항"
            for level, ratio in bloom_distribution.items()
        ])

        # Few-shot 예시 구성
        few_shot_text = ""
        if few_shot_examples:
            few_shot_text = f"\n\n## 모범 문항 예시 (이 수준으로 작성)\n{json.dumps(few_shot_examples[:3], ensure_ascii=False, indent=2)}"

        user_prompt = f"""## 교재 내용
{content_summary}

## 시각자료 분석 (VLM)
{vlm_text}

## 학습목표
{chr(10).join(f'- {obj}' for obj in learning_objectives)}

## 참고 자료
{context_text}
{few_shot_text}

## 생성 요구사항
- 총 {num_questions}문항 생성
- Bloom's Taxonomy 수준별 배분:
{distribution_text}
- Application/Analysis 문항은 반드시 시나리오(scenario) 포함
- 모든 오답에 distractor_rationale(선택 이유) 필수 작성

위 내용을 바탕으로 평가문항을 JSON 형식으로 작성해주세요."""

        tm = get_template_manager()
        result = await self._call_llm(
            tm.get_prompt("question_generation"), user_prompt,
            temperature=tm.get_temperature("question_generation"),
            max_tokens=tm.get_max_tokens("question_generation"),
        )

        # 결과 파싱
        if isinstance(result, dict) and "questions" in result:
            questions = result["questions"]
        elif isinstance(result, list):
            questions = result
        else:
            questions = [result]

        # === Pass 2: 오답 매력도 개선 ===
        if enable_two_pass and questions:
            questions = await self._refine_distractors(questions)

        return questions

    async def _refine_distractors(self, questions: list[dict]) -> list[dict]:
        """
        Pass 2: 비기능적 오답 자동 감지 및 교체

        각 문항의 오답을 검토하여 비기능적 오답을 매력적 오답으로 개선
        """
        refined_questions = []

        for q in questions:
            # 오답 개선 요청
            user_prompt = f"""## 검토 대상 문항
{json.dumps(q, ensure_ascii=False, indent=2)}

이 문항의 오답 보기들을 검토하고, 비기능적 오답이 있으면 개선해주세요."""

            try:
                _tm = get_template_manager()
                refinement = await self._call_llm(
                    _tm.get_prompt("distractor_refinement"), user_prompt,
                    temperature=_tm.get_temperature("distractor_refinement"),
                    max_tokens=_tm.get_max_tokens("distractor_refinement"),
                )

                # 개선된 보기 적용
                if refinement.get("improved_options"):
                    q["options"] = refinement["improved_options"]
                if refinement.get("improved_distractor_rationale"):
                    q["distractor_rationale"] = refinement["improved_distractor_rationale"]
                q["_pass2_applied"] = True
                q["_pass2_changes"] = refinement.get("changes_made", [])

            except Exception:
                q["_pass2_applied"] = False

            refined_questions.append(q)

        return refined_questions

    async def generate_scenario_questions(
        self,
        slides: list[dict],
        learning_objectives: list[str],
        rag_context: list[dict],
        num_scenarios: int = 5,
    ) -> list[dict]:
        """
        시나리오 기반 복합 문항 세트 생성

        하나의 시나리오에서 2~3개 연계 문항 (Application + Analysis)
        원자력 운전/정비 실무 의사결정 상황
        """
        content_summary = "\n\n".join([
            f"[p.{s['page_number']}] {s.get('title', '')}: {s.get('content', '')[:200]}"
            for s in slides[:10]
        ])

        # VLM 분석 결과에서 시나리오 소재 추출
        vlm_context_parts = []
        for s in slides:
            for analysis in s.get("vlm_analyses", []):
                parts = []
                if analysis.get("visual_type"):
                    parts.append(f"유형: {analysis['visual_type']}")
                if analysis.get("description"):
                    parts.append(f"설명: {analysis['description'][:150]}")
                if analysis.get("key_components"):
                    comps = analysis["key_components"]
                    if isinstance(comps, list):
                        parts.append(f"구성요소: {', '.join(str(c) for c in comps[:5])}")
                if analysis.get("related_systems"):
                    systems = analysis["related_systems"]
                    if isinstance(systems, list):
                        parts.append(f"관련계통: {', '.join(str(s) for s in systems[:3])}")
                if analysis.get("safety_notes"):
                    parts.append(f"안전사항: {analysis['safety_notes']}")
                if parts:
                    vlm_context_parts.append(
                        f"[시각자료 p.{s['page_number']}] " + " | ".join(parts)
                    )
        vlm_context = "\n".join(vlm_context_parts[:8]) if vlm_context_parts else "(시각자료 분석 없음)"

        context_text = "\n".join([
            f"[참고] {ctx['text'][:200]}" for ctx in rag_context[:3]
        ])

        user_prompt = f"""## 교재 내용
{content_summary}

## 시각자료 분석 결과 (VLM) — 시나리오 소재로 활용
{vlm_context}

## 학습목표
{chr(10).join(f'- {obj}' for obj in learning_objectives)}

## 참고 자료
{context_text}

## 요구사항
위 교재 내용과 시각자료(도면/계통도) 분석 결과를 활용하여 {num_scenarios}개의 실무 시나리오를 만들고,
각 시나리오에서 2~3개의 연계 문항을 생성해주세요.

시나리오는 원자력 발전소 현장에서 실제 발생할 수 있는 상황이어야 합니다.
- 시나리오 1~2개: 운전 관련 (비정상/비상 상황 대응)
- 시나리오 1~2개: 정비/시험 관련 (작업 절차, 판단)
- 시나리오 1개: 안전 관련 (방사선 방호, 산업안전)

각 시나리오에 구체적 수치(온도, 압력, 유량, 방사선량 등)를 포함하세요.

## 출력 형식 (JSON)
{{
    "scenario_sets": [
        {{
            "scenario_id": "S001",
            "scenario_type": "비정상운전",
            "scenario_text": "구체적 상황 묘사 (3~5문장, 수치 포함)",
            "related_systems": ["관련 계통"],
            "questions": [
                {{
                    "question_id": "S001-Q1",
                    "bloom_level": "Application",
                    "question_text": "문제",
                    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
                    "correct_answer": "B",
                    "distractor_rationale": {{"A": "...", "C": "...", "D": "..."}},
                    "explanation": "해설"
                }},
                ...
            ]
        }}
    ]
}}"""

        _tm = get_template_manager()
        result = await self._call_llm(
            _tm.get_prompt("question_generation") + "\n\n" + _tm.get_prompt("scenario_generation"),
            user_prompt,
            temperature=_tm.get_temperature("scenario_generation"),
            max_tokens=_tm.get_max_tokens("scenario_generation"),
        )

        # 시나리오 세트에서 개별 문항 추출
        scenario_sets = result.get("scenario_sets", [result] if "scenario_id" in result else [])
        all_questions = []

        for scenario in scenario_sets:
            scenario_text = scenario.get("scenario_text", "")
            for q in scenario.get("questions", []):
                q["scenario"] = scenario_text
                q["scenario_id"] = scenario.get("scenario_id", "")
                q["scenario_type"] = scenario.get("scenario_type", "")
                all_questions.append(q)

        return all_questions

    async def validate_questions(
        self,
        questions: list[dict],
        rag_search_fn,
    ) -> list[dict]:
        """생성된 문항 품질 검증 (기본)"""
        validated = []
        for q in questions:
            search_results = await rag_search_fn(q.get("question_text", ""))
            q["validation"] = {
                "fact_checked": len(search_results) > 0,
                "reference_found": bool(search_results),
                "references": [r["text"][:100] for r in search_results[:3]],
            }
            validated.append(q)
        return validated
