"""
강의 스크립트 생성 엔진
- 슬라이드별 강의 멘트 자동 생성
- SAT 학습목표 연계
- RAG 기반 보충 설명 포함
"""
import json
from typing import Optional

import httpx

from src.utils.http_client import get_ssl_verify

from src.prompts.template_manager import get_template_manager


class ScriptGenerator:
    """Solar LLM 기반 강의 스크립트 생성기"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
        model: str = "solar-pro3",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @staticmethod
    def _format_vlm_analyses(vlm_analyses: list[dict]) -> str:
        """VLM 분석 결과를 프롬프트에 삽입할 텍스트로 변환"""
        if not vlm_analyses:
            return "이 슬라이드에는 복잡한 시각자료가 없습니다."

        sections = []
        for i, analysis in enumerate(vlm_analyses, 1):
            parts = [f"### 시각자료 {i}"]
            if analysis.get("title"):
                parts.append(f"- 제목: {analysis['title']}")
            if analysis.get("visual_type"):
                parts.append(f"- 유형: {analysis['visual_type']}")
            if analysis.get("description"):
                parts.append(f"- 설명: {analysis['description']}")
            if analysis.get("key_components"):
                components = analysis["key_components"]
                if isinstance(components, list):
                    parts.append(f"- 주요 구성요소: {', '.join(str(c) for c in components)}")
                else:
                    parts.append(f"- 주요 구성요소: {components}")
            if analysis.get("difficulty_level"):
                parts.append(f"- 난이도: {analysis['difficulty_level']}")
            if analysis.get("related_systems"):
                systems = analysis["related_systems"]
                if isinstance(systems, list):
                    parts.append(f"- 관련 계통: {', '.join(str(s) for s in systems)}")
                else:
                    parts.append(f"- 관련 계통: {systems}")
            if analysis.get("teaching_points"):
                tp = analysis["teaching_points"]
                if isinstance(tp, list):
                    parts.append("- **교육 포인트:**")
                    for point in tp:
                        parts.append(f"  - {point}")
                else:
                    parts.append(f"- **교육 포인트:** {tp}")
            if analysis.get("safety_notes"):
                sn = analysis["safety_notes"]
                if isinstance(sn, list):
                    parts.append("- **안전 관련 사항:**")
                    for note in sn:
                        parts.append(f"  - {note}")
                else:
                    parts.append(f"- **안전 관련 사항:** {sn}")
            sections.append("\n".join(parts))

        return "\n\n".join(sections)

    async def generate_slide_script(
        self,
        slide: dict,
        learning_objectives: list[str],
        rag_context: list[dict],
        audience_level: str = "intermediate",
        previous_slide_summary: str = "",
        curated_prompt: str = "",
    ) -> dict:
        """
        단일 슬라이드에 대한 강의 스크립트 생성

        Args:
            slide: 파싱된 슬라이드 데이터
            learning_objectives: SAT 학습목표 리스트
            rag_context: RAG 검색 결과 (관련 지식)
            audience_level: 대상 수준
            previous_slide_summary: 이전 슬라이드 요약 (흐름 연결용)
            curated_prompt: Curated System Prompt (SAT/Bloom/terminology 등 추가 시스템 컨텍스트)

        Returns:
            생성된 스크립트
        """
        # RAG 컨텍스트 구성
        context_text = "\n\n".join([
            f"[참고자료 {i+1}] {ctx['text']}"
            for i, ctx in enumerate(rag_context[:5])
        ])

        # VLM 시각자료 분석 결과 구성
        vlm_analyses = slide.get("vlm_analyses", [])
        vlm_text = self._format_vlm_analyses(vlm_analyses)

        user_prompt = f"""## 슬라이드 정보
- 페이지: {slide['page_number']}
- 제목: {slide.get('title', '(제목 없음)')}
- 내용: {slide.get('content', '')}
- 표: {json.dumps(slide.get('tables', []), ensure_ascii=False)[:500]}
- 이미지/차트 설명: {json.dumps([img.get('description', '') for img in slide.get('images', []) + slide.get('charts', [])], ensure_ascii=False)[:500]}

## 시각자료 분석 결과 (VLM)
{vlm_text}

## 학습목표
{chr(10).join(f'- {obj}' for obj in learning_objectives)}

## 대상 수준
{audience_level}

## 이전 슬라이드 요약
{previous_slide_summary or '(첫 슬라이드)'}

## 참고 자료 (RAG)
{context_text}

위 정보를 바탕으로 이 슬라이드에 대한 강의 스크립트를 JSON 형식으로 작성해주세요.
시각자료 분석 결과가 있는 경우, visual_explanation 섹션에서 해당 시각자료를 구체적으로 설명하고 교육 포인트와 안전 관련 사항을 강조해주세요.
visual_references에는 참조한 시각자료의 제목이나 설명을 포함해주세요."""

        # 시스템 메시지 구성: YAML 템플릿 + Curated Prompt (있는 경우)
        tm = get_template_manager()
        system_content = tm.get_prompt("script_generation")
        if curated_prompt:
            system_content = (
                f"{system_content}\n\n"
                f"## 추가 교수설계 컨텍스트\n{curated_prompt}"
            )

        async with httpx.AsyncClient(timeout=60.0, verify=get_ssl_verify()) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": tm.get_temperature("script_generation"),
                    "max_tokens": tm.get_max_tokens("script_generation"),
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)

    async def generate_full_script(
        self,
        slides: list[dict],
        learning_objectives: list[str],
        rag_search_fn,
        audience_level: str = "intermediate",
        curated_prompt: str = "",
    ) -> list[dict]:
        """
        전체 슬라이드에 대한 강의 스크립트 일괄 생성

        Args:
            slides: 파싱된 슬라이드 데이터 리스트
            learning_objectives: SAT 학습목표 리스트
            rag_search_fn: RAG 검색 함수
            audience_level: 대상 수준
            curated_prompt: Curated System Prompt (SAT/Bloom/terminology 등 추가 시스템 컨텍스트)
        """
        scripts = []
        previous_summary = ""

        for slide in slides:
            # 슬라이드 내용 기반 RAG 검색
            query = f"{slide.get('title', '')} {slide.get('content', '')[:200]}"
            rag_context = await rag_search_fn(query)

            script = await self.generate_slide_script(
                slide=slide,
                learning_objectives=learning_objectives,
                rag_context=rag_context,
                audience_level=audience_level,
                previous_slide_summary=previous_summary,
                curated_prompt=curated_prompt,
            )
            scripts.append(script)

            # 다음 슬라이드를 위한 요약
            previous_summary = script.get("script", {}).get("summary", "")

        return scripts
