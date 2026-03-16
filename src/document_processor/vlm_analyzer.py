"""
VLM 기반 강의안 시각자료 심층 분석 모듈
- 원자력 설비도면, P&ID, 계통도, 차트를 VLM으로 해석
- 슬라이드별 시각자료에 대한 자연어 설명 생성
- 강의 스크립트 생성 시 시각자료 설명의 품질을 결정하는 핵심 모듈
- v4: 원자력 용어사전 컨텍스트 주입으로 도면 식별 정확도 향상
"""
import json
from typing import Optional

import httpx

from src.utils.http_client import get_ssl_verify

from src.domain.nuclear_glossary import NuclearGlossaryManager
from src.prompts.template_manager import get_template_manager


class VLMAnalyzer:
    """VLM 기반 시각자료 심층 분석기 (v4: 용어사전 연계)"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
        model: str = "solar-pro3",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._glossary = NuclearGlossaryManager()
        self._glossary_context = self._glossary.to_vlm_context()

    async def analyze_visual(
        self,
        image_description: str,
        slide_context: str = "",
        surrounding_text: str = "",
    ) -> dict:
        """
        Document Parse Enhanced에서 추출된 시각자료 설명을 심층 분석

        Args:
            image_description: Document Parse Enhanced가 생성한 이미지 설명
            slide_context: 해당 슬라이드의 텍스트 맥락
            surrounding_text: 이미지 전후 텍스트

        Returns:
            구조화된 시각자료 분석 결과
        """
        user_prompt = f"""## 시각자료 정보
Document Parse 추출 설명: {image_description}

## 슬라이드 맥락
{slide_context}

## 전후 텍스트
{surrounding_text}

{self._glossary_context}

위 시각자료를 원자력 교육 관점에서 분석하고, JSON 형식으로 교육용 설명을 생성해주세요.
용어사전의 약어와 정식 명칭을 참조하여 구성요소를 정확하게 식별하세요."""

        async with httpx.AsyncClient(timeout=60.0, verify=get_ssl_verify()) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": get_template_manager().get_prompt("vlm_analysis")},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": get_template_manager().get_temperature("vlm_analysis"),
                    "max_tokens": get_template_manager().get_max_tokens("vlm_analysis"),
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)

    async def enrich_slides_with_vlm(
        self,
        slides: list[dict],
    ) -> list[dict]:
        """
        파싱된 슬라이드의 모든 시각자료를 VLM으로 보강

        복잡한 시각자료가 있는 슬라이드만 선별하여 VLM 분석 수행
        """
        enriched = []
        for slide in slides:
            enriched_slide = {**slide}

            if not slide["metadata"].get("has_complex_visuals"):
                enriched.append(enriched_slide)
                continue

            slide_text = f"{slide.get('title', '')} {slide.get('content', '')}"
            vlm_analyses = []

            # 이미지 분석
            for img in slide.get("images", []):
                if img.get("description"):
                    analysis = await self.analyze_visual(
                        image_description=img["description"],
                        slide_context=slide_text,
                    )
                    vlm_analyses.append(analysis)

            # 차트 분석
            for chart in slide.get("charts", []):
                if chart.get("description") or chart.get("data"):
                    desc = chart.get("description", "")
                    if chart.get("data"):
                        desc += f"\n데이터: {json.dumps(chart['data'], ensure_ascii=False)[:500]}"
                    analysis = await self.analyze_visual(
                        image_description=desc,
                        slide_context=slide_text,
                    )
                    vlm_analyses.append(analysis)

            enriched_slide["vlm_analyses"] = vlm_analyses
            enriched_slide["metadata"]["vlm_enriched"] = True
            enriched.append(enriched_slide)

        return enriched


class GoldenContextBuilder:
    """
    Golden Context 방식: 참고자료를 함께 파싱하여 LLM 컨텍스트 구성

    PoC 단계에서 풀스케일 RAG 대신 사용하는 경량 전략
    """

    def __init__(self, parser):
        """
        Args:
            parser: DocumentParser 인스턴스
        """
        self.parser = parser

    async def build_context(
        self,
        reference_files: list[str],
        max_context_chars: int = 30000,
    ) -> str:
        """
        참고자료 파일들을 파싱하여 컨텍스트 텍스트로 변환

        Args:
            reference_files: 참고자료 파일 경로 리스트
            max_context_chars: 최대 컨텍스트 길이

        Returns:
            구조화된 컨텍스트 텍스트
        """
        context_parts = []
        remaining_chars = max_context_chars

        for file_path in reference_files:
            if remaining_chars <= 0:
                break

            parse_result = await self.parser.parse_document(
                file_path, mode="auto", output_format="text"
            )
            slides = self.parser.parse_result_to_slides(parse_result)

            file_text = f"\n[참고자료: {file_path}]\n"
            for slide in slides:
                page_text = ""
                if slide.get("title"):
                    page_text += f"## {slide['title']}\n"
                if slide.get("content"):
                    page_text += f"{slide['content']}\n"
                for table in slide.get("tables", []):
                    page_text += f"[표] {table}\n"

                if len(file_text) + len(page_text) > remaining_chars:
                    break
                file_text += page_text

            context_parts.append(file_text[:remaining_chars])
            remaining_chars -= len(file_text)

        return "\n\n".join(context_parts)

    def build_curated_prompt(
        self,
        sat_guide: str = "",
        bloom_guide: str = "",
        terminology: dict = None,
        example_scripts: list[str] = None,
        example_questions: list[dict] = None,
        question_guidelines: str = "",
    ) -> str:
        """
        Curated System Prompt 구성

        SME로부터 확보한 도메인 지식을 프롬프트에 내장

        Args:
            sat_guide: SAT 원칙 가이드 텍스트
            bloom_guide: Bloom's Taxonomy 가이드
            terminology: 핵심 용어 사전 {용어: 정의}
            example_scripts: 모범 스크립트 예시
            example_questions: 모범 문항 예시
            question_guidelines: 문항 작성 기준

        Returns:
            조합된 시스템 프롬프트 보강 텍스트
        """
        parts = []

        if sat_guide:
            parts.append(f"## SAT 교수설계 원칙\n{sat_guide}")

        if bloom_guide:
            parts.append(f"## Bloom's Taxonomy 가이드\n{bloom_guide}")

        if terminology:
            term_text = "\n".join(
                f"- **{term}**: {defn}" for term, defn in terminology.items()
            )
            parts.append(f"## 원자력 핵심 용어\n{term_text}")

        if example_scripts:
            for i, script in enumerate(example_scripts, 1):
                parts.append(f"## 모범 스크립트 예시 {i}\n{script}")

        if example_questions:
            q_text = json.dumps(example_questions, ensure_ascii=False, indent=2)
            parts.append(f"## 모범 문항 예시\n{q_text}")

        if question_guidelines:
            parts.append(f"## 문항 작성 기준\n{question_guidelines}")

        return "\n\n---\n\n".join(parts)
