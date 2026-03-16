"""
Upstage Document Parse를 활용한 강의안 파싱 모듈
- Standard mode: 일반 텍스트/레이아웃 추출
- Enhanced mode: 복잡한 시각자료(도면, 차트, P&ID) 해석
"""
import json
import os
from pathlib import Path
from typing import Optional

import httpx

from src.utils.http_client import get_ssl_verify


class DocumentParser:
    """Upstage Document Parse API 래퍼"""

    def __init__(self, api_key: str, base_url: str = "https://api.upstage.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def parse_document(
        self,
        file_path: str,
        mode: str = "auto",  # standard | enhanced | auto
        output_format: str = "html",
        coordinates: bool = False,
        chart_recognition: bool = True,
    ) -> dict:
        """
        강의안 파일을 파싱하여 구조화된 데이터로 변환

        Upstage Document Parse API (v1/document-digitization)
        - model: "document-parse" (latest stable)
        - mode: "standard" | "enhanced" | "auto"
        - output_formats: ["html"] | ["text"] | ["markdown"]
        - 동기 API: 최대 100페이지, 타임아웃 5분

        Args:
            file_path: PDF/PPTX 파일 경로
            mode: 파싱 모드 (auto 권장 - 페이지별 복잡도 자동 판단)
            output_format: 출력 형식 (html | text | markdown)
            coordinates: 바운딩 박스 좌표 포함 여부
            chart_recognition: 차트 → 표 변환 활성화

        Returns:
            파싱 결과 (페이지별 구조화 데이터)
        """
        url = f"{self.base_url}/document-digitization"

        # MIME 타입 추론
        suffix = Path(file_path).suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        mime_type = mime_map.get(suffix, "application/octet-stream")

        async with httpx.AsyncClient(timeout=300.0, verify=get_ssl_verify()) as client:
            with open(file_path, "rb") as f:
                files = {"document": (Path(file_path).name, f, mime_type)}
                data = {
                    "model": "document-parse",
                    "mode": mode,
                    "output_formats": json.dumps([output_format]),
                    "ocr": "auto",
                    "coordinates": str(coordinates).lower(),
                    "chart_recognition": str(chart_recognition).lower(),
                }

                response = await client.post(
                    url, headers=self.headers, files=files, data=data
                )
                response.raise_for_status()
                return response.json()

    def parse_result_to_slides(self, parse_result: dict) -> list[dict]:
        """
        파싱 결과를 슬라이드 단위로 구조화

        Returns:
            [
                {
                    "page_number": 1,
                    "title": "슬라이드 제목",
                    "content": "본문 내용 (HTML)",
                    "tables": [...],
                    "images": [...],
                    "charts": [...],
                    "equations": [...],
                    "metadata": {
                        "has_complex_visuals": True,
                        "element_count": 5
                    }
                },
                ...
            ]
        """
        slides = []
        elements = parse_result.get("elements", [])

        current_page = None
        current_slide = None

        for element in elements:
            page = element.get("page", 1)

            if page != current_page:
                if current_slide:
                    slides.append(current_slide)
                current_page = page
                current_slide = {
                    "page_number": page,
                    "title": "",
                    "content": "",
                    "tables": [],
                    "images": [],
                    "charts": [],
                    "equations": [],
                    "metadata": {"has_complex_visuals": False, "element_count": 0},
                }

            if current_slide is None:
                continue

            category = element.get("category", "paragraph")
            html_content = element.get("html", element.get("text", ""))

            if category in ("heading", "heading1"):
                if not current_slide["title"]:
                    current_slide["title"] = html_content
                else:
                    current_slide["content"] += f"\n{html_content}"
            elif category == "table":
                current_slide["tables"].append(html_content)
                current_slide["metadata"]["has_complex_visuals"] = True
            elif category in ("figure", "image"):
                current_slide["images"].append({
                    "description": element.get("description", ""),
                    "html": html_content,
                })
                current_slide["metadata"]["has_complex_visuals"] = True
            elif category == "chart":
                current_slide["charts"].append({
                    "description": element.get("description", ""),
                    "data": element.get("data", {}),
                })
                current_slide["metadata"]["has_complex_visuals"] = True
            elif category == "equation":
                current_slide["equations"].append(html_content)
            else:
                current_slide["content"] += f"\n{html_content}"

            current_slide["metadata"]["element_count"] += 1

        if current_slide:
            slides.append(current_slide)

        return slides

    async def save_parsed_result(
        self, slides: list[dict], output_path: str
    ) -> str:
        """파싱 결과를 JSON으로 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(slides, f, ensure_ascii=False, indent=2)
        return output_path
