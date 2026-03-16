"""
안전 중요도 태깅 시스템 (IAEA TECDOC-2082 Graded Approach)

원자력 교육 문항에 안전 중요도 등급을 부여하여,
안전 핵심 영역은 강화 검증 / 일반 영역은 표준 검증 적용.

등급 체계:
- safety_critical: 원자로 안전계통, 비상운전, 방사선 방호 관련
- safety_related: 안전 관련 보조계통, 기술기준 준수 사항
- general: 일반 운전, 정비, 관리 업무

IAEA 근거:
- TECDOC-2082 (2025): "평가의 엄격도는 안전 중요도에 비례해야 한다"
- 안전 핵심 문항은 QA 검증 강화, 오답 매력도 기준 상향
"""
import json
from typing import Optional

import httpx

from src.utils.http_client import get_ssl_verify


# 안전 중요도 키워드 사전
SAFETY_KEYWORDS = {
    "safety_critical": {
        "systems": [
            "비상노심냉각계통", "ECCS", "격납건물", "containment",
            "원자로보호계통", "RPS", "안전주입계통", "SIS",
            "보조급수계통", "AFWS", "잔열제거계통", "RHRS",
            "비상디젤발전기", "EDG", "안전밸브", "가압기",
        ],
        "topics": [
            "LOCA", "냉각재상실", "증기발생기세관파손", "SGTR",
            "정전", "SBO", "비상운전절차서", "EOP",
            "방사선비상", "방사선방호", "피폭", "선량한도",
            "노심손상", "핵연료손상", "임계", "반응도",
        ],
        "regulations": [
            "운영제한조건", "기술기준", "안전등급", "1E등급",
            "내진설계", "단일고장기준",
        ],
    },
    "safety_related": {
        "systems": [
            "화학 및 체적 제어계통", "CVCS", "냉각재정화계통",
            "기기냉각수계통", "CCW", "필수냉각수계통", "ESW",
            "환기계통", "HVAC", "전원계통",
            "계측제어계통", "I&C",
        ],
        "topics": [
            "비정상운전", "기기고장", "경보대응",
            "정기시험", "감시시험", "작업허가",
            "방사성폐기물", "수화학", "부식",
            "격리", "정비절차", "안전점검",
        ],
        "regulations": [
            "예방정비", "안전문화", "인적오류",
            "사건보고", "시정조치",
        ],
    },
}

# 안전 등급별 QA 기준
SAFETY_GRADE_QA_CRITERIA = {
    "safety_critical": {
        "min_quality_score": 85,
        "require_two_pass": True,
        "require_sme_review": True,
        "max_nonfunctional_distractors": 0,
        "require_reference_verification": True,
        "revalidation_interval_months": 6,
    },
    "safety_related": {
        "min_quality_score": 75,
        "require_two_pass": True,
        "require_sme_review": True,
        "max_nonfunctional_distractors": 1,
        "require_reference_verification": True,
        "revalidation_interval_months": 12,
    },
    "general": {
        "min_quality_score": 65,
        "require_two_pass": False,
        "require_sme_review": False,
        "max_nonfunctional_distractors": 1,
        "require_reference_verification": False,
        "revalidation_interval_months": 24,
    },
}


class SafetySignificanceTagger:
    """문항에 안전 중요도 등급을 자동 부여"""

    @staticmethod
    def tag_question(question: dict) -> dict:
        """
        문항의 텍스트를 분석하여 안전 중요도 등급 부여

        Returns:
            question dict에 safety_significance 관련 필드 추가
        """
        # 문항 텍스트 추출
        text_parts = [
            question.get("question_text", ""),
            question.get("scenario", ""),
            question.get("explanation", ""),
            " ".join(question.get("keywords", [])),
        ]
        # 옵션 텍스트도 포함
        options = question.get("options", {})
        if isinstance(options, dict):
            text_parts.extend(options.values())
        combined_text = " ".join(str(p) for p in text_parts).lower()

        # 키워드 매칭으로 등급 판정
        critical_hits = []
        related_hits = []

        for category, keywords in SAFETY_KEYWORDS["safety_critical"].items():
            for kw in keywords:
                if kw.lower() in combined_text:
                    critical_hits.append(kw)

        for category, keywords in SAFETY_KEYWORDS["safety_related"].items():
            for kw in keywords:
                if kw.lower() in combined_text:
                    related_hits.append(kw)

        # 등급 결정
        if len(critical_hits) >= 2 or any(
            kw.lower() in combined_text
            for kw in ["LOCA", "비상운전", "노심손상", "방사선비상"]
        ):
            grade = "safety_critical"
            matched_keywords = critical_hits
        elif critical_hits or len(related_hits) >= 2:
            grade = "safety_related"
            matched_keywords = critical_hits + related_hits
        else:
            grade = "general"
            matched_keywords = related_hits

        # 결과 추가
        question["safety_significance"] = {
            "grade": grade,
            "matched_keywords": list(set(matched_keywords))[:10],
            "qa_criteria": SAFETY_GRADE_QA_CRITERIA[grade],
        }

        return question

    @staticmethod
    def tag_batch(questions: list[dict]) -> list[dict]:
        """문항 세트 전체에 안전 중요도 태깅"""
        tagged = []
        for q in questions:
            tagged.append(SafetySignificanceTagger.tag_question(q))
        return tagged

    @staticmethod
    def get_safety_statistics(questions: list[dict]) -> dict:
        """안전 등급별 통계"""
        stats = {"safety_critical": 0, "safety_related": 0, "general": 0}
        for q in questions:
            grade = q.get("safety_significance", {}).get("grade", "general")
            stats[grade] = stats.get(grade, 0) + 1

        total = len(questions)
        return {
            "total": total,
            "distribution": stats,
            "critical_ratio": f"{stats['safety_critical']/max(total,1)*100:.1f}%",
            "safety_related_ratio": f"{stats['safety_related']/max(total,1)*100:.1f}%",
            "general_ratio": f"{stats['general']/max(total,1)*100:.1f}%",
            "qa_recommendation": (
                "안전 핵심 문항이 전체의 30% 이상 — 2-Pass + SME 검토 필수"
                if stats["safety_critical"] / max(total, 1) > 0.3
                else "안전 핵심 문항 비율 적정"
            ),
        }


class SafetyAwareLLMTagger:
    """LLM 기반 정밀 안전 중요도 태깅 (키워드 매칭 보완용)"""

    SAFETY_TAGGING_PROMPT = """당신은 원자력 안전 전문가입니다.
아래 문항의 안전 중요도를 평가해주세요.

## 등급 기준
- **safety_critical**: 원자로 안전기능 직접 관련, 비상운전, 방사선 방호, 노심 보호
  → 해당 문항의 오류가 발전소 안전에 직접 영향을 줄 수 있는 지식/역량 평가
- **safety_related**: 안전 관련 보조기능, 기술기준, 안전문화, 예방정비
  → 간접적으로 안전에 영향, 절차 준수 관련
- **general**: 일반 운전지식, 관리업무, 기초이론
  → 안전과 직접적 관련 없는 일반 직무지식

## 출력 (JSON)
{
    "grade": "safety_critical|safety_related|general",
    "reasoning": "판단 근거 (1~2문장)",
    "safety_functions_involved": ["관련 안전기능 리스트"]
}
"""

    def __init__(self, api_key: str, base_url: str = "https://api.upstage.ai/v1",
                 model: str = "solar-pro"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def tag_with_llm(self, question: dict) -> dict:
        """LLM으로 정밀 안전 중요도 태깅"""
        user_prompt = f"""## 평가 대상 문항
{json.dumps(question, ensure_ascii=False, indent=2)}

위 문항의 안전 중요도를 평가해주세요."""

        async with httpx.AsyncClient(timeout=30.0, verify=get_ssl_verify()) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SAFETY_TAGGING_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 512,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            llm_tag = json.loads(content)

        # 키워드 기반 태깅과 병합 (LLM 결과 우선)
        question["safety_significance"] = {
            "grade": llm_tag.get("grade", "general"),
            "reasoning": llm_tag.get("reasoning", ""),
            "safety_functions": llm_tag.get("safety_functions_involved", []),
            "qa_criteria": SAFETY_GRADE_QA_CRITERIA.get(
                llm_tag.get("grade", "general"), SAFETY_GRADE_QA_CRITERIA["general"]
            ),
            "tagged_by": "llm",
        }
        return question


class HybridSafetyTagger:
    """
    Hybrid 안전 태깅 전략

    1차: 키워드 기반 빠른 태깅 (전체 문항)
    2차: LLM 정밀 태깅 (safety_critical 판정 문항만 → 비용 효율)

    효과:
    - 전체 문항 LLM 태깅 대비 API 호출 ~70% 절감
    - safety_critical 정확도 향상 (키워드 오탐 보정)
    - safety_related ↔ safety_critical 경계 문항 정밀 판별
    """

    def __init__(self, api_key: str, base_url: str = "https://api.upstage.ai/v1",
                 model: str = "solar-pro"):
        self.keyword_tagger = SafetySignificanceTagger()
        self.llm_tagger = SafetyAwareLLMTagger(api_key, base_url, model)

    async def tag_batch_hybrid(
        self,
        questions: list[dict],
        llm_threshold: str = "safety_critical",
    ) -> dict:
        """
        Hybrid 배치 태깅

        Args:
            questions: 문항 리스트
            llm_threshold: LLM 2차 검증 대상
                - "safety_critical": critical만 LLM 검증 (기본, 비용 효율)
                - "safety_related": related 이상 LLM 검증
                - "all": 전체 LLM 검증 (비용 높음)

        Returns:
            {questions, stats, llm_overrides}
        """
        # 1차: 키워드 태깅 (전체)
        keyword_tagged = self.keyword_tagger.tag_batch(questions)

        # LLM 검증 대상 필터
        llm_targets = []
        if llm_threshold == "all":
            llm_targets = keyword_tagged
        elif llm_threshold == "safety_related":
            llm_targets = [
                q for q in keyword_tagged
                if q.get("safety_significance", {}).get("grade")
                in ("safety_critical", "safety_related")
            ]
        else:  # safety_critical (기본)
            llm_targets = [
                q for q in keyword_tagged
                if q.get("safety_significance", {}).get("grade") == "safety_critical"
            ]

        # 2차: LLM 정밀 태깅
        llm_overrides = []
        for q in llm_targets:
            keyword_grade = q.get("safety_significance", {}).get("grade", "general")
            try:
                q = await self.llm_tagger.tag_with_llm(q)
                llm_grade = q.get("safety_significance", {}).get("grade", "general")
                if keyword_grade != llm_grade:
                    llm_overrides.append({
                        "question_id": q.get("question_id", ""),
                        "keyword_grade": keyword_grade,
                        "llm_grade": llm_grade,
                        "reasoning": q.get("safety_significance", {}).get("reasoning", ""),
                    })
            except Exception:
                # LLM 실패 시 키워드 결과 유지
                q["safety_significance"]["tagged_by"] = "keyword_only"

        stats = self.keyword_tagger.get_safety_statistics(keyword_tagged)
        stats["llm_verified_count"] = len(llm_targets)
        stats["llm_override_count"] = len(llm_overrides)

        return {
            "questions": keyword_tagged,
            "stats": stats,
            "llm_overrides": llm_overrides,
        }
