"""
원자력 전문 용어사전 모듈

Curated Prompt에 주입되는 원자력 도메인 용어 및 약어 사전.
SME가 검수한 정의와 현장 용어 매핑을 관리.

용도:
1. LLM 프롬프트에 용어 정의 주입 → 생성 정확도 향상
2. 문항 QA에서 용어 일관성 검증
3. VLM 분석 시 도면 구성요소 식별 보조
4. 교육생용 용어 해설 자동 생성

구조:
- 계통별 분류 (RCS, ECCS, 전원계통 등)
- 한글/영문/약어 3중 매핑
- 안전등급 연계 (safety_critical/related/general)
- IAEA/KINS/KEPCO 표준 용어 기반
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GlossaryEntry:
    """용어사전 항목"""
    term_ko: str                    # 한글 용어
    term_en: str = ""               # 영문 용어
    abbreviation: str = ""          # 약어
    definition: str = ""            # 정의
    system_category: str = ""       # 계통 분류
    safety_grade: str = "general"   # safety_critical | safety_related | general
    field_usage: str = ""           # 현장에서 실제 사용하는 표현
    related_terms: list = field(default_factory=list)  # 관련 용어
    iaea_reference: str = ""        # IAEA 표준 문서 참조
    sme_verified: bool = False      # SME 검증 여부


# ============================================================
# 기본 용어사전 (시드 데이터)
# ============================================================

NUCLEAR_GLOSSARY: list[dict] = [
    # === 1차 냉각재계통 (RCS) ===
    {
        "term_ko": "원자로냉각재계통",
        "term_en": "Reactor Coolant System",
        "abbreviation": "RCS",
        "definition": "원자로에서 발생한 열을 증기발생기로 전달하는 1차측 순환계통. 원자로용기, 냉각재펌프, 가압기, 증기발생기, 배관으로 구성.",
        "system_category": "1차계통",
        "safety_grade": "safety_critical",
        "field_usage": "1차측, 1차계통",
        "related_terms": ["가압기", "RCP", "증기발생기"],
    },
    {
        "term_ko": "가압기",
        "term_en": "Pressurizer",
        "abbreviation": "PZR",
        "definition": "1차 냉각재계통의 압력을 제어하는 장치. 전열기(heater)와 살수(spray)로 압력 조절. 정상 운전 시 155 kg/cm² 유지.",
        "system_category": "1차계통",
        "safety_grade": "safety_critical",
        "field_usage": "가압기, PZR",
        "related_terms": ["가압기 안전밸브", "PORV", "가압기 수위"],
    },
    {
        "term_ko": "원자로냉각재펌프",
        "term_en": "Reactor Coolant Pump",
        "abbreviation": "RCP",
        "definition": "1차 냉각재를 강제 순환시키는 펌프. 루프당 1대, 총 2~4대 설치. 정전 시 관성효과로 유량 유지(flywheel).",
        "system_category": "1차계통",
        "safety_grade": "safety_critical",
        "related_terms": ["RCS", "강제순환", "자연순환"],
    },
    {
        "term_ko": "증기발생기",
        "term_en": "Steam Generator",
        "abbreviation": "SG",
        "definition": "1차측 고온 냉각재의 열을 2차측 급수로 전달하여 증기를 생산하는 열교환기. U-tube형 수천개의 세관으로 구성.",
        "system_category": "1차계통",
        "safety_grade": "safety_critical",
        "field_usage": "SG, 증발기",
        "related_terms": ["세관파손", "SGTR", "2차측"],
    },

    # === 비상노심냉각계통 (ECCS) ===
    {
        "term_ko": "비상노심냉각계통",
        "term_en": "Emergency Core Cooling System",
        "abbreviation": "ECCS",
        "definition": "냉각재상실사고(LOCA) 시 원자로 노심에 비상 냉각수를 주입하는 안전계통. 고압주입/저압주입/축압기(SIT)로 구성.",
        "system_category": "안전계통",
        "safety_grade": "safety_critical",
        "related_terms": ["안전주입", "SIS", "LOCA", "SIT"],
    },
    {
        "term_ko": "안전주입계통",
        "term_en": "Safety Injection System",
        "abbreviation": "SIS",
        "definition": "LOCA 감지 시 자동 작동하여 고압으로 붕산수를 원자로에 주입하는 계통. 안전주입(SI) 신호에 의해 기동.",
        "system_category": "안전계통",
        "safety_grade": "safety_critical",
        "related_terms": ["ECCS", "안전주입신호", "RWST"],
    },
    {
        "term_ko": "보조급수계통",
        "term_en": "Auxiliary Feedwater System",
        "abbreviation": "AFWS",
        "definition": "주급수 상실 시 증기발생기에 급수를 공급하는 안전계통. 터빈구동 펌프(1대) + 전동구동 펌프(2대) 구성.",
        "system_category": "안전계통",
        "safety_grade": "safety_critical",
        "related_terms": ["주급수", "SG", "ATWS"],
    },
    {
        "term_ko": "비상디젤발전기",
        "term_en": "Emergency Diesel Generator",
        "abbreviation": "EDG",
        "definition": "소내전원 상실(LOOP) 시 안전계통에 비상전원을 공급하는 디젤발전기. LOOP 또는 SI 신호로 자동 기동, 10초 내 전압 확립.",
        "system_category": "전원계통",
        "safety_grade": "safety_critical",
        "field_usage": "비디발, EDG",
        "related_terms": ["LOOP", "SBO", "안전모선"],
    },

    # === 원자로보호계통 ===
    {
        "term_ko": "원자로보호계통",
        "term_en": "Reactor Protection System",
        "abbreviation": "RPS",
        "definition": "원자로 운전변수가 설정치를 초과하면 자동으로 원자로를 정지(Trip)시키는 계통. 2/4 일치논리 적용.",
        "system_category": "계측제어",
        "safety_grade": "safety_critical",
        "related_terms": ["원자로트립", "ESFAS", "AMSAC"],
    },
    {
        "term_ko": "공학적안전설비작동계통",
        "term_en": "Engineered Safety Features Actuation System",
        "abbreviation": "ESFAS",
        "definition": "사고 감지 시 안전주입, 격납건물 격리, 보조급수 등 공학적안전설비를 자동 작동시키는 계통.",
        "system_category": "계측제어",
        "safety_grade": "safety_critical",
        "related_terms": ["RPS", "SI", "격납건물격리"],
    },

    # === 사고/비상 용어 ===
    {
        "term_ko": "냉각재상실사고",
        "term_en": "Loss of Coolant Accident",
        "abbreviation": "LOCA",
        "definition": "1차 냉각재계통 배관 파단으로 냉각재가 유출되는 사고. 파단 크기에 따라 대형/중형/소형 LOCA로 분류.",
        "system_category": "사고분류",
        "safety_grade": "safety_critical",
        "related_terms": ["ECCS", "EOP", "파단"],
    },
    {
        "term_ko": "증기발생기세관파손",
        "term_en": "Steam Generator Tube Rupture",
        "abbreviation": "SGTR",
        "definition": "SG 세관이 파손되어 1차측 냉각재가 2차측으로 누출되는 사고. 2차측 방사능 상승이 핵심 진단 지표.",
        "system_category": "사고분류",
        "safety_grade": "safety_critical",
        "related_terms": ["SG", "1차/2차 경계", "방사능"],
    },
    {
        "term_ko": "소내정전",
        "term_en": "Station Blackout",
        "abbreviation": "SBO",
        "definition": "외부전원과 비상전원(EDG)이 모두 상실된 상태. 축전지와 터빈구동 보조급수만으로 안전기능 유지 필요.",
        "system_category": "사고분류",
        "safety_grade": "safety_critical",
        "related_terms": ["EDG", "LOOP", "축전지", "SG"],
    },
    {
        "term_ko": "비상운전절차서",
        "term_en": "Emergency Operating Procedure",
        "abbreviation": "EOP",
        "definition": "비상 상황 발생 시 운전원이 따라야 할 표준화된 절차서. 증상 기반(Symptom-based) 접근법 적용.",
        "system_category": "절차서",
        "safety_grade": "safety_critical",
        "related_terms": ["E-0", "ES-", "FR-", "ECA-"],
    },

    # === 안전 관련 보조계통 ===
    {
        "term_ko": "화학및체적제어계통",
        "term_en": "Chemical and Volume Control System",
        "abbreviation": "CVCS",
        "definition": "1차 냉각재의 화학적 조성(붕소 농도) 제어 및 체적(수위) 조절을 담당하는 계통.",
        "system_category": "보조계통",
        "safety_grade": "safety_related",
        "field_usage": "CVCS, 화체계",
        "related_terms": ["붕소", "충전", "감압"],
    },
    {
        "term_ko": "기기냉각수계통",
        "term_en": "Component Cooling Water System",
        "abbreviation": "CCW",
        "definition": "안전 관련 기기에 냉각수를 공급하는 중간 냉각계통. 방사성 물질의 환경 유출을 방지하는 경계 역할.",
        "system_category": "보조계통",
        "safety_grade": "safety_related",
        "related_terms": ["ESW", "열교환기"],
    },
    {
        "term_ko": "잔열제거계통",
        "term_en": "Residual Heat Removal System",
        "abbreviation": "RHRS",
        "definition": "원자로 정지 후 노심의 잔열(붕괴열)을 제거하는 계통. 저온정지(CSD) 달성 및 유지에 필수.",
        "system_category": "안전계통",
        "safety_grade": "safety_critical",
        "related_terms": ["붕괴열", "CSD", "냉각"],
    },

    # === 규제/기준 용어 ===
    {
        "term_ko": "운영제한조건",
        "term_en": "Limiting Conditions for Operation",
        "abbreviation": "LCO",
        "definition": "발전소 운전 중 충족해야 하는 최소 안전 조건. 위반 시 조치시간 내 복구 또는 출력 감발/정지 필요.",
        "system_category": "규제",
        "safety_grade": "safety_critical",
        "field_usage": "기술기준, LCO",
        "related_terms": ["기술사양서", "조치시간", "감시시험"],
    },
    {
        "term_ko": "안전문화",
        "term_en": "Safety Culture",
        "abbreviation": "",
        "definition": "조직과 개인이 안전을 최우선으로 인식하고 행동하는 특성과 태도의 집합. IAEA INSAG-4에서 정의.",
        "system_category": "조직",
        "safety_grade": "safety_related",
        "related_terms": ["INSAG-4", "방어심층", "인적오류"],
    },
    {
        "term_ko": "방어심층",
        "term_en": "Defence in Depth",
        "abbreviation": "DiD",
        "definition": "다중 방벽과 다중 안전계통으로 방사성물질 유출을 방지하는 원칙. 5단계: 예방→감시→안전계통→사고관리→비상대응.",
        "system_category": "안전원칙",
        "safety_grade": "safety_critical",
        "related_terms": ["다중성", "다양성", "독립성"],
    },

    # === 일반 운전 용어 ===
    {
        "term_ko": "반응도",
        "term_en": "Reactivity",
        "abbreviation": "ρ",
        "definition": "원자로 핵분열 반응의 증감을 나타내는 물리량. 양의 반응도: 출력 증가, 음의 반응도: 출력 감소. 단위: pcm, dk/k.",
        "system_category": "원자로물리",
        "safety_grade": "safety_critical",
        "related_terms": ["임계", "제어봉", "붕소"],
    },
    {
        "term_ko": "임계",
        "term_en": "Criticality",
        "abbreviation": "",
        "definition": "핵분열 연쇄반응이 자기 지속적으로 유지되는 상태 (keff = 1). 미임계(keff < 1), 초임계(keff > 1).",
        "system_category": "원자로물리",
        "safety_grade": "safety_critical",
        "related_terms": ["반응도", "keff", "제어봉"],
    },
    {
        "term_ko": "붕소",
        "term_en": "Boron",
        "abbreviation": "B-10",
        "definition": "중성자를 흡수하여 반응도를 제어하는 물질. 1차 냉각재에 붕산(H3BO3) 형태로 용해. 농도 조절로 장기 반응도 제어.",
        "system_category": "원자로물리",
        "safety_grade": "safety_related",
        "related_terms": ["CVCS", "반응도", "붕소희석"],
    },
]


class NuclearGlossaryManager:
    """
    원자력 용어사전 관리자

    JSON 파일 기반 (PoC). 커스텀 용어 추가/수정 지원.
    """

    def __init__(self, storage_dir: str = "data/glossary"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        self._load()

    def _file_path(self) -> Path:
        return self.storage_dir / "nuclear_glossary.json"

    def _load(self):
        fpath = self._file_path()
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                self._entries = json.load(f)
        else:
            # 기본 사전 초기화
            self._entries = NUCLEAR_GLOSSARY
            self._save()

    def _save(self):
        with open(self._file_path(), "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        safety_grade: Optional[str] = None,
    ) -> list[dict]:
        """용어 검색"""
        results = []
        q = query.lower()
        for entry in self._entries:
            if category and entry.get("system_category") != category:
                continue
            if safety_grade and entry.get("safety_grade") != safety_grade:
                continue
            searchable = " ".join([
                entry.get("term_ko", ""),
                entry.get("term_en", ""),
                entry.get("abbreviation", ""),
                entry.get("definition", ""),
            ]).lower()
            if q in searchable:
                results.append(entry)
        return results

    def get_by_abbreviation(self, abbr: str) -> Optional[dict]:
        """약어로 용어 조회"""
        for entry in self._entries:
            if entry.get("abbreviation", "").upper() == abbr.upper():
                return entry
        return None

    def get_by_system(self, system_category: str) -> list[dict]:
        """계통별 용어 목록"""
        return [e for e in self._entries if e.get("system_category") == system_category]

    def get_safety_critical_terms(self) -> list[dict]:
        """안전 핵심 용어만 추출"""
        return [e for e in self._entries if e.get("safety_grade") == "safety_critical"]

    def add_entry(self, entry: dict):
        """용어 추가 (중복 시 업데이트)"""
        for i, existing in enumerate(self._entries):
            if (existing.get("term_ko") == entry.get("term_ko") or
                (existing.get("abbreviation") and
                 existing.get("abbreviation") == entry.get("abbreviation"))):
                self._entries[i] = entry
                self._save()
                return
        self._entries.append(entry)
        self._save()

    def to_prompt_text(
        self,
        category: Optional[str] = None,
        safety_grade: Optional[str] = None,
        max_entries: int = 50,
    ) -> str:
        """
        Curated Prompt용 용어사전 텍스트 생성

        LLM 프롬프트에 직접 주입할 수 있는 포맷으로 변환
        """
        entries = self._entries
        if category:
            entries = [e for e in entries if e.get("system_category") == category]
        if safety_grade:
            entries = [e for e in entries if e.get("safety_grade") == safety_grade]

        entries = entries[:max_entries]

        lines = ["## 원자력 핵심 용어사전\n"]
        current_category = ""
        for entry in sorted(entries, key=lambda x: x.get("system_category", "")):
            cat = entry.get("system_category", "기타")
            if cat != current_category:
                current_category = cat
                lines.append(f"\n### {cat}")

            abbr = f" ({entry['abbreviation']})" if entry.get("abbreviation") else ""
            en = f" [{entry['term_en']}]" if entry.get("term_en") else ""
            field = f" (현장: {entry['field_usage']})" if entry.get("field_usage") else ""
            grade_mark = " ⚠️" if entry.get("safety_grade") == "safety_critical" else ""

            lines.append(
                f"- **{entry['term_ko']}{abbr}**{en}{field}{grade_mark}: "
                f"{entry.get('definition', '')}"
            )

        return "\n".join(lines)

    def to_vlm_context(self, system_categories: list[str] = None) -> str:
        """
        VLM 분석용 도메인 컨텍스트 생성

        VLM이 원자력 도면의 구성요소를 더 정확하게 식별하도록
        관련 계통/설비 용어를 컨텍스트로 제공
        """
        entries = self._entries
        if system_categories:
            entries = [
                e for e in entries
                if e.get("system_category") in system_categories
            ]

        lines = ["## 원자력 설비/계통 용어 참조\n"]
        lines.append("아래 용어를 참조하여 도면의 구성요소를 정확하게 식별하세요.\n")

        for entry in entries:
            abbr = entry.get("abbreviation", "")
            ko = entry.get("term_ko", "")
            en = entry.get("term_en", "")
            parts = [ko]
            if abbr:
                parts.append(f"({abbr})")
            if en:
                parts.append(f"[{en}]")
            lines.append(f"- {' '.join(parts)}")

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """용어사전 통계"""
        total = len(self._entries)
        by_category = {}
        by_grade = {}
        verified = 0
        for e in self._entries:
            cat = e.get("system_category", "기타")
            by_category[cat] = by_category.get(cat, 0) + 1
            grade = e.get("safety_grade", "general")
            by_grade[grade] = by_grade.get(grade, 0) + 1
            if e.get("sme_verified"):
                verified += 1
        return {
            "total_terms": total,
            "by_category": by_category,
            "by_safety_grade": by_grade,
            "sme_verified": verified,
            "verification_rate": round(verified / max(total, 1) * 100, 1),
        }
