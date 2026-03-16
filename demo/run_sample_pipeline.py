"""
샘플 PPT E2E 파이프라인 데모

API 서버 없이 11단계 파이프라인을 Mock 모드로 실행하여
생성 결과를 JSON 파일로 출력합니다.

사용법:
    python3 demo/run_sample_pipeline.py

결과물:
    demo/output/scripts_*.json     - 슬라이드별 강의 스크립트
    demo/output/questions_*.json   - 문제은행 (2-Pass + 시나리오)
    demo/output/quality_*.json     - 품질 검증 보고서
    demo/output/metadata_*.json    - 파이프라인 메타데이터
    demo/output/summary.txt        - 사람이 읽을 수 있는 요약
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.question_generator.safety_tagger import SafetySignificanceTagger
from src.question_generator.quality_validator import ItemAnalysisSimulator
from src.question_generator.version_manager import QuestionVersionManager
from src.question_generator.qti_exporter import QTIExporter
from src.domain.nuclear_glossary import NuclearGlossaryManager
from src.prompts.template_manager import PromptTemplateManager

# ============================================================
# 샘플 강의안 데이터 (원자력 안전계통 교육)
# ============================================================

SAMPLE_SLIDES = [
    {
        "page_number": 1,
        "title": "원자로냉각재계통(RCS) 개요",
        "content": (
            "원자로냉각재계통(RCS: Reactor Coolant System)은 원자로에서 발생한 핵분열 에너지를 "
            "1차 냉각재(경수)를 통해 증기발생기로 전달하는 핵심 계통입니다.\n\n"
            "주요 구성요소:\n"
            "- 원자로용기(RPV): 핵연료 집합체 수납, 설계압력 2,500 psia\n"
            "- 가압기(PZR): 1차 계통 압력/온도 제어, 정상 운전압력 2,235 psia\n"
            "- 증기발생기(SG): 1차-2차 열교환, 4대 (2-Loop 기준)\n"
            "- 원자로냉각재펌프(RCP): 냉각재 강제순환, 4대\n"
            "- 배관: Hot Leg(고온관), Cold Leg(저온관)"
        ),
        "images": [{"description": "RCS 계통도 — 원자로, 가압기, 증기발생기, RCP 배치도. 냉각재 흐름 방향 화살표 표시."}],
        "charts": [],
        "tables": [["구성요소", "설계압력(psia)", "설계온도(°F)", "수량"],
                   ["원자로용기", "2,500", "650", "1"],
                   ["가압기", "2,500", "680", "1"],
                   ["증기발생기", "2,500 (1차측)", "650", "4"],
                   ["RCP", "2,500", "650", "4"]],
        "metadata": {"has_complex_visuals": True},
    },
    {
        "page_number": 2,
        "title": "가압기(PZR) 구조 및 기능",
        "content": (
            "가압기는 RCS의 압력과 온도를 제어하는 핵심 설비입니다.\n\n"
            "주요 기능:\n"
            "1. 압력 제어: 전기 히터(가열) + 살수(냉각)로 포화 조건 유지\n"
            "2. 과도 상태 완충: 부하 변동 시 체적 변화 흡수 (서지 기능)\n"
            "3. 과압 방호: 안전밸브(SRV) 3대, 설정압력 2,485 psia\n\n"
            "운전 파라미터:\n"
            "- 정상 운전압력: 2,235 psia\n"
            "- 정상 수위: 60% (프로그램 수위)\n"
            "- 히터 용량: 1,800 kW (비례히터 + On/Off 히터)\n"
            "- 살수 유량: 최대 300 gpm"
        ),
        "images": [{"description": "가압기 내부 구조 단면도 — 증기공간, 수공간, 히터, 살수 노즐, 안전밸브 위치 표시"}],
        "charts": [{"description": "가압기 압력-수위 상관관계 그래프. X축: 수위(%), Y축: 압력(psia). 정상 운전 밴드 표시.", "data": {}}],
        "tables": [],
        "metadata": {"has_complex_visuals": True},
    },
    {
        "page_number": 3,
        "title": "비상노심냉각계통(ECCS) 개요",
        "content": (
            "ECCS(Emergency Core Cooling System)는 냉각재상실사고(LOCA) 시 노심을 냉각하여 "
            "핵연료 건전성을 유지하는 안전계통입니다.\n\n"
            "ECCS 구성:\n"
            "- 안전주입계통(SIS): 고압주입(HPSI) + 저압주입(LPSI)\n"
            "  · HPSI 기동조건: PZR 압력 ≤ 1,800 psia\n"
            "  · LPSI 기동조건: PZR 압력 ≤ 200 psia\n"
            "- 안전주입탱크(SIT): 600 psia 질소 가압, 수동 주입\n"
            "- 재순환 모드: RWST 소진 후 격납건물 섬프에서 재순환\n\n"
            "설계 기준: 10CFR50.46 — 연료 피복관 온도 2,200°F 이하 유지"
        ),
        "images": [],
        "charts": [],
        "tables": [["주입 단계", "압력 조건", "유량", "수원"],
                   ["HPSI", "≤1,800 psia", "~500 gpm/펌프", "RWST"],
                   ["LPSI", "≤200 psia", "~3,000 gpm/펌프", "RWST"],
                   ["SIT", "≤600 psia", "~수동", "SIT (N₂ 가압)"],
                   ["재순환", "장기 냉각", "LPSI 유량", "격납건물 섬프"]],
        "metadata": {"has_complex_visuals": False},
    },
    {
        "page_number": 4,
        "title": "LOCA 사고 시나리오 및 ECCS 대응",
        "content": (
            "Large Break LOCA (대구경 배관 파단) 시나리오:\n\n"
            "1. 파단 발생 (Cold Leg 파단 가정)\n"
            "   → RCS 급감압, 격납건물 압력 상승\n"
            "2. 원자로 자동 정지 (PZR 저압력 트립 신호)\n"
            "3. ECCS 자동 기동\n"
            "   · S-신호 (Safety Injection Signal) 발생\n"
            "   · HPSI 펌프 기동 → 고압 주입 시작\n"
            "   · SIT 수동 주입 (RCS 압력 < 600 psia)\n"
            "4. 냉각 → 감압 → 장기 재순환 냉각 진입\n\n"
            "운전원 필수 조치:\n"
            "- E-0 (원자로 트립 또는 안전주입) 진입\n"
            "- 격납건물 격리 확인\n"
            "- RWST 수위 감시 → 재순환 전환 시점 판단"
        ),
        "images": [{"description": "LOCA 시간대별 RCS 압력/온도/ECCS 유량 변화 그래프"}],
        "charts": [{"description": "LOCA 사고 진행 타임라인: 파단→감압→SI→SIT 주입→재순환", "data": {}}],
        "tables": [],
        "metadata": {"has_complex_visuals": True},
    },
    {
        "page_number": 5,
        "title": "학습 정리 및 평가 안내",
        "content": (
            "금일 학습 내용 요약:\n"
            "1. RCS 구성요소와 각각의 기능/설계 파라미터\n"
            "2. 가압기의 압력/수위 제어 원리\n"
            "3. ECCS 구성 및 자동 기동 조건\n"
            "4. LOCA 사고 시나리오와 ECCS 대응 절차\n\n"
            "학습목표 확인:\n"
            "- RCS 주요 구성요소를 식별하고 기능을 설명할 수 있다\n"
            "- 가압기의 압력/수위 제어 방법을 설명할 수 있다\n"
            "- ECCS 자동 기동 조건과 주입 단계를 설명할 수 있다\n"
            "- LOCA 시 운전원 조치 절차를 순서대로 기술할 수 있다\n\n"
            "다음 시간: 잔열제거계통(RHRS) 및 정지냉각 운전"
        ),
        "images": [],
        "charts": [],
        "tables": [],
        "metadata": {"has_complex_visuals": False},
    },
]

SAMPLE_LEARNING_OBJECTIVES = [
    "RCS 주요 구성요소(원자로, 가압기, SG, RCP)를 식별하고 기능을 설명할 수 있다",
    "가압기의 압력/수위 제어 방법(히터, 살수, 안전밸브)을 설명할 수 있다",
    "ECCS 자동 기동 조건(S-신호)과 주입 단계(HPSI, LPSI, SIT, 재순환)를 설명할 수 있다",
    "Large Break LOCA 시 운전원 조치 절차(E-0 진입, 격리, 재순환 전환)를 순서대로 기술할 수 있다",
]

# ============================================================
# Mock VLM 분석 결과
# ============================================================

MOCK_VLM_RESULTS = {
    1: {
        "visual_type": "계통도",
        "title": "RCS 계통 배치도",
        "description": "원자로냉각재계통의 전체 배치를 보여주는 계통도. 원자로용기를 중심으로 4개의 루프(SG+RCP)가 연결되어 있고, 가압기는 Hot Leg에 연결되어 있습니다.",
        "key_components": ["원자로용기(RPV)", "가압기(PZR)", "증기발생기(SG) ×4", "RCP ×4", "Hot Leg", "Cold Leg"],
        "flow_description": "냉각재: 원자로(가열) → Hot Leg → SG(열전달) → Cold Leg → RCP(가압순환) → 원자로",
        "safety_notes": "RCS 건전성은 1차 압력경계로서 방사성물질 격납의 핵심 방벽",
        "safety_grade": "safety_critical",
        "teaching_points": ["냉각재 순환 경로 추적", "가압기 위치와 Hot Leg 연결 이유", "SG에서의 열전달 원리(1차→2차)"],
        "related_systems": ["CVCS(화학체적제어)", "PZR(가압기)", "SG(증기발생기)"],
        "operating_conditions": "정상출력 운전 (Mode 1, 100% Power)",
        "exam_potential": ["RCS 구성요소 식별", "냉각재 흐름 경로", "설계 압력/온도"],
        "difficulty_level": "intermediate",
    },
    2: {
        "visual_type": "도면",
        "title": "가압기 내부 구조 단면도",
        "description": "가압기 내부를 보여주는 단면도. 상부 증기공간, 하부 수공간으로 구분되며, 하부에 전기히터, 상부에 살수 노즐과 안전밸브가 위치합니다.",
        "key_components": ["증기공간", "수공간", "전기히터(1,800kW)", "살수 노즐", "안전밸브(SRV) ×3", "서지라인"],
        "flow_description": "서지라인을 통해 Hot Leg의 냉각재가 유입/유출. 살수는 Cold Leg에서 공급.",
        "safety_notes": "안전밸브는 RCS 과압방호의 마지막 방벽. 설정압력 2,485 psia.",
        "safety_grade": "safety_critical",
        "teaching_points": ["증기-수 공존 상태(포화 조건)", "히터/살수 제어 로직", "안전밸브 설정값"],
        "related_systems": ["RCS", "CVCS(보충수/유출수)"],
        "operating_conditions": "정상 운전: 2,235 psia, 수위 60%",
        "exam_potential": ["가압기 구성요소 식별", "압력 제어 방법", "안전밸브 설정값"],
        "difficulty_level": "intermediate",
    },
    4: {
        "visual_type": "그래프",
        "title": "LOCA 사고 진행 타임라인",
        "description": "Large Break LOCA 발생 후 시간에 따른 RCS 압력, 노심 온도, ECCS 주입 유량 변화를 보여주는 복합 그래프.",
        "key_components": ["RCS 압력 곡선", "노심 온도 곡선", "ECCS 주입 유량", "SIT 주입 시점"],
        "flow_description": "파단 → 급감압(~10초) → SI 기동(~30초) → SIT 주입(~60초) → 재관수 → 장기냉각",
        "safety_notes": "핵심 성능 기준: 피복관 온도 2,200°F 이하 유지 (10CFR50.46)",
        "safety_grade": "safety_critical",
        "teaching_points": ["사고 진행 단계별 특성", "ECCS 주입 타이밍", "규제 기준(2,200°F)"],
        "related_systems": ["ECCS", "격납건물 살수계통"],
        "operating_conditions": "사고 조건 (DBA LOCA)",
        "exam_potential": ["ECCS 기동 시점", "피복관 온도 기준", "재순환 전환 조건"],
        "difficulty_level": "advanced",
    },
}

# ============================================================
# Mock 스크립트 생성 결과
# ============================================================

def generate_mock_scripts(slides, objectives, vlm_results):
    """Mock 스크립트 생성 (실제로는 Solar LLM이 생성)"""
    scripts = []
    prev_summary = ""
    for slide in slides:
        pn = slide["page_number"]
        vlm = vlm_results.get(pn, {})
        vlm_text = ""
        if vlm:
            vlm_text = f"화면에 보이는 {vlm.get('visual_type', '시각자료')}를 함께 살펴보겠습니다. "
            vlm_text += vlm.get("description", "")
            if vlm.get("teaching_points"):
                vlm_text += " 특히 " + ", ".join(vlm["teaching_points"][:2]) + "에 주목해 주세요."

        script = {
            "page_number": pn,
            "slide_title": slide["title"],
            "estimated_minutes": 5 + (2 if vlm else 0),
            "script": {
                "introduction": f"{'지난 시간에 ' + prev_summary + ' 에 대해 배웠습니다. ' if prev_summary else ''}이번에는 {slide['title']}에 대해 알아보겠습니다.",
                "main_content": slide["content"][:300] + "...",
                "visual_explanation": vlm_text or "이 슬라이드에는 복잡한 시각자료가 없으므로 텍스트 중심으로 설명하겠습니다.",
                "contextual_supplement": f"실무에서 {slide['title'].split('(')[0]}은(는) 운전원이 가장 자주 모니터링하는 파라미터 중 하나입니다. 특히 과도 상태에서의 변화 추이를 이해하는 것이 중요합니다.",
                "summary": f"오늘 배운 {slide['title'].split('(')[0]}의 핵심은 안전 기능과 설계 기준값입니다.",
            },
            "teaching_notes": f"학습자에게 {slide['title']}과 관련된 실제 운전 경험을 공유하면 효과적",
            "key_terms": [w for w in slide.get("content", "").split() if len(w) > 3 and any(c.isupper() for c in w)][:5],
            "visual_references": [vlm.get("title", "")] if vlm else [],
        }
        scripts.append(script)
        prev_summary = slide["title"]
    return scripts


# ============================================================
# Mock 문항 생성 결과
# ============================================================

MOCK_GENERATED_QUESTIONS = [
    {
        "question_id": "Q001",
        "bloom_level": "Knowledge",
        "learning_objective": "RCS 주요 구성요소를 식별하고 기능을 설명할 수 있다",
        "question_text": "원자로냉각재계통(RCS)의 정상 운전압력은?",
        "question_type": "multiple_choice",
        "options": {"A": "1,800 psia", "B": "2,235 psia", "C": "2,485 psia", "D": "2,500 psia"},
        "correct_answer": "B",
        "distractor_rationale": {
            "A": "ECCS(HPSI) 기동 설정값으로, RCS 정상압력과 혼동 가능",
            "C": "가압기 안전밸브 설정값으로, 정상 운전압력과 혼동 가능",
            "D": "RCS 설계압력으로, 운전압력과 혼동 가능",
        },
        "explanation": "RCS 정상 운전압력은 2,235 psia이며, 가압기에 의해 유지됩니다. 2,500 psia는 설계압력, 2,485 psia는 안전밸브 설정값, 1,800 psia는 HPSI 기동 설정값입니다.",
        "difficulty": "easy",
        "keywords": ["RCS", "운전압력", "가압기"],
        "source_page": 1,
    },
    {
        "question_id": "Q002",
        "bloom_level": "Comprehension",
        "learning_objective": "가압기의 압력/수위 제어 방법을 설명할 수 있다",
        "question_text": "가압기에서 RCS 압력이 상승할 때 사용하는 제어 방법으로 올바른 것은?",
        "question_type": "multiple_choice",
        "options": {"A": "전기 히터 투입", "B": "살수(Spray) 작동", "C": "보충수 주입 증가", "D": "안전밸브 수동 개방"},
        "correct_answer": "B",
        "distractor_rationale": {
            "A": "히터는 압력 '상승' 시 사용하므로 반대 상황에 적용. 압력 제어 방향을 혼동하는 학습자가 선택 가능",
            "C": "보충수는 수위 제어 수단으로, 압력 제어와 혼동 가능",
            "D": "안전밸브는 비정상 고압 시 자동 개방되는 최후 수단으로, 정상 제어와 혼동 가능",
        },
        "explanation": "RCS 압력 상승 시 가압기 살수(Spray)를 작동하여 증기를 응축시켜 압력을 낮춥니다. 반대로 압력 하강 시에는 전기 히터를 투입합니다.",
        "difficulty": "medium",
        "keywords": ["가압기", "살수", "압력 제어"],
        "source_page": 2,
    },
    {
        "question_id": "Q003",
        "bloom_level": "Application",
        "learning_objective": "ECCS 자동 기동 조건과 주입 단계를 설명할 수 있다",
        "scenario": "출력 100% 정상 운전 중 RCS 압력이 2,235 psia에서 급격히 하강하여 1,750 psia까지 떨어졌다. 격납건물 압력이 상승 중이다.",
        "question_text": "이 시점에서 자동으로 기동되어야 하는 계통은?",
        "question_type": "multiple_choice",
        "options": {"A": "잔열제거계통(RHRS)", "B": "고압안전주입(HPSI)", "C": "저압안전주입(LPSI)", "D": "격납건물 살수계통만"},
        "correct_answer": "B",
        "distractor_rationale": {
            "A": "RHRS는 정지냉각 시 사용하며, 사고 초기에는 적용 불가. RCS 압력이 RHRS 운전 범위(~400 psia)보다 훨씬 높음",
            "C": "LPSI 기동조건은 200 psia 이하로, 현재 1,750 psia에서는 기동 불가. 기동 조건을 정확히 모르는 학습자가 선택 가능",
            "D": "격납건물 살수는 격납건물 압력 기반이며, ECCS와 동시 기동 가능하나 핵심 대응은 ECCS 주입",
        },
        "explanation": "HPSI 기동조건은 PZR 압력 ≤ 1,800 psia이며, 현재 1,750 psia이므로 HPSI가 자동 기동됩니다. LPSI는 200 psia 이하에서 기동하며, RHRS는 정상 정지 시 사용합니다.",
        "difficulty": "medium",
        "keywords": ["ECCS", "HPSI", "자동기동", "S-신호"],
        "source_page": 3,
    },
    {
        "question_id": "Q004",
        "bloom_level": "Analysis",
        "learning_objective": "LOCA 시 운전원 조치 절차를 순서대로 기술할 수 있다",
        "scenario": "Large Break LOCA 발생 후 15분 경과. ECCS가 정상 작동 중이며, RWST 수위가 급격히 감소하고 있다. 현재 RWST 수위 25%(재순환 전환 설정값: 20%).",
        "question_text": "운전원이 이 시점에서 가장 우선적으로 수행해야 할 조치는?",
        "question_type": "multiple_choice",
        "options": {
            "A": "ECCS 펌프를 정지하여 RWST 소진을 지연시킨다",
            "B": "재순환 전환 절차를 준비하고 RWST 수위를 계속 감시한다",
            "C": "즉시 재순환 모드로 전환한다",
            "D": "보조급수계통을 기동하여 SG를 통한 냉각을 강화한다",
        },
        "correct_answer": "B",
        "distractor_rationale": {
            "A": "ECCS 정지는 노심 냉각을 중단시켜 극히 위험. 그러나 RWST 보존이라는 논리에 혼동 가능",
            "C": "설정값(20%)에 아직 도달하지 않았으므로 조기 전환은 부적절. 적극적 조치로 보여 선택 가능",
            "D": "AFW는 2차측 냉각이며, LOCA 대응의 핵심이 아님. 냉각 관련이라 혼동 가능",
        },
        "explanation": "RWST 수위가 설정값(20%)에 접근하고 있으므로, 재순환 전환 절차를 미리 준비하면서 수위를 계속 감시하는 것이 올바른 대응입니다. 설정값 도달 전 조기 전환이나 ECCS 정지는 부적절합니다.",
        "difficulty": "hard",
        "keywords": ["LOCA", "재순환", "RWST", "운전원 조치"],
        "source_page": 4,
    },
]


# ============================================================
# 메인 데모 실행
# ============================================================

async def run_demo():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("KHNP Education AI Platform — 샘플 PPT E2E 데모")
    print("=" * 60)

    # Step 1-2: 슬라이드 + VLM 보강
    print("\n[Step 1-2] 슬라이드 파싱 + VLM 시각자료 분석...")
    enriched_slides = []
    for slide in SAMPLE_SLIDES:
        es = {**slide}
        vlm = MOCK_VLM_RESULTS.get(slide["page_number"])
        if vlm:
            es["vlm_analyses"] = [vlm]
            es["metadata"] = {**slide["metadata"], "vlm_enriched": True}
        enriched_slides.append(es)
    vlm_count = sum(1 for s in enriched_slides if s.get("metadata", {}).get("vlm_enriched"))
    print(f"  → {len(enriched_slides)}개 슬라이드, {vlm_count}개 VLM 보강")

    # Step 3: Golden Context (이 데모에서는 슬라이드 자체가 참고자료)
    print("\n[Step 3] Golden Context 구성...")
    print("  → 자체 참고자료 없음 (슬라이드 내용 활용)")

    # Step 4: 스크립트 생성
    print("\n[Step 4] 강의 스크립트 생성...")
    scripts = generate_mock_scripts(enriched_slides, SAMPLE_LEARNING_OBJECTIVES, MOCK_VLM_RESULTS)
    total_minutes = sum(s["estimated_minutes"] for s in scripts)
    print(f"  → {len(scripts)}개 스크립트, 예상 강의시간 {total_minutes}분")

    # Step 5: 문항 생성
    print("\n[Step 5] 문제은행 생성 (2-Pass)...")
    questions = MOCK_GENERATED_QUESTIONS
    bloom_dist = {}
    for q in questions:
        bl = q.get("bloom_level", "Unknown")
        bloom_dist[bl] = bloom_dist.get(bl, 0) + 1
    print(f"  → {len(questions)}개 문항: {bloom_dist}")

    # Step 6: 안전 태깅
    print("\n[Step 6] 안전 중요도 태깅 (IAEA Graded Approach)...")
    tagger = SafetySignificanceTagger()
    questions = tagger.tag_batch(questions)
    safety_stats = tagger.get_safety_statistics(questions)
    print(f"  → 분포: {safety_stats.get('distribution', {})}")

    # Step 7: 품질 검증 (Mock)
    print("\n[Step 7] 품질 검증 (Mock QA)...")
    qa_results = []
    for q in questions:
        qa_results.append({
            "question_id": q["question_id"],
            "overall_quality": "pass",
            "quality_score": 85,
        })
    quality_report = {
        "statistics": {
            "total": len(questions),
            "passed": len(questions),
            "pass_rate": "100.0%",
            "avg_quality_score": 85.0,
            "bloom_coverage": bloom_dist,
        },
    }
    print(f"  → {len(qa_results)}개 검증 완료, 통과율 100%")

    # Step 8: 난이도 추정
    print("\n[Step 8] 예상 난이도 분석 (IRT 기초)...")
    for q in questions:
        q["estimated_item_analysis"] = ItemAnalysisSimulator.estimate_difficulty(q)
    difficulties = [q["estimated_item_analysis"]["estimated_difficulty_index"] for q in questions]
    print(f"  → 평균 난이도: {sum(difficulties)/len(difficulties):.2f}")

    # Step 9: 버전 관리
    print("\n[Step 9] 문항 버전 관리 등록...")
    vmgr = QuestionVersionManager()
    for q in questions:
        vmgr.create_question(q, created_by="AI:Demo")
    print(f"  → {len(questions)}개 문항 등록 완료")

    # Step 10: 스크립트 품질
    print("\n[Step 10] 스크립트 품질 평가...")
    # 간이 평가
    script_quality = {
        "overall_score": 78.5,
        "completeness": 100.0,
        "structure_adherence": 85.0,
        "visual_utilization": 66.7,
        "length_appropriateness": 70.0,
        "recommendation": "보통: 시각자료 활용을 더 강화하세요.",
    }
    print(f"  → 종합 점수: {script_quality['overall_score']}")

    # Step 11: QTI 내보내기
    print("\n[Step 11] QTI 2.1 내보내기...")
    qti_xml = QTIExporter.export_to_qti_xml(questions, "RCS/ECCS 교육평가")
    qti_json = QTIExporter.export_to_json(questions)
    print(f"  → QTI XML: {len(qti_xml)}자, JSON: {len(qti_json)}자")

    # 용어사전 통계
    glossary = NuclearGlossaryManager()
    glossary_stats = glossary.get_statistics()

    # 프롬프트 템플릿 정보
    tm = PromptTemplateManager()
    prompt_versions = tm.get_all_versions()

    # ========== 결과 저장 ==========
    print("\n" + "=" * 60)
    print("결과 저장 중...")

    with open(output_dir / f"scripts_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(scripts, f, ensure_ascii=False, indent=2)

    with open(output_dir / f"questions_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2, default=str)

    with open(output_dir / f"quality_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    with open(output_dir / f"qti_{timestamp}.xml", "w", encoding="utf-8") as f:
        f.write(qti_xml)

    # 사람이 읽을 수 있는 요약
    summary_lines = [
        "=" * 60,
        "KHNP Education AI Platform — E2E 데모 결과",
        f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"강의 제목: 원자력 안전계통 교육 (RCS, PZR, ECCS, LOCA)",
        f"슬라이드 수: {len(enriched_slides)}",
        f"VLM 보강 슬라이드: {vlm_count}",
        f"학습목표: {len(SAMPLE_LEARNING_OBJECTIVES)}개",
        "",
        "--- 강의 스크립트 ---",
    ]
    for s in scripts:
        summary_lines.append(f"  [p.{s['page_number']}] {s['slide_title']} ({s['estimated_minutes']}분)")
        summary_lines.append(f"    도입: {s['script']['introduction'][:80]}...")
    summary_lines.append(f"  총 예상 강의시간: {total_minutes}분")
    summary_lines.append("")
    summary_lines.append("--- 문제은행 ---")
    for q in questions:
        tag = q.get("safety_significance", {}).get("grade", "?")
        summary_lines.append(f"  [{q['question_id']}] [{q['bloom_level']}] [{tag}] {q['question_text'][:60]}...")
        summary_lines.append(f"    정답: {q['correct_answer']}")
    summary_lines.append("")
    summary_lines.append(f"안전 통계: {safety_stats.get('distribution', {})}")
    summary_lines.append(f"Bloom 분포: {bloom_dist}")
    summary_lines.append(f"평균 난이도: {sum(difficulties)/len(difficulties):.2f}")
    summary_lines.append(f"스크립트 품질: {script_quality['overall_score']}")
    summary_lines.append(f"용어사전: {glossary_stats['total_terms']}개 용어")
    summary_lines.append(f"프롬프트 버전: {prompt_versions}")
    summary_lines.append("")
    summary_lines.append("--- 파일 출력 ---")
    summary_lines.append(f"  scripts_{timestamp}.json")
    summary_lines.append(f"  questions_{timestamp}.json")
    summary_lines.append(f"  quality_{timestamp}.json")
    summary_lines.append(f"  qti_{timestamp}.xml")

    summary_text = "\n".join(summary_lines)
    with open(output_dir / f"summary_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\n결과 파일: {output_dir}/")
    print("데모 완료!")


if __name__ == "__main__":
    asyncio.run(run_demo())
