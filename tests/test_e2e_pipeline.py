"""
E2E 통합 테스트 - 11단계 파이프라인 Mock 검증

외부 API 호출 없이 전체 파이프라인 로직을 검증합니다.
- DocumentParser → VLM → GoldenContext → Script → Question → Safety → QA → IRT → Version → Quality
"""

import json
import pytest
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline import EducationAIPipeline, PipelineInput, PipelineOutput
from src.question_generator.safety_tagger import SafetySignificanceTagger
from src.question_generator.quality_validator import ItemAnalysisSimulator
from src.question_generator.version_manager import QuestionVersionManager, QuestionLifecycle
from src.question_generator.sme_review import SMEReviewWorkflow, ReviewStatus
from src.question_generator.qti_exporter import QTIExporter
from src.question_generator.adaptive_engine import (
    AdaptiveDifficultyEngine, TraineeProfile, ExamConfig,
)
from src.domain.nuclear_glossary import NuclearGlossaryManager
from src.prompts.template_manager import PromptTemplateManager


# ============================================================
# Mock 데이터
# ============================================================

MOCK_SLIDES = [
    {
        "page_number": 1,
        "title": "원자로냉각재계통 개요",
        "content": "RCS(Reactor Coolant System)는 원자로에서 발생한 열을 증기발생기로 전달하는 1차 계통입니다.",
        "images": [{"description": "RCS 계통도 — 원자로, 가압기, 증기발생기, RCP 배치"}],
        "charts": [],
        "tables": [],
        "metadata": {"has_complex_visuals": True},
    },
    {
        "page_number": 2,
        "title": "비상노심냉각계통 (ECCS)",
        "content": "ECCS는 냉각재상실사고(LOCA) 시 노심을 냉각하는 안전계통입니다. SIS, AFWS 등으로 구성됩니다.",
        "images": [],
        "charts": [{"description": "ECCS 주입 유량 vs 1차 계통 압력 그래프", "data": {"x": "pressure_psi", "y": "flow_gpm"}}],
        "tables": [["구분", "설계압력", "유량"], ["SIS", "1800 psi", "500 gpm"]],
        "metadata": {"has_complex_visuals": True},
    },
    {
        "page_number": 3,
        "title": "안전주입계통 운전 절차",
        "content": "SIS 자동 기동 조건: 가압기 저압력 1800 psia, 격납건물 고압력 4 psig",
        "images": [],
        "charts": [],
        "tables": [],
        "metadata": {"has_complex_visuals": False},
    },
]

MOCK_VLM_ANALYSIS = {
    "visual_type": "계통도",
    "title": "RCS 계통도",
    "description": "원자로냉각재계통의 전체 배치를 보여주는 계통도입니다. 원자로, 가압기, 4개의 증기발생기, RCP가 표시되어 있습니다.",
    "key_components": ["원자로", "가압기(PZR)", "증기발생기(SG)", "원자로냉각재펌프(RCP)"],
    "flow_description": "냉각재는 원자로 → Hot Leg → SG → Cold Leg → RCP → 원자로 경로로 순환합니다.",
    "safety_notes": "RCS 건전성은 1차 방벽으로서 방사성물질 격납의 핵심입니다.",
    "safety_grade": "safety_critical",
    "teaching_points": ["냉각재 순환 경로", "가압기의 압력/수위 제어 역할", "SG에서의 열전달 원리"],
    "related_systems": ["가압기(PZR)", "증기발생기(SG)", "화학체적제어계통(CVCS)"],
    "operating_conditions": "정상출력 운전 (Mode 1)",
    "exam_potential": ["RCS 구성요소 식별", "냉각재 흐름 경로", "가압기 기능"],
    "difficulty_level": "intermediate",
}

MOCK_SCRIPT = {
    "page_number": 1,
    "slide_title": "원자로냉각재계통 개요",
    "estimated_minutes": 5,
    "script": {
        "introduction": "지금부터 원자력 발전소의 핵심 계통인 원자로냉각재계통에 대해 알아보겠습니다.",
        "main_content": "RCS는 원자로에서 발생한 핵분열 에너지를 1차 냉각재를 통해 증기발생기로 전달합니다.",
        "visual_explanation": "계통도를 보시면 원자로, 가압기, SG, RCP가 루프 형태로 연결되어 있습니다.",
        "contextual_supplement": "실제 한빛원전의 RCS는 2-루프 구성으로 총 4대의 SG를 가지고 있습니다.",
        "summary": "RCS는 1차 방벽으로서 방사성물질 격납의 핵심이며, 다음 시간에는 가압기 상세를 다루겠습니다.",
    },
    "teaching_notes": "학습자에게 냉각재 순환 경로를 손으로 따라가며 설명하면 효과적",
    "key_terms": ["RCS", "가압기", "증기발생기", "RCP"],
    "visual_references": ["RCS 계통도"],
}

MOCK_QUESTIONS = [
    {
        "question_id": "Q001",
        "bloom_level": "Knowledge",
        "learning_objective": "RCS 구성요소를 식별할 수 있다",
        "question_text": "원자로냉각재계통(RCS)의 주요 구성요소가 아닌 것은?",
        "question_type": "multiple_choice",
        "options": {"A": "원자로", "B": "가압기", "C": "터빈", "D": "증기발생기"},
        "correct_answer": "C",
        "distractor_rationale": {
            "A": "RCS의 핵심 구성요소이므로 정답이 아님",
            "B": "압력 제어를 위한 RCS 구성요소",
            "D": "1차-2차 열교환이 이루어지는 RCS 구성요소",
        },
        "explanation": "터빈은 2차 계통(BOP)의 구성요소이며, RCS(1차 계통)에는 포함되지 않습니다.",
        "difficulty": "easy",
        "keywords": ["RCS", "구성요소", "1차 계통"],
        "source_page": 1,
    },
    {
        "question_id": "Q002",
        "bloom_level": "Application",
        "learning_objective": "ECCS 자동 기동 조건을 적용할 수 있다",
        "scenario": "출력 100% 운전 중 가압기 압력이 1850 psia에서 급격히 하강하고 있다.",
        "question_text": "SIS 자동 기동이 발생하는 가압기 압력 설정값은?",
        "question_type": "multiple_choice",
        "options": {"A": "1500 psia", "B": "1800 psia", "C": "2000 psia", "D": "2235 psia"},
        "correct_answer": "B",
        "distractor_rationale": {
            "A": "SIS 기동 압력보다 낮은 값으로, 이미 기동된 후의 압력 수준",
            "C": "정상 운전 압력 범위에 해당하여 혼동 가능",
            "D": "설계 압력으로 SIS 기동 조건과 혼동 가능",
        },
        "explanation": "SIS는 가압기 저압력 1800 psia에서 자동 기동됩니다.",
        "difficulty": "medium",
        "keywords": ["SIS", "ECCS", "자동 기동", "가압기 압력"],
        "source_page": 3,
    },
]

MOCK_QA_RESULT = {
    "question_id": "Q001",
    "overall_quality": "pass",
    "quality_score": 85,
    "checks": {
        "clarity": {"pass": True, "issue": ""},
        "correct_answer": {"pass": True, "issue": ""},
        "distractor_quality": {"pass": True, "nonfunctional_distractors": [], "suggestions": {}},
        "difficulty_appropriate": {"pass": True, "issue": ""},
        "sat_alignment": {"pass": True, "issue": ""},
    },
    "revision_suggestions": [],
    "revised_question": None,
}


# ============================================================
# 테스트: 개별 모듈
# ============================================================


class TestSafetyTagger:
    """안전 중요도 태깅 모듈 테스트"""

    def test_tag_single_question(self):
        tagger = SafetySignificanceTagger()
        tagged = tagger.tag_batch(MOCK_QUESTIONS[:1])
        assert len(tagged) == 1
        assert "safety_significance" in tagged[0]
        grade = tagged[0]["safety_significance"]["grade"]
        assert grade in ("safety_critical", "safety_related", "general")

    def test_tag_batch(self):
        tagger = SafetySignificanceTagger()
        tagged = tagger.tag_batch(MOCK_QUESTIONS)
        assert len(tagged) == 2
        stats = tagger.get_safety_statistics(tagged)
        assert "distribution" in stats

    def test_safety_critical_detection(self):
        """RCS 관련 문항은 safety_critical이어야 함"""
        tagger = SafetySignificanceTagger()
        tagged = tagger.tag_batch([MOCK_QUESTIONS[0]])
        # RCS는 safety_critical 키워드 포함
        grade = tagged[0]["safety_significance"]["grade"]
        assert grade in ("safety_critical", "safety_related")


class TestItemAnalysis:
    """문항 분석 시뮬레이터 테스트"""

    def test_estimate_difficulty_knowledge(self):
        result = ItemAnalysisSimulator.estimate_difficulty(MOCK_QUESTIONS[0])
        assert "estimated_difficulty_index" in result
        assert result["bloom_level"] == "Knowledge"
        # Knowledge는 상대적으로 낮은 난이도
        assert result["estimated_difficulty_index"] < 0.5

    def test_estimate_difficulty_application(self):
        result = ItemAnalysisSimulator.estimate_difficulty(MOCK_QUESTIONS[1])
        assert result["bloom_level"] == "Application"
        # Application은 상대적으로 높은 난이도
        assert result["estimated_difficulty_index"] > 0.4

    def test_calibrate_without_history(self):
        result = ItemAnalysisSimulator.calibrate_difficulty_from_history(
            MOCK_QUESTIONS[0], []
        )
        assert not result["calibrated"]
        assert result["source"] == "ai_estimate"

    def test_calibrate_with_history(self):
        history = [
            {"exam_id": "E1", "difficulty_index": 0.4, "discrimination_index": 0.35, "total_responses": 100},
            {"exam_id": "E2", "difficulty_index": 0.45, "discrimination_index": 0.38, "total_responses": 120},
        ]
        result = ItemAnalysisSimulator.calibrate_difficulty_from_history(
            MOCK_QUESTIONS[0], history
        )
        assert result["calibrated"]
        assert result["confidence"] == "high"  # 100+120 >= 200
        assert result["irt_ready"]

    def test_post_exam_analysis(self):
        responses = [
            {"student_id": f"S{i}", "selected": "C", "correct": True, "total_score": 80 + i}
            for i in range(30)
        ] + [
            {"student_id": f"S{i+30}", "selected": "A", "correct": False, "total_score": 40 + i}
            for i in range(20)
        ]
        result = ItemAnalysisSimulator.analyze_post_exam("Q001", responses)
        assert result["total_responses"] == 50
        assert result["difficulty_index"] == 0.6  # 30/50
        assert "discrimination_index" in result


class TestVersionManager:
    """문항 버전 관리 테스트"""

    def test_create_question(self):
        mgr = QuestionVersionManager()
        record = mgr.create_question(MOCK_QUESTIONS[0], created_by="test")
        assert record.question_id == "Q001"
        assert record.lifecycle_status == QuestionLifecycle.DRAFT
        assert len(record.versions) == 1

    def test_change_status(self):
        mgr = QuestionVersionManager()
        mgr.create_question(MOCK_QUESTIONS[0], created_by="test")
        mgr.change_status("Q001", QuestionLifecycle.SME_REVIEW, changed_by="test")
        record = mgr._records["Q001"]
        assert record.lifecycle_status == QuestionLifecycle.SME_REVIEW

    def test_add_version(self):
        mgr = QuestionVersionManager()
        mgr.create_question(MOCK_QUESTIONS[0], created_by="test")
        modified = {**MOCK_QUESTIONS[0], "difficulty": "hard"}
        mgr.add_version("Q001", modified, change_type="revision", changed_by="SME", change_summary="난이도 조정")
        record = mgr._records["Q001"]
        assert len(record.versions) == 2

    def test_statistics(self, tmp_path):
        mgr = QuestionVersionManager(storage_dir=str(tmp_path / "qv"))
        for q in MOCK_QUESTIONS:
            mgr.create_question(q, created_by="test")
        stats = mgr.get_statistics()
        assert stats["total_questions"] == 2


class TestSMEReview:
    """SME 검토 워크플로우 테스트"""

    def test_submit_for_review(self):
        wf = SMEReviewWorkflow()
        review_id = wf.submit_for_review(MOCK_QUESTIONS[0], reviewer_name="Dr_Kim")
        assert isinstance(review_id, str)
        assert "REV-" in review_id

    def test_get_pending(self):
        wf = SMEReviewWorkflow()
        wf.submit_for_review(MOCK_QUESTIONS[0], reviewer_name="Dr_Kim")
        wf.submit_for_review(MOCK_QUESTIONS[1], reviewer_name="Dr_Lee")
        pending = wf.get_pending_reviews()
        assert len(pending) >= 2

    def test_submit_revision(self):
        wf = SMEReviewWorkflow()
        review_id = wf.submit_for_review(MOCK_QUESTIONS[0], reviewer_name="Dr_Kim")
        revised = {**MOCK_QUESTIONS[0], "explanation": "개선된 해설"}
        result = wf.submit_revision(
            review_id,
            status=ReviewStatus.APPROVED,
            revised_question=revised,
            overall_comment="해설 보강",
        )
        assert result is not None


class TestQTIExporter:
    """QTI 내보내기 테스트"""

    def test_export_qti_xml(self):
        xml = QTIExporter.export_to_qti_xml(MOCK_QUESTIONS, "테스트 평가")
        assert "<?xml version" in xml
        assert "assessmentTest" in xml
        assert "assessmentItemRef" in xml

    def test_export_json(self):
        json_str = QTIExporter.export_to_json(MOCK_QUESTIONS)
        data = json.loads(json_str)
        assert data["format"] == "khnp-qbank-v3"
        assert len(data["questions"]) == 2

    def test_export_csv(self):
        csv_str = QTIExporter.export_to_csv(MOCK_QUESTIONS)
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 3  # header + 2 questions


class TestAdaptiveEngine:
    """적응형 시험 엔진 테스트"""

    def test_irt_probability(self):
        # 능력 = 난이도 → 확률 ≈ 0.5
        p = AdaptiveDifficultyEngine.irt_probability(theta=0.0, difficulty=0.0, discrimination=1.0)
        assert abs(p - 0.5) < 0.01

        # 능력 >> 난이도 → 확률 → 1
        p_high = AdaptiveDifficultyEngine.irt_probability(theta=3.0, difficulty=0.0, discrimination=1.0)
        assert p_high > 0.9

    def test_exam_config_defaults(self):
        config = ExamConfig()
        assert config.total_questions == 20
        assert config.safety_critical_min >= 1

    def test_trainee_profile(self):
        trainee = TraineeProfile(trainee_id="T001")
        assert trainee.ability_estimate == 0.0


class TestNuclearGlossary:
    """원자력 용어사전 테스트"""

    def test_search_korean(self):
        mgr = NuclearGlossaryManager()
        results = mgr.search("가압기")
        assert len(results) > 0
        assert any("PZR" in str(r) for r in results)

    def test_search_abbreviation(self):
        mgr = NuclearGlossaryManager()
        results = mgr.search("ECCS")
        assert len(results) > 0

    def test_get_by_abbreviation(self):
        mgr = NuclearGlossaryManager()
        term = mgr.get_by_abbreviation("RCS")
        assert term is not None
        assert "원자로냉각재계통" in str(term)

    def test_to_prompt_text(self):
        mgr = NuclearGlossaryManager()
        text = mgr.to_prompt_text()
        assert "원자력 전문 용어" in text or len(text) > 100

    def test_to_vlm_context(self):
        mgr = NuclearGlossaryManager()
        ctx = mgr.to_vlm_context()
        assert len(ctx) > 50

    def test_statistics(self):
        mgr = NuclearGlossaryManager()
        stats = mgr.get_statistics()
        assert stats["total_terms"] >= 20

    def test_filter_by_category(self):
        mgr = NuclearGlossaryManager()
        results = mgr.search("", category="안전계통")
        assert len(results) > 0


class TestPromptTemplateManager:
    """프롬프트 템플릿 매니저 테스트"""

    def test_load_templates(self):
        tm = PromptTemplateManager()
        templates = tm.list_templates()
        assert len(templates) >= 6  # 6개 프롬프트 섹션

    def test_get_script_prompt(self):
        tm = PromptTemplateManager()
        prompt = tm.get_prompt("script_generation")
        assert "SAT" in prompt
        assert "슬라이드" in prompt

    def test_get_vlm_prompt(self):
        tm = PromptTemplateManager()
        prompt = tm.get_prompt("vlm_analysis")
        assert "원자력" in prompt
        assert "P&ID" in prompt

    def test_get_question_prompt(self):
        tm = PromptTemplateManager()
        prompt = tm.get_prompt("question_generation")
        assert "Bloom" in prompt
        assert "오답" in prompt

    def test_get_config(self):
        tm = PromptTemplateManager()
        config = tm.get_config("script_generation")
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 4096

    def test_override(self):
        tm = PromptTemplateManager()
        original = tm.get_prompt("script_generation")
        tm.set_override("script_generation", "CUSTOM PROMPT")
        assert tm.get_prompt("script_generation") == "CUSTOM PROMPT"
        tm.clear_override("script_generation")
        assert tm.get_prompt("script_generation") == original

    def test_versions(self):
        tm = PromptTemplateManager()
        versions = tm.get_all_versions()
        assert "script_generation" in versions
        assert versions["script_generation"] == "v4.0"


# ============================================================
# 테스트: E2E 파이프라인 (Mock)
# ============================================================


class TestE2EPipeline:
    """전체 파이프라인 E2E 테스트 (Mock API 호출)"""

    @pytest.fixture
    def mock_pipeline(self):
        """API 호출을 모킹한 파이프라인"""
        with patch.dict("os.environ", {"UPSTAGE_API_KEY": "test-key"}):
            pipeline = EducationAIPipeline(api_key="test-key")

            # Mock: Document Parser
            pipeline.parser.parse_document = AsyncMock(return_value={"content": {"pages": MOCK_SLIDES}})
            pipeline.parser.parse_result_to_slides = MagicMock(return_value=MOCK_SLIDES)

            # Mock: VLM Analyzer
            pipeline.vlm.analyze_visual = AsyncMock(return_value=MOCK_VLM_ANALYSIS)
            enriched = []
            for s in MOCK_SLIDES:
                es = {**s}
                if s["metadata"].get("has_complex_visuals"):
                    es["vlm_analyses"] = [MOCK_VLM_ANALYSIS]
                    es["metadata"] = {**s["metadata"], "vlm_enriched": True}
                enriched.append(es)
            pipeline.vlm.enrich_slides_with_vlm = AsyncMock(return_value=enriched)

            # Mock: Golden Context Builder
            pipeline.context_builder.build_context = AsyncMock(
                return_value="[참고자료] RCS 설계 기준: 설계압력 2500 psia, 설계온도 650°F"
            )

            # Mock: Script Generator
            pipeline.script_gen.generate_slide_script = AsyncMock(return_value=MOCK_SCRIPT)
            pipeline.script_gen.generate_full_script = AsyncMock(
                return_value=[MOCK_SCRIPT] * 3
            )

            # Mock: Question Generator
            pipeline.question_gen.generate_questions = AsyncMock(return_value=MOCK_QUESTIONS)

            # Mock: QA Validator
            pipeline.qa_validator.validate_single = AsyncMock(return_value=MOCK_QA_RESULT)
            pipeline.qa_validator.validate_batch = AsyncMock(return_value={
                "individual_results": [MOCK_QA_RESULT] * 2,
                "statistics": {
                    "total": 2, "passed": 2, "needs_revision": 0, "rejected": 0,
                    "pass_rate": "100.0%", "avg_quality_score": 85.0,
                    "bloom_coverage": {"Knowledge": 1, "Application": 1},
                    "common_issues": [],
                },
                "recommendations": [],
            })

            yield pipeline

    @pytest.mark.asyncio
    async def test_full_pipeline_run(self, mock_pipeline):
        """전체 11단계 파이프라인 실행 테스트"""
        input_data = PipelineInput(
            slide_file="test_lecture.pptx",
            reference_files=["ref1.pdf"],
            learning_objectives=[
                "RCS 구성요소를 식별할 수 있다",
                "ECCS 자동 기동 조건을 설명할 수 있다",
            ],
            audience_level="intermediate",
            num_questions=2,
        )

        output = await mock_pipeline.run(input_data)

        # 검증: 슬라이드 파싱
        assert len(output.slides) == 3
        assert output.metadata["total_slides"] == 3

        # 검증: VLM 보강
        assert output.metadata["vlm_enriched_slides"] >= 1

        # 검증: 스크립트 생성
        assert len(output.scripts) == 3
        assert output.metadata["total_scripts"] == 3

        # 검증: 문항 생성
        assert len(output.questions) == 2
        assert output.metadata["total_questions"] == 2

        # 검증: 안전 태깅
        assert "safety_statistics" in output.metadata
        for q in output.questions:
            assert "safety_significance" in q

        # 검증: 품질 보고서
        assert output.quality_report["statistics"]["total"] == 2
        assert output.quality_report["statistics"]["pass_rate"] == "100.0%"

        # 검증: 난이도 추정
        for q in output.questions:
            assert "estimated_item_analysis" in q

        # 검증: 버전 관리
        assert output.metadata["revalidation_due_count"] >= 0

        # 검증: 스크립트 품질
        assert "overall_score" in output.script_quality
        assert output.script_quality["total_slides"] == 3

    @pytest.mark.asyncio
    async def test_pipeline_metadata(self, mock_pipeline):
        """메타데이터 완전성 테스트"""
        input_data = PipelineInput(
            slide_file="test.pptx",
            learning_objectives=["테스트 학습목표"],
        )
        output = await mock_pipeline.run(input_data)

        assert "started_at" in output.metadata
        assert "completed_at" in output.metadata
        assert "input_file" in output.metadata
        assert "qa_pass_rate" in output.metadata

    @pytest.mark.asyncio
    async def test_pipeline_output_serializable(self, mock_pipeline):
        """출력이 JSON 직렬화 가능한지 테스트"""
        input_data = PipelineInput(
            slide_file="test.pptx",
            learning_objectives=["테스트"],
        )
        output = await mock_pipeline.run(input_data)

        # 각 필드가 JSON 직렬화 가능해야 함
        json.dumps(output.scripts, ensure_ascii=False)
        json.dumps(output.questions, ensure_ascii=False, default=str)
        json.dumps(output.quality_report, ensure_ascii=False)
        json.dumps(output.metadata, ensure_ascii=False, default=str)
        json.dumps(output.script_quality, ensure_ascii=False)


# ============================================================
# 테스트: LLM 하이브리드 라우터
# ============================================================

from src.llm_router.router import (
    LLMRouter, ModelProvider, ModelConfig, TaskType,
    RoutingPolicy, DEFAULT_ROUTING_POLICIES, create_default_router,
)


class TestLLMRouter:
    """LLM 하이브리드 라우터 단위 테스트"""

    def test_default_policies_count(self):
        """기본 라우팅 정책 7개 확인"""
        assert len(DEFAULT_ROUTING_POLICIES) == 7

    def test_question_generation_routes_to_atomic(self):
        """문항 생성은 AtomicGPT 우선"""
        policy = next(p for p in DEFAULT_ROUTING_POLICIES if p.task_type == TaskType.QUESTION_GENERATION)
        assert policy.primary == ModelProvider.ATOMIC_GPT
        assert policy.fallback == ModelProvider.SOLAR_PRO

    def test_vlm_analysis_routes_to_solar(self):
        """VLM 분석은 Solar Pro 전용"""
        policy = next(p for p in DEFAULT_ROUTING_POLICIES if p.task_type == TaskType.VLM_ANALYSIS)
        assert policy.primary == ModelProvider.SOLAR_PRO

    def test_safety_review_requires_cross_check(self):
        """안전등급 검토는 크로스체크 필수"""
        policy = next(p for p in DEFAULT_ROUTING_POLICIES if p.task_type == TaskType.SAFETY_REVIEW)
        assert policy.require_cross_check is True
        assert policy.confidence_threshold == 0.9

    def test_create_default_router_solar_only(self):
        """Solar Pro만 활성화된 기본 라우터"""
        router = create_default_router(solar_api_key="test-key")
        assert router.models[ModelProvider.SOLAR_PRO].enabled is True
        assert router.models[ModelProvider.ATOMIC_GPT].enabled is False

    def test_create_default_router_both(self):
        """양쪽 모두 활성화된 라우터"""
        router = create_default_router(
            solar_api_key="solar-key",
            atomic_api_key="atomic-key",
        )
        assert router.models[ModelProvider.SOLAR_PRO].enabled is True
        assert router.models[ModelProvider.ATOMIC_GPT].enabled is True

    def test_confidence_estimation(self):
        """신뢰도 추정 로직"""
        assert LLMRouter._estimate_confidence({}) == 0.0
        assert LLMRouter._estimate_confidence({"key": "value"}) == 0.6  # JSON ok + key exists
        assert LLMRouter._estimate_confidence(
            {"key": "a" * 100, "another": "data"}
        ) == 1.0  # JSON ok + keys + content length

    def test_agreement_computation(self):
        """두 모델 응답 동의율 계산"""
        resp_a = {"answer": "B", "explanation": "RCS 정상 압력"}
        resp_b = {"answer": "B", "explanation": "RCS 정상 압력"}
        agreement = LLMRouter._compute_agreement(resp_a, resp_b)
        assert agreement == 1.0

        resp_c = {"answer": "B", "explanation": "다른 설명"}
        agreement2 = LLMRouter._compute_agreement(resp_a, resp_c)
        assert 0.0 < agreement2 < 1.0

    def test_statistics_empty(self):
        """빈 호출 이력의 통계"""
        router = create_default_router(solar_api_key="test")
        stats = router.get_statistics()
        assert stats["total_calls"] == 0


# ============================================================
# 테스트: 프롬프트 A/B 테스트
# ============================================================

from src.prompts.ab_testing import PromptABTestManager, PromptVariant, ExperimentResult


class TestABTesting:
    """프롬프트 A/B 테스트 프레임워크"""

    def test_create_experiment(self, tmp_path):
        mgr = PromptABTestManager(str(tmp_path / "ab"))
        exp_id = mgr.create_experiment(
            "question_generation",
            [PromptVariant("A", "prompt A", "baseline"), PromptVariant("B", "prompt B", "improved")],
        )
        assert exp_id.startswith("EXP-")
        assert len(mgr.list_experiments()) == 1

    def test_assign_variant(self, tmp_path):
        mgr = PromptABTestManager(str(tmp_path / "ab"))
        exp_id = mgr.create_experiment(
            "question_generation",
            [PromptVariant("A", "prompt A"), PromptVariant("B", "prompt B")],
        )
        v = mgr.assign_variant(exp_id)
        assert v is not None
        assert v.variant_id in ("A", "B")

    def test_record_and_analyze(self, tmp_path):
        mgr = PromptABTestManager(str(tmp_path / "ab"))
        exp_id = mgr.create_experiment(
            "question_generation",
            [PromptVariant("A", "prompt A"), PromptVariant("B", "prompt B")],
            min_samples=2,
        )
        # Record results
        for _ in range(2):
            mgr.record_result(exp_id, ExperimentResult(variant_id="A", qa_pass=True, quality_score=80.0))
            mgr.record_result(exp_id, ExperimentResult(variant_id="B", qa_pass=True, quality_score=92.0))

        results = mgr.get_experiment_results(exp_id)
        assert results["variant_stats"]["A"]["avg_quality_score"] == 80.0
        assert results["variant_stats"]["B"]["avg_quality_score"] == 92.0
        assert results["winner"]["variant_id"] == "B"

    def test_auto_complete(self, tmp_path):
        mgr = PromptABTestManager(str(tmp_path / "ab"))
        exp_id = mgr.create_experiment(
            "test", [PromptVariant("A", "a"), PromptVariant("B", "b")], min_samples=1,
        )
        mgr.record_result(exp_id, ExperimentResult(variant_id="A", quality_score=70))
        mgr.record_result(exp_id, ExperimentResult(variant_id="B", quality_score=90))
        results = mgr.get_experiment_results(exp_id)
        assert results["status"] == "completed"

    def test_persistence(self, tmp_path):
        storage = str(tmp_path / "ab")
        mgr1 = PromptABTestManager(storage)
        exp_id = mgr1.create_experiment("test", [PromptVariant("A", "a")])
        mgr1.record_result(exp_id, ExperimentResult(variant_id="A", quality_score=85))

        # 새 인스턴스에서 로드
        mgr2 = PromptABTestManager(storage)
        exps = mgr2.list_experiments()
        assert len(exps) == 1
        assert exps[0]["total_results"] == 1


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
