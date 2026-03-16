"""
KHNP Education AI Platform - FastAPI Backend
강의안 업로드 → 파싱 → RAG 증강 → 스크립트/문제은행 생성 API
"""
import logging
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env 파일 자동 로드

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.document_processor.parser import DocumentParser
from src.document_processor.vlm_analyzer import VLMAnalyzer, GoldenContextBuilder
from src.script_generator.generator import ScriptGenerator
from src.question_generator.generator import QuestionGenerator
from src.question_generator.quality_validator import (
    QuestionQualityValidator,
    ItemAnalysisSimulator,
)
from src.pipeline import EducationAIPipeline, PipelineInput, PipelineOutput
from src.question_generator.sme_review import SMEReviewWorkflow, ReviewStatus
from src.question_generator.version_manager import QuestionVersionManager
from src.question_generator.safety_tagger import HybridSafetyTagger
from src.question_generator.qti_exporter import QTIExporter
from src.question_generator.adaptive_engine import (
    AdaptiveDifficultyEngine, TraineeProfile, ExamConfig,
)
from src.domain.nuclear_glossary import NuclearGlossaryManager
from src.prompts.template_manager import get_template_manager
from src.llm_router.router import (
    LLMRouter, ModelProvider, TaskType, create_default_router,
)
from src.prompts.ab_testing import (
    PromptABTestManager, PromptVariant, ExperimentResult,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="KHNP Education AI Platform",
    description="한국수력원자력 인재개발원 AI 기반 교육컨텐츠 생성 시스템",
    version="0.1.0-poc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("UPSTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
    return key


def _get_base_url() -> str:
    return os.environ.get("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")


# ---------------------------------------------------------------------------
# Lazy-initialized singleton instances
# ---------------------------------------------------------------------------

_parser: DocumentParser | None = None
_vlm: VLMAnalyzer | None = None
_script_gen: ScriptGenerator | None = None
_question_gen: QuestionGenerator | None = None
_qa_validator: QuestionQualityValidator | None = None
_pipeline: EducationAIPipeline | None = None
_sme_workflow: SMEReviewWorkflow | None = None
_version_mgr: QuestionVersionManager | None = None
_hybrid_tagger: HybridSafetyTagger | None = None


def _init_components() -> None:
    """첫 요청 시 한 번만 초기화"""
    global _parser, _vlm, _script_gen, _question_gen, _qa_validator, _pipeline
    global _sme_workflow, _version_mgr, _hybrid_tagger

    if _parser is not None:
        return

    api_key = _get_api_key()
    base_url = _get_base_url()

    _parser = DocumentParser(api_key, base_url)
    _vlm = VLMAnalyzer(api_key, base_url)
    _script_gen = ScriptGenerator(api_key, base_url)
    _question_gen = QuestionGenerator(api_key, base_url)
    _qa_validator = QuestionQualityValidator(api_key, base_url)
    _pipeline = EducationAIPipeline(api_key, base_url)
    _sme_workflow = SMEReviewWorkflow()
    _version_mgr = QuestionVersionManager()
    _hybrid_tagger = HybridSafetyTagger(api_key, base_url)


# ---------------------------------------------------------------------------
# In-memory cache: file_id -> parsed slides
# ---------------------------------------------------------------------------

_parsed_cache: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """스크립트/문제 생성 요청"""
    file_id: str
    learning_objectives: list[str] = []
    audience_level: str = "intermediate"
    num_questions: int = 20
    bloom_distribution: dict | None = None
    enable_two_pass: bool = True


class ScenarioRequest(BaseModel):
    """시나리오 기반 문항 생성 요청"""
    file_id: str
    learning_objectives: list[str] = []
    num_scenarios: int = 5


class SMEReviewRequest(BaseModel):
    """SME 검토 등록 요청"""
    questions: list[dict]
    reviewer_name: str
    reviewer_role: str = ""


class SMERevisionRequest(BaseModel):
    """SME 수정 제출 요청"""
    review_id: str
    status: str  # approved | revised | rejected | escalated
    revised_question: dict | None = None
    overall_comment: str = ""
    quality_score_after: float = 0.0
    review_time_minutes: float = 0.0
    few_shot_candidate: bool = False
    few_shot_reason: str = ""


class ExamResponseRequest(BaseModel):
    """시험 응시 결과 수집"""
    exam_id: str
    question_id: str
    selected_answer: str
    is_correct: bool
    response_time_seconds: float = 0.0
    trainee_id: str = ""


class TraineeFeedbackRequest(BaseModel):
    """교육생 만족도 피드백"""
    session_id: str
    trainee_id: str = ""
    overall_satisfaction: int = 0  # 1-5 Likert
    content_relevance: int = 0
    difficulty_perception: str = ""  # too_easy | appropriate | too_hard
    free_text: str = ""


class ExportRequest(BaseModel):
    """문항 내보내기 요청"""
    questions: list[dict]
    format: str = "json"  # json | qti | csv
    title: str = "KHNP Question Bank"


class AdaptiveExamRequest(BaseModel):
    """적응형 시험 구성 요청"""
    trainee_id: str = "anonymous"
    ability_estimate: float = 0.0
    weak_topics: list[str] = []
    total_questions: int = 20
    safety_critical_min: int = 5
    safety_related_min: int = 5
    file_id: str = ""  # 문항 풀 소스


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_file_path(file_id: str) -> Path:
    """file_id 로부터 실제 파일 경로를 반환하거나 404 를 raise"""
    file_path = Path("data/raw_slides") / file_id
    if not file_path.exists():
        raise HTTPException(404, f"파일을 찾을 수 없습니다: {file_id}")
    return file_path


def _require_parsed(file_id: str) -> list[dict]:
    """캐시된 파싱 결과를 반환하거나, 아직 파싱되지 않았으면 400 을 raise"""
    slides = _parsed_cache.get(file_id)
    if slides is None:
        raise HTTPException(
            400,
            f"파싱되지 않은 파일입니다. 먼저 POST /api/v1/parse/{file_id} 를 호출하세요.",
        )
    return slides


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version="0.1.0-poc")


@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    """강의안 파일 업로드"""
    if not file.filename:
        raise HTTPException(400, "파일명이 없습니다.")

    allowed_extensions = {".pdf", ".pptx", ".ppt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"지원하지 않는 파일 형식: {ext}")

    upload_dir = Path("data/raw_slides")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    return {
        "file_id": file.filename,
        "file_path": str(file_path),
        "size_bytes": len(content),
        "message": "업로드 완료. /api/v1/parse 로 파싱을 시작하세요.",
    }


@app.post("/api/v1/parse/{file_id}")
async def parse_document(file_id: str):
    """업로드된 강의안 파싱 (Upstage Document Parse)"""
    _init_components()
    assert _parser is not None
    assert _vlm is not None

    file_path = _resolve_file_path(file_id)

    try:
        parse_result = await _parser.parse_document(str(file_path), mode="auto")
        slides = _parser.parse_result_to_slides(parse_result)

        # VLM 으로 시각자료 보강
        enriched_slides = await _vlm.enrich_slides_with_vlm(slides)

        # 캐시에 저장 (이후 generate 단계에서 사용)
        _parsed_cache[file_id] = enriched_slides

        return {
            "file_id": file_id,
            "status": "parsed",
            "slides_count": len(enriched_slides),
            "vlm_enriched_count": sum(
                1 for s in enriched_slides
                if s.get("metadata", {}).get("vlm_enriched")
            ),
            "message": "파싱 완료. /api/v1/generate/* 엔드포인트로 생성을 시작하세요.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document parse failed for %s", file_id)
        raise HTTPException(500, f"문서 파싱 중 오류가 발생했습니다: {exc}") from exc


@app.post("/api/v1/generate/script")
async def generate_script(request: GenerateRequest):
    """강의 스크립트 생성"""
    _init_components()
    assert _script_gen is not None

    slides = _require_parsed(request.file_id)

    try:
        scripts: list[dict] = []
        previous_summary = ""

        for slide in slides:
            script = await _script_gen.generate_slide_script(
                slide=slide,
                learning_objectives=request.learning_objectives,
                rag_context=[],  # Golden Context 없이 슬라이드 단독 생성
                audience_level=request.audience_level,
                previous_slide_summary=previous_summary,
            )
            scripts.append(script)
            previous_summary = script.get("script", {}).get("summary", "")

        return {
            "file_id": request.file_id,
            "status": "generated",
            "scripts_count": len(scripts),
            "scripts": scripts,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Script generation failed for %s", request.file_id)
        raise HTTPException(500, f"스크립트 생성 중 오류가 발생했습니다: {exc}") from exc


@app.post("/api/v1/generate/questions")
async def generate_questions(request: GenerateRequest):
    """문제은행 생성 (2-Pass 엔진)"""
    _init_components()
    assert _question_gen is not None

    slides = _require_parsed(request.file_id)

    try:
        questions = await _question_gen.generate_questions(
            slides=slides,
            learning_objectives=request.learning_objectives,
            rag_context=[],  # Golden Context 없이 슬라이드 단독 생성
            bloom_distribution=request.bloom_distribution,
            num_questions=request.num_questions,
            enable_two_pass=request.enable_two_pass,
        )

        return {
            "file_id": request.file_id,
            "status": "generated",
            "engine": "2-pass" if request.enable_two_pass else "1-pass",
            "questions_count": len(questions),
            "questions": questions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Question generation failed for %s", request.file_id)
        raise HTTPException(500, f"문제 생성 중 오류가 발생했습니다: {exc}") from exc


@app.post("/api/v1/generate/scenario-questions")
async def generate_scenario_questions(request: ScenarioRequest):
    """시나리오 기반 문항 생성 (운전/정비 실무 의사결정형)"""
    _init_components()
    assert _question_gen is not None

    slides = _require_parsed(request.file_id)

    try:
        questions = await _question_gen.generate_scenario_questions(
            slides=slides,
            learning_objectives=request.learning_objectives,
            rag_context=[],  # Golden Context 없이 슬라이드 단독 생성
            num_scenarios=request.num_scenarios,
        )

        return {
            "file_id": request.file_id,
            "status": "generated",
            "num_scenarios": request.num_scenarios,
            "questions_count": len(questions),
            "questions": questions,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Scenario question generation failed for %s", request.file_id,
        )
        raise HTTPException(
            500, f"시나리오 문항 생성 중 오류가 발생했습니다: {exc}",
        ) from exc


@app.post("/api/v1/generate/full-pipeline")
async def run_full_pipeline(request: GenerateRequest):
    """전체 파이프라인 실행 (파싱→VLM→스크립트→문제→QA)"""
    _init_components()
    assert _pipeline is not None

    file_path = _resolve_file_path(request.file_id)

    try:
        pipeline_input = PipelineInput(
            slide_file=str(file_path),
            learning_objectives=request.learning_objectives,
            audience_level=request.audience_level,
            num_questions=request.num_questions,
            bloom_distribution=request.bloom_distribution or {
                "Knowledge": 0.3,
                "Comprehension": 0.3,
                "Application": 0.25,
                "Analysis": 0.15,
            },
        )

        result: PipelineOutput = await _pipeline.run(pipeline_input)

        # 파이프라인 결과로 캐시도 갱신
        _parsed_cache[request.file_id] = result.slides

        return {
            "file_id": request.file_id,
            "status": "completed",
            "metadata": result.metadata,
            "slides_count": len(result.slides),
            "scripts_count": len(result.scripts),
            "scripts": result.scripts,
            "questions_count": len(result.questions),
            "questions": result.questions,
            "quality_report": result.quality_report,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Full pipeline failed for %s", request.file_id)
        raise HTTPException(
            500, f"파이프라인 실행 중 오류가 발생했습니다: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# SME Review Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/sme/submit-for-review")
async def sme_submit_for_review(request: SMEReviewRequest):
    """AI 생성 문항을 SME 검토 큐에 등록"""
    _init_components()
    assert _sme_workflow is not None

    review_ids = _sme_workflow.submit_batch_for_review(
        questions=request.questions,
        reviewer_name=request.reviewer_name,
        reviewer_role=request.reviewer_role,
    )
    return {
        "status": "submitted",
        "review_ids": review_ids,
        "count": len(review_ids),
    }


@app.get("/api/v1/sme/pending")
async def sme_get_pending(reviewer_name: str = ""):
    """검토 대기 중인 문항 목록"""
    _init_components()
    assert _sme_workflow is not None

    pending = _sme_workflow.get_pending_reviews(
        reviewer_name=reviewer_name or None
    )
    return {"pending": pending, "count": len(pending)}


@app.post("/api/v1/sme/submit-revision")
async def sme_submit_revision(request: SMERevisionRequest):
    """SME 수정/승인/반려 제출"""
    _init_components()
    assert _sme_workflow is not None

    try:
        record = _sme_workflow.submit_revision(
            review_id=request.review_id,
            status=request.status,
            revised_question=request.revised_question,
            overall_comment=request.overall_comment,
            quality_score_after=request.quality_score_after,
            review_time_minutes=request.review_time_minutes,
            few_shot_candidate=request.few_shot_candidate,
            few_shot_reason=request.few_shot_reason,
        )
        from dataclasses import asdict as _asdict
        return {"status": "recorded", "review": _asdict(record)}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/v1/sme/insights")
async def sme_get_insights():
    """SME 수정 패턴 → 프롬프트 개선 인사이트"""
    _init_components()
    assert _sme_workflow is not None

    return _sme_workflow.get_improvement_insights()


@app.get("/api/v1/sme/statistics")
async def sme_get_statistics():
    """SME 검토 워크플로우 통계"""
    _init_components()
    assert _sme_workflow is not None

    return _sme_workflow.get_statistics()


# ---------------------------------------------------------------------------
# Question Version Management Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/questions/{question_id}/versions")
async def get_question_versions(question_id: str):
    """문항 버전 이력 조회"""
    _init_components()
    assert _version_mgr is not None

    history = _version_mgr.get_version_history(question_id)
    if not history:
        raise HTTPException(404, f"문항을 찾을 수 없습니다: {question_id}")
    return {"question_id": question_id, "versions": history}


@app.get("/api/v1/questions/{question_id}")
async def get_question_detail(question_id: str):
    """문항 상세 정보 (현재 버전 + 메타데이터)"""
    _init_components()
    assert _version_mgr is not None

    record = _version_mgr.get_question(question_id)
    if not record:
        raise HTTPException(404, f"문항을 찾을 수 없습니다: {question_id}")
    return record


@app.get("/api/v1/questions/revalidation/due")
async def get_revalidation_due():
    """재검증 필요 문항 목록"""
    _init_components()
    assert _version_mgr is not None

    due = _version_mgr.get_revalidation_due()
    return {"due_count": len(due), "questions": due}


@app.get("/api/v1/questions/statistics/overview")
async def get_question_statistics():
    """문항 버전 관리 전체 통계"""
    _init_components()
    assert _version_mgr is not None

    return _version_mgr.get_statistics()


# ---------------------------------------------------------------------------
# Education Feedback Loop Endpoints (Kirkpatrick Level 2-3)
# ---------------------------------------------------------------------------

# In-memory feedback store (PoC — Phase 2에서 DB 전환)
_exam_responses: list[dict] = []
_trainee_feedback: list[dict] = []


@app.post("/api/v1/feedback/exam-response")
async def submit_exam_response(request: ExamResponseRequest):
    """시험 응시 결과 수집 (IRT 교정 데이터)"""
    _exam_responses.append(request.model_dump())
    return {"status": "recorded", "total_responses": len(_exam_responses)}


@app.post("/api/v1/feedback/exam-responses/batch")
async def submit_exam_responses_batch(responses: list[ExamResponseRequest]):
    """시험 응시 결과 배치 수집"""
    for r in responses:
        _exam_responses.append(r.model_dump())
    return {"status": "recorded", "batch_size": len(responses), "total": len(_exam_responses)}


@app.post("/api/v1/feedback/trainee-satisfaction")
async def submit_trainee_feedback(request: TraineeFeedbackRequest):
    """교육생 만족도 피드백 수집 (Kirkpatrick Level 1)"""
    _trainee_feedback.append(request.model_dump())
    return {"status": "recorded", "total_feedback": len(_trainee_feedback)}


@app.get("/api/v1/feedback/exam-stats/{question_id}")
async def get_exam_stats_for_question(question_id: str):
    """특정 문항의 응시 통계 (정답률, 선택지 분포)"""
    responses = [r for r in _exam_responses if r["question_id"] == question_id]
    if not responses:
        return {"question_id": question_id, "total_responses": 0}

    total = len(responses)
    correct = sum(1 for r in responses if r["is_correct"])
    answer_dist: dict[str, int] = {}
    for r in responses:
        ans = r.get("selected_answer", "")
        answer_dist[ans] = answer_dist.get(ans, 0) + 1

    avg_time = sum(r.get("response_time_seconds", 0) for r in responses) / total

    return {
        "question_id": question_id,
        "total_responses": total,
        "correct_rate": round(correct / total * 100, 1),
        "answer_distribution": answer_dist,
        "avg_response_time_seconds": round(avg_time, 1),
        "irt_ready": total >= 200,
    }


@app.get("/api/v1/feedback/satisfaction-summary")
async def get_satisfaction_summary():
    """교육 만족도 종합 요약"""
    if not _trainee_feedback:
        return {"total_feedback": 0}

    total = len(_trainee_feedback)
    avg_satisfaction = sum(f.get("overall_satisfaction", 0) for f in _trainee_feedback) / total
    avg_relevance = sum(f.get("content_relevance", 0) for f in _trainee_feedback) / total

    difficulty_dist: dict[str, int] = {}
    for f in _trainee_feedback:
        d = f.get("difficulty_perception", "unknown")
        difficulty_dist[d] = difficulty_dist.get(d, 0) + 1

    return {
        "total_feedback": total,
        "avg_satisfaction": round(avg_satisfaction, 2),
        "avg_content_relevance": round(avg_relevance, 2),
        "difficulty_distribution": difficulty_dist,
    }


# ---------------------------------------------------------------------------
# Hybrid Safety Tagging Endpoint
# ---------------------------------------------------------------------------


@app.post("/api/v1/safety/tag-hybrid")
async def tag_safety_hybrid(questions: list[dict], threshold: str = "safety_critical"):
    """Hybrid 안전 태깅 (키워드 1차 + LLM 2차)"""
    _init_components()
    assert _hybrid_tagger is not None

    try:
        result = await _hybrid_tagger.tag_batch_hybrid(
            questions=questions, llm_threshold=threshold,
        )
        return result
    except Exception as exc:
        logger.exception("Hybrid safety tagging failed")
        raise HTTPException(500, f"안전 태깅 중 오류: {exc}") from exc


# ---------------------------------------------------------------------------
# QTI Export Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/export/questions")
async def export_questions(request: ExportRequest):
    """문항 내보내기 (QTI/JSON/CSV)"""
    fmt = request.format.lower()
    if fmt == "qti":
        xml_str = QTIExporter.export_to_qti_xml(request.questions, request.title)
        return {"format": "qti", "content": xml_str}
    elif fmt == "csv":
        csv_str = QTIExporter.export_to_csv(request.questions)
        return {"format": "csv", "content": csv_str}
    else:
        json_str = QTIExporter.export_to_json(request.questions)
        return {"format": "json", "content": json_str}


# ---------------------------------------------------------------------------
# Adaptive Exam Endpoints
# ---------------------------------------------------------------------------

_adaptive_engine = AdaptiveDifficultyEngine()


@app.post("/api/v1/exam/adaptive-select")
async def adaptive_select_questions(request: AdaptiveExamRequest):
    """적응형 시험 문항 선별"""
    _init_components()

    # 문항 풀 확보
    if request.file_id:
        pool = _parsed_cache.get(request.file_id)
        if not pool:
            raise HTTPException(400, "파싱된 문항 풀이 없습니다.")
        # 슬라이드가 아닌 문항 풀이 필요 — 여기서는 생성된 문항 캐시 활용
        pool = []  # 실제로는 생성된 문항 DB에서 조회

    trainee = TraineeProfile(
        trainee_id=request.trainee_id,
        ability_estimate=request.ability_estimate,
        weak_topics=request.weak_topics,
    )
    config = ExamConfig(
        total_questions=request.total_questions,
        safety_critical_min=request.safety_critical_min,
        safety_related_min=request.safety_related_min,
    )

    # 버전 관리에서 active 문항 풀 조회
    assert _version_mgr is not None
    all_records = _version_mgr.get_statistics()
    # PoC: 문항 풀이 비어있으면 안내
    if all_records.get("total_questions", 0) == 0:
        return {
            "status": "no_pool",
            "message": "문항 풀이 비어있습니다. 먼저 문항을 생성하세요.",
        }

    return {
        "status": "ready",
        "trainee_id": request.trainee_id,
        "config": {
            "total_questions": config.total_questions,
            "safety_critical_min": config.safety_critical_min,
        },
        "message": "적응형 시험 엔진 준비 완료. 문항 풀 축적 후 활성화됩니다.",
    }


# ---------------------------------------------------------------------------
# Nuclear Glossary Endpoints
# ---------------------------------------------------------------------------

_glossary = NuclearGlossaryManager()


@app.get("/api/v1/glossary/search")
async def glossary_search(q: str, category: str = "", safety_grade: str = ""):
    """용어사전 검색"""
    results = _glossary.search(
        q, category=category or None, safety_grade=safety_grade or None,
    )
    return {"query": q, "results": results, "count": len(results)}


@app.get("/api/v1/glossary/statistics")
async def glossary_statistics():
    """용어사전 통계"""
    return _glossary.get_statistics()


@app.get("/api/v1/glossary/prompt-text")
async def glossary_prompt_text(category: str = "", safety_grade: str = ""):
    """Curated Prompt용 용어사전 텍스트"""
    text = _glossary.to_prompt_text(
        category=category or None, safety_grade=safety_grade or None,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Integrated Dashboard Endpoint
# ---------------------------------------------------------------------------


@app.get("/api/v1/dashboard")
async def get_dashboard():
    """
    통합 대시보드 — 전체 시스템 현황을 한눈에

    모든 모듈의 통계를 집계하여 반환
    """
    _init_components()

    dashboard = {
        "platform": "KHNP Education AI Platform",
        "version": "v9-poc",
        "modules": {
            "total": 19,
            "api_endpoints": 33,
        },
    }

    # 문항 버전 관리 통계
    if _version_mgr:
        dashboard["question_bank"] = _version_mgr.get_statistics()

    # SME 검토 통계
    if _sme_workflow:
        dashboard["sme_review"] = _sme_workflow.get_statistics()

    # 용어사전 통계
    dashboard["glossary"] = _glossary.get_statistics()

    # 교육 피드백 통계
    dashboard["feedback"] = {
        "total_exam_responses": len(_exam_responses),
        "total_trainee_feedback": len(_trainee_feedback),
        "irt_ready_questions": sum(
            1 for qid in set(r["question_id"] for r in _exam_responses)
            if sum(1 for r in _exam_responses if r["question_id"] == qid) >= 200
        ) if _exam_responses else 0,
    }

    # 파싱 캐시 현황
    dashboard["parsing_cache"] = {
        "cached_files": len(_parsed_cache),
        "total_cached_slides": sum(len(v) for v in _parsed_cache.values()),
    }

    # 시스템 구성요소 상태
    dashboard["components"] = {
        "document_parser": "ready" if _parser else "not_initialized",
        "vlm_analyzer": "ready" if _vlm else "not_initialized",
        "script_generator": "ready" if _script_gen else "not_initialized",
        "question_generator": "ready" if _question_gen else "not_initialized",
        "qa_validator": "ready" if _qa_validator else "not_initialized",
        "pipeline": "ready" if _pipeline else "not_initialized",
        "sme_workflow": "ready" if _sme_workflow else "not_initialized",
        "version_manager": "ready" if _version_mgr else "not_initialized",
        "hybrid_tagger": "ready" if _hybrid_tagger else "not_initialized",
        "glossary": "ready",
        "adaptive_engine": "ready",
        "qti_exporter": "ready",
        "llm_router": "ready",
    }

    # LLM 라우터 통계
    router = _get_router()
    dashboard["llm_router"] = router.get_statistics()
    dashboard["llm_router"]["models"] = {
        p.value: {"enabled": c.enabled, "model": c.model_name}
        for p, c in router.models.items()
    }

    return dashboard


# ---------------------------------------------------------------------------
# Prompt Template Management Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/prompts/list")
async def list_prompt_templates():
    """프롬프트 템플릿 목록 조회"""
    tm = get_template_manager()
    return {
        "templates": tm.list_templates(),
        "versions": tm.get_all_versions(),
    }


@app.get("/api/v1/prompts/{key}")
async def get_prompt_template(key: str):
    """특정 프롬프트 템플릿 조회"""
    tm = get_template_manager()
    config = tm.get_config(key)
    if not config:
        raise HTTPException(404, f"프롬프트 템플릿 '{key}'을 찾을 수 없습니다.")
    return {
        "key": key,
        "version": config.get("version", "unknown"),
        "system_prompt": config.get("system_prompt", ""),
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens"),
    }


class PromptOverrideRequest(BaseModel):
    key: str
    system_prompt: str


@app.post("/api/v1/prompts/override")
async def override_prompt(request: PromptOverrideRequest):
    """프롬프트 런타임 오버라이드 (A/B 테스트용)"""
    tm = get_template_manager()
    tm.set_override(request.key, request.system_prompt)
    return {
        "status": "overridden",
        "key": request.key,
        "prompt_length": len(request.system_prompt),
    }


@app.delete("/api/v1/prompts/override/{key}")
async def clear_prompt_override(key: str):
    """프롬프트 오버라이드 해제"""
    tm = get_template_manager()
    tm.clear_override(key)
    return {"status": "cleared", "key": key}


# ---------------------------------------------------------------------------
# LLM Hybrid Router Endpoints
# ---------------------------------------------------------------------------

_llm_router: LLMRouter | None = None


def _get_router() -> LLMRouter:
    global _llm_router
    if _llm_router is None:
        api_key = os.environ.get("UPSTAGE_API_KEY", "")
        atomic_key = os.environ.get("ATOMIC_GPT_API_KEY", "")
        atomic_url = os.environ.get("ATOMIC_GPT_BASE_URL", "http://localhost:8080/v1")
        _llm_router = create_default_router(
            solar_api_key=api_key,
            atomic_api_key=atomic_key,
            atomic_base_url=atomic_url,
        )
    return _llm_router


@app.get("/api/v1/router/statistics")
async def get_router_statistics():
    """LLM 라우터 호출 통계"""
    router = _get_router()
    stats = router.get_statistics()
    models_status = {
        provider.value: {
            "enabled": config.enabled,
            "model_name": config.model_name,
            "base_url": config.base_url[:30] + "...",
        }
        for provider, config in router.models.items()
    }
    return {**stats, "models": models_status}


@app.get("/api/v1/router/policies")
async def get_routing_policies():
    """현재 라우팅 정책 목록"""
    router = _get_router()
    policies = []
    for task_type in TaskType:
        policy = router.get_policy(task_type)
        policies.append({
            "task_type": task_type.value,
            "primary": policy.primary.value,
            "fallback": policy.fallback.value,
            "require_cross_check": policy.require_cross_check,
            "confidence_threshold": policy.confidence_threshold,
            "description": policy.description,
        })
    return {"policies": policies}


# ---------------------------------------------------------------------------
# Education Feedback Loop — 환류 자동화
# ---------------------------------------------------------------------------


@app.get("/api/v1/feedback/loop-status")
async def get_feedback_loop_status():
    """교육 환류 루프 현황 — IRT 교정 준비도 + 프롬프트 개선 트리거 상태"""
    _init_components()

    # 문항별 응시 데이터 집계
    question_response_counts: dict[str, int] = {}
    for r in _exam_responses:
        qid = r.get("question_id", "")
        question_response_counts[qid] = question_response_counts.get(qid, 0) + 1

    total_questions_tracked = len(question_response_counts)
    irt_ready = sum(1 for cnt in question_response_counts.values() if cnt >= 200)
    ctt_ready = sum(1 for cnt in question_response_counts.values() if cnt >= 50)

    # 만족도 기반 프롬프트 개선 트리거
    prompt_improvement_triggers = []
    if _trainee_feedback:
        avg_sat = sum(f.get("overall_satisfaction", 0) for f in _trainee_feedback) / len(_trainee_feedback)
        if avg_sat < 3.5:
            prompt_improvement_triggers.append({
                "trigger": "low_satisfaction",
                "value": round(avg_sat, 2),
                "threshold": 3.5,
                "recommendation": "교육 만족도 3.5 미만 — 스크립트 생성 프롬프트 개선 권장",
            })

        difficulty_dist: dict[str, int] = {}
        for f in _trainee_feedback:
            d = f.get("difficulty_perception", "")
            if d:
                difficulty_dist[d] = difficulty_dist.get(d, 0) + 1
        total_diff = sum(difficulty_dist.values())
        if total_diff > 0:
            too_hard_pct = difficulty_dist.get("too_hard", 0) / total_diff
            too_easy_pct = difficulty_dist.get("too_easy", 0) / total_diff
            if too_hard_pct > 0.3:
                prompt_improvement_triggers.append({
                    "trigger": "too_difficult",
                    "value": round(too_hard_pct * 100, 1),
                    "threshold": 30.0,
                    "recommendation": "30%+ 교육생이 '너무 어려움' — 문항 난이도 분포 하향 조정 권장",
                })
            if too_easy_pct > 0.4:
                prompt_improvement_triggers.append({
                    "trigger": "too_easy",
                    "value": round(too_easy_pct * 100, 1),
                    "threshold": 40.0,
                    "recommendation": "40%+ 교육생이 '너무 쉬움' — Application/Analysis 비중 상향 권장",
                })

    # SME 반려율 기반 트리거
    if _sme_workflow:
        sme_stats = _sme_workflow.get_statistics()
        total_reviews = sme_stats.get("total_reviews", 0)
        rejected = sme_stats.get("rejected_count", 0)
        if total_reviews > 10 and rejected / max(total_reviews, 1) > 0.2:
            prompt_improvement_triggers.append({
                "trigger": "high_sme_rejection",
                "value": round(rejected / total_reviews * 100, 1),
                "threshold": 20.0,
                "recommendation": "SME 반려율 20%+ — 문항 생성 프롬프트 or Few-shot 예시 보강 필요",
            })

    return {
        "feedback_loop": {
            "total_exam_responses": len(_exam_responses),
            "total_trainee_feedback": len(_trainee_feedback),
            "questions_tracked": total_questions_tracked,
            "irt_ready_questions": irt_ready,
            "ctt_ready_questions": ctt_ready,
            "irt_readiness_pct": round(irt_ready / max(total_questions_tracked, 1) * 100, 1),
        },
        "prompt_improvement_triggers": prompt_improvement_triggers,
        "loop_phase": (
            "Phase 3: IRT 적응형" if irt_ready >= 10
            else "Phase 2: CTT 기반 교정" if ctt_ready >= 5
            else "Phase 1: 데이터 수집 중"
        ),
        "next_milestone": (
            f"IRT 활성화까지 {max(10 - irt_ready, 0)}개 문항 추가 필요"
            if ctt_ready >= 5
            else f"CTT 분석까지 {max(5 - ctt_ready, 0)}개 문항 추가 필요 (각 50+ 응답)"
        ),
    }


@app.post("/api/v1/feedback/trigger-calibration")
async def trigger_calibration():
    """수동 IRT 교정 트리거 — 축적된 응시 데이터로 난이도 파라미터 갱신"""
    _init_components()
    assert _version_mgr is not None

    calibrated_count = 0
    question_responses: dict[str, list[dict]] = {}
    for r in _exam_responses:
        qid = r.get("question_id", "")
        if qid not in question_responses:
            question_responses[qid] = []
        question_responses[qid].append(r)

    results = []
    for qid, responses in question_responses.items():
        if len(responses) < 30:
            continue
        # 시험 후 분석
        analysis = ItemAnalysisSimulator.analyze_post_exam(qid, responses)
        results.append(analysis)
        calibrated_count += 1

    return {
        "status": "calibration_complete",
        "calibrated_questions": calibrated_count,
        "results": results[:20],  # 상위 20개만 반환
    }


# ---------------------------------------------------------------------------
# Prompt A/B Testing Endpoints
# ---------------------------------------------------------------------------

_ab_manager = PromptABTestManager()


class ABExperimentRequest(BaseModel):
    prompt_key: str
    variants: list[dict]  # [{"variant_id": "A", "system_prompt": "...", "description": "..."}]
    description: str = ""
    min_samples: int = 30


class ABResultRequest(BaseModel):
    experiment_id: str
    variant_id: str
    qa_pass: bool = False
    quality_score: float = 0.0
    distractor_quality: float = 0.0
    nonfunctional_distractors: int = 0
    generation_time_ms: float = 0.0


@app.post("/api/v1/ab-test/create")
async def create_ab_experiment(request: ABExperimentRequest):
    """프롬프트 A/B 테스트 실험 생성"""
    variants = [
        PromptVariant(
            variant_id=v["variant_id"],
            system_prompt=v.get("system_prompt", ""),
            description=v.get("description", ""),
            temperature=v.get("temperature"),
            max_tokens=v.get("max_tokens"),
        )
        for v in request.variants
    ]
    exp_id = _ab_manager.create_experiment(
        prompt_key=request.prompt_key,
        variants=variants,
        description=request.description,
        min_samples=request.min_samples,
    )
    return {"experiment_id": exp_id, "variants": [v.variant_id for v in variants]}


@app.get("/api/v1/ab-test/list")
async def list_ab_experiments():
    """실험 목록 조회"""
    return {"experiments": _ab_manager.list_experiments()}


@app.get("/api/v1/ab-test/{experiment_id}")
async def get_ab_experiment(experiment_id: str):
    """실험 결과 분석"""
    return _ab_manager.get_experiment_results(experiment_id)


@app.post("/api/v1/ab-test/{experiment_id}/assign")
async def assign_ab_variant(experiment_id: str):
    """변형 할당 (균등 분배)"""
    variant = _ab_manager.assign_variant(experiment_id)
    if not variant:
        raise HTTPException(404, "활성 실험을 찾을 수 없습니다.")
    return {
        "variant_id": variant.variant_id,
        "description": variant.description,
        "system_prompt_length": len(variant.system_prompt),
    }


@app.post("/api/v1/ab-test/{experiment_id}/record")
async def record_ab_result(experiment_id: str, request: ABResultRequest):
    """실험 결과 기록"""
    result = ExperimentResult(
        variant_id=request.variant_id,
        qa_pass=request.qa_pass,
        quality_score=request.quality_score,
        distractor_quality=request.distractor_quality,
        nonfunctional_distractors=request.nonfunctional_distractors,
        generation_time_ms=request.generation_time_ms,
    )
    _ab_manager.record_result(experiment_id, result)
    return {"status": "recorded", "experiment_id": experiment_id}


# ---------------------------------------------------------------------------
# 웹 정적 파일 서빙
# ---------------------------------------------------------------------------

_web_dir = Path(__file__).resolve().parent.parent.parent / "web"


@app.get("/")
async def serve_index():
    """메인 웹 페이지"""
    index_file = _web_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "KHNP Education AI Platform API", "docs": "/docs"}


# StaticFiles는 모든 라우트 뒤에 마운트 (catch-all 방지)
_static_dir = _web_dir / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static-files")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
