"""
KHNP Education AI Platform - End-to-End 파이프라인 오케스트레이터

전체 흐름 (v3 - 11단계):
  1. 강의안 + 참고자료 업로드
  2. Document Parse (Standard/Enhanced/Auto)
  3. VLM 시각자료 심층 분석
  4. Golden Context 구성 (참고자료 파싱 + Curated Prompt)
  5. 강의 스크립트 생성
  6. 문제은행 생성 (Few-shot + 2-Pass)
  7. 안전 중요도 태깅 (Graded Approach)
  8. 품질 검증 (QA Pipeline)
  9. 예상 난이도 분석 (Item Analysis)
  10. 버전 관리 등록 + 자동 폐기 후보 추출
  11. 스크립트 품질 자동 평가
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.document_processor.parser import DocumentParser
from src.document_processor.vlm_analyzer import VLMAnalyzer, GoldenContextBuilder
from src.script_generator.generator import ScriptGenerator
from src.question_generator.generator import QuestionGenerator
from src.question_generator.quality_validator import (
    QuestionQualityValidator,
    ItemAnalysisSimulator,
)
from src.question_generator.safety_tagger import SafetySignificanceTagger
from src.question_generator.few_shot_manager import FewShotManager
from src.question_generator.version_manager import QuestionVersionManager, QuestionLifecycle


@dataclass
class PipelineInput:
    """파이프라인 입력"""
    slide_file: str                          # 강의안 PDF/PPTX
    reference_files: list[str] = field(default_factory=list)  # 참고자료
    learning_objectives: list[str] = field(default_factory=list)
    audience_level: str = "intermediate"     # beginner|intermediate|advanced
    num_questions: int = 20
    bloom_distribution: dict = field(default_factory=lambda: {
        "Knowledge": 0.3,
        "Comprehension": 0.3,
        "Application": 0.25,
        "Analysis": 0.15,
    })
    curated_prompt_data: dict = field(default_factory=dict)


@dataclass
class PipelineOutput:
    """파이프라인 출력"""
    slides: list[dict] = field(default_factory=list)
    scripts: list[dict] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    quality_report: dict = field(default_factory=dict)
    script_quality: dict = field(default_factory=dict)
    retirement_candidates: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class EducationAIPipeline:
    """메인 파이프라인 오케스트레이터"""

    def __init__(self, api_key: str, base_url: str = "https://api.upstage.ai/v1"):
        self.parser = DocumentParser(api_key, base_url)
        self.vlm = VLMAnalyzer(api_key, base_url)
        self.context_builder = GoldenContextBuilder(self.parser)
        self.script_gen = ScriptGenerator(api_key, base_url)
        self.question_gen = QuestionGenerator(api_key, base_url)
        self.qa_validator = QuestionQualityValidator(api_key, base_url)
        self.safety_tagger = SafetySignificanceTagger()
        self.few_shot_mgr = FewShotManager()
        self.version_mgr = QuestionVersionManager()

    async def run(self, input_data: PipelineInput) -> PipelineOutput:
        """전체 파이프라인 실행"""
        output = PipelineOutput()
        output.metadata["started_at"] = datetime.now().isoformat()
        output.metadata["input_file"] = input_data.slide_file

        # === Step 1: Document Parse ===
        parse_result = await self.parser.parse_document(
            input_data.slide_file, mode="auto"
        )
        slides = self.parser.parse_result_to_slides(parse_result)
        output.metadata["total_slides"] = len(slides)

        # === Step 2: VLM 시각자료 심층 분석 ===
        enriched_slides = await self.vlm.enrich_slides_with_vlm(slides)
        output.slides = enriched_slides
        vlm_count = sum(
            1 for s in enriched_slides if s.get("metadata", {}).get("vlm_enriched")
        )
        output.metadata["vlm_enriched_slides"] = vlm_count

        # === Step 3: Golden Context 구성 ===
        golden_context = ""
        if input_data.reference_files:
            golden_context = await self.context_builder.build_context(
                input_data.reference_files
            )
        output.metadata["reference_files_count"] = len(input_data.reference_files)

        curated_prompt = ""
        if input_data.curated_prompt_data:
            curated_prompt = self.context_builder.build_curated_prompt(
                **input_data.curated_prompt_data
            )

        # === Step 4: 강의 스크립트 생성 ===
        async def rag_search_fn(query):
            """Golden Context에서 관련 부분 추출 (간이 RAG)"""
            if not golden_context:
                return []
            # 간단한 키워드 매칭 (PoC 수준)
            paragraphs = golden_context.split("\n\n")
            query_words = set(query.lower().split())
            scored = []
            for para in paragraphs:
                if len(para.strip()) < 20:
                    continue
                para_words = set(para.lower().split())
                overlap = len(query_words & para_words)
                if overlap > 0:
                    scored.append({"text": para, "score": overlap})
            scored.sort(key=lambda x: -x["score"])
            return scored[:5]

        scripts = await self.script_gen.generate_full_script(
            slides=enriched_slides,
            learning_objectives=input_data.learning_objectives,
            rag_search_fn=rag_search_fn,
            audience_level=input_data.audience_level,
            curated_prompt=curated_prompt,
        )
        output.scripts = scripts

        # === Step 5: 문제은행 생성 (Few-shot + 2-Pass) ===
        rag_context = await rag_search_fn(
            " ".join(input_data.learning_objectives[:3])
        )
        few_shot_examples = self.few_shot_mgr.get_examples(
            category="question", sme_approved_only=True, max_count=3,
        )
        questions = await self.question_gen.generate_questions(
            slides=enriched_slides,
            learning_objectives=input_data.learning_objectives,
            rag_context=rag_context,
            bloom_distribution=input_data.bloom_distribution,
            num_questions=input_data.num_questions,
            few_shot_examples=few_shot_examples if few_shot_examples else None,
        )

        # === Step 6: 안전 중요도 태깅 (IAEA TECDOC-2082 Graded Approach) ===
        questions = self.safety_tagger.tag_batch(questions)
        safety_stats = self.safety_tagger.get_safety_statistics(questions)
        output.metadata["safety_statistics"] = safety_stats
        output.questions = questions

        # === Step 7: 품질 검증 ===
        quality_report = await self.qa_validator.validate_batch(
            questions=questions,
            reference_context=golden_context[:5000],
        )
        output.quality_report = quality_report

        # === Step 8: 예상 난이도 분석 ===
        for q in output.questions:
            q["estimated_item_analysis"] = ItemAnalysisSimulator.estimate_difficulty(q)

        # === Step 9: 버전 관리 등록 + 자동 폐기 후보 추출 ===
        for q in output.questions:
            self.version_mgr.create_question(q, created_by="AI:Pipeline")

        # 재검증 필요 문항 확인
        revalidation_due = self.version_mgr.get_revalidation_due()
        output.metadata["revalidation_due_count"] = len(revalidation_due)

        # 자동 폐기 후보: QA 미통과 + safety_critical인 문항
        retirement_candidates = []
        qa_results = quality_report.get("results", [])
        for qa_r in qa_results:
            qid = qa_r.get("question_id", "")
            passed = qa_r.get("passed", True)
            grade = ""
            for q in output.questions:
                if q.get("question_id") == qid:
                    grade = q.get("safety_significance", {}).get("grade", "general")
                    break
            if not passed and grade == "safety_critical":
                retirement_candidates.append({
                    "question_id": qid,
                    "safety_grade": grade,
                    "reason": "QA 미통과 + safety_critical",
                    "action": "SME 재검토 또는 재생성 권장",
                })
        output.retirement_candidates = retirement_candidates

        # === Step 10: 스크립트 품질 자동 평가 ===
        output.script_quality = self._evaluate_script_quality(scripts, enriched_slides)

        output.metadata["completed_at"] = datetime.now().isoformat()
        output.metadata["total_scripts"] = len(scripts)
        output.metadata["total_questions"] = len(questions)
        output.metadata["qa_pass_rate"] = quality_report.get("statistics", {}).get("pass_rate", "N/A")
        output.metadata["retirement_candidates_count"] = len(retirement_candidates)

        return output

    @staticmethod
    def _evaluate_script_quality(scripts: list[dict], slides: list[dict]) -> dict:
        """
        스크립트 품질 자동 평가

        평가 기준:
        1. 완전성: 모든 슬라이드에 스크립트가 생성되었는가
        2. 시각자료 활용: VLM 데이터가 있는 슬라이드에 시각 설명이 포함되었는가
        3. 구조 준수: 5단계 구조가 모두 포함되었는가
        4. 분량 적정성: 슬라이드당 스크립트 길이가 적정한가
        """
        total_slides = len(slides)
        total_scripts = len(scripts)
        vlm_slides = sum(
            1 for s in slides if s.get("vlm_analyses") or s.get("metadata", {}).get("vlm_enriched")
        )

        # 구조 검사
        required_sections = ["introduction", "main_content", "summary"]
        optional_sections = ["visual_explanation", "contextual_supplement"]
        structure_scores = []
        visual_utilization = []
        length_scores = []

        for i, script in enumerate(scripts):
            script_body = script.get("script", {})

            # 구조 점수
            present = sum(1 for s in required_sections if script_body.get(s))
            optional_present = sum(1 for s in optional_sections if script_body.get(s))
            struct_score = (present / len(required_sections)) * 70 + (optional_present / len(optional_sections)) * 30
            structure_scores.append(struct_score)

            # 시각자료 활용 점수
            if i < len(slides):
                has_vlm = bool(slides[i].get("vlm_analyses") or slides[i].get("metadata", {}).get("vlm_enriched"))
                has_visual_script = bool(script_body.get("visual_explanation"))
                if has_vlm:
                    visual_utilization.append(100.0 if has_visual_script else 0.0)

            # 분량 점수 (200~800자 적정)
            total_text = " ".join(str(v) for v in script_body.values() if isinstance(v, str))
            char_count = len(total_text)
            if 200 <= char_count <= 800:
                length_scores.append(100.0)
            elif 100 <= char_count < 200 or 800 < char_count <= 1200:
                length_scores.append(70.0)
            else:
                length_scores.append(40.0)

        completeness = round(total_scripts / max(total_slides, 1) * 100, 1)
        avg_structure = round(sum(structure_scores) / max(len(structure_scores), 1), 1)
        avg_visual = round(sum(visual_utilization) / max(len(visual_utilization), 1), 1) if visual_utilization else 0.0
        avg_length = round(sum(length_scores) / max(len(length_scores), 1), 1)

        overall = round((completeness * 0.2 + avg_structure * 0.3 + avg_visual * 0.3 + avg_length * 0.2), 1)

        return {
            "overall_score": overall,
            "completeness": completeness,
            "structure_adherence": avg_structure,
            "visual_utilization": avg_visual,
            "length_appropriateness": avg_length,
            "total_slides": total_slides,
            "total_scripts": total_scripts,
            "vlm_slides": vlm_slides,
            "recommendation": (
                "우수: 스크립트 품질이 전반적으로 양호합니다." if overall >= 80
                else "보통: 시각자료 활용 또는 구조 준수를 개선하세요." if overall >= 60
                else "개선 필요: 스크립트 생성 품질을 점검하세요."
            ),
        }

    async def save_output(self, output: PipelineOutput, output_dir: str):
        """결과를 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 스크립트 저장
        with open(f"{output_dir}/scripts_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(output.scripts, f, ensure_ascii=False, indent=2)

        # 문제은행 저장
        with open(f"{output_dir}/questions_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(output.questions, f, ensure_ascii=False, indent=2)

        # 품질 리포트 저장
        with open(f"{output_dir}/quality_report_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(output.quality_report, f, ensure_ascii=False, indent=2)

        # 메타데이터 저장
        with open(f"{output_dir}/metadata_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(output.metadata, f, ensure_ascii=False, indent=2)

        return output_dir
