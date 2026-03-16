"""
SME 검토 워크플로우 모듈

AI 생성 문항 → SME 검토/수정 → Diff 분석 → 프롬프트 개선 학습 체계

핵심 설계:
- AI 초안과 SME 수정본의 diff를 체계적으로 추적
- 반복되는 수정 패턴을 자동 감지하여 프롬프트 개선에 반영
- 안전등급별 차별화된 검토 기준 적용 (IAEA TECDOC-2082)
- SME 피드백 누적으로 Few-shot 예시 자동 후보 추천

연구 근거:
- SME 수정 이력 학습 시 2세대 이후 수정률 40% 감소 (내부 목표)
- 안전 관련 문항의 SME 검토는 규제 요건 (IAEA NS-G-2.8)
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from collections import Counter


class ReviewStatus(str, Enum):
    PENDING = "pending"           # SME 검토 대기
    IN_REVIEW = "in_review"       # SME 검토 중
    APPROVED = "approved"         # 승인 (수정 없음)
    REVISED = "revised"           # 수정 후 승인
    REJECTED = "rejected"         # 반려 (재생성 필요)
    ESCALATED = "escalated"       # 상위 검토자 에스컬레이션


class RevisionCategory(str, Enum):
    """SME 수정 유형 분류"""
    TECHNICAL_ERROR = "technical_error"           # 기술적 오류 (잘못된 수치, 절차)
    DISTRACTOR_QUALITY = "distractor_quality"     # 오답지 품질 (비기능적 → 기능적)
    QUESTION_CLARITY = "question_clarity"         # 문항 명확성 (모호한 표현)
    SAFETY_ACCURACY = "safety_accuracy"           # 안전 관련 정확성
    TERMINOLOGY = "terminology"                   # 용어 부정확 (현장 용어 vs 교재 용어)
    BLOOM_MISMATCH = "bloom_mismatch"             # Bloom 수준 부적합
    SCENARIO_REALISM = "scenario_realism"         # 시나리오 현실성 부족
    EXPLANATION_DEPTH = "explanation_depth"        # 해설 깊이 부족
    REGULATORY_COMPLIANCE = "regulatory_compliance"  # 규정/절차서 불일치
    OTHER = "other"


@dataclass
class FieldDiff:
    """단일 필드 변경 기록"""
    field_path: str          # e.g., "options.B", "explanation", "question_text"
    original_value: str
    revised_value: str
    category: str = ""       # RevisionCategory
    sme_comment: str = ""    # SME의 수정 사유


@dataclass
class ReviewRecord:
    """단일 문항 검토 기록"""
    review_id: str
    question_id: str
    reviewer_name: str
    reviewer_role: str = ""          # "교수", "현장전문가", "규제전문가"
    status: str = ReviewStatus.PENDING
    safety_grade: str = ""           # safety_critical | safety_related | general

    # AI 원본
    original_question: dict = field(default_factory=dict)

    # SME 수정본
    revised_question: dict = field(default_factory=dict)

    # 변경 내역
    diffs: list = field(default_factory=list)  # list of FieldDiff as dict
    revision_categories: list = field(default_factory=list)  # 수정 유형 목록

    # 메타
    overall_comment: str = ""
    quality_score_before: float = 0.0  # AI QA 점수
    quality_score_after: float = 0.0   # SME 판정 점수
    review_time_minutes: float = 0.0   # 검토 소요 시간
    created_at: str = ""
    completed_at: str = ""

    # Few-shot 후보 여부
    few_shot_candidate: bool = False
    few_shot_reason: str = ""


class SMEReviewWorkflow:
    """
    SME 검토 워크플로우 관리

    JSON 파일 기반 저장 (PoC). Phase 2에서 DB 전환.

    워크플로우:
    1. submit_for_review() - AI 생성 문항을 검토 큐에 등록
    2. start_review() - SME가 검토 시작
    3. submit_revision() - SME가 수정/승인/반려 제출
    4. analyze_diffs() - 수정 패턴 자동 분석
    5. get_improvement_insights() - 프롬프트 개선 인사이트 추출
    """

    def __init__(self, storage_dir: str = "data/sme_reviews"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._reviews: dict[str, ReviewRecord] = {}
        self._load_all()

    def _reviews_file(self) -> Path:
        return self.storage_dir / "reviews.json"

    def _insights_file(self) -> Path:
        return self.storage_dir / "improvement_insights.json"

    def _load_all(self):
        """저장된 검토 기록 로드"""
        fpath = self._reviews_file()
        if not fpath.exists():
            return
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            rec = ReviewRecord(**item)
            self._reviews[rec.review_id] = rec

    def _save_all(self):
        """전체 검토 기록 저장"""
        with open(self._reviews_file(), "w", encoding="utf-8") as f:
            json.dump(
                [asdict(r) for r in self._reviews.values()],
                f, ensure_ascii=False, indent=2,
            )

    # ---------------------------------------------------------------
    # Step 1: 검토 등록
    # ---------------------------------------------------------------
    def submit_for_review(
        self,
        question: dict,
        reviewer_name: str,
        reviewer_role: str = "",
        quality_score: float = 0.0,
    ) -> str:
        """AI 생성 문항을 SME 검토 큐에 등록"""
        question_id = question.get("question_id", "unknown")
        review_id = f"REV-{question_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        record = ReviewRecord(
            review_id=review_id,
            question_id=question_id,
            reviewer_name=reviewer_name,
            reviewer_role=reviewer_role,
            status=ReviewStatus.PENDING,
            safety_grade=question.get("safety_grade", "general"),
            original_question=question,
            quality_score_before=quality_score,
            created_at=datetime.now().isoformat(),
        )
        self._reviews[review_id] = record
        self._save_all()
        return review_id

    def submit_batch_for_review(
        self,
        questions: list[dict],
        reviewer_name: str,
        reviewer_role: str = "",
    ) -> list[str]:
        """배치 검토 등록"""
        review_ids = []
        for q in questions:
            rid = self.submit_for_review(
                q, reviewer_name, reviewer_role,
                quality_score=q.get("quality_score", 0.0),
            )
            review_ids.append(rid)
        return review_ids

    # ---------------------------------------------------------------
    # Step 2: 검토 시작
    # ---------------------------------------------------------------
    def start_review(self, review_id: str) -> ReviewRecord:
        """SME가 검토를 시작함"""
        rec = self._reviews.get(review_id)
        if not rec:
            raise ValueError(f"검토 기록을 찾을 수 없습니다: {review_id}")
        rec.status = ReviewStatus.IN_REVIEW
        self._save_all()
        return rec

    # ---------------------------------------------------------------
    # Step 3: 수정/승인/반려 제출
    # ---------------------------------------------------------------
    def submit_revision(
        self,
        review_id: str,
        status: str,
        revised_question: Optional[dict] = None,
        overall_comment: str = "",
        quality_score_after: float = 0.0,
        review_time_minutes: float = 0.0,
        few_shot_candidate: bool = False,
        few_shot_reason: str = "",
    ) -> ReviewRecord:
        """
        SME 검토 결과 제출

        Args:
            review_id: 검토 ID
            status: approved | revised | rejected | escalated
            revised_question: 수정된 문항 (revised 시 필수)
            overall_comment: 종합 의견
            quality_score_after: SME 판정 품질 점수
            review_time_minutes: 검토 소요 시간 (분)
            few_shot_candidate: Few-shot 모범 문항 후보 추천 여부
            few_shot_reason: 추천 사유
        """
        rec = self._reviews.get(review_id)
        if not rec:
            raise ValueError(f"검토 기록을 찾을 수 없습니다: {review_id}")

        rec.status = status
        rec.overall_comment = overall_comment
        rec.quality_score_after = quality_score_after
        rec.review_time_minutes = review_time_minutes
        rec.completed_at = datetime.now().isoformat()
        rec.few_shot_candidate = few_shot_candidate
        rec.few_shot_reason = few_shot_reason

        if status == ReviewStatus.REVISED and revised_question:
            rec.revised_question = revised_question
            rec.diffs = self._compute_diffs(rec.original_question, revised_question)
            rec.revision_categories = self._classify_revisions(rec.diffs)

        self._save_all()
        return rec

    # ---------------------------------------------------------------
    # Diff 계산
    # ---------------------------------------------------------------
    @staticmethod
    def _compute_diffs(original: dict, revised: dict) -> list[dict]:
        """원본과 수정본의 차이를 필드 단위로 계산"""
        diffs = []
        compare_fields = [
            "question_text", "correct_answer", "explanation",
            "learning_objective", "bloom_level", "scenario",
        ]

        for fld in compare_fields:
            orig_val = str(original.get(fld, ""))
            rev_val = str(revised.get(fld, ""))
            if orig_val != rev_val and (orig_val or rev_val):
                diffs.append(asdict(FieldDiff(
                    field_path=fld,
                    original_value=orig_val,
                    revised_value=rev_val,
                )))

        # 선택지 비교
        orig_opts = original.get("options", {})
        rev_opts = revised.get("options", {})
        all_keys = set(list(orig_opts.keys()) + list(rev_opts.keys()))
        for key in sorted(all_keys):
            orig_val = str(orig_opts.get(key, ""))
            rev_val = str(rev_opts.get(key, ""))
            if orig_val != rev_val:
                diffs.append(asdict(FieldDiff(
                    field_path=f"options.{key}",
                    original_value=orig_val,
                    revised_value=rev_val,
                )))

        # 오답 근거 비교
        orig_rat = original.get("distractor_rationale", {})
        rev_rat = revised.get("distractor_rationale", {})
        all_rat_keys = set(list(orig_rat.keys()) + list(rev_rat.keys()))
        for key in sorted(all_rat_keys):
            orig_val = str(orig_rat.get(key, ""))
            rev_val = str(rev_rat.get(key, ""))
            if orig_val != rev_val:
                diffs.append(asdict(FieldDiff(
                    field_path=f"distractor_rationale.{key}",
                    original_value=orig_val,
                    revised_value=rev_val,
                )))

        return diffs

    @staticmethod
    def _classify_revisions(diffs: list[dict]) -> list[str]:
        """수정 내역으로부터 수정 유형 자동 분류"""
        categories = set()
        for diff in diffs:
            path = diff.get("field_path", "")
            cat = diff.get("category", "")

            # SME가 직접 분류한 경우 우선
            if cat:
                categories.add(cat)
                continue

            # 필드 경로 기반 자동 분류
            if path.startswith("options."):
                categories.add(RevisionCategory.DISTRACTOR_QUALITY)
            elif path == "question_text":
                categories.add(RevisionCategory.QUESTION_CLARITY)
            elif path == "explanation":
                categories.add(RevisionCategory.EXPLANATION_DEPTH)
            elif path == "bloom_level":
                categories.add(RevisionCategory.BLOOM_MISMATCH)
            elif path == "scenario":
                categories.add(RevisionCategory.SCENARIO_REALISM)
            elif path == "correct_answer":
                categories.add(RevisionCategory.TECHNICAL_ERROR)
            elif path.startswith("distractor_rationale."):
                categories.add(RevisionCategory.DISTRACTOR_QUALITY)

        return list(categories)

    # ---------------------------------------------------------------
    # Step 4: 수정 패턴 분석
    # ---------------------------------------------------------------
    def analyze_revision_patterns(self) -> dict:
        """
        전체 검토 이력에서 반복되는 수정 패턴을 분석

        Returns:
            {
                total_reviews, revision_rate, avg_review_time,
                category_distribution, field_change_frequency,
                safety_grade_stats, top_issues, few_shot_candidates
            }
        """
        completed = [
            r for r in self._reviews.values()
            if r.status in (ReviewStatus.APPROVED, ReviewStatus.REVISED, ReviewStatus.REJECTED)
        ]
        if not completed:
            return {"total_reviews": 0, "message": "검토 완료된 문항이 없습니다."}

        revised = [r for r in completed if r.status == ReviewStatus.REVISED]
        rejected = [r for r in completed if r.status == ReviewStatus.REJECTED]

        # 수정 유형 분포
        all_categories = []
        for r in revised:
            all_categories.extend(r.revision_categories)
        category_counts = dict(Counter(all_categories).most_common())

        # 필드별 변경 빈도
        field_changes = []
        for r in revised:
            for d in r.diffs:
                field_changes.append(d.get("field_path", ""))
        field_counts = dict(Counter(field_changes).most_common())

        # 안전등급별 통계
        safety_stats = {}
        for grade in ["safety_critical", "safety_related", "general"]:
            grade_reviews = [r for r in completed if r.safety_grade == grade]
            grade_revised = [r for r in grade_reviews if r.status == ReviewStatus.REVISED]
            if grade_reviews:
                safety_stats[grade] = {
                    "total": len(grade_reviews),
                    "revision_rate": round(len(grade_revised) / len(grade_reviews) * 100, 1),
                    "avg_review_time": round(
                        sum(r.review_time_minutes for r in grade_reviews) / len(grade_reviews), 1
                    ),
                }

        # 평균 검토 시간
        review_times = [r.review_time_minutes for r in completed if r.review_time_minutes > 0]
        avg_review_time = round(sum(review_times) / max(len(review_times), 1), 1)

        # Few-shot 후보 목록
        candidates = [
            {
                "review_id": r.review_id,
                "question_id": r.question_id,
                "reason": r.few_shot_reason,
                "quality_score": r.quality_score_after,
            }
            for r in completed if r.few_shot_candidate
        ]

        return {
            "total_reviews": len(completed),
            "approved": len([r for r in completed if r.status == ReviewStatus.APPROVED]),
            "revised": len(revised),
            "rejected": len(rejected),
            "revision_rate": round(len(revised) / len(completed) * 100, 1),
            "rejection_rate": round(len(rejected) / len(completed) * 100, 1),
            "avg_review_time_minutes": avg_review_time,
            "category_distribution": category_counts,
            "field_change_frequency": field_counts,
            "safety_grade_stats": safety_stats,
            "few_shot_candidates": candidates,
        }

    # ---------------------------------------------------------------
    # Step 5: 프롬프트 개선 인사이트
    # ---------------------------------------------------------------
    def get_improvement_insights(self) -> dict:
        """
        수정 패턴에서 프롬프트 개선 인사이트를 추출

        핵심: 반복되는 수정 패턴 → 시스템 프롬프트/Few-shot에 반영
        """
        patterns = self.analyze_revision_patterns()
        if patterns.get("total_reviews", 0) == 0:
            return {"insights": [], "message": "분석할 데이터가 부족합니다."}

        insights = []
        cat_dist = patterns.get("category_distribution", {})
        total = patterns.get("total_reviews", 1)

        # 오답지 품질 문제가 잦은 경우
        distractor_count = cat_dist.get(RevisionCategory.DISTRACTOR_QUALITY, 0)
        if distractor_count > 0:
            rate = round(distractor_count / total * 100, 1)
            insights.append({
                "priority": "high" if rate > 30 else "medium",
                "category": "distractor_quality",
                "frequency": distractor_count,
                "rate_percent": rate,
                "recommendation": (
                    f"오답지 수정 비율 {rate}%. "
                    "2-Pass 검증 프롬프트에 수정된 오답지 패턴을 Few-shot으로 추가 권장. "
                    "특히 '너무 쉽게 배제 가능한 오답'과 '정답과 무관한 오답' 패턴 강화 필요."
                ),
                "prompt_action": "QUESTION_PROMPT의 distractor 생성 지침 강화",
            })

        # 기술적 오류가 잦은 경우
        tech_count = cat_dist.get(RevisionCategory.TECHNICAL_ERROR, 0)
        if tech_count > 0:
            rate = round(tech_count / total * 100, 1)
            insights.append({
                "priority": "critical" if rate > 20 else "high",
                "category": "technical_error",
                "frequency": tech_count,
                "rate_percent": rate,
                "recommendation": (
                    f"기술적 오류 수정 비율 {rate}%. "
                    "Golden Context의 참고자료 품질 점검 필요. "
                    "RAG 검색 정확도 개선 또는 참고자료 추가 필요."
                ),
                "prompt_action": "Golden Context에 기술 검증 체크리스트 추가",
            })

        # 용어 문제
        term_count = cat_dist.get(RevisionCategory.TERMINOLOGY, 0)
        if term_count > 0:
            insights.append({
                "priority": "medium",
                "category": "terminology",
                "frequency": term_count,
                "recommendation": (
                    "현장 용어와 교재 용어 불일치 사례 발생. "
                    "Curated Prompt의 용어집(glossary) 업데이트 필요."
                ),
                "prompt_action": "시스템 프롬프트 용어집 섹션 보강",
            })

        # 시나리오 현실성
        scenario_count = cat_dist.get(RevisionCategory.SCENARIO_REALISM, 0)
        if scenario_count > 0:
            insights.append({
                "priority": "high",
                "category": "scenario_realism",
                "frequency": scenario_count,
                "recommendation": (
                    "시나리오의 계기 수치/절차 현실성 부족. "
                    "실제 비상운전절차서(EOP) 수치를 Few-shot 예시에 포함 권장."
                ),
                "prompt_action": "SCENARIO_PROMPT에 실제 EOP 수치 범위 명시",
            })

        # 안전 관련 정확성
        safety_count = cat_dist.get(RevisionCategory.SAFETY_ACCURACY, 0)
        if safety_count > 0:
            insights.append({
                "priority": "critical",
                "category": "safety_accuracy",
                "frequency": safety_count,
                "recommendation": (
                    "안전 관련 문항의 정확성 문제. "
                    "Safety-critical 문항에 대한 LLM 2차 검증 강화 필요. "
                    "관련 규정/기술기준 문서를 Golden Context에 필수 포함."
                ),
                "prompt_action": "안전 문항 생성 시 관련 규정 원문 첨부 필수화",
            })

        # Bloom 수준 부적합
        bloom_count = cat_dist.get(RevisionCategory.BLOOM_MISMATCH, 0)
        if bloom_count > 0:
            insights.append({
                "priority": "medium",
                "category": "bloom_mismatch",
                "frequency": bloom_count,
                "recommendation": (
                    "Bloom 수준 판정 오류. "
                    "각 Bloom 수준별 문항 특성을 Few-shot 예시에서 더 명확히 구분 필요."
                ),
                "prompt_action": "Bloom 수준별 문항 차이를 예시로 명시",
            })

        # 인사이트 우선순위 정렬
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        insights.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

        result = {
            "generated_at": datetime.now().isoformat(),
            "total_reviews_analyzed": patterns["total_reviews"],
            "revision_rate": patterns.get("revision_rate", 0),
            "insights": insights,
            "few_shot_candidates_count": len(patterns.get("few_shot_candidates", [])),
        }

        # 인사이트 저장
        with open(self._insights_file(), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    # ---------------------------------------------------------------
    # 조회 헬퍼
    # ---------------------------------------------------------------
    def get_pending_reviews(self, reviewer_name: Optional[str] = None) -> list[dict]:
        """대기 중인 검토 목록"""
        pending = [
            r for r in self._reviews.values()
            if r.status in (ReviewStatus.PENDING, ReviewStatus.IN_REVIEW)
        ]
        if reviewer_name:
            pending = [r for r in pending if r.reviewer_name == reviewer_name]
        return [asdict(r) for r in pending]

    def get_review(self, review_id: str) -> Optional[dict]:
        """특정 검토 기록 조회"""
        rec = self._reviews.get(review_id)
        return asdict(rec) if rec else None

    def get_reviews_by_question(self, question_id: str) -> list[dict]:
        """특정 문항의 전체 검토 이력"""
        return [
            asdict(r) for r in self._reviews.values()
            if r.question_id == question_id
        ]

    def get_statistics(self) -> dict:
        """검토 워크플로우 전체 통계"""
        all_reviews = list(self._reviews.values())
        status_counts = dict(Counter(r.status for r in all_reviews))
        return {
            "total": len(all_reviews),
            "by_status": status_counts,
            "patterns": self.analyze_revision_patterns(),
        }
