"""
문항 버전 관리 및 이력 추적 시스템

핵심 기능:
- 문항 생성/수정/폐기 전 이력 추적 (Git-like versioning)
- AI 생성 → SME 수정 → 시험 출제 → IRT 교정 전체 라이프사이클 관리
- 안전등급별 재검증 주기 자동 관리 (IAEA Graded Approach)
- 문항 폐기 사유 및 대체 문항 추적

설계 원칙:
- 원본 보존: 모든 버전은 불변(immutable), 수정 시 새 버전 생성
- 추적성: 누가, 언제, 왜 변경했는지 완전 추적
- 규제 대응: 안전 관련 문항의 변경 이력은 감사 추적 가능
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class QuestionLifecycle(str, Enum):
    """문항 생명주기 상태"""
    DRAFT = "draft"                 # AI 초안 생성
    SME_REVIEW = "sme_review"       # SME 검토 중
    APPROVED = "approved"           # SME 승인, 출제 가능
    ACTIVE = "active"               # 시험 출제 활용 중
    CALIBRATING = "calibrating"     # IRT 교정 중 (응시 데이터 수집)
    CALIBRATED = "calibrated"       # IRT 교정 완료
    REVALIDATION = "revalidation"   # 재검증 필요 (주기 도래)
    RETIRED = "retired"             # 폐기 (더 이상 출제 불가)
    SUPERSEDED = "superseded"       # 개정판으로 대체됨


class ChangeType(str, Enum):
    """변경 유형"""
    CREATED = "created"             # 최초 생성
    AI_REVISED = "ai_revised"       # AI 2-Pass 수정
    SME_REVISED = "sme_revised"     # SME 수정
    IRT_CALIBRATED = "irt_calibrated"  # IRT 파라미터 교정
    REVALIDATED = "revalidated"     # 주기적 재검증
    STATUS_CHANGED = "status_changed"  # 상태 변경
    METADATA_UPDATED = "metadata_updated"  # 메타데이터 업데이트
    RETIRED = "retired"             # 폐기


@dataclass
class VersionEntry:
    """단일 버전 엔트리"""
    version: int
    change_type: str
    changed_by: str              # "AI", "SME:홍길동", "SYSTEM:IRT"
    changed_at: str
    question_snapshot: dict      # 해당 시점 문항 전체 스냅샷
    change_summary: str = ""     # 변경 요약
    lifecycle_status: str = ""
    diff_from_previous: list = field(default_factory=list)  # 이전 버전과의 diff


@dataclass
class QuestionVersionRecord:
    """문항 버전 관리 레코드"""
    question_id: str
    current_version: int = 0
    lifecycle_status: str = QuestionLifecycle.DRAFT
    safety_grade: str = "general"

    # 재검증 관리
    last_validated_at: str = ""
    next_revalidation_at: str = ""
    revalidation_interval_months: int = 12

    # 출제 이력
    times_used_in_exams: int = 0
    last_used_in_exam: str = ""

    # IRT 파라미터
    irt_difficulty: float = 0.0
    irt_discrimination: float = 0.0
    irt_confidence: str = "none"  # none | low | medium | high

    # 대체 문항 추적
    superseded_by: str = ""      # 대체된 경우 새 문항 ID
    supersedes: str = ""         # 이 문항이 대체한 원래 문항 ID
    retirement_reason: str = ""

    # 버전 이력
    versions: list = field(default_factory=list)  # list of VersionEntry as dict

    created_at: str = ""
    updated_at: str = ""


class QuestionVersionManager:
    """
    문항 버전 관리자

    JSON 파일 기반 (PoC). Phase 2에서 DB 전환.
    """

    # 안전등급별 재검증 주기 (개월)
    REVALIDATION_INTERVALS = {
        "safety_critical": 6,
        "safety_related": 12,
        "general": 24,
    }

    def __init__(self, storage_dir: str = "data/question_versions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, QuestionVersionRecord] = {}
        self._load_all()

    def _index_file(self) -> Path:
        return self.storage_dir / "version_index.json"

    def _question_file(self, question_id: str) -> Path:
        safe_id = question_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_id}.json"

    def _load_all(self):
        idx = self._index_file()
        if not idx.exists():
            return
        with open(idx, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        for qid in index_data.get("question_ids", []):
            qfile = self._question_file(qid)
            if qfile.exists():
                with open(qfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._records[qid] = QuestionVersionRecord(**data)

    def _save_record(self, qid: str):
        rec = self._records.get(qid)
        if not rec:
            return
        with open(self._question_file(qid), "w", encoding="utf-8") as f:
            json.dump(asdict(rec), f, ensure_ascii=False, indent=2)
        self._save_index()

    def _save_index(self):
        with open(self._index_file(), "w", encoding="utf-8") as f:
            json.dump(
                {"question_ids": list(self._records.keys()),
                 "total": len(self._records),
                 "updated_at": datetime.now().isoformat()},
                f, ensure_ascii=False, indent=2,
            )

    # ---------------------------------------------------------------
    # 버전 생성
    # ---------------------------------------------------------------
    def create_question(
        self,
        question: dict,
        created_by: str = "AI",
    ) -> QuestionVersionRecord:
        """새 문항 등록 (v1 생성)"""
        qid = question.get("question_id", f"Q-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        safety = question.get("safety_grade", "general")
        interval = self.REVALIDATION_INTERVALS.get(safety, 12)

        now = datetime.now()
        entry = VersionEntry(
            version=1,
            change_type=ChangeType.CREATED,
            changed_by=created_by,
            changed_at=now.isoformat(),
            question_snapshot=question,
            change_summary="최초 생성",
            lifecycle_status=QuestionLifecycle.DRAFT,
        )

        rec = QuestionVersionRecord(
            question_id=qid,
            current_version=1,
            lifecycle_status=QuestionLifecycle.DRAFT,
            safety_grade=safety,
            revalidation_interval_months=interval,
            versions=[asdict(entry)],
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )

        self._records[qid] = rec
        self._save_record(qid)
        return rec

    def add_version(
        self,
        question_id: str,
        question: dict,
        change_type: str,
        changed_by: str,
        change_summary: str = "",
        new_status: Optional[str] = None,
    ) -> QuestionVersionRecord:
        """기존 문항에 새 버전 추가"""
        rec = self._records.get(question_id)
        if not rec:
            raise ValueError(f"문항을 찾을 수 없습니다: {question_id}")

        new_ver = rec.current_version + 1
        now = datetime.now()

        # 이전 버전과의 diff (간이)
        prev_snapshot = {}
        if rec.versions:
            prev_snapshot = rec.versions[-1].get("question_snapshot", {})
        diff = self._simple_diff(prev_snapshot, question)

        entry = VersionEntry(
            version=new_ver,
            change_type=change_type,
            changed_by=changed_by,
            changed_at=now.isoformat(),
            question_snapshot=question,
            change_summary=change_summary,
            lifecycle_status=new_status or rec.lifecycle_status,
            diff_from_previous=diff,
        )

        rec.versions.append(asdict(entry))
        rec.current_version = new_ver
        rec.updated_at = now.isoformat()

        if new_status:
            rec.lifecycle_status = new_status

        self._save_record(question_id)
        return rec

    @staticmethod
    def _simple_diff(old: dict, new: dict) -> list[dict]:
        """간이 diff 계산"""
        diffs = []
        all_keys = set(list(old.keys()) + list(new.keys()))
        skip_keys = {"estimated_item_analysis", "safety_keywords_matched"}
        for key in sorted(all_keys - skip_keys):
            old_val = old.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                diffs.append({
                    "field": key,
                    "old": str(old_val)[:200] if old_val else "",
                    "new": str(new_val)[:200] if new_val else "",
                })
        return diffs

    # ---------------------------------------------------------------
    # 상태 전이
    # ---------------------------------------------------------------
    def change_status(
        self,
        question_id: str,
        new_status: str,
        changed_by: str,
        reason: str = "",
    ) -> QuestionVersionRecord:
        """문항 상태 변경"""
        rec = self._records.get(question_id)
        if not rec:
            raise ValueError(f"문항을 찾을 수 없습니다: {question_id}")

        old_status = rec.lifecycle_status
        rec.lifecycle_status = new_status
        rec.updated_at = datetime.now().isoformat()

        # 상태별 후속 처리
        if new_status == QuestionLifecycle.APPROVED:
            rec.last_validated_at = datetime.now().isoformat()

        if new_status == QuestionLifecycle.RETIRED:
            rec.retirement_reason = reason

        # 상태 변경도 버전 기록에 남김
        current_q = {}
        if rec.versions:
            current_q = rec.versions[-1].get("question_snapshot", {})

        entry = VersionEntry(
            version=rec.current_version,  # 버전 번호는 유지
            change_type=ChangeType.STATUS_CHANGED,
            changed_by=changed_by,
            changed_at=datetime.now().isoformat(),
            question_snapshot=current_q,
            change_summary=f"상태 변경: {old_status} → {new_status}. {reason}",
            lifecycle_status=new_status,
        )
        rec.versions.append(asdict(entry))

        self._save_record(question_id)
        return rec

    def retire_and_replace(
        self,
        old_question_id: str,
        new_question: dict,
        reason: str,
        retired_by: str,
    ) -> tuple:
        """문항 폐기 + 대체 문항 등록"""
        # 기존 문항 폐기
        old_rec = self.change_status(
            old_question_id, QuestionLifecycle.RETIRED,
            changed_by=retired_by, reason=reason,
        )

        # 새 문항 등록
        new_rec = self.create_question(new_question, created_by=retired_by)
        new_rec.supersedes = old_question_id
        old_rec.superseded_by = new_rec.question_id
        old_rec.lifecycle_status = QuestionLifecycle.SUPERSEDED

        self._save_record(old_question_id)
        self._save_record(new_rec.question_id)

        return old_rec, new_rec

    def record_exam_usage(self, question_id: str, exam_id: str = ""):
        """시험 출제 기록"""
        rec = self._records.get(question_id)
        if not rec:
            return
        rec.times_used_in_exams += 1
        rec.last_used_in_exam = datetime.now().isoformat()
        if rec.lifecycle_status == QuestionLifecycle.APPROVED:
            rec.lifecycle_status = QuestionLifecycle.ACTIVE
        self._save_record(question_id)

    def update_irt_params(
        self,
        question_id: str,
        difficulty: float,
        discrimination: float,
        confidence: str,
    ):
        """IRT 파라미터 업데이트"""
        rec = self._records.get(question_id)
        if not rec:
            return
        rec.irt_difficulty = difficulty
        rec.irt_discrimination = discrimination
        rec.irt_confidence = confidence
        if confidence in ("medium", "high"):
            rec.lifecycle_status = QuestionLifecycle.CALIBRATED
        self._save_record(question_id)

    # ---------------------------------------------------------------
    # 재검증 관리
    # ---------------------------------------------------------------
    def get_revalidation_due(self) -> list[dict]:
        """재검증이 필요한 문항 목록 (주기 도래)"""
        now = datetime.now()
        due_list = []
        for rec in self._records.values():
            if rec.lifecycle_status in (
                QuestionLifecycle.RETIRED, QuestionLifecycle.SUPERSEDED
            ):
                continue
            if not rec.last_validated_at:
                continue
            from datetime import timedelta
            validated = datetime.fromisoformat(rec.last_validated_at)
            interval = timedelta(days=rec.revalidation_interval_months * 30)
            if now > validated + interval:
                due_list.append({
                    "question_id": rec.question_id,
                    "safety_grade": rec.safety_grade,
                    "last_validated": rec.last_validated_at,
                    "overdue_days": (now - (validated + interval)).days,
                    "times_used": rec.times_used_in_exams,
                })
        due_list.sort(key=lambda x: (-1 if x["safety_grade"] == "safety_critical" else 0, -x["overdue_days"]))
        return due_list

    # ---------------------------------------------------------------
    # 조회
    # ---------------------------------------------------------------
    def get_question(self, question_id: str) -> Optional[dict]:
        rec = self._records.get(question_id)
        return asdict(rec) if rec else None

    def get_current_snapshot(self, question_id: str) -> Optional[dict]:
        """최신 버전의 문항 데이터"""
        rec = self._records.get(question_id)
        if not rec or not rec.versions:
            return None
        return rec.versions[-1].get("question_snapshot")

    def get_version_history(self, question_id: str) -> list[dict]:
        """문항의 전체 버전 이력"""
        rec = self._records.get(question_id)
        if not rec:
            return []
        return [
            {
                "version": v.get("version"),
                "change_type": v.get("change_type"),
                "changed_by": v.get("changed_by"),
                "changed_at": v.get("changed_at"),
                "change_summary": v.get("change_summary"),
                "lifecycle_status": v.get("lifecycle_status"),
                "diff_fields": [d.get("field") for d in v.get("diff_from_previous", [])],
            }
            for v in rec.versions
        ]

    def get_statistics(self) -> dict:
        """전체 문항 버전 관리 통계"""
        records = list(self._records.values())
        if not records:
            return {"total": 0}

        status_counts = {}
        safety_counts = {}
        for r in records:
            status_counts[r.lifecycle_status] = status_counts.get(r.lifecycle_status, 0) + 1
            safety_counts[r.safety_grade] = safety_counts.get(r.safety_grade, 0) + 1

        avg_versions = round(
            sum(r.current_version for r in records) / len(records), 1
        )

        return {
            "total_questions": len(records),
            "by_lifecycle": status_counts,
            "by_safety_grade": safety_counts,
            "avg_versions_per_question": avg_versions,
            "revalidation_due": len(self.get_revalidation_due()),
            "total_exam_usages": sum(r.times_used_in_exams for r in records),
            "irt_calibrated": sum(
                1 for r in records
                if r.irt_confidence in ("medium", "high")
            ),
        }
