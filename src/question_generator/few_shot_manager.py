"""
Few-shot 예시 관리 모듈

SME가 검수한 모범 문항과 강의 스크립트를 관리하여
AI 생성 품질을 앵커링하는 인프라.

연구 근거:
- Few-shot 예시 제공 시 비기능적 오답 비율 50% → 16% 감소
- 도메인 특화 예시가 범용 예시보다 효과적
- 3~5건의 예시가 최적 (과다 시 컨텍스트 소모)
"""
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class FewShotExample:
    """단일 Few-shot 예시"""
    example_id: str
    category: str  # "question" | "script" | "scenario"
    bloom_level: str = ""  # Knowledge | Comprehension | Application | Analysis
    safety_grade: str = ""  # safety_critical | safety_related | general
    topic_tags: list[str] = field(default_factory=list)
    content: dict = field(default_factory=dict)
    sme_approved: bool = False
    sme_name: str = ""
    quality_score: float = 0.0
    usage_count: int = 0


class FewShotManager:
    """
    Few-shot 예시 저장소

    JSON 파일 기반 간이 저장소 (PoC 단계).
    Phase 2에서는 DB로 전환 가능.
    """

    def __init__(self, storage_dir: str = "data/few_shot_examples"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[FewShotExample]] = {}

    def _get_file_path(self, category: str) -> Path:
        return self.storage_dir / f"{category}_examples.json"

    def _load_category(self, category: str) -> list[FewShotExample]:
        """카테고리별 예시 로드"""
        if category in self._cache:
            return self._cache[category]

        file_path = self._get_file_path(category)
        if not file_path.exists():
            self._cache[category] = []
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = [
            FewShotExample(**item) for item in data
        ]
        self._cache[category] = examples
        return examples

    def _save_category(self, category: str):
        """카테고리별 예시 저장"""
        examples = self._cache.get(category, [])
        file_path = self._get_file_path(category)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(ex) for ex in examples],
                f, ensure_ascii=False, indent=2,
            )

    def add_example(self, example: FewShotExample):
        """예시 추가"""
        examples = self._load_category(example.category)

        # 중복 체크
        existing_ids = {ex.example_id for ex in examples}
        if example.example_id in existing_ids:
            # 업데이트
            examples = [
                example if ex.example_id == example.example_id else ex
                for ex in examples
            ]
            self._cache[example.category] = examples
        else:
            examples.append(example)

        self._save_category(example.category)

    def get_examples(
        self,
        category: str,
        bloom_level: Optional[str] = None,
        safety_grade: Optional[str] = None,
        topic_tags: Optional[list[str]] = None,
        sme_approved_only: bool = True,
        max_count: int = 5,
    ) -> list[dict]:
        """
        조건에 맞는 Few-shot 예시 검색

        Args:
            category: question | script | scenario
            bloom_level: 특정 Bloom 수준 필터
            safety_grade: 안전 등급 필터
            topic_tags: 주제 태그 필터 (OR 매칭)
            sme_approved_only: SME 승인된 것만
            max_count: 최대 반환 수

        Returns:
            예시 content 리스트 (프롬프트 주입용)
        """
        examples = self._load_category(category)

        # 필터링
        filtered = []
        for ex in examples:
            if sme_approved_only and not ex.sme_approved:
                continue
            if bloom_level and ex.bloom_level and ex.bloom_level != bloom_level:
                continue
            if safety_grade and ex.safety_grade and ex.safety_grade != safety_grade:
                continue
            filtered.append(ex)

        # 주제 태그 매칭 시 관련도 점수 부여
        if topic_tags:
            scored = []
            tag_set = set(t.lower() for t in topic_tags)
            for ex in filtered:
                ex_tags = set(t.lower() for t in ex.topic_tags)
                overlap = len(tag_set & ex_tags)
                scored.append((overlap, ex.quality_score, ex))
            scored.sort(key=lambda x: (-x[0], -x[1]))
            filtered = [item[2] for item in scored]

        # 품질 점수 기반 정렬 (높은 것 우선)
        if not topic_tags:
            filtered.sort(key=lambda x: -x.quality_score)

        # 사용 횟수 기록
        selected = filtered[:max_count]
        for ex in selected:
            ex.usage_count += 1

        if selected:
            self._save_category(category)

        return [ex.content for ex in selected]

    def get_statistics(self) -> dict:
        """전체 예시 저장소 통계"""
        stats = {}
        for category in ["question", "script", "scenario"]:
            examples = self._load_category(category)
            approved = sum(1 for ex in examples if ex.sme_approved)
            stats[category] = {
                "total": len(examples),
                "sme_approved": approved,
                "avg_quality": (
                    round(
                        sum(ex.quality_score for ex in examples) / max(len(examples), 1),
                        1,
                    )
                ),
                "bloom_distribution": {},
            }
            for ex in examples:
                level = ex.bloom_level or "Unspecified"
                stats[category]["bloom_distribution"][level] = (
                    stats[category]["bloom_distribution"].get(level, 0) + 1
                )
        return stats

    def seed_default_examples(self):
        """
        기본 모범 문항 예시 시드 데이터 생성

        SME가 실제 예시를 등록하기 전까지 사용할 기본 예시
        """
        default_questions = [
            FewShotExample(
                example_id="seed-q-001",
                category="question",
                bloom_level="Knowledge",
                safety_grade="safety_critical",
                topic_tags=["비상디젤발전기", "EDG", "전원계통"],
                content={
                    "question_id": "SEED-Q001",
                    "bloom_level": "Knowledge",
                    "learning_objective": "비상디젤발전기(EDG)의 기동 조건을 설명할 수 있다",
                    "question_text": "비상디젤발전기(EDG)가 자동 기동되는 신호로 올바른 것은?",
                    "options": {
                        "A": "소내 전원 상실(LOOP) 신호",
                        "B": "원자로 냉각재 펌프 정지 신호",
                        "C": "주급수 펌프 정지 신호",
                        "D": "터빈 트립 신호",
                    },
                    "correct_answer": "A",
                    "distractor_rationale": {
                        "B": "RCP 정지 시 EDG가 아닌 다른 보호기능이 작동하나, 비상전원과 혼동 가능",
                        "C": "주급수 정지 시 보조급수가 기동되는 것과 EDG 기동을 혼동할 수 있음",
                        "D": "터빈 트립은 원자로 트립으로 이어질 수 있어 EDG와 연관짓기 쉬움",
                    },
                    "explanation": "EDG는 소내 전원 상실(LOOP) 또는 안전주입(SI) 신호에 의해 자동 기동됨",
                },
                sme_approved=True,
                sme_name="시드데이터",
                quality_score=90.0,
            ),
            FewShotExample(
                example_id="seed-q-002",
                category="question",
                bloom_level="Application",
                safety_grade="safety_critical",
                topic_tags=["LOCA", "비상운전", "ECCS"],
                content={
                    "question_id": "SEED-Q002",
                    "bloom_level": "Application",
                    "learning_objective": "소형 LOCA 발생 시 운전원의 초기 대응 절차를 수행할 수 있다",
                    "scenario": "100% 출력 운전 중 가압기 압력이 153 kg/cm²에서 급격히 하강하기 시작하며, "
                    "가압기 수위도 동시 하강 중이다. 격납건물 내 습도가 상승하는 경보가 발생했다.",
                    "question_text": "위 상황에서 운전원이 가장 먼저 확인해야 할 사항은?",
                    "options": {
                        "A": "안전주입(SI) 신호 자동 발생 여부 확인",
                        "B": "증기발생기 수위 변화 확인",
                        "C": "원자로 냉각재 펌프 진동 확인",
                        "D": "1차측 냉각재 샘플링 실시",
                    },
                    "correct_answer": "A",
                    "distractor_rationale": {
                        "B": "SG 수위도 LOCA 시 변하므로 혼동 가능하나, SI 확인이 최우선",
                        "C": "RCP 진동은 LOCA의 2차 영향이며, 초기 대응 순서에서 후순위",
                        "D": "샘플링은 진단에 도움되나 비상 대응 시 즉각 조치가 아님",
                    },
                    "explanation": "LOCA 징후 시 최우선 확인은 안전주입(SI) 자동작동 여부. "
                    "SI가 미작동이면 수동 기동 필요.",
                },
                sme_approved=True,
                sme_name="시드데이터",
                quality_score=95.0,
            ),
            FewShotExample(
                example_id="seed-q-003",
                category="question",
                bloom_level="Analysis",
                safety_grade="safety_critical",
                topic_tags=["증기발생기", "세관파손", "SGTR"],
                content={
                    "question_id": "SEED-Q003",
                    "bloom_level": "Analysis",
                    "learning_objective": "증기발생기 세관 파손(SGTR) 사고를 진단하고 적절한 대응 전략을 수립할 수 있다",
                    "scenario": "75% 출력 운전 중 가압기 수위가 서서히 하강하며, "
                    "1번 증기발생기(SG) 2차측 방사능 계측기 경보가 발생했다. "
                    "가압기 압력은 155→152 kg/cm²로 완만히 감소 중이며, "
                    "1번 SG 수위는 정상 범위에서 약간 상승하고 있다.",
                    "question_text": "위 현상들을 종합 분석한 결과 가장 가능성 높은 사고 유형은?",
                    "options": {
                        "A": "증기발생기 세관 파손(SGTR)",
                        "B": "소형 냉각재 상실사고(Small LOCA)",
                        "C": "가압기 안전밸브 오개방",
                        "D": "화학 및 체적 제어계통(CVCS) 누설",
                    },
                    "correct_answer": "A",
                    "distractor_rationale": {
                        "B": "가압기 압력/수위 하강은 Small LOCA와 유사하나, SG 2차측 방사능 상승은 SGTR 고유 징후",
                        "C": "가압기 안전밸브 오개방도 1차측 압력/수위 하강을 야기하나, SG 방사능은 변화 없음",
                        "D": "CVCS 누설도 1차측 재고량 감소를 야기하나, 감소 속도가 느리고 SG 방사능 무관",
                    },
                    "explanation": "핵심 판별 지표: SG 2차측 방사능 상승 + 1차측 압력/수위 하강 + "
                    "해당 SG 수위 상승 → SGTR 3대 징후. Small LOCA와의 차이는 SG 2차측 방사능.",
                },
                sme_approved=True,
                sme_name="시드데이터",
                quality_score=95.0,
            ),
        ]

        for q in default_questions:
            self.add_example(q)

        return len(default_questions)
