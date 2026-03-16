"""
적응형 난이도 조절 엔진

IRT (Item Response Theory) 기반 문항 선별 + 안전등급 연계
교육생 수준에 맞는 최적 문항 세트를 자동 구성.

설계 원칙 (IAEA NP-T-2.8):
- 안전 핵심 문항은 난이도와 무관하게 반드시 포함
- 일반 문항은 교육생 수준에 맞게 난이도 조절
- Spaced Repetition 연계: 이전 오답 문항 재출제

핵심 알고리즘:
- Phase 1 (PoC): 규칙 기반 난이도 조절 (IRT 데이터 부족)
- Phase 2: 실제 IRT 파라미터 기반 적응형 선별 (200+ 응답 축적 후)
"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TraineeProfile:
    """교육생 프로필"""
    trainee_id: str
    ability_estimate: float = 0.0  # IRT θ (표준 정규, 0 = 평균)
    exam_history: list[dict] = field(default_factory=list)  # 이전 응시 이력
    weak_topics: list[str] = field(default_factory=list)  # 취약 주제
    safety_cert_level: str = ""  # 현재 안전 인증 수준


@dataclass
class ExamConfig:
    """시험 구성 설정"""
    total_questions: int = 20
    safety_critical_min: int = 5       # 안전 핵심 최소 문항 수
    safety_related_min: int = 5        # 안전 관련 최소 문항 수
    bloom_distribution: dict = field(default_factory=lambda: {
        "Knowledge": 0.3,
        "Comprehension": 0.3,
        "Application": 0.25,
        "Analysis": 0.15,
    })
    target_difficulty: float = 0.5     # 목표 정답률 (0.5 = 적정)
    include_weak_topics: bool = True   # 취약 주제 문항 포함
    weak_topic_ratio: float = 0.2      # 취약 주제 문항 비율
    include_previous_errors: bool = True  # 이전 오답 재출제


class AdaptiveDifficultyEngine:
    """
    적응형 난이도 조절 엔진

    문항 풀에서 교육생 수준에 최적화된 문항 세트를 선별.
    """

    @staticmethod
    def irt_probability(theta: float, difficulty: float, discrimination: float = 1.0) -> float:
        """
        2PL IRT 모델: 정답 확률 계산

        P(θ) = 1 / (1 + exp(-a(θ - b)))
        θ: 교육생 능력
        b: 문항 난이도
        a: 문항 변별도
        """
        exponent = -discrimination * (theta - difficulty)
        exponent = max(-10, min(10, exponent))  # overflow 방지
        return 1.0 / (1.0 + math.exp(exponent))

    @staticmethod
    def information_function(theta: float, difficulty: float, discrimination: float = 1.0) -> float:
        """
        문항 정보 함수: I(θ) = a² × P(θ) × (1 - P(θ))

        높을수록 해당 능력 수준에서 변별력이 큼
        """
        p = AdaptiveDifficultyEngine.irt_probability(theta, difficulty, discrimination)
        return discrimination ** 2 * p * (1 - p)

    def select_questions(
        self,
        question_pool: list[dict],
        trainee: Optional[TraineeProfile] = None,
        config: Optional[ExamConfig] = None,
    ) -> list[dict]:
        """
        최적 문항 세트 선별

        전략:
        1. 안전 핵심/관련 문항 최소 수량 먼저 확보 (Graded Approach)
        2. 취약 주제 문항 배치 (Spaced Repetition 원리)
        3. Bloom 분포에 맞게 배분
        4. 능력 수준에 맞는 난이도 최적화 (정보 함수 최대화)
        """
        if config is None:
            config = ExamConfig()
        if trainee is None:
            trainee = TraineeProfile(trainee_id="anonymous")

        selected = []
        used_ids = set()
        remaining = config.total_questions

        # === Phase 1: 안전 핵심 문항 확보 (필수) ===
        critical_pool = [
            q for q in question_pool
            if q.get("safety_significance", {}).get("grade") == "safety_critical"
        ]
        critical_selected = self._select_by_difficulty(
            critical_pool, trainee.ability_estimate,
            min(config.safety_critical_min, remaining), used_ids,
        )
        selected.extend(critical_selected)
        remaining -= len(critical_selected)

        # === Phase 2: 안전 관련 문항 확보 ===
        related_pool = [
            q for q in question_pool
            if q.get("safety_significance", {}).get("grade") == "safety_related"
            and q.get("question_id") not in used_ids
        ]
        related_selected = self._select_by_difficulty(
            related_pool, trainee.ability_estimate,
            min(config.safety_related_min, remaining), used_ids,
        )
        selected.extend(related_selected)
        remaining -= len(related_selected)

        # === Phase 3: 취약 주제 문항 (Spaced Repetition) ===
        if config.include_weak_topics and trainee.weak_topics and remaining > 0:
            weak_count = max(1, int(config.total_questions * config.weak_topic_ratio))
            weak_count = min(weak_count, remaining)
            weak_pool = [
                q for q in question_pool
                if q.get("question_id") not in used_ids
                and any(
                    topic.lower() in " ".join(q.get("keywords", [])).lower()
                    or topic.lower() in q.get("question_text", "").lower()
                    for topic in trainee.weak_topics
                )
            ]
            weak_selected = self._select_by_difficulty(
                weak_pool, trainee.ability_estimate, weak_count, used_ids,
            )
            selected.extend(weak_selected)
            remaining -= len(weak_selected)

        # === Phase 4: 이전 오답 재출제 ===
        if config.include_previous_errors and trainee.exam_history and remaining > 0:
            error_ids = set()
            for hist in trainee.exam_history[-3:]:  # 최근 3회
                for resp in hist.get("responses", []):
                    if not resp.get("is_correct", True):
                        error_ids.add(resp.get("question_id", ""))
            error_pool = [
                q for q in question_pool
                if q.get("question_id") in error_ids
                and q.get("question_id") not in used_ids
            ]
            error_count = min(3, remaining, len(error_pool))
            for q in error_pool[:error_count]:
                selected.append(q)
                used_ids.add(q.get("question_id"))
                remaining -= 1

        # === Phase 5: Bloom 분포 맞춰 나머지 채우기 ===
        if remaining > 0:
            general_pool = [
                q for q in question_pool
                if q.get("question_id") not in used_ids
            ]
            bloom_selected = self._select_by_bloom_and_difficulty(
                general_pool, trainee.ability_estimate,
                config.bloom_distribution, remaining, used_ids,
            )
            selected.extend(bloom_selected)

        # 순서 섞기 (안전 문항이 앞에 몰리지 않게)
        random.shuffle(selected)

        return selected

    def _select_by_difficulty(
        self,
        pool: list[dict],
        theta: float,
        count: int,
        used_ids: set,
    ) -> list[dict]:
        """능력 수준에 맞는 문항 선별 (정보 함수 최대화)"""
        if not pool or count <= 0:
            return []

        scored = []
        for q in pool:
            qid = q.get("question_id", "")
            if qid in used_ids:
                continue

            # IRT 파라미터 추출 (없으면 추정치 사용)
            item_analysis = q.get("estimated_item_analysis", {})
            difficulty = item_analysis.get("difficulty_index", 0.5)
            # difficulty_index(정답률)를 IRT 난이도 파라미터로 변환
            # 정답률 0.5 → 난이도 0, 정답률 0.9 → 난이도 -1.5, 정답률 0.1 → 난이도 +1.5
            irt_b = -1.5 * (2 * difficulty - 1) if difficulty > 0 else 0
            discrimination = item_analysis.get("discrimination_index", 0.5)
            irt_a = max(0.3, discrimination * 2)

            info = self.information_function(theta, irt_b, irt_a)
            scored.append((info, q))

        scored.sort(key=lambda x: -x[0])

        selected = []
        for _, q in scored[:count]:
            selected.append(q)
            used_ids.add(q.get("question_id", ""))

        return selected

    def _select_by_bloom_and_difficulty(
        self,
        pool: list[dict],
        theta: float,
        bloom_dist: dict,
        count: int,
        used_ids: set,
    ) -> list[dict]:
        """Bloom 분포 + 난이도 최적화"""
        bloom_targets = {}
        for level, ratio in bloom_dist.items():
            bloom_targets[level] = max(1, round(count * ratio))

        selected = []
        # Bloom 수준별로 선별
        for level, target in bloom_targets.items():
            if len(selected) >= count:
                break
            level_pool = [
                q for q in pool
                if q.get("bloom_level", "") == level
                and q.get("question_id") not in used_ids
            ]
            level_selected = self._select_by_difficulty(
                level_pool, theta,
                min(target, count - len(selected)), used_ids,
            )
            selected.extend(level_selected)

        # 부족하면 아무거나 채우기
        if len(selected) < count:
            remaining_pool = [
                q for q in pool if q.get("question_id") not in used_ids
            ]
            for q in remaining_pool[:count - len(selected)]:
                selected.append(q)
                used_ids.add(q.get("question_id", ""))

        return selected

    @staticmethod
    def update_ability_estimate(
        current_theta: float,
        responses: list[dict],
    ) -> float:
        """
        EAP (Expected A Posteriori) 간이 능력 추정

        응시 결과를 반영하여 교육생 능력 추정치 업데이트.
        Phase 1에서는 간이 MLE를 사용.
        """
        if not responses:
            return current_theta

        # 간이 업데이트: 정답률 기반 보정
        correct_count = sum(1 for r in responses if r.get("is_correct", False))
        total = len(responses)
        observed_rate = correct_count / total

        # 정답률 → θ 근사 변환
        # P=0.5 → θ=0, P=0.8 → θ≈1, P=0.2 → θ≈-1
        if observed_rate >= 0.99:
            new_theta = 2.5
        elif observed_rate <= 0.01:
            new_theta = -2.5
        else:
            # logit 변환
            new_theta = math.log(observed_rate / (1 - observed_rate))
            new_theta = max(-3.0, min(3.0, new_theta))

        # 기존 추정치와 가중 평균 (베이지안 업데이트 간이 버전)
        weight = min(total / 50, 0.8)  # 문항 수가 많을수록 새 데이터 비중 증가
        updated = current_theta * (1 - weight) + new_theta * weight

        return round(updated, 3)

    @staticmethod
    def generate_exam_report(
        trainee: TraineeProfile,
        responses: list[dict],
        questions: list[dict],
    ) -> dict:
        """시험 결과 리포트 생성"""
        total = len(responses)
        correct = sum(1 for r in responses if r.get("is_correct", False))

        # 안전등급별 성적
        safety_scores = {"safety_critical": {"correct": 0, "total": 0},
                         "safety_related": {"correct": 0, "total": 0},
                         "general": {"correct": 0, "total": 0}}

        # Bloom 수준별 성적
        bloom_scores = {}

        # 취약 주제 추출
        wrong_topics = []

        q_map = {q.get("question_id"): q for q in questions}

        for resp in responses:
            qid = resp.get("question_id", "")
            q = q_map.get(qid, {})
            grade = q.get("safety_significance", {}).get("grade", "general")
            bloom = q.get("bloom_level", "Unknown")

            safety_scores[grade]["total"] += 1
            if bloom not in bloom_scores:
                bloom_scores[bloom] = {"correct": 0, "total": 0}
            bloom_scores[bloom]["total"] += 1

            if resp.get("is_correct"):
                safety_scores[grade]["correct"] += 1
                bloom_scores[bloom]["correct"] += 1
            else:
                wrong_topics.extend(q.get("keywords", []))

        # 취약 주제 빈도 분석
        topic_freq = {}
        for t in wrong_topics:
            topic_freq[t] = topic_freq.get(t, 0) + 1
        weak_topics = sorted(topic_freq.items(), key=lambda x: -x[1])[:5]

        return {
            "trainee_id": trainee.trainee_id,
            "total_questions": total,
            "correct_count": correct,
            "score_percent": round(correct / max(total, 1) * 100, 1),
            "ability_estimate": trainee.ability_estimate,
            "safety_scores": {
                grade: {
                    "correct": v["correct"],
                    "total": v["total"],
                    "rate": round(v["correct"] / max(v["total"], 1) * 100, 1),
                }
                for grade, v in safety_scores.items() if v["total"] > 0
            },
            "bloom_scores": {
                level: {
                    "correct": v["correct"],
                    "total": v["total"],
                    "rate": round(v["correct"] / max(v["total"], 1) * 100, 1),
                }
                for level, v in bloom_scores.items()
            },
            "weak_topics": [{"topic": t, "error_count": c} for t, c in weak_topics],
            "recommendations": _generate_recommendations(
                correct / max(total, 1), safety_scores, bloom_scores
            ),
        }


def _generate_recommendations(
    overall_rate: float,
    safety_scores: dict,
    bloom_scores: dict,
) -> list[str]:
    """성적 기반 학습 추천"""
    recs = []

    # 전체 성적 기반
    if overall_rate < 0.6:
        recs.append("전체 정답률이 60% 미만입니다. 기초 개념 복습이 필요합니다.")
    elif overall_rate < 0.8:
        recs.append("전체 정답률이 양호하나, 취약 영역 보강이 필요합니다.")

    # 안전 핵심 성적
    critical = safety_scores.get("safety_critical", {})
    if critical.get("total", 0) > 0:
        crit_rate = critical["correct"] / critical["total"]
        if crit_rate < 0.8:
            recs.append(
                f"안전 핵심 문항 정답률 {crit_rate*100:.0f}%: "
                "비상운전/안전계통 관련 재학습이 필요합니다. "
                "EOP/비상운전절차서 복습을 권장합니다."
            )

    # Bloom 수준별
    for level in ["Application", "Analysis"]:
        data = bloom_scores.get(level, {})
        if data.get("total", 0) >= 2:
            rate = data["correct"] / data["total"]
            if rate < 0.5:
                recs.append(
                    f"{level} 수준 문항 정답률이 낮습니다. "
                    "단순 암기가 아닌 적용/분석 역량 강화가 필요합니다."
                )

    if not recs:
        recs.append("전반적으로 우수한 성적입니다. 고급 시나리오 문항에 도전해보세요.")

    return recs
