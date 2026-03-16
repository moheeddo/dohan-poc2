"""
문제은행 품질 검증 파이프라인
- 문항 분석(Item Analysis) 자동화: 난이도, 변별도, 오답 매력도
- SAT 학습목표 커버리지 검증
- Bloom's Taxonomy 수준 분포 검증
- AI 생성 문항의 특이 패턴 감지

연구 근거:
- AI 생성 문항의 비기능적 오답 비율이 인간 출제 대비 높음 (33.75% vs 13.75%)
- 난이도 지수(Difficulty Index) 평균: AI 0.50, 인간 0.53 (유의차 없음)
- 변별도(Discrimination Index): AI 73.33% 적합, 인간 86.67% 적합
→ AI 문항의 오답 매력도 검증이 가장 중요한 QA 포인트
"""
import json
from typing import Optional

import httpx

from src.utils.http_client import get_ssl_verify

from src.prompts.template_manager import get_template_manager


class QuestionQualityValidator:
    """AI 생성 문항의 품질 검증 및 자동 개선"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
        model: str = "solar-pro3",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def validate_single(
        self,
        question: dict,
        reference_context: str = "",
    ) -> dict:
        """
        단일 문항 품질 검증

        Args:
            question: 검증할 문항
            reference_context: 정답 확인용 참고자료

        Returns:
            검증 결과
        """
        user_prompt = f"""## 검증 대상 문항
{json.dumps(question, ensure_ascii=False, indent=2)}

## 참고자료 (정답 확인용)
{reference_context[:3000] if reference_context else "(참고자료 없음 - 문항 자체 품질만 검증)"}

위 문항을 검증하고 JSON으로 결과를 출력해주세요.
특히 오답 매력도를 집중 검증해주세요."""

        async with httpx.AsyncClient(timeout=60.0, verify=get_ssl_verify()) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": get_template_manager().get_prompt("quality_validation")},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": get_template_manager().get_temperature("quality_validation"),
                    "max_tokens": get_template_manager().get_max_tokens("quality_validation"),
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)

    async def validate_batch(
        self,
        questions: list[dict],
        reference_context: str = "",
    ) -> dict:
        """
        문항 세트 전체 검증 + 통계 분석

        Returns:
            {
                "individual_results": [...],
                "statistics": {
                    "total": N,
                    "passed": N,
                    "needs_revision": N,
                    "rejected": N,
                    "avg_quality_score": float,
                    "bloom_coverage": {...},
                    "common_issues": [...]
                },
                "recommendations": [...]
            }
        """
        results = []
        for q in questions:
            result = await self.validate_single(q, reference_context)
            results.append(result)

        # 통계 분석
        total = len(results)
        passed = sum(1 for r in results if r.get("overall_quality") == "pass")
        needs_rev = sum(1 for r in results if r.get("overall_quality") == "needs_revision")
        rejected = sum(1 for r in results if r.get("overall_quality") == "reject")

        scores = [r.get("quality_score", 0) for r in results]
        avg_score = sum(scores) / max(len(scores), 1)

        # Bloom 수준 커버리지
        bloom_counts = {}
        for q in questions:
            level = q.get("bloom_level", "Unknown")
            bloom_counts[level] = bloom_counts.get(level, 0) + 1

        # 빈번한 이슈 집계
        issue_counts = {}
        for r in results:
            checks = r.get("checks", {})
            for check_name, check_result in checks.items():
                if isinstance(check_result, dict) and not check_result.get("pass", True):
                    issue_counts[check_name] = issue_counts.get(check_name, 0) + 1

        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])

        # 권고사항 생성
        recommendations = []
        if rejected > total * 0.2:
            recommendations.append(
                f"전체 {total}문항 중 {rejected}문항({rejected/total*100:.0f}%)이 부적합. "
                "프롬프트 또는 참고자료 보강 필요."
            )
        if any(name == "distractor_quality" for name, _ in common_issues[:3]):
            recommendations.append(
                "오답 매력도가 가장 빈번한 이슈. "
                "프롬프트에 '각 오답을 선택할 수 있는 합리적 이유를 명시하라'는 지시 추가 권장."
            )
        if len(bloom_counts) < 3:
            recommendations.append(
                f"Bloom 수준이 {len(bloom_counts)}개만 커버됨. "
                "Knowledge/Comprehension/Application/Analysis 4단계 균형 필요."
            )

        return {
            "individual_results": results,
            "statistics": {
                "total": total,
                "passed": passed,
                "needs_revision": needs_rev,
                "rejected": rejected,
                "pass_rate": f"{passed/max(total,1)*100:.1f}%",
                "avg_quality_score": round(avg_score, 1),
                "bloom_coverage": bloom_counts,
                "common_issues": common_issues[:5],
            },
            "recommendations": recommendations,
        }


class ItemAnalysisSimulator:
    """
    문항 분석 시뮬레이터

    실제 시험 전에 AI로 문항의 예상 난이도/변별도를 추정.
    실제 시험 후에는 실제 응답 데이터로 검증.
    """

    # 기준: IAEA SAT Evaluation 가이드라인
    DIFFICULTY_RANGES = {
        "too_easy": (0.0, 0.2),     # 정답률 80%+ → 너무 쉬움
        "easy": (0.2, 0.4),          # 정답률 60-80%
        "appropriate": (0.4, 0.7),   # 정답률 30-60% → 적정
        "difficult": (0.7, 0.85),    # 정답률 15-30%
        "too_difficult": (0.85, 1.0), # 정답률 15% 미만
    }

    # 변별도 기준
    DISCRIMINATION_THRESHOLDS = {
        "excellent": 0.40,    # 0.40+ : 우수
        "good": 0.30,         # 0.30-0.39 : 양호
        "acceptable": 0.20,   # 0.20-0.29 : 수용 가능
        "poor": 0.0,          # 0.20 미만 : 수정 필요
    }

    @staticmethod
    def estimate_difficulty(question: dict) -> dict:
        """
        문항의 예상 난이도 추정 (시험 전)

        Bloom 수준, 문항 길이, 보기 수 등으로 추정
        """
        bloom_difficulty_map = {
            "Knowledge": 0.3,       # 보통 쉬움
            "Comprehension": 0.45,  # 중간
            "Application": 0.6,     # 어려움
            "Analysis": 0.75,       # 매우 어려움
        }

        bloom = question.get("bloom_level", "Comprehension")
        base_difficulty = bloom_difficulty_map.get(bloom, 0.5)

        # 보기 수에 따른 조정 (4지선다 기준)
        options = question.get("options", {})
        if len(options) > 4:
            base_difficulty += 0.05
        elif len(options) < 4:
            base_difficulty -= 0.05

        return {
            "estimated_difficulty_index": round(base_difficulty, 2),
            "difficulty_category": next(
                (cat for cat, (low, high) in ItemAnalysisSimulator.DIFFICULTY_RANGES.items()
                 if low <= base_difficulty < high),
                "appropriate"
            ),
            "bloom_level": bloom,
        }

    @staticmethod
    def calibrate_difficulty_from_history(
        question: dict,
        exam_history: list[dict],
    ) -> dict:
        """
        시험 이력 기반 난이도 파라미터 보정 (IRT 기초)

        CTT(Classical Test Theory) 기반 보정 후,
        충분한 데이터(200+ 응답) 축적 시 IRT 파라미터 추정 준비.

        Args:
            question: 문항 데이터 (estimated_item_analysis 포함)
            exam_history: 과거 시험 결과 리스트
                [{
                    "exam_id": str,
                    "difficulty_index": float,  # 실측
                    "discrimination_index": float,
                    "total_responses": int,
                    "nonfunctional_distractors": list,
                }]

        Returns:
            보정된 난이도 파라미터
        """
        if not exam_history:
            # 이력 없으면 AI 추정값 그대로 반환
            estimate = question.get("estimated_item_analysis", {})
            return {
                "calibrated": False,
                "source": "ai_estimate",
                "difficulty_index": estimate.get("estimated_difficulty_index", 0.5),
                "confidence": "low",
                "total_exam_count": 0,
                "recommendation": "첫 시험 후 실측 데이터로 보정 필요",
            }

        # 실측 난이도 가중 평균 (최근 시험에 가중치 부여)
        total_weight = 0.0
        weighted_difficulty = 0.0
        weighted_discrimination = 0.0
        total_responses = 0

        for i, exam in enumerate(exam_history):
            # 최근 시험일수록 높은 가중치 (지수 감소)
            weight = 2.0 ** (-i * 0.5)  # 최근 1.0, 그 전 0.71, ...
            resp_count = exam.get("total_responses", 0)
            # 응답 수가 많을수록 신뢰도 높음
            resp_weight = min(resp_count / 100, 1.0)
            combined_weight = weight * resp_weight

            weighted_difficulty += exam.get("difficulty_index", 0.5) * combined_weight
            weighted_discrimination += exam.get("discrimination_index", 0.3) * combined_weight
            total_weight += combined_weight
            total_responses += resp_count

        if total_weight > 0:
            calibrated_difficulty = weighted_difficulty / total_weight
            calibrated_discrimination = weighted_discrimination / total_weight
        else:
            calibrated_difficulty = 0.5
            calibrated_discrimination = 0.3

        # 신뢰도 판정
        if total_responses >= 200:
            confidence = "high"
            irt_ready = True
        elif total_responses >= 50:
            confidence = "medium"
            irt_ready = False
        else:
            confidence = "low"
            irt_ready = False

        # 난이도 범주 재판정
        difficulty_category = next(
            (cat for cat, (low, high) in ItemAnalysisSimulator.DIFFICULTY_RANGES.items()
             if low <= 1 - calibrated_difficulty < high),
            "appropriate"
        )

        # 권고사항
        recommendations = []
        if calibrated_difficulty > 0.9:
            recommendations.append("난이도 너무 높음 — 문항 단순화 또는 힌트 추가 검토")
        elif calibrated_difficulty < 0.3:
            recommendations.append("난이도 너무 낮음 — 오답 매력도 강화 또는 복합 판단 요구")
        if calibrated_discrimination < 0.2:
            recommendations.append("변별도 부족 — 문항 수정 또는 교체 필요")

        # 비기능적 오답 이력 확인
        nf_history = [
            exam.get("nonfunctional_distractors", []) for exam in exam_history
        ]
        persistent_nf = set()
        for nf_list in nf_history:
            for nf in nf_list:
                persistent_nf.add(nf)
        if persistent_nf:
            recommendations.append(
                f"지속적 비기능적 오답: {', '.join(persistent_nf)} — 교체 필요"
            )

        return {
            "calibrated": True,
            "source": "exam_history",
            "difficulty_index": round(calibrated_difficulty, 3),
            "discrimination_index": round(calibrated_discrimination, 3),
            "difficulty_category": difficulty_category,
            "confidence": confidence,
            "total_exam_count": len(exam_history),
            "total_responses": total_responses,
            "irt_ready": irt_ready,
            "persistent_nonfunctional_distractors": list(persistent_nf),
            "recommendations": recommendations,
        }

    @staticmethod
    def analyze_post_exam(
        question_id: str,
        responses: list[dict],
    ) -> dict:
        """
        시험 후 실제 응답 데이터 기반 문항 분석

        Args:
            question_id: 문항 ID
            responses: [{"student_id": str, "selected": "A", "total_score": float}, ...]

        Returns:
            난이도 지수, 변별도, 오답 분포
        """
        if not responses:
            return {"error": "응답 데이터 없음"}

        total = len(responses)
        correct_count = sum(1 for r in responses if r.get("correct", False))

        # 난이도 지수 (Difficulty Index) = 정답자 수 / 전체 응답자 수
        difficulty_index = correct_count / total

        # 변별도 (Discrimination Index)
        # 상위 27% vs 하위 27% 정답률 차이
        sorted_responses = sorted(responses, key=lambda r: r.get("total_score", 0), reverse=True)
        n_group = max(int(total * 0.27), 1)
        upper_group = sorted_responses[:n_group]
        lower_group = sorted_responses[-n_group:]

        upper_correct = sum(1 for r in upper_group if r.get("correct", False))
        lower_correct = sum(1 for r in lower_group if r.get("correct", False))

        discrimination_index = (upper_correct - lower_correct) / n_group

        # 변별도 등급
        disc_grade = "poor"
        for grade, threshold in sorted(
            ItemAnalysisSimulator.DISCRIMINATION_THRESHOLDS.items(),
            key=lambda x: -x[1]
        ):
            if discrimination_index >= threshold:
                disc_grade = grade
                break

        # 보기별 선택 분포
        option_distribution = {}
        for r in responses:
            selected = r.get("selected", "?")
            option_distribution[selected] = option_distribution.get(selected, 0) + 1

        # 비기능적 오답 감지 (선택률 5% 미만)
        nonfunctional = [
            opt for opt, count in option_distribution.items()
            if count / total < 0.05 and opt != r.get("correct_answer")
        ]

        return {
            "question_id": question_id,
            "total_responses": total,
            "difficulty_index": round(difficulty_index, 3),
            "difficulty_category": next(
                (cat for cat, (low, high) in ItemAnalysisSimulator.DIFFICULTY_RANGES.items()
                 if low <= 1 - difficulty_index < high),
                "appropriate"
            ),
            "discrimination_index": round(discrimination_index, 3),
            "discrimination_grade": disc_grade,
            "option_distribution": option_distribution,
            "nonfunctional_distractors": nonfunctional,
            "action": (
                "유지" if disc_grade in ("excellent", "good")
                else "검토 필요" if disc_grade == "acceptable"
                else "수정 또는 교체 필요"
            ),
        }
