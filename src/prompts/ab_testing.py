"""
프롬프트 A/B 테스트 프레임워크

프롬프트 변형(Variant)을 정의하고 생성 결과의 품질을 비교 추적.
실험 결과를 축적하여 최적 프롬프트를 데이터 기반으로 선택.

흐름:
  1. 실험 생성: create_experiment(key, variants)
  2. 변형 할당: assign_variant(experiment_id) → variant_id
  3. 결과 기록: record_result(experiment_id, variant_id, metrics)
  4. 분석: get_experiment_results(experiment_id) → 통계 비교
"""
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class PromptVariant:
    """프롬프트 변형"""
    variant_id: str            # e.g. "A", "B", "C"
    system_prompt: str         # 프롬프트 텍스트
    description: str = ""      # 변경 사항 설명
    temperature: float | None = None   # None이면 기본값 사용
    max_tokens: int | None = None


@dataclass
class ExperimentResult:
    """단일 실험 결과"""
    variant_id: str
    timestamp: str = ""
    # QA 메트릭
    qa_pass: bool = False
    quality_score: float = 0.0
    # 오답 매력도
    distractor_quality: float = 0.0
    nonfunctional_distractors: int = 0
    # SME 평가
    sme_approved: bool | None = None
    sme_score: float | None = None
    # 생성 메타
    generation_time_ms: float = 0.0
    token_count: int = 0
    # 교육 효과 (시험 후)
    actual_difficulty: float | None = None
    actual_discrimination: float | None = None


@dataclass
class ABExperiment:
    """A/B 테스트 실험"""
    experiment_id: str
    prompt_key: str            # e.g. "question_generation"
    variants: dict[str, PromptVariant] = field(default_factory=dict)
    results: list[ExperimentResult] = field(default_factory=list)
    created_at: str = ""
    status: str = "active"     # active | completed | archived
    min_samples_per_variant: int = 30
    description: str = ""

    def is_complete(self) -> bool:
        """모든 변형이 최소 샘플 수를 충족했는지"""
        counts = self._variant_counts()
        return all(
            counts.get(vid, 0) >= self.min_samples_per_variant
            for vid in self.variants
        )

    def _variant_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.variant_id] = counts.get(r.variant_id, 0) + 1
        return counts


class PromptABTestManager:
    """프롬프트 A/B 테스트 관리자"""

    def __init__(self, storage_dir: str = "data/ab_tests"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._experiments: dict[str, ABExperiment] = {}
        self._load_all()

    def _load_all(self) -> None:
        for f in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                exp = ABExperiment(
                    experiment_id=data["experiment_id"],
                    prompt_key=data["prompt_key"],
                    created_at=data.get("created_at", ""),
                    status=data.get("status", "active"),
                    min_samples_per_variant=data.get("min_samples_per_variant", 30),
                    description=data.get("description", ""),
                )
                for vid, vdata in data.get("variants", {}).items():
                    exp.variants[vid] = PromptVariant(
                        variant_id=vid,
                        system_prompt=vdata.get("system_prompt", ""),
                        description=vdata.get("description", ""),
                        temperature=vdata.get("temperature"),
                        max_tokens=vdata.get("max_tokens"),
                    )
                for rdata in data.get("results", []):
                    exp.results.append(ExperimentResult(**{
                        k: v for k, v in rdata.items()
                        if k in ExperimentResult.__dataclass_fields__
                    }))
                self._experiments[exp.experiment_id] = exp
            except Exception:
                continue

    def _save(self, exp: ABExperiment) -> None:
        data = {
            "experiment_id": exp.experiment_id,
            "prompt_key": exp.prompt_key,
            "created_at": exp.created_at,
            "status": exp.status,
            "min_samples_per_variant": exp.min_samples_per_variant,
            "description": exp.description,
            "variants": {
                vid: {
                    "variant_id": v.variant_id,
                    "system_prompt": v.system_prompt,
                    "description": v.description,
                    "temperature": v.temperature,
                    "max_tokens": v.max_tokens,
                }
                for vid, v in exp.variants.items()
            },
            "results": [
                {k: getattr(r, k) for k in ExperimentResult.__dataclass_fields__}
                for r in exp.results
            ],
        }
        path = self.storage_dir / f"{exp.experiment_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_experiment(
        self,
        prompt_key: str,
        variants: list[PromptVariant],
        description: str = "",
        min_samples: int = 30,
    ) -> str:
        """새 A/B 실험 생성"""
        exp_id = f"EXP-{prompt_key}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        exp = ABExperiment(
            experiment_id=exp_id,
            prompt_key=prompt_key,
            created_at=datetime.now().isoformat(),
            description=description,
            min_samples_per_variant=min_samples,
        )
        for v in variants:
            exp.variants[v.variant_id] = v
        self._experiments[exp_id] = exp
        self._save(exp)
        return exp_id

    def assign_variant(self, experiment_id: str) -> Optional[PromptVariant]:
        """실험에서 변형 할당 (균등 분배)"""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != "active":
            return None

        # 가장 적은 샘플을 가진 변형 할당
        counts = exp._variant_counts()
        min_count = min(counts.get(vid, 0) for vid in exp.variants)
        candidates = [
            vid for vid in exp.variants
            if counts.get(vid, 0) == min_count
        ]
        selected = random.choice(candidates)
        return exp.variants[selected]

    def record_result(
        self,
        experiment_id: str,
        result: ExperimentResult,
    ) -> None:
        """실험 결과 기록"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"실험을 찾을 수 없습니다: {experiment_id}")

        result.timestamp = datetime.now().isoformat()
        exp.results.append(result)

        # 자동 완료 체크
        if exp.is_complete():
            exp.status = "completed"

        self._save(exp)

    def get_experiment_results(self, experiment_id: str) -> dict:
        """실험 결과 분석"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {"error": f"실험을 찾을 수 없습니다: {experiment_id}"}

        variant_stats = {}
        for vid in exp.variants:
            v_results = [r for r in exp.results if r.variant_id == vid]
            if not v_results:
                variant_stats[vid] = {"count": 0}
                continue

            n = len(v_results)
            avg_quality = sum(r.quality_score for r in v_results) / n
            qa_pass_rate = sum(1 for r in v_results if r.qa_pass) / n * 100
            avg_distractor = sum(r.distractor_quality for r in v_results) / n
            avg_nf = sum(r.nonfunctional_distractors for r in v_results) / n
            avg_gen_time = sum(r.generation_time_ms for r in v_results) / n

            # SME 평가 있는 것만
            sme_results = [r for r in v_results if r.sme_approved is not None]
            sme_approval_rate = (
                sum(1 for r in sme_results if r.sme_approved) / len(sme_results) * 100
                if sme_results else None
            )

            # 실제 난이도/변별도 (시험 후 데이터 있는 것만)
            actual_diff = [r.actual_difficulty for r in v_results if r.actual_difficulty is not None]
            actual_disc = [r.actual_discrimination for r in v_results if r.actual_discrimination is not None]

            variant_stats[vid] = {
                "count": n,
                "description": exp.variants[vid].description,
                "avg_quality_score": round(avg_quality, 2),
                "qa_pass_rate": round(qa_pass_rate, 1),
                "avg_distractor_quality": round(avg_distractor, 2),
                "avg_nonfunctional_distractors": round(avg_nf, 2),
                "avg_generation_time_ms": round(avg_gen_time, 1),
                "sme_approval_rate": round(sme_approval_rate, 1) if sme_approval_rate is not None else None,
                "actual_difficulty_avg": round(sum(actual_diff) / len(actual_diff), 3) if actual_diff else None,
                "actual_discrimination_avg": round(sum(actual_disc) / len(actual_disc), 3) if actual_disc else None,
            }

        # 승자 판정
        winner = None
        if all(s.get("count", 0) >= exp.min_samples_per_variant for s in variant_stats.values()):
            best_vid = max(
                variant_stats,
                key=lambda vid: variant_stats[vid].get("avg_quality_score", 0),
            )
            winner = {
                "variant_id": best_vid,
                "reason": f"평균 품질 점수 최고: {variant_stats[best_vid]['avg_quality_score']}",
                "recommendation": f"프롬프트 키 '{exp.prompt_key}'를 변형 '{best_vid}'로 업데이트 권장",
            }

        return {
            "experiment_id": experiment_id,
            "prompt_key": exp.prompt_key,
            "status": exp.status,
            "description": exp.description,
            "total_results": len(exp.results),
            "min_samples_per_variant": exp.min_samples_per_variant,
            "variant_stats": variant_stats,
            "winner": winner,
        }

    def list_experiments(self) -> list[dict]:
        """실험 목록"""
        return [
            {
                "experiment_id": exp.experiment_id,
                "prompt_key": exp.prompt_key,
                "status": exp.status,
                "description": exp.description,
                "variants": list(exp.variants.keys()),
                "total_results": len(exp.results),
                "is_complete": exp.is_complete(),
                "created_at": exp.created_at,
            }
            for exp in self._experiments.values()
        ]

    def get_active_experiment(self, prompt_key: str) -> Optional[str]:
        """특정 프롬프트 키의 활성 실험 ID 반환"""
        for exp in self._experiments.values():
            if exp.prompt_key == prompt_key and exp.status == "active":
                return exp.experiment_id
        return None
