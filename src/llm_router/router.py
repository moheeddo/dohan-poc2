"""
LLM 하이브리드 라우터 — AtomicGPT + Solar Pro

아키텍처:
  ┌─────────────┐
  │  Task Input  │
  └──────┬──────┘
         │  classify_task()
  ┌──────▼──────┐
  │  LLMRouter   │
  │  (정책 기반)  │
  └──┬───────┬──┘
     │       │
  ┌──▼──┐ ┌─▼────┐
  │Atomic│ │Solar │  ← 1차 라우팅
  │ GPT  │ │ Pro  │
  └──┬──┘ └──┬───┘
     │       │
  ┌──▼───────▼──┐
  │  품질 평가    │  ← confidence check
  │  (threshold)  │
  └──────┬──────┘
         │  fallback if low confidence
  ┌──────▼──────┐
  │  최종 응답    │
  └─────────────┘

라우팅 정책:
  1. 원자력 도메인 지식 문항/스크립트 → AtomicGPT 우선
  2. 범용 교수설계/구조화 → Solar Pro 우선
  3. 안전등급(safety_critical) 문항 → AtomicGPT 필수 + Solar Pro 크로스체크
  4. 시각자료(VLM) 분석 → Solar Pro (AtomicGPT는 VLM 미지원)

AtomicGPT 참고:
  - KAERI 개발, 8B/70B 파라미터
  - AtomBench에서 Llama 대비 5-24% 우수 (원자력 도메인)
  - 2025년 기준 연구용, 상용 API 미확정 → 인터페이스만 선정의
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

from src.utils.http_client import get_ssl_verify


class ModelProvider(str, Enum):
    """사용 가능한 LLM 프로바이더"""
    SOLAR_PRO = "solar_pro"       # Upstage Solar Pro (범용, VLM 지원)
    ATOMIC_GPT = "atomic_gpt"     # KAERI AtomicGPT (원자력 특화)


class TaskType(str, Enum):
    """태스크 유형별 분류"""
    SCRIPT_GENERATION = "script_generation"         # 강의 스크립트 생성
    QUESTION_GENERATION = "question_generation"     # 문항 생성
    DISTRACTOR_REFINEMENT = "distractor_refinement" # 오답 개선
    VLM_ANALYSIS = "vlm_analysis"                   # 시각자료 분석
    QUALITY_VALIDATION = "quality_validation"       # 품질 검증
    SCENARIO_GENERATION = "scenario_generation"     # 시나리오 생성
    SAFETY_REVIEW = "safety_review"                 # 안전등급 문항 크로스체크


@dataclass
class ModelConfig:
    """개별 모델 설정"""
    provider: ModelProvider
    base_url: str
    api_key: str
    model_name: str
    max_tokens: int = 4096
    timeout: float = 120.0
    enabled: bool = True


@dataclass
class RoutingPolicy:
    """라우팅 정책 정의"""
    task_type: TaskType
    primary: ModelProvider           # 1차 선택 모델
    fallback: ModelProvider          # 실패 시 대체 모델
    require_cross_check: bool = False  # 두 모델 모두 호출 후 비교
    confidence_threshold: float = 0.7  # 이 이하면 fallback 호출
    description: str = ""


# 기본 라우팅 정책 테이블
DEFAULT_ROUTING_POLICIES: list[RoutingPolicy] = [
    RoutingPolicy(
        task_type=TaskType.SCRIPT_GENERATION,
        primary=ModelProvider.SOLAR_PRO,
        fallback=ModelProvider.ATOMIC_GPT,
        description="스크립트 생성은 구조화 능력이 중요 → Solar Pro 우선",
    ),
    RoutingPolicy(
        task_type=TaskType.QUESTION_GENERATION,
        primary=ModelProvider.ATOMIC_GPT,
        fallback=ModelProvider.SOLAR_PRO,
        description="원자력 도메인 지식 정확도가 핵심 → AtomicGPT 우선",
    ),
    RoutingPolicy(
        task_type=TaskType.DISTRACTOR_REFINEMENT,
        primary=ModelProvider.ATOMIC_GPT,
        fallback=ModelProvider.SOLAR_PRO,
        description="오답 매력도는 도메인 지식에 의존 → AtomicGPT 우선",
    ),
    RoutingPolicy(
        task_type=TaskType.VLM_ANALYSIS,
        primary=ModelProvider.SOLAR_PRO,
        fallback=ModelProvider.SOLAR_PRO,
        description="VLM은 Solar Pro만 지원 (AtomicGPT VLM 미지원)",
    ),
    RoutingPolicy(
        task_type=TaskType.QUALITY_VALIDATION,
        primary=ModelProvider.SOLAR_PRO,
        fallback=ModelProvider.ATOMIC_GPT,
        description="품질 검증은 범용 추론 능력 활용 → Solar Pro 우선",
    ),
    RoutingPolicy(
        task_type=TaskType.SCENARIO_GENERATION,
        primary=ModelProvider.ATOMIC_GPT,
        fallback=ModelProvider.SOLAR_PRO,
        description="원전 시나리오는 도메인 지식 필수 → AtomicGPT 우선",
    ),
    RoutingPolicy(
        task_type=TaskType.SAFETY_REVIEW,
        primary=ModelProvider.ATOMIC_GPT,
        fallback=ModelProvider.SOLAR_PRO,
        require_cross_check=True,
        confidence_threshold=0.9,
        description="안전등급 문항은 양쪽 모두 검증 필수",
    ),
]


@dataclass
class RoutingResult:
    """라우팅 + LLM 호출 결과"""
    response: dict
    provider_used: ModelProvider
    fallback_used: bool = False
    cross_checked: bool = False
    cross_check_agreement: Optional[float] = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)


class LLMRouter:
    """하이브리드 LLM 라우터"""

    def __init__(
        self,
        models: dict[ModelProvider, ModelConfig],
        policies: list[RoutingPolicy] | None = None,
    ):
        self.models = models
        self._policies = {p.task_type: p for p in (policies or DEFAULT_ROUTING_POLICIES)}
        self._call_history: list[dict] = []

    def get_policy(self, task_type: TaskType) -> RoutingPolicy:
        return self._policies.get(task_type, RoutingPolicy(
            task_type=task_type,
            primary=ModelProvider.SOLAR_PRO,
            fallback=ModelProvider.SOLAR_PRO,
        ))

    async def route_and_call(
        self,
        task_type: TaskType,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        safety_grade: str = "general",
    ) -> RoutingResult:
        """
        태스크를 최적 모델에 라우팅하고 호출

        Args:
            task_type: 태스크 유형
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            temperature: LLM temperature
            max_tokens: 최대 토큰
            safety_grade: 안전등급 (safety_critical이면 크로스체크 강제)

        Returns:
            라우팅 + 응답 결과
        """
        policy = self.get_policy(task_type)

        # safety_critical 문항은 크로스체크 강제
        force_cross_check = (safety_grade == "safety_critical")
        do_cross_check = policy.require_cross_check or force_cross_check

        # 1차 호출: primary 모델
        primary_config = self.models.get(policy.primary)
        result = RoutingResult(response={}, provider_used=policy.primary)

        if primary_config and primary_config.enabled:
            try:
                import time
                start = time.monotonic()
                resp = await self._call_model(
                    primary_config, system_prompt, user_prompt,
                    temperature, max_tokens,
                )
                result.latency_ms = (time.monotonic() - start) * 1000
                result.response = resp
                result.confidence = self._estimate_confidence(resp)
            except Exception:
                result.confidence = 0.0

        # confidence 부족 → fallback
        if result.confidence < policy.confidence_threshold:
            fallback_config = self.models.get(policy.fallback)
            if fallback_config and fallback_config.enabled and policy.fallback != policy.primary:
                try:
                    resp = await self._call_model(
                        fallback_config, system_prompt, user_prompt,
                        temperature, max_tokens,
                    )
                    result.response = resp
                    result.provider_used = policy.fallback
                    result.fallback_used = True
                    result.confidence = self._estimate_confidence(resp)
                except Exception:
                    pass  # 두 모델 모두 실패 → 빈 응답 반환

        # 크로스체크: 두 모델 응답 비교
        if do_cross_check and not result.fallback_used:
            other_provider = (
                policy.fallback if result.provider_used == policy.primary
                else policy.primary
            )
            other_config = self.models.get(other_provider)
            if other_config and other_config.enabled:
                try:
                    other_resp = await self._call_model(
                        other_config, system_prompt, user_prompt,
                        temperature, max_tokens,
                    )
                    agreement = self._compute_agreement(result.response, other_resp)
                    result.cross_checked = True
                    result.cross_check_agreement = agreement

                    # 동의율 낮으면 플래그
                    if agreement < 0.5:
                        result.response["_cross_check_warning"] = (
                            f"두 모델 간 동의율 {agreement:.0%} — SME 검토 권장"
                        )
                except Exception:
                    pass

        # 이력 기록
        self._call_history.append({
            "task_type": task_type.value,
            "provider": result.provider_used.value,
            "fallback_used": result.fallback_used,
            "cross_checked": result.cross_checked,
            "confidence": result.confidence,
            "latency_ms": round(result.latency_ms, 1),
        })

        return result

    async def _call_model(
        self,
        config: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """개별 모델 API 호출"""
        async with httpx.AsyncClient(timeout=config.timeout, verify=get_ssl_verify()) as client:
            response = await client.post(
                f"{config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {config.api_key}"},
                json={
                    "model": config.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            # 토큰 사용량 추출
            usage = data.get("usage", {})
            parsed["_token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
            return parsed

    @staticmethod
    def _estimate_confidence(response: dict) -> float:
        """응답의 신뢰도 추정 (0.0 ~ 1.0)

        기준:
        - 응답이 비어있으면 0.0
        - JSON 구조가 올바르면 +0.3
        - 필수 필드가 있으면 +0.3
        - 내용이 충분하면 +0.4
        """
        if not response:
            return 0.0

        score = 0.3  # JSON 파싱 성공

        # 의미있는 키가 있는지
        meaningful_keys = [
            k for k in response.keys()
            if not k.startswith("_") and response[k]
        ]
        if meaningful_keys:
            score += 0.3

        # 내용 길이 체크
        content_str = json.dumps(response, ensure_ascii=False)
        if len(content_str) > 100:
            score += 0.4

        return min(score, 1.0)

    @staticmethod
    def _compute_agreement(resp_a: dict, resp_b: dict) -> float:
        """두 모델 응답 간 동의율 추정 (0.0 ~ 1.0)

        간이 방식: 공통 키 비율 + 값 유사도
        """
        if not resp_a or not resp_b:
            return 0.0

        keys_a = {k for k in resp_a if not k.startswith("_")}
        keys_b = {k for k in resp_b if not k.startswith("_")}

        if not keys_a and not keys_b:
            return 1.0

        common_keys = keys_a & keys_b
        all_keys = keys_a | keys_b
        key_overlap = len(common_keys) / max(len(all_keys), 1)

        # 공통 키의 값 유사도 (문자열화 후 비교)
        value_matches = 0
        for key in common_keys:
            str_a = json.dumps(resp_a[key], ensure_ascii=False, sort_keys=True)
            str_b = json.dumps(resp_b[key], ensure_ascii=False, sort_keys=True)
            if str_a == str_b:
                value_matches += 1
            elif str_a[:50] == str_b[:50]:  # 부분 일치
                value_matches += 0.5

        value_similarity = value_matches / max(len(common_keys), 1)

        return round((key_overlap * 0.4 + value_similarity * 0.6), 2)

    def get_statistics(self) -> dict:
        """라우팅 통계"""
        if not self._call_history:
            return {"total_calls": 0}

        total = len(self._call_history)
        by_provider = {}
        by_task = {}
        fallback_count = 0
        cross_check_count = 0

        for call in self._call_history:
            provider = call["provider"]
            by_provider[provider] = by_provider.get(provider, 0) + 1

            task = call["task_type"]
            by_task[task] = by_task.get(task, 0) + 1

            if call["fallback_used"]:
                fallback_count += 1
            if call["cross_checked"]:
                cross_check_count += 1

        avg_latency = sum(c["latency_ms"] for c in self._call_history) / total
        avg_confidence = sum(c["confidence"] for c in self._call_history) / total

        return {
            "total_calls": total,
            "by_provider": by_provider,
            "by_task_type": by_task,
            "fallback_rate": round(fallback_count / total * 100, 1),
            "cross_check_rate": round(cross_check_count / total * 100, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_confidence": round(avg_confidence, 2),
        }


def create_default_router(
    solar_api_key: str,
    solar_base_url: str = "https://api.upstage.ai/v1",
    atomic_api_key: str = "",
    atomic_base_url: str = "http://localhost:8080/v1",
) -> LLMRouter:
    """기본 라우터 팩토리 — Solar Pro 활성, AtomicGPT 대기

    AtomicGPT가 준비되면 atomic_api_key만 설정하면 자동 활성화됩니다.
    """
    models = {
        ModelProvider.SOLAR_PRO: ModelConfig(
            provider=ModelProvider.SOLAR_PRO,
            base_url=solar_base_url,
            api_key=solar_api_key,
            model_name="solar-pro3",
            max_tokens=8192,
            enabled=True,
        ),
        ModelProvider.ATOMIC_GPT: ModelConfig(
            provider=ModelProvider.ATOMIC_GPT,
            base_url=atomic_base_url,
            api_key=atomic_api_key,
            model_name="atomic-gpt-70b",
            max_tokens=8192,
            enabled=bool(atomic_api_key),  # API 키 있을 때만 활성화
        ),
    }
    return LLMRouter(models=models)
