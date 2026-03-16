"""프롬프트 템플릿 매니저

YAML 파일에서 프롬프트를 로드하여 코드 변경 없이 LLM 동작 튜닝 가능.
- 핫리로드: 파일 수정 시 자동 반영 (캐시 무효화)
- 버전 추적: 프롬프트 변경 이력 관리
- 오버라이드: 런타임에 특정 프롬프트만 교체 가능
"""

import os
from pathlib import Path
from typing import Optional

import yaml


_TEMPLATES_DIR = Path(__file__).parent
_DEFAULT_TEMPLATE_FILE = _TEMPLATES_DIR / "templates.yaml"

# 모듈 수준 싱글톤
_manager_instance: Optional["PromptTemplateManager"] = None


class PromptTemplateManager:
    """YAML 기반 프롬프트 템플릿 로더"""

    def __init__(self, template_path: str | Path | None = None):
        self._path = Path(template_path) if template_path else _DEFAULT_TEMPLATE_FILE
        self._templates: dict = {}
        self._overrides: dict[str, str] = {}
        self._file_mtime: float = 0.0
        self._load()

    def _load(self) -> None:
        """YAML 파일 로드 (파일이 변경되었으면 재로드)"""
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return

        if mtime != self._file_mtime:
            with open(self._path, encoding="utf-8") as f:
                self._templates = yaml.safe_load(f) or {}
            self._file_mtime = mtime

    def get_prompt(self, key: str) -> str:
        """프롬프트 시스템 메시지 조회

        Args:
            key: 프롬프트 키 (예: "script_generation", "vlm_analysis")

        Returns:
            시스템 프롬프트 텍스트
        """
        # 오버라이드 우선
        if key in self._overrides:
            return self._overrides[key]

        # 핫리로드 체크
        self._load()

        section = self._templates.get(key, {})
        if isinstance(section, dict):
            return section.get("system_prompt", "")
        return ""

    def get_config(self, key: str) -> dict:
        """프롬프트 섹션의 전체 설정 조회 (temperature, max_tokens 등 포함)"""
        self._load()
        section = self._templates.get(key, {})
        return section if isinstance(section, dict) else {}

    def get_temperature(self, key: str) -> float:
        """프롬프트별 temperature 조회"""
        config = self.get_config(key)
        return config.get("temperature", 0.3)

    def get_max_tokens(self, key: str) -> int:
        """프롬프트별 max_tokens 조회"""
        config = self.get_config(key)
        return config.get("max_tokens", 4096)

    def get_version(self, key: str) -> str:
        """프롬프트 버전 조회"""
        config = self.get_config(key)
        return config.get("version", "unknown")

    def set_override(self, key: str, prompt: str) -> None:
        """런타임 프롬프트 오버라이드 (A/B 테스트용)"""
        self._overrides[key] = prompt

    def clear_override(self, key: str) -> None:
        """오버라이드 해제"""
        self._overrides.pop(key, None)

    def list_templates(self) -> list[dict]:
        """사용 가능한 템플릿 목록"""
        self._load()
        result = []
        for key, section in self._templates.items():
            if key == "metadata":
                continue
            if isinstance(section, dict) and "system_prompt" in section:
                result.append({
                    "key": key,
                    "version": section.get("version", "unknown"),
                    "has_override": key in self._overrides,
                    "prompt_length": len(section.get("system_prompt", "")),
                })
        return result

    def get_all_versions(self) -> dict[str, str]:
        """모든 템플릿의 버전 정보"""
        self._load()
        return {
            key: section.get("version", "unknown")
            for key, section in self._templates.items()
            if key != "metadata" and isinstance(section, dict)
        }


def get_template_manager(template_path: str | Path | None = None) -> PromptTemplateManager:
    """싱글톤 매니저 인스턴스 반환"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = PromptTemplateManager(template_path)
    return _manager_instance


# 편의 함수: 코드에서 직접 호출
def get_prompt(key: str) -> str:
    """프롬프트 키로 시스템 프롬프트 조회 (편의 함수)"""
    return get_template_manager().get_prompt(key)
