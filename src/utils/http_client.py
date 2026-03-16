"""
HTTP 클라이언트 공통 설정

회사 네트워크 SSL 프록시(ePrism) 대응:
  SSL_VERIFY=false 환경변수 설정 시 SSL 검증 비활성화
"""
import os

import httpx


def get_ssl_verify() -> bool:
    """SSL 검증 설정 반환"""
    val = os.environ.get("SSL_VERIFY", "true").lower()
    return val not in ("false", "0", "no")


def create_client(timeout: float = 120.0, **kwargs) -> httpx.AsyncClient:
    """SSL 설정이 적용된 AsyncClient 생성"""
    return httpx.AsyncClient(
        timeout=timeout,
        verify=get_ssl_verify(),
        **kwargs,
    )
