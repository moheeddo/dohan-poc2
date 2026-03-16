"""Question Generator - SAT 기반 문제은행 자동 생성

v4 모듈 구성:
- generator: 2-Pass 문항 생성 + 시나리오형 문항
- quality_validator: QA 파이프라인 + Item Analysis + IRT 교정
- safety_tagger: IAEA TECDOC-2082 안전 중요도 태깅 + Hybrid 전략
- few_shot_manager: SME 모범 문항 관리
- sme_review: SME 검토 워크플로우 (AI→SME→Diff학습→프롬프트개선)
- version_manager: 문항 버전 관리 및 라이프사이클 추적
- qti_exporter: QTI 2.1 표준 내보내기 (CBT 연동)
- adaptive_engine: 적응형 난이도 조절 (IRT + 안전등급 연계)
"""
