"""
KHNP Education AI Platform - Configuration
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UpstageConfig:
    """Upstage API 설정 (PoC: 클라우드 API / Production: 온프레미스)"""
    api_key: str = ""
    base_url: str = "https://api.upstage.ai/v1"
    document_parse_endpoint: str = "/document-digitization"
    embedding_endpoint: str = "/embeddings"
    chat_endpoint: str = "/chat/completions"
    document_parse_model: str = "document-parse"  # or "document-parse-nightly" for enhanced
    embedding_model: str = "embedding-query"
    solar_model: str = "solar-pro"

    # 온프레미스 전환 시
    on_premise_llm_url: str = "http://localhost:8000"
    on_premise_doc_parse_url: str = "http://localhost:8001"


@dataclass
class RAGConfig:
    """RAG 엔진 설정"""
    vector_db_type: str = "chromadb"  # chromadb | milvus
    chromadb_path: str = "./data/chromadb"
    collection_name: str = "khnp_knowledge"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class SATConfig:
    """SAT 체계 설정"""
    taxonomy_levels: list = field(default_factory=lambda: [
        "Knowledge",      # 지식
        "Comprehension",  # 이해
        "Application",    # 적용
        "Analysis",       # 분석
    ])
    default_question_distribution: dict = field(default_factory=lambda: {
        "Knowledge": 0.3,
        "Comprehension": 0.3,
        "Application": 0.25,
        "Analysis": 0.15,
    })


@dataclass
class GenerationConfig:
    """생성 엔진 설정"""
    script_max_tokens: int = 4096
    question_max_tokens: int = 2048
    temperature_script: float = 0.7
    temperature_question: float = 0.3
    questions_per_slide: int = 3
    script_style: str = "formal"  # formal | conversational
    target_audience: str = "intermediate"  # beginner | intermediate | advanced


@dataclass
class AppConfig:
    """전체 앱 설정"""
    upstage: UpstageConfig = field(default_factory=UpstageConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    sat: SATConfig = field(default_factory=SATConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    data_dir: str = "./data"
    output_dir: str = "./data/output"
    log_level: str = "INFO"
