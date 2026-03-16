"""
지식 기반 RAG 엔진
- 교재, 규정, 절차서, 사고사례 등 도메인 지식 임베딩
- Hybrid Search (Dense + Sparse) 지원
- SAT 학습목표 기반 메타데이터 필터링
"""
import os
from typing import Optional

import chromadb
import httpx

from src.utils.http_client import get_ssl_verify


class KnowledgeBase:
    """ChromaDB 기반 벡터 지식 저장소"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.upstage.ai/v1",
        db_path: str = "./data/chromadb",
        collection_name: str = "khnp_knowledge",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def embed_text(self, text: str, model: str = "embedding-query") -> list[float]:
        """Upstage Embedding API로 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트 (최대 4,000 토큰, 512 이하 권장)
            model: "embedding-query" (검색 쿼리용) 또는 "embedding-passage" (문서 저장용)
        """
        async with httpx.AsyncClient(timeout=30.0, verify=get_ssl_verify()) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "input": text,
                },
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ):
        """문서 청크를 지식 베이스에 추가"""
        embedding = await self.embed_text(text, model="embedding-passage")
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )

    async def add_documents_batch(
        self,
        chunks: list[dict],
    ):
        """
        다수 문서 청크를 일괄 추가

        Args:
            chunks: [{"id": str, "text": str, "metadata": dict}, ...]
        """
        for chunk in chunks:
            await self.add_document(
                doc_id=chunk["id"],
                text=chunk["text"],
                metadata=chunk.get("metadata", {}),
            )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        쿼리와 유사한 지식 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            metadata_filter: 메타데이터 필터 (과목, 역량단위 등)

        Returns:
            검색 결과 리스트
        """
        query_embedding = await self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

        return search_results


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """시맨틱 청킹 (문단 단위 우선, 크기 초과 시 분할)"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > chunk_size:
                # 긴 문단은 문장 단위로 분할
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk += (" " if current_chunk else "") + word
                    else:
                        chunks.append(current_chunk)
                        # overlap 적용
                        overlap_words = current_chunk.split()[-chunk_overlap // 10 :]
                        current_chunk = " ".join(overlap_words) + " " + word
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
