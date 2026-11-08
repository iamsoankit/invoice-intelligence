"""Retrieval over the policy knowledge base.

Three retrievers share one interface:
  * BM25Retriever      - lexical, zero dependencies, used offline.
  * EmbeddingRetriever - dense vectors from Gemini embeddings, cached on disk.
  * HybridRetriever    - Reciprocal Rank Fusion of the two.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from . import config
from .schemas import PolicyChunk, RetrievedChunk

_HEADING = re.compile(r"^##\s+\[(?P<id>[A-Z]+-\d+)\]\s+(?P<title>.+)$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its must not of on or the this to with will be".split()
)


def load_policy_chunks(directory: Path = config.KNOWLEDGE_BASE_DIR) -> list[PolicyChunk]:
    """Split every markdown file into one chunk per `## [ID] Title` section."""
    chunks: list[PolicyChunk] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        matches = list(_HEADING.finditer(text))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end():end].strip()
            chunks.append(
                PolicyChunk(id=match["id"], title=match["title"].strip(), text=body, source=path.name)
            )
    return chunks


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[PolicyChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.title} {c.text}")) for c in self.chunks]
        self.doc_len = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.doc_len) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        terms = tokenize(query)
        scored = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.doc_len, strict=True):
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[t] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=round(score, 4), retriever=self.name))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingRetriever:
    """Dense retriever backed by Gemini embeddings.

    gemini-embedding-2 takes task instructions as text prefixes (not `task_type`)
    and aggregates list inputs into a single vector, so each text is embedded
    in its own call. Document vectors are cached by content hash.
    """

    name = "embedding"

    def __init__(self, chunks: Sequence[PolicyChunk], client, model: str = config.EMBEDDING_MODEL,
                 dim: int = config.EMBEDDING_DIM, cache_dir: Path = config.CACHE_DIR):
        self.chunks = list(chunks)
        self.client, self.model, self.dim = client, model, dim
        self.cache_path = cache_dir / f"policy_index_{model}_{dim}.json"
        self.vectors = self._load_or_build_index()

    def _embed(self, text: str) -> list[float]:
        from google.genai import types

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.dim),
        )
        return list(response.embeddings[0].values)

    @staticmethod
    def _doc_text(chunk: PolicyChunk) -> str:
        return f"title: {chunk.title} | text: {chunk.text}"

    def _load_or_build_index(self) -> dict[str, list[float]]:
        cache = {}
        if self.cache_path.exists():
            cache = json.loads(self.cache_path.read_text())
        vectors, dirty = {}, False
        for chunk in self.chunks:
            key = hashlib.sha256(self._doc_text(chunk).encode()).hexdigest()
            if key not in cache:
                cache[key] = self._embed(self._doc_text(chunk))
                dirty = True
            vectors[chunk.id] = cache[key]
        if dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache))
        return vectors

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        q = self._embed(f"task: search result | query: {query}")
        scored = [
            RetrievedChunk(chunk=c, score=round(_cosine(q, self.vectors[c.id]), 4), retriever=self.name)
            for c in self.chunks
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)) across retrievers."""

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60):
        self.retrievers, self.rrf_k = list(retrievers), rrf_k

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        fused: dict[str, float] = {}
        by_id: dict[str, PolicyChunk] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, k=k * 2), start=1):
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1 / (self.rrf_k + rank)
                by_id[hit.chunk.id] = hit.chunk
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [RetrievedChunk(chunk=by_id[cid], score=round(s, 5), retriever=self.name) for cid, s in ranked]
