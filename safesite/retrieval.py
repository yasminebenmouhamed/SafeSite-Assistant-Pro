"""Similarity, MMR and lexical+vector reciprocal-rank-fusion retrieval."""
from __future__ import annotations
import re
from collections import defaultdict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def doc_key(doc: Document) -> str:
    return doc.metadata.get('chunk_id') or f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{doc.page_content}"


def rrf_merge(rankings: list[list[Document]], top_k: int, constant: int = 60) -> list[Document]:
    scores = defaultdict(float)
    docs = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            key = doc_key(doc)
            scores[key] += 1 / (constant + rank)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)[:top_k]]


def lexical_search(query: str, docs: list[Document], k: int) -> list[Document]:
    if not docs or not tokenize(query):
        return []
    corpus = [tokenize(d.page_content) or ['_empty_'] for d in docs]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    return [docs[i] for i in sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k] if scores[i] > 0]


def search(store, query: str, k: int, mode: str = 'hybrid', source: str | None = None) -> list[Document]:
    where = {'source': source} if source else None
    kwargs = {'filter': where} if where else {}
    if mode == 'mmr':
        return store.max_marginal_relevance_search(query, k=k, fetch_k=max(k * 5, 20), **kwargs)
    vector_results = store.similarity_search(query, k=max(k * 3, 12) if mode == 'hybrid' else k, **kwargs)
    if mode == 'similarity':
        return vector_results
    # Lexical candidates come from all stored chunks. For large deployments,
    # replace this local BM25 scan with a dedicated indexed search engine.
    all_data = store.get(where=where, include=['documents', 'metadatas']) if where else store.get(include=['documents', 'metadatas'])
    all_docs = [Document(page_content=text, metadata=meta or {}) for text, meta in zip(all_data.get('documents') or [], all_data.get('metadatas') or [])]
    lexical_results = lexical_search(query, all_docs, k=max(12, k * 3))
    return rrf_merge([vector_results, lexical_results], top_k=k)
