"""
TF-IDF baseline ranker (default model).

Uses the existing serialized InvertedIndex (with TF + positions) to compute
cosine-similarity scores efficiently (posting-list based).

Scoring (default):
  w_td = (1 + log(tf_td)) * idf_t
  w_tq = (1 + log(tf_tq)) * idf_t
  score(d, q) = dot(q, d) / (||q|| * ||d||)

IDF:
  idf_t = log((N + 1) / (df_t + 1)) + 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.index.inverted_index import InvertedIndex, tokenize


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: int
    score: float
    text: str
    meta: Dict[str, Any]


class TfidfRanker:
    def __init__(self, inv: InvertedIndex, docstore: List[Dict[str, Any]]):
        self.inv = inv
        self.docstore = docstore

        # map doc_id -> doc object for quick lookup
        self._docs_by_id: Dict[int, Dict[str, Any]] = {int(d["doc_id"]): d for d in docstore}

        # precompute df + idf for each term in index
        self._df: Dict[str, int] = {t: len(postings) for t, postings in inv.index.items()}
        self._idf: Dict[str, float] = {t: self._idf_formula(self._df[t]) for t in self._df}

        # precompute document vector norms for cosine normalization
        self._doc_norm: Dict[int, float] = self._build_doc_norms()

    def _idf_formula(self, df: int) -> float:
        # smooth idf
        return math.log((self.inv.num_docs + 1.0) / (df + 1.0)) + 1.0

    def _tf_weight(self, tf: int) -> float:
        # log tf
        return 1.0 + math.log(tf) if tf > 0 else 0.0

    def _build_doc_norms(self) -> Dict[int, float]:
        accum: Dict[int, float] = {doc_id: 0.0 for doc_id in self.inv.doc_len.keys()}
        for term, postings in self.inv.index.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for p in postings:
                w = self._tf_weight(int(p.tf)) * idf
                accum[int(p.doc_id)] += w * w
        return {doc_id: math.sqrt(v) if v > 0 else 1.0 for doc_id, v in accum.items()}

    def _query_weights(self, query: str) -> Tuple[Dict[str, float], float]:
        tokens = tokenize(query)
        if not tokens:
            return {}, 1.0

        tf_q: Dict[str, int] = {}
        for t in tokens:
            tf_q[t] = tf_q.get(t, 0) + 1

        w_q: Dict[str, float] = {}
        norm2 = 0.0
        for t, tf in tf_q.items():
            idf = self._idf.get(t)
            if idf is None:
                continue  # OOV term not in index
            w = self._tf_weight(tf) * idf
            w_q[t] = w
            norm2 += w * w

        q_norm = math.sqrt(norm2) if norm2 > 0 else 1.0
        return w_q, q_norm

    def rank(
        self,
        query: str,
        top_k: int = 20,
        candidate_doc_ids: Optional[List[int]] = None,
    ) -> List[ScoredDoc]:
        """
        Rank documents for a query using TF-IDF cosine similarity.

        If candidate_doc_ids is provided, scores are only computed for that subset
        (useful for pre-filtering).
        """
        w_q, q_norm = self._query_weights(query)
        if not w_q:
            return []

        # scoring via posting lists
        scores: Dict[int, float] = {}

        allowed: Optional[set[int]] = None
        if candidate_doc_ids is not None:
            allowed = set(int(x) for x in candidate_doc_ids)

        for term, wqt in w_q.items():
            postings = self.inv.index.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for p in postings:
                doc_id = int(p.doc_id)
                if allowed is not None and doc_id not in allowed:
                    continue
                wdt = self._tf_weight(int(p.tf)) * idf
                scores[doc_id] = scores.get(doc_id, 0.0) + (wqt * wdt)

        # cosine normalize
        results: List[ScoredDoc] = []
        for doc_id, dot in scores.items():
            d_norm = self._doc_norm.get(doc_id, 1.0)
            score = dot / (q_norm * d_norm)

            d = self._docs_by_id.get(doc_id)
            if d is None:
                continue
            results.append(
                ScoredDoc(
                    doc_id=doc_id,
                    score=float(score),
                    text=str(d.get("text", "")),
                    meta=dict(d.get("meta", {})),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
