"""
Personal-model reranking.

This is intentionally lightweight / rules-based so you can:
- keep a clean baseline ranker (TF-IDF only)
- optionally apply personalization when the user provides a profile

The dataset metadata fields (from docstore.jsonl) include:
  Age, Athleticism, Travel, Frequency, Interests
(and may include Season/Preference depending on your CSV)

We treat "Both" as a wildcard match for categorical fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tfidf import ScoredDoc


def _norm_str(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


def _split_interests(s: Any) -> List[str]:
    # dataset Interests is like: "culture, history, spiritual, sightseeing"
    s = _norm_str(s)
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


@dataclass(frozen=True)
class UserProfile:
    """
    All fields optional: if a user does not select a field, we don't use it.

    age_group: "young" | "older"
    athleticism: "low" | "medium" | "high"
    travel_style: "solo" | "friends" | "family"
    visit_type: "first" | "frequent"
    interests: list of interest tags (lowercase strings)
    """
    age_group: Optional[str] = None
    athleticism: Optional[str] = None
    travel_style: Optional[str] = None
    visit_type: Optional[str] = None
    interests: Optional[Sequence[str]] = None


class PersonalReranker:
    def __init__(
        self,
        w_age: float = 0.08,
        w_athleticism: float = 0.10,
        w_travel: float = 0.08,
        w_frequency: float = 0.06,
        w_interests: float = 0.18,
        mismatch_penalty: float = 0.06,
    ):
        """
        We apply: new_score = base_score * (1 + boost - penalty)

        Weights are intentionally small so baseline relevance stays dominant.
        Tune as needed with your team.
        """
        self.w_age = w_age
        self.w_athleticism = w_athleticism
        self.w_travel = w_travel
        self.w_frequency = w_frequency
        self.w_interests = w_interests
        self.mismatch_penalty = mismatch_penalty

    def rerank(self, results: List[ScoredDoc], profile: UserProfile, top_k: Optional[int] = None) -> List[ScoredDoc]:
        scored: List[Tuple[float, ScoredDoc]] = []
        for r in results:
            boost, penalty = self._personal_signal(r.meta, profile)
            new_score = r.score * (1.0 + boost - penalty)
            scored.append((new_score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [
            ScoredDoc(doc_id=r.doc_id, score=float(s), text=r.text, meta=r.meta)
            for s, r in scored
        ]
        return reranked if top_k is None else reranked[:top_k]

    def _match_or_both(self, doc_val: Any, user_val: str) -> Optional[bool]:
        """
        Returns:
          True  if matches (or doc is Both)
          False if definitely mismatch
          None  if doc missing / user missing
        """
        d = _norm_str(doc_val)
        u = _norm_str(user_val)
        if not u:
            return None
        if not d:
            return None
        if d == "both":
            return True
        return d == u

    def _personal_signal(self, meta: Dict[str, Any], profile: UserProfile) -> Tuple[float, float]:
        boost = 0.0
        penalty = 0.0

        # Age
        if profile.age_group:
            m = self._match_or_both(meta.get("Age"), profile.age_group)
            if m is True:
                boost += self.w_age
            elif m is False:
                penalty += self.mismatch_penalty

        # Athleticism
        if profile.athleticism:
            m = self._match_or_both(meta.get("Athleticism"), profile.athleticism)
            if m is True:
                boost += self.w_athleticism
            elif m is False:
                penalty += self.mismatch_penalty

        # Travel style
        if profile.travel_style:
            m = self._match_or_both(meta.get("Travel"), profile.travel_style)
            if m is True:
                boost += self.w_travel
            elif m is False:
                penalty += self.mismatch_penalty

        # Visit type / frequency
        if profile.visit_type:
            m = self._match_or_both(meta.get("Frequency"), profile.visit_type)
            if m is True:
                boost += self.w_frequency
            elif m is False:
                penalty += self.mismatch_penalty

        # Interests overlap (soft signal)
        user_interests = [ _norm_str(x) for x in (profile.interests or []) if _norm_str(x) ]
        if user_interests:
            doc_interests = _split_interests(meta.get("Interests"))
            if doc_interests:
                overlap = len(set(user_interests) & set(doc_interests))
                # normalize by user interests count so it doesn't reward huge tag lists too much
                overlap_ratio = overlap / max(1, len(set(user_interests)))
                boost += self.w_interests * overlap_ratio

        # clamp
        boost = max(0.0, min(0.6, boost))
        penalty = max(0.0, min(0.6, penalty))
        return boost, penalty
