"""
Convenience search entrypoint:
- baseline TF-IDF ranker (default)
- optional personal-model reranking when profile is provided
- lightweight query rewriting for structured metadata (controlled vocabulary)

Usage (from repo root):
  python -m src.rank.search_model --query "nature hiking" --top_k 10
  python -m src.rank.search_model --query "outdoor" --top_k 10
  python -m src.rank.search_model --query "nature hiking" --top_k 10 --personal \
      --age older --athleticism low --travel family --frequency first \
      --interests nature,culture,food
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from src.config import Config
from src.index.inverted_index import InvertedIndex
from .tfidf import TfidfRanker
from .personal_model import PersonalReranker, UserProfile


# -----------------------------
# IO helpers
# -----------------------------

def load_docstore_jsonl(path) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs

def rewrite_query(query: str) -> str:
    """
    Map user-friendly intents to field-aware query tokens.

    This bridges free-text user input (e.g., "outdoor")
    to structured metadata indexed with prefixes
    (e.g., "preference:outdoor", "interests:nature").
    """
    q = query.lower().strip()

    QUERY_MAP = {
        # preference / environment
        "outdoor": [
            "preference:outdoor",
            "interests:nature",
            "interests:hiking",
            "interests:scenery",
        ],
        "indoor": [
            "preference:indoor",
        ],

        # travel style
        "family": ["travel:family"],
        "friends": ["travel:friends"],
        "solo": ["travel:solo"],

        # common activity intents
        "hiking": ["interests:hiking", "interests:adventure"],
        "nature": ["interests:nature", "interests:scenery"],
        "culture": ["interests:culture", "interests:history"],
        "food": ["interests:food", "interests:culinary"],
        "relax": ["interests:relaxation", "interests:wellness"],
    }

    # exact intent match → structured rewrite
    if q in QUERY_MAP:
        return " ".join(QUERY_MAP[q])

    # otherwise, fall back to original query
    return query


def build_profile_from_args(args: argparse.Namespace) -> Optional[UserProfile]:
    if not args.personal:
        return None

    interests = None
    if args.interests:
        interests = [x.strip().lower() for x in args.interests.split(",") if x.strip()]

    return UserProfile(
        age_group=args.age,
        athleticism=args.athleticism,
        travel_style=args.travel,
        visit_type=args.frequency,
        interests=interests,
    )


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=10)

    # optional personalization
    parser.add_argument("--personal", action="store_true", help="enable personal-model reranking")
    parser.add_argument("--age", default=None, help="young|older")
    parser.add_argument("--athleticism", default=None, help="low|medium|high")
    parser.add_argument("--travel", default=None, help="solo|friends|family")
    parser.add_argument("--frequency", default=None, help="first|frequent")
    parser.add_argument("--interests", default=None, help="comma-separated, e.g. nature,culture,food")

    args = parser.parse_args()

    # load data
    cfg = Config()
    inv = InvertedIndex.load(cfg.INDEX_PATH)
    docstore = load_docstore_jsonl(cfg.DOCSTORE_PATH)

    # rewrite query (key step)
    rewritten_query = rewrite_query(args.query)

    ranker = TfidfRanker(inv, docstore)
    base = ranker.rank(rewritten_query, top_k=args.top_k)

    profile = build_profile_from_args(args)
    if profile is not None:
        reranker = PersonalReranker()
        final = reranker.rerank(base, profile, top_k=args.top_k)
    else:
        final = base

    # output
    for i, r in enumerate(final, start=1):
        print(f"{i:02d}. score={r.score:.4f}  {r.text}")
        m = r.meta or {}
        print(
            f"    meta: Age={m.get('Age')}, "
            f"Athleticism={m.get('Athleticism')}, "
            f"Travel={m.get('Travel')}, "
            f"Frequency={m.get('Frequency')}, "
            f"Interests={m.get('Interests')}"
        )


if __name__ == "__main__":
    main()
