import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

# Token pattern:
# - latin/number chunk: [a-z0-9]+
# - japanese chunk: hiragana/katakana/kanji continuous blocks
TOKEN_RE = re.compile(
    r"[a-z0-9]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+|[\u4e00-\u9fff]+",
    re.IGNORECASE
)

def tokenize(text: str) -> List[str]:
    text = "" if text is None else str(text).lower()
    return TOKEN_RE.findall(text)

@dataclass
class Posting:
    doc_id: int
    tf: int
    positions: List[int]

class InvertedIndex:
    """
    In-memory inverted index with positional information.
    """
    def __init__(self):
        self.index: Dict[str, List[Posting]] = defaultdict(list)
        self.doc_len: Dict[int, int] = {}
        self.num_docs: int = 0

    def add_document(self, doc_id: int, text: str):
        tokens = tokenize(text)
        self.doc_len[doc_id] = len(tokens)

        # positions
        pos_map = defaultdict(list)
        for i, t in enumerate(tokens):
            pos_map[t].append(i)

        tf_counter = Counter(tokens)
        for term, tf in tf_counter.items():
            self.index[term].append(Posting(doc_id=doc_id, tf=tf, positions=pos_map[term]))

        self.num_docs += 1

    def build(self, docs: List[Dict[str, Any]], text_key: str = "text", id_key: str = "doc_id"):
        for d in docs:
            self.add_document(int(d[id_key]), d[text_key])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_docs": self.num_docs,
            "doc_len": self.doc_len,
            "index": {
                term: [{"doc_id": p.doc_id, "tf": p.tf, "positions": p.positions} for p in postings]
                for term, postings in self.index.items()
            },
        }

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "InvertedIndex":
        ii = InvertedIndex()
        ii.num_docs = int(obj.get("num_docs", 0))
        ii.doc_len = {int(k): int(v) for k, v in obj.get("doc_len", {}).items()}
        for term, postings in obj.get("index", {}).items():
            ii.index[term] = [Posting(doc_id=int(p["doc_id"]), tf=int(p["tf"]), positions=list(p["positions"])) for p in postings]
        return ii

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @staticmethod
    def load(path: Path) -> "InvertedIndex":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return InvertedIndex.from_dict(obj)
