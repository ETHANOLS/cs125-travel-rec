import re
from typing import Dict, Any, List
import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text

def row_to_doc(row: pd.Series, text_col: str = "Activity") -> Dict[str, Any]:
    """
    Logical view:
      - doc_id: assigned outside (row index)
      - text: Activity
      - meta: other columns
    """
    text = normalize_text(row.get(text_col, ""))

    meta = {}
    for k, v in row.items():
        if k == text_col:
            continue
        # keep metadata, but normalize to string for consistency
        meta[k] = normalize_text(v)

    return {"text": text, "meta": meta}

def build_docstore(df: pd.DataFrame, text_col: str = "Activity") -> List[Dict[str, Any]]:
    docs = []
    for doc_id, row in df.iterrows():
        doc = row_to_doc(row, text_col=text_col)
        doc["doc_id"] = int(doc_id)
        docs.append(doc)
    return docs
