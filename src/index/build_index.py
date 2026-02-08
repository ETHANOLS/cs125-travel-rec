import json
from pathlib import Path
from typing import List, Dict, Any

from src.config import Config
from src.ingest.load_data import load_japan_activities
from src.ingest.clean import build_docstore
from src.index.inverted_index import InvertedIndex

def save_docstore_jsonl(docs: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def main():
    cfg = Config()

    # 1) load
    df = load_japan_activities(cfg)

    # 2) logical view
    docs = build_docstore(df, text_col="Activity")

    # 3) build index
    inv = InvertedIndex()
    inv.build(docs, text_key="text", id_key="doc_id")

    # 4) save to data/processed
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    save_docstore_jsonl(docs, cfg.DOCSTORE_PATH)
    inv.save(cfg.INDEX_PATH)

    print("Done.")
    print(f"- docs: {len(docs)} -> {cfg.DOCSTORE_PATH}")
    print(f"- index terms: {len(inv.index)} -> {cfg.INDEX_PATH}")

if __name__ == "__main__":
    main()
