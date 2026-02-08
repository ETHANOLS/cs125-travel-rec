from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # repo root: .../cs125-travel-rec
    REPO_ROOT: Path = Path(__file__).resolve().parents[1]

    # raw input
    RAW_DATA_PATH: Path = REPO_ROOT / "data" / "raw" / "Japan_Activites.csv"

    # processed outputs
    PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed"
    INDEX_DIR: Path = PROCESSED_DIR / "index"

    # outputs
    DOCSTORE_PATH: Path = PROCESSED_DIR / "docstore.jsonl"
    INDEX_PATH: Path = INDEX_DIR / "inverted_index.json"
