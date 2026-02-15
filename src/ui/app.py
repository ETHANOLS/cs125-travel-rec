from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import streamlit as st

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.config import Config
from src.index.inverted_index import InvertedIndex
from src.rank.personal_model import PersonalReranker, UserProfile
from src.rank.search_model import load_docstore_jsonl, rewrite_query
from src.rank.tfidf import ScoredDoc, TfidfRanker


INTEREST_CATEGORIES: Dict[str, List[str]] = {
    "Nature & Outdoors": ["nature", "scenery", "hiking", "adventure", "wildlife", "beaches"],
    "Culture & History": ["culture", "history", "spiritual", "education", "sightseeing"],
    "Food & Drink": ["food", "culinary", "relaxation", "wellness"],
    "Arts & Entertainment": ["art", "entertainment", "photography", "music", "pop culture", "anime", "festivals"],
    "Shopping & Nightlife": ["shopping", "fashion", "night life"],
    "Games & Otaku": ["crane games", "video games", "retro games", "trading cards", "pokemon", "nintendo"],
    "Sports & Active": ["sports"],
}


@dataclass(frozen=True)
class UiContext:
    season: str | None = None
    setting: str | None = None


def _split_interests(raw: Any) -> List[str]:
    s = "" if raw is None else str(raw).strip().lower()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def get_location(name: str) -> str | None:
    n = name.lower()
    if "kyoto" in n:
        return "Kyoto"
    if any(x in n for x in ["tokyo", "akihabara", "harajuku", "ginza", "shibuya", "shinjuku", "asakusa", "odaiba", "akiba"]):
        return "Tokyo"
    if any(x in n for x in ["osaka", "dotonbori", "den den"]):
        return "Osaka"
    if any(x in n for x in ["hokkaido", "sapporo", "niseko", "furano"]):
        return "Hokkaido"
    if "okinawa" in n:
        return "Okinawa"
    if "hiroshima" in n:
        return "Hiroshima"
    if "nara" in n:
        return "Nara"
    if "hakone" in n:
        return "Hakone"
    if "kanazawa" in n:
        return "Kanazawa"
    if "nagoya" in n:
        return "Nagoya"
    if "yokohama" in n:
        return "Yokohama"
    if "kamakura" in n:
        return "Kamakura"
    if "fuji" in n:
        return "Mt. Fuji"
    if "nagasaki" in n:
        return "Nagasaki"
    if "beppu" in n:
        return "Beppu"
    if "nikko" in n:
        return "Nikko"
    if "matsumoto" in n:
        return "Matsumoto"
    if "himeji" in n:
        return "Himeji"
    if "miyajima" in n:
        return "Miyajima"
    if "kumamoto" in n:
        return "Kumamoto"
    if "takao" in n:
        return "Mt. Takao"
    if "shirakawa" in n:
        return "Shirakawa-go"
    if "naoshima" in n:
        return "Naoshima"
    if "yakushima" in n:
        return "Yakushima"
    if "alps" in n:
        return "Japanese Alps"
    return None


def get_emoji(interests: Sequence[str]) -> str:
    if any(i in interests for i in ["hiking", "adventure"]):
        return "⛰️"
    if any(i in interests for i in ["spiritual", "culture"]):
        return "⛩️"
    if any(i in interests for i in ["food", "culinary"]):
        return "🍜"
    if any(i in interests for i in ["art", "photography"]):
        return "🎨"
    if "wildlife" in interests:
        return "🦌"
    if "festivals" in interests:
        return "🎆"
    if any(i in interests for i in ["relaxation", "wellness"]):
        return "♨️"
    if any(i in interests for i in ["entertainment", "pop culture", "anime"]):
        return "🎮"
    if any(i in interests for i in ["shopping", "fashion"]):
        return "🛍️"
    if any(i in interests for i in ["history", "education"]):
        return "🏯"
    if "sports" in interests:
        return "🏅"
    if "beaches" in interests:
        return "🏖️"
    if any(i in interests for i in ["crane games", "video games", "trading cards"]):
        return "🕹️"
    if any(i in interests for i in ["nature", "scenery"]):
        return "🌸"
    if "night life" in interests:
        return "🌙"
    return "📍"


def build_why(meta: Dict[str, Any], profile: UserProfile, ctx: UiContext, selected_interests: Sequence[str]) -> List[str]:
    why: List[str] = []
    age = str(meta.get("Age", "")).strip().lower()
    athleticism = str(meta.get("Athleticism", "")).strip().lower()
    travel = str(meta.get("Travel", "")).strip().lower()
    frequency = str(meta.get("Frequency", "")).strip().lower()
    season = str(meta.get("Season", "")).strip().lower()
    pref = str(meta.get("Preference", "")).strip().lower()
    doc_interests = _split_interests(meta.get("Interests"))

    overlap = sorted(set(selected_interests) & set(doc_interests))
    if overlap:
        why.append(f"Matches interests: {', '.join(overlap[:3])}")
    if profile.age_group and age in {profile.age_group, "both"}:
        why.append("Fits traveler type")
    if profile.athleticism and athleticism in {profile.athleticism, "both"}:
        why.append("Good activity level fit")
    if profile.travel_style and travel in {profile.travel_style, "both"}:
        why.append("Good for your travel style")
    if profile.visit_type and frequency in {profile.visit_type, "both"}:
        why.append("Matches first-time vs repeat preference")
    if ctx.season and season in {ctx.season.lower(), "all"}:
        why.append(f"Great for {ctx.season}")
    if ctx.setting and pref in {ctx.setting.lower(), "both"}:
        why.append(f"{ctx.setting} as preferred")

    return why[:3]


def meta_match(meta: Dict[str, Any], key: str, value: str) -> bool:
    m = str(meta.get(key, "")).strip().lower()
    return m in {value.lower(), "both", "all"}


def _augment_query(query: str, interests: Sequence[str], ctx: UiContext) -> str:
    parts: List[str] = []
    q = query.strip()
    if q:
        parts.append(rewrite_query(q))
    if interests:
        parts.extend([f"interests:{i}" for i in interests])
    if ctx.setting:
        parts.append(f"preference:{ctx.setting.lower()}")
    if ctx.season:
        parts.append(f"season:{ctx.season.lower()}")

    return " ".join(parts).strip()


def _init_state() -> None:
    defaults = {
        "saved_ids": set(),
        "hidden_ids": set(),
        "view_mode": "discover",
        "age_group": "",
        "athleticism": "",
        "travel_style": "",
        "visit_type": "",
        "interests": [],
        "season": "",
        "setting": "",
        "query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _has_profile(profile: UserProfile) -> bool:
    return any(
        [
            bool(profile.age_group),
            bool(profile.athleticism),
            bool(profile.travel_style),
            bool(profile.visit_type),
            bool(profile.interests),
        ]
    )


@st.cache_resource(show_spinner=False)
def load_engine() -> tuple[TfidfRanker, List[Dict[str, Any]]]:
    cfg = Config()
    inv = InvertedIndex.load(cfg.INDEX_PATH)
    docs = load_docstore_jsonl(cfg.DOCSTORE_PATH)
    return TfidfRanker(inv, docs), docs


def render_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #f5f0e8;
  --panel: #faf7f2;
  --text: #2d2420;
  --muted: #8a7a6c;
  --line: #e0d6ca;
  --line-light: #ede6dc;
  --accent: #c62828;
  --accent-soft: #fce4e4;
  --white: #ffffff;
  --font-display: 'Cormorant Garamond', Georgia, serif;
  --font-body: 'DM Sans', -apple-system, sans-serif;
}

/* ── Global ────────────────────────────────────── */
.stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: var(--font-body) !important;
}

/* Subtle cross-hatch texture overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.025;
  background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

/* ── Typography ────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
  font-family: var(--font-display) !important;
  color: var(--text) !important;
  letter-spacing: -0.01em;
}

p, span, label, div, li,
.stMarkdown p,
.stTextInput label,
.stRadio label,
.stSelectbox label,
.stCheckbox label {
  font-family: var(--font-body) !important;
}

/* ── Header / Title ────────────────────────────── */
.stApp h1:first-of-type {
  font-size: 2.4rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  margin-bottom: 0 !important;
}

/* Title accent color for 旅 kanji */
.header-kanji {
  color: var(--accent);
  font-family: var(--font-display);
  font-size: 2.8rem;
  font-weight: 700;
  line-height: 1;
  margin-right: 0.25rem;
}

.header-bar {
  padding: 1rem 0 0.75rem 0;
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.2rem;
}

.header-bar h1 {
  font-family: var(--font-display) !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  margin: 0 !important;
  padding: 0 !important;
  line-height: 1.15;
}

.header-sub {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  font-weight: 500;
  margin-top: 0.15rem;
}

/* ── Sidebar Sections ──────────────────────────── */
section[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--line) !important;
}

.sidebar-section-title {
  font-family: var(--font-display);
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.5rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--line-light);
}

.sidebar-label {
  display: block;
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  font-weight: 600;
  margin-bottom: 0.5rem;
  margin-top: 0.9rem;
}

/* ── Radio Buttons ─────────────────────────────── */
.stRadio > div {
  gap: 0.35rem !important;
}

.stRadio [role="radiogroup"] {
  gap: 0.35rem !important;
  flex-wrap: wrap !important;
}

.stRadio [role="radiogroup"] label {
  background: var(--white) !important;
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  padding: 0.3rem 0.75rem !important;
  font-size: 0.78rem !important;
  color: #2d2420 !important;
  transition: all 0.15s ease;
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: unset !important;
  min-width: fit-content !important;
}

.stRadio [role="radiogroup"] label p,
.stRadio [role="radiogroup"] label span,
.stRadio [role="radiogroup"] label div {
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: unset !important;
  font-size: 0.78rem !important;
  color: #2d2420 !important;
}

.stRadio [role="radiogroup"] label[data-checked="true"],
.stRadio [role="radiogroup"] label:has(input:checked) {
  background: var(--text) !important;
  color: var(--bg) !important;
  border-color: var(--text) !important;
}

.stRadio [role="radiogroup"] label[data-checked="true"] p,
.stRadio [role="radiogroup"] label[data-checked="true"] span,
.stRadio [role="radiogroup"] label[data-checked="true"] div,
.stRadio [role="radiogroup"] label:has(input:checked) p,
.stRadio [role="radiogroup"] label:has(input:checked) span,
.stRadio [role="radiogroup"] label:has(input:checked) div {
  color: var(--bg) !important;
}

/* ── Selectbox ─────────────────────────────────── */
.stSelectbox [data-baseweb="select"] {
  border-radius: 10px !important;
}

.stSelectbox [data-baseweb="select"] > div {
  background: var(--white) !important;
  border-color: var(--line) !important;
  font-family: var(--font-body) !important;
  font-size: 0.85rem !important;
}

/* ── Search Input ──────────────────────────────── */
.stTextInput > div > div {
  border-radius: 12px !important;
  border: 2px solid var(--line) !important;
  background: var(--white) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
  padding: 0.1rem 0.5rem !important;
  transition: border-color 0.2s;
}

.stTextInput > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 2px 12px rgba(198,40,40,0.08) !important;
}

.stTextInput input {
  font-family: var(--font-body) !important;
  font-size: 1rem !important;
  color: var(--text) !important;
}

.stTextInput input::placeholder {
  color: #9e8e82 !important;
  font-style: italic;
}

/* ── Chips & Tags ──────────────────────────────── */
.chip {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  margin: 0 0.25rem 0.3rem 0;
  font-size: 0.7rem;
  font-weight: 500;
  color: #5d4e42;
  background: var(--white);
  font-family: var(--font-body);
}

.chip-loc {
  background: #fef3e2;
  color: #bf6c00;
  border-color: #fef3e2;
}

.chip-season {
  background: #e8f5e9;
  color: #2e7d32;
  border-color: #e8f5e9;
}

.chip-diff {
  background: var(--line-light);
  color: #6d5e52;
  border-color: var(--line-light);
}

.chip-type {
  background: #e3f2fd;
  color: #1565c0;
  border-color: #e3f2fd;
}

.chip-interest {
  background: var(--bg);
  color: var(--muted);
  border-color: var(--bg);
  font-size: 0.65rem;
  padding: 0.1rem 0.45rem;
}

.chip-interest-match {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: var(--accent-soft);
  font-size: 0.65rem;
  padding: 0.1rem 0.45rem;
}

/* ── Result Cards ──────────────────────────────── */
.card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem 0.7rem 1.1rem;
  background: var(--white);
  margin-bottom: 1rem;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
  animation: fadeSlideIn 0.4s ease both;
}

.card:hover {
  box-shadow: 0 6px 24px rgba(45,36,32,0.08);
  transform: translateY(-1px);
}

.card-dismissed {
  border: 1px dashed var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem 0.7rem 1.1rem;
  background: var(--white);
  margin-bottom: 1rem;
  opacity: 0.7;
}

.card h3, .card h4 {
  font-family: var(--font-display) !important;
  font-size: 1.2rem !important;
  font-weight: 700 !important;
  line-height: 1.25;
  margin: 0.1rem 0 0.45rem 0 !important;
  color: var(--text) !important;
}

.card-rank {
  font-family: var(--font-display);
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--accent);
}

.card-match-bar {
  display: inline-block;
  width: 60px;
  height: 4px;
  border-radius: 2px;
  background: var(--line-light);
  overflow: hidden;
  vertical-align: middle;
  margin: 0 0.4rem;
}

.card-match-fill {
  display: block;
  height: 100%;
  border-radius: 2px;
  transition: width 0.5s ease;
}

.match-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.3rem;
  gap: 0.5rem;
}

.match-pct {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--muted);
}

.tags-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.2rem;
  margin-bottom: 0.4rem;
}

.interests-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.2rem;
  margin-bottom: 0.4rem;
}

/* ── Why Box ───────────────────────────────────── */
.why {
  background: var(--panel);
  border: 1px solid var(--line-light);
  border-radius: 8px;
  padding: 0.5rem 0.65rem;
  margin-top: 0.5rem;
  font-size: 0.78rem;
  line-height: 1.55;
  color: #5d4e42;
}

.why-label {
  display: block;
  font-size: 0.62rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #a09488;
  margin-bottom: 0.15rem;
}

.why-reason {
  display: block;
  color: #5d4e42;
  font-size: 0.78rem;
  line-height: 1.55;
}

/* ── Buttons ───────────────────────────────────── */
.stButton > button {
  font-family: var(--font-body) !important;
  border-radius: 10px !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.35rem 1rem !important;
  transition: all 0.15s ease !important;
  border: 1px solid var(--line) !important;
  background: var(--white) !important;
  color: var(--muted) !important;
}

.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-soft) !important;
}

/* ── View Tabs ─────────────────────────────────── */
.view-tabs {
  display: flex;
  gap: 0.4rem;
  margin-bottom: 0.5rem;
}

/* ── Interest Categories ────────────────────────── */
.interest-cat-label {
  font-family: var(--font-display);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text);
  margin-top: 0.7rem;
  margin-bottom: 0.3rem;
}

/* ── Checkbox (chip-style) ─────────────────────── */
.stCheckbox {
  margin-bottom: 0.15rem !important;
}

.stCheckbox label {
  background: var(--white) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  padding: 0.15rem 0.5rem !important;
  cursor: pointer;
  transition: all 0.15s ease;
  display: inline-flex !important;
  align-items: center !important;
}

.stCheckbox label:has(input:checked) {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

.stCheckbox label span,
.stCheckbox label p {
  font-size: 0.78rem !important;
  color: #5d4e42 !important;
  font-family: var(--font-body) !important;
}

.stCheckbox label:has(input:checked) span,
.stCheckbox label:has(input:checked) p {
  color: #fff !important;
}

/* Hide the default checkbox square */
.stCheckbox [data-testid="stCheckbox"] > div:first-child {
  display: none !important;
}

/* ── Info / Empty state ────────────────────────── */
.empty-state {
  text-align: center;
  padding: 3rem 1.5rem;
}

.empty-icon {
  font-size: 3.2rem;
  margin-bottom: 0.8rem;
}

.empty-title {
  font-family: var(--font-display);
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.4rem;
}

.empty-text {
  font-size: 0.9rem;
  color: var(--muted);
  max-width: 28rem;
  margin: 0 auto;
  line-height: 1.6;
}

.result-count {
  font-size: 0.78rem;
  color: var(--muted);
  font-weight: 500;
  margin-top: 0.5rem;
  margin-bottom: 0.8rem;
}

/* ── View Header ───────────────────────────────── */
.view-header {
  margin-bottom: 1.2rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid var(--line);
}

.view-header h2 {
  font-family: var(--font-display) !important;
  font-size: 1.7rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  margin-bottom: 0.2rem !important;
}

.view-header p {
  font-size: 0.85rem;
  color: var(--muted);
  line-height: 1.5;
}

/* ── Scrollbar ─────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c8b8a8; border-radius: 3px; }

/* ── Animation ─────────────────────────────────── */
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Hide default Streamlit chrome ─────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
  background: var(--bg) !important;
}

/* ── Force radio text readability (high specificity) ── */
div[data-testid="stRadio"] label[data-baseweb="radio"] p,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] [role="radiogroup"] label p,
.stRadio label p,
[data-testid="stHorizontalBlock"] .stRadio label p {
  color: #2d2420 !important;
  opacity: 1 !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"][data-checked="true"] p,
div[data-testid="stRadio"] label:has(input:checked) p,
.stRadio label:has(input:checked) p {
  color: #f5f0e8 !important;
  opacity: 1 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Tabisaki", page_icon="旅", layout="wide")
    _init_state()
    render_css()

    st.markdown(
        "<div class='header-bar'>"
        "<h1><span class='header-kanji'>旅</span> Tabisaki</h1>"
        "<div class='header-sub'>Japan Activity Recommender</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        ranker, docs = load_engine()
    except FileNotFoundError:
        st.error("Processed index/docstore not found. Run: `python -m src.index.build_index` first.")
        return

    col_left, col_main = st.columns([1.2, 2.2], gap="large")

    with col_left:
        st.markdown("<div class='sidebar-section-title'>Your Profile</div>", unsafe_allow_html=True)
        st.markdown("<span class='sidebar-label'>Traveler Type</span>", unsafe_allow_html=True)
        st.session_state.age_group = st.radio("Traveler Type", ["", "young", "older"], format_func=lambda x: {"": "Any", "young": "🎒 Young Explorer", "older": "🎩 Seasoned Traveler"}[x], horizontal=True, label_visibility="collapsed")
        st.markdown("<span class='sidebar-label'>Activity Level</span>", unsafe_allow_html=True)
        st.session_state.athleticism = st.radio("Activity Level", ["", "low", "medium", "high"], format_func=lambda x: {"": "Any", "low": "🚶 Easy", "medium": "🚴 Moderate", "high": "🧗 Athletic"}[x], horizontal=True, label_visibility="collapsed")
        st.markdown("<span class='sidebar-label'>Traveling With</span>", unsafe_allow_html=True)
        st.session_state.travel_style = st.radio("Traveling With", ["", "solo", "friends", "family"], format_func=lambda x: {"": "Any", "solo": "🧘 Solo", "friends": "👯 Friends", "family": "👨‍👩‍👧 Family"}[x], horizontal=True, label_visibility="collapsed")
        st.markdown("<span class='sidebar-label'>Visit Type</span>", unsafe_allow_html=True)
        st.session_state.visit_type = st.radio("Visit Type", ["", "first", "frequent"], format_func=lambda x: {"": "Any", "first": "✨ First Visit", "frequent": "🔄 Return Trip"}[x], horizontal=True, label_visibility="collapsed")

        st.markdown("<div class='sidebar-section-title' style='margin-top:1rem;'>Interests</div>", unsafe_allow_html=True)
        selected: List[str] = st.session_state.interests
        for cat, items in INTEREST_CATEGORIES.items():
            st.markdown(f"<div class='interest-cat-label'>{cat}</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(items), 3))
            for idx, item in enumerate(items):
                col = cols[idx % min(len(items), 3)]
                checked = item in selected
                with col:
                    if st.checkbox(item, value=checked, key=f"interest_{cat}_{item}"):
                        if item not in selected:
                            selected.append(item)
                    else:
                        if item in selected:
                            selected.remove(item)
        st.session_state.interests = selected

        st.markdown("<div class='sidebar-section-title' style='margin-top:1rem;'>Context</div>", unsafe_allow_html=True)
        st.markdown("<span class='sidebar-label'>Season</span>", unsafe_allow_html=True)
        st.session_state.season = st.radio("Season", ["", "Spring", "Summer", "Fall", "Winter"], format_func=lambda x: {"": "Any", "Spring": "🌸 Spring", "Summer": "☀️ Summer", "Fall": "🍁 Fall", "Winter": "❄️ Winter"}.get(x, x), horizontal=True, label_visibility="collapsed")
        st.markdown("<span class='sidebar-label'>Setting</span>", unsafe_allow_html=True)
        st.session_state.setting = st.radio("Setting", ["", "Indoor", "Outdoor"], format_func=lambda x: {"": "Any", "Indoor": "🏛️ Indoor", "Outdoor": "🌿 Outdoor"}.get(x, x), horizontal=True, label_visibility="collapsed")

        st.markdown("<div class='sidebar-section-title' style='margin-top:1rem;'>View</div>", unsafe_allow_html=True)
        vm = st.radio("View", ["discover", "saved", "hidden"], format_func=lambda x: {"discover": "🔍 Discover", "saved": "♥ Saved", "hidden": "✕ Hidden"}[x], horizontal=True, label_visibility="collapsed")
        st.session_state.view_mode = vm

    with col_main:
        st.session_state.query = st.text_input(
            "Search activities",
            value=st.session_state.query,
            placeholder="Try: temple, hiking, food, anime...",
            label_visibility="collapsed",
        )

        profile = UserProfile(
            age_group=st.session_state.age_group or None,
            athleticism=st.session_state.athleticism or None,
            travel_style=st.session_state.travel_style or None,
            visit_type=st.session_state.visit_type or None,
            interests=st.session_state.interests or None,
        )
        ctx = UiContext(season=st.session_state.season or None, setting=st.session_state.setting or None)

        retrieval_query = _augment_query(st.session_state.query, st.session_state.interests, ctx)

        # Show welcome state if user hasn't set anything yet
        has_any_input = bool(retrieval_query) or _has_profile(profile)
        if not has_any_input and vm == "discover":
            st.markdown(
                "<div class='empty-state'>"
                "<div class='empty-icon'>⛩️</div>"
                "<div class='empty-title'>Discover Japan</div>"
                "<div class='empty-text'>Set your profile, choose a season, or search for activities to get personalized recommendations.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        if not retrieval_query:
            if profile.interests:
                retrieval_query = " ".join([f"interests:{i}" for i in profile.interests])
            else:
                retrieval_query = "interests:nature interests:culture interests:history"

        base = ranker.rank(retrieval_query, top_k=50)

        filtered: List[ScoredDoc] = []
        for r in base:
            if ctx.season and not meta_match(r.meta, "Season", ctx.season):
                continue
            if ctx.setting and not meta_match(r.meta, "Preference", ctx.setting):
                continue
            filtered.append(r)

        final = PersonalReranker().rerank(filtered, profile, top_k=30) if _has_profile(profile) else filtered[:30]

        hidden_ids = st.session_state.hidden_ids
        saved_ids = st.session_state.saved_ids
        discover_results = [r for r in final if r.doc_id not in hidden_ids]
        saved_results = [r for r in final if r.doc_id in saved_ids]

        doc_map: Dict[int, Dict[str, Any]] = {int(d["doc_id"]): d for d in docs}
        hidden_results: List[ScoredDoc] = []
        for did in hidden_ids:
            d = doc_map.get(int(did))
            if d:
                hidden_results.append(ScoredDoc(doc_id=int(did), score=0.0, text=str(d.get("text", "")), meta=dict(d.get("meta", {}))))

        results = discover_results if vm == "discover" else (saved_results if vm == "saved" else hidden_results)

        if vm == "discover":
            st.markdown(f"<div class='result-count'>{len(discover_results)} recommendation{'s' if len(discover_results) != 1 else ''}</div>", unsafe_allow_html=True)
        elif vm == "saved":
            st.markdown("<div class='view-header'><h2>♥ Saved Activities</h2>"
                        f"<p>{len(saved_results)} activit{'y' if len(saved_results) == 1 else 'ies'} saved for your trip</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='view-header'><h2>✕ Hidden Activities</h2>"
                        f"<p>{len(hidden_results)} activit{'y' if len(hidden_results) == 1 else 'ies'} hidden — click Restore to bring them back</p></div>", unsafe_allow_html=True)

        if not results:
            st.markdown(
                "<div class='empty-state'>"
                "<div class='empty-icon'>🔎</div>"
                "<div class='empty-title'>No results</div>"
                "<div class='empty-text'>Adjust your filters, search terms, or switch views to find activities.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        max_score = max([r.score for r in discover_results], default=1.0)

        for i, r in enumerate(results, start=1):
            meta = r.meta or {}
            doc_interests = _split_interests(meta.get("Interests"))
            emoji = get_emoji(doc_interests)
            loc = get_location(r.text)
            score_pct = int(round((r.score / max_score) * 100)) if vm != "hidden" and max_score > 0 else 0

            card_class = "card-dismissed" if vm == "hidden" else "card"
            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)

            if vm != "hidden":
                fill_color = "#c62828" if score_pct > 80 else "#e65100" if score_pct > 50 else "#5d4037"
                st.markdown(
                    f"<div class='match-row'>"
                    f"<span class='card-rank'>#{i}</span>"
                    f"<span>"
                    f"<span class='card-match-bar'><span class='card-match-fill' style='width:{score_pct}%;background:{fill_color};'></span></span>"
                    f"<span class='match-pct'>{score_pct}%</span>"
                    f"</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(f"### {emoji} {r.text}")

            tag_parts: List[str] = []
            if loc:
                tag_parts.append(f"<span class='chip chip-loc'>📍 {loc}</span>")
            if meta.get("Season") and str(meta.get("Season")) != "All":
                tag_parts.append(f"<span class='chip chip-season'>{meta.get('Season')}</span>")
            if meta.get("Athleticism"):
                ath_val = str(meta.get("Athleticism"))
                ath_label = {"Low": "Easy", "Medium": "Moderate", "High": "Challenging"}.get(ath_val, ath_val)
                tag_parts.append(f"<span class='chip chip-diff'>{ath_label}</span>")
            if meta.get("Preference"):
                tag_parts.append(f"<span class='chip chip-type'>{meta.get('Preference')}</span>")
            if tag_parts:
                st.markdown(f"<div class='tags-row'>{''.join(tag_parts)}</div>", unsafe_allow_html=True)

            if doc_interests:
                interest_parts = []
                for x in doc_interests:
                    cls = "chip-interest-match" if x in st.session_state.interests else "chip-interest"
                    interest_parts.append(f"<span class='chip {cls}'>{x}</span>")
                st.markdown(f"<div class='interests-row'>{''.join(interest_parts)}</div>", unsafe_allow_html=True)

            why = build_why(meta, profile, ctx, st.session_state.interests)
            if why and vm == "discover":
                st.markdown(
                    "<div class='why'><span class='why-label'>Why recommended</span>"
                    + "".join([f"<span class='why-reason'>• {x}</span>" for x in why])
                    + "</div>",
                    unsafe_allow_html=True,
                )

            c1, c2 = st.columns([1, 1])
            if vm == "discover":
                with c1:
                    if r.doc_id in saved_ids:
                        if st.button("♥ Saved", key=f"unsave_{r.doc_id}"):
                            saved_ids.remove(r.doc_id)
                            st.rerun()
                    else:
                        if st.button("♡ Save", key=f"save_{r.doc_id}"):
                            saved_ids.add(r.doc_id)
                            st.rerun()
                with c2:
                    if st.button("Not for me", key=f"hide_{r.doc_id}"):
                        hidden_ids.add(r.doc_id)
                        st.rerun()
            elif vm == "saved":
                if st.button("♥ Remove", key=f"saved_remove_{r.doc_id}"):
                    if r.doc_id in saved_ids:
                        saved_ids.remove(r.doc_id)
                    st.rerun()
            else:
                if st.button("↩ Restore", key=f"restore_{r.doc_id}"):
                    if r.doc_id in hidden_ids:
                        hidden_ids.remove(r.doc_id)
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()