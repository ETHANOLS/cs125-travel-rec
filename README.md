# cs125-travel-rec

Travel activity recommendation system with ingestion, indexing, retrieval, and personalization.

## Run Pipeline

```
python -m src.index.build_index
```

This builds:
- `data/processed/docstore.jsonl`
- `data/processed/index/inverted_index.json`

## Run CLI Search

```
python -m src.rank.search_model --query "nature hiking" --top_k 10
python -m src.rank.search_model --query "outdoor" --top_k 10 --personal --age older --athleticism low --travel family --frequency first --interests nature,culture
```

## Run UI

```
streamlit run src/ui/app.py
```

UI supports:
- profile controls (age, activity level, travel style, visit type, interests)
- context controls (season, indoor/outdoor)
- discover / saved / hidden views
- ranked recommendations with brief "why" explanations
