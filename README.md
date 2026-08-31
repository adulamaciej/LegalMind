# ⚖️ LegalMind — Multi-Agent System for ECHR Case Analysis

![Tests](https://github.com/adulamaciej/LegalMind/actions/workflows/tests.yml/badge.svg)


🔗 **Live demo:** [legalmind-adulamaciej.streamlit.app](https://legalmind-adulamaciej.streamlit.app)

Multi-agent LLM system simulating a legal debate and delivering a verdict on European Court of Human Rights (ECHR) cases, built on the Claude API. Combines RAG-based precedent retrieval with an adversarial prosecutor/defender debate, culminating in a judge agent's ruling.

## Overview

Given ECHR case facts (dataset example or custom input), the system runs a 5-agent pipeline:

1. **Facts Agent** — extracts structured facts, with chunked summarization for long cases
2. **Precedent Agent (RAG)** — retrieves similar past cases from ChromaDB and analyzes relevance
3. **Prosecutor & Defender** — argue opposing sides across two rounds each
4. **Judge Agent** — delivers verdict: violation status, articles, confidence, reasoning


Facts → Precedent (RAG) → Debate (Prosecutor ↔ Defender) → Judge → Verdict


**Tech stack:** Claude API (Sonnet 5), ChromaDB, Streamlit, HuggingFace `datasets`, LexGLUE/ECtHR dataset.

## Dataset

[LexGLUE `ecthr_a`](https://huggingface.co/datasets/coastalcph/lex_glue) — 9,000 train / 1,000 test cases, multi-label (10 possible ECHR articles).

**Key EDA findings:**
- Case length varies widely (mean 24 paragraphs, range 1–558); ~9% exceed 50 paragraphs → chunked summarization used instead of hard truncation
- Significant class imbalance: Article 6 = 52% of train labels, Article 9 = only 41 cases → per-article precision/recall/F1 used instead of aggregate accuracy
- 24% of cases have 2+ violations → multi-label verdict handling required
- Long/truncated cases skew toward Article 2/3/5 (life, torture, liberty), 2.5–5x overrepresented
- Train/test split is consistent overall, but Article 6's share drops from 52%→30%, and Article 9 has only 5 test cases (too few for reliable metrics)

Full analysis in `notebooks/eda.ipynb`.

## Research question & evaluation


**Hypothesis:** Does adversarial debate improve verdict accuracy vs. judge ruling on facts + precedents alone? Tested via two variants: **A** (full pipeline) vs. **B** (no debate).

**Results (pilot, random test cases, budget-constrained):**
- Debate usually didn't change the verdict — A and B matched in most cases
- In a few cases, debate corrected over-predicted articles (e.g., dropped a spurious Article 6 flag)
- Partial match (≥1 correct article) was consistently high (66–100%); exact match varied more, reflecting a tendency to over-predict extra articles
- **Limitation:** sample size is too small for statistical confidence — a pilot finding, not a definitive answer

**Debugging highlights:**
- Judge occasionally hallucinated non-existent article codes (e.g. "13") → fixed with explicit allowed-codes list in prompt + code-level filtering
- Judge showed Article 6 bias (predicted by EDA's class imbalance finding) → initial prompt fix overcorrected into false negatives → rebalanced for accuracy on both sides

**Model Version**
Haiku 4.5 gave noticeably lower exact-match accuracy than Sonnet 5 across evaluation batches, suggesting model capability — not just prompting — drove verdict quality. This motivated routing  to Sonnet 5 despite higher API cost.


## RAG-stage evaluation

**Metrics** — in `evaluation/rag_metrics.py`, reported per-case and in aggregate by `evaluation/evaluation.py`, persisted to `evaluation/results.json` (reproducible):

| Metric | What it measures | Cost |
|---|---|---|
| **precision@k** | Of the *k* retrieved precedents, the fraction sharing ≥1 violated article with the query case (dataset labels, no LLM). No-violation query cases are excluded — precision is undefined there, not zero. | free |
| **similarity ↔ relevance correlation** | Point-biserial correlation between cosine similarity and label-overlap relevance — does vector similarity actually predict legal relevance? | free |
| **ungrounded article codes** | Article codes cited in the analysis that no retrieved precedent's labels support (regex + set arithmetic). | free |
| **faithfulness** | LLM-as-judge (`JUDGE_MODEL`, default Sonnet 5, kept separate from the pipeline model): splits the analysis into atomic claims and scores the fraction directly supported by the retrieved precedent text. Truncated judge responses are salvaged, not discarded. | 1 judge call / case |

**Retrieval-pipeline changes driven by these metrics:**
- **Chunked index** — one embedding per 3-paragraph chunk (was: first 10 paragraphs of each case as one vector). ECHR fact sections open with procedural boilerplate and the default MiniLM embedder truncates at ~256 word-pieces, so the discriminative facts never reached the vector. ~63k chunks vs 9k case-docs.
- **Query on extracted facts** — retrieval queries with the Facts Agent's summary + key events + alleged violations, not the raw opening paragraphs.
- **Case-level retrieval** — over-fetch chunk hits, collapse to distinct cases scored by best-matching chunk, return the top chunks as context.
- **Grounded precedent analysis** — `analyze_precedents` now receives 1500 chars/precedent (matching what the faithfulness judge sees) and is told to use only the shown excerpts, no outside ECHR knowledge.

**Run:**
```bash
python rag/indexer.py    # re-index — chunked schema; delete data/chroma first if it exists
python eval.py           # 20 cases, faithfulness judge on; writes evaluation/results.json
```
`eval.py` loads `.env` before importing the pipeline. Numbers remain small-sample — directional, not definitive.


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-api-key
python rag/indexer.py   # one-time: chunk + index all 9,000 training cases into ChromaDB
```

## Usage

```bash
streamlit run app.py              # Web UI
python main.py --example 5        # CLI: dataset example
python main.py --text "..."       # CLI: custom case
python eval.py                    # Run evaluation (verdict A/B + RAG-stage metrics)
pytest tests/ -v                  # Run unit tests
```


## Docker

A `Dockerfile` is included for containerized deployment.


## Known limitations / future work

- Small evaluation sample (API cost constraints) — larger-scale testing needed for confidence

- RAG-precedent hypothesis not rigorously tested at scale; retrieval is chunked semantic similarity over extracted case facts (see *RAG-stage evaluation*), still lacking legal-structure-aware ranking or cross-encoder re-ranking

- Rare-class metrics (e.g. Article 9) remain statistically unreliable given low frequency in the dataset