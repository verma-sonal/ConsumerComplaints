# ConsumerComplaints

DSBA Master Thesis — **Generative AI for Consumer Financial Complaint Understanding**

> **Status:** Active research code and notes for Sonal Verma’s thesis. This README consolidates scope, approach, Colab entry points, and how to reproduce results.

---

## Overview
This project evaluates how modern Large Language Models (LLMs) and representation learning can:
- **Classify** CFPB consumer complaints from free‑text into fine‑grained categories.
- **Generate Themes and Tags** that reveal hidden topics not captured by official labels.
- **Surface trends** across time, products, and companies to aid operations and compliance.

Why it matters: Complaint volume and heterogeneity overwhelm manual triage. Transparent automation reduces load and highlights emerging issues—e.g., a **human trafficking** signal discovered via LLM‑generated tags.

---

## Terminology
- **Theme ≙ Label.** We standardize on **Theme** throughout this repo and thesis.
- **Tag:** short, granular descriptors at the complaint level (e.g., *identity theft*, *loan modification*, *human trafficking*).

---

## Project Phases

### Phase 1 — Cluster‑level Theme generation (EDA‑oriented)
- **Flow:** SBERT embeddings → **HDBSCAN**/KMeans → LLM names each cluster.
- **Model tried:** LLaMA.
- **Takeaways:**
  - Compute heavy; moved experiments to **Colab Pro**.
  - Cluster names varied across runs/seeds and were often underspecified.
  - **Decision:** keep for **EDA only**; not used for production labeling.
- **Notes:**
  - Removed an earlier **FAISS** branch (no retrieval needed here).
  - HDBSCAN yields a **noise** cluster; we report its size and examples.
  - “Clear cluster” = high density + stable membership across seeds with coherent exemplar narratives.

### Phase 2 — **Per‑complaint** Theme & Tag generation (primary)
- **Flow:** Each complaint narrative → LLM → **Theme** (and **Tags**) with structured output.
- **Models:** **Gemini** and **LLaMA** (separate runs).
- **Engineering:**
  - Provider **rate limits** handled via retries + exponential backoff.
  - Checkpointing with `SAVE_EVERY` to avoid progress loss.
  - Typical throughput: **>12 hours per year** of data under rate limits.
  - Notebook constants: `MAX_RETRIES = 3`, `SAVE_EVERY = 10`.
- **Outcome:** Clearer, more actionable Themes; Tags added **granularity** for dashboards and trend analysis.

**Key finding:** LLM Tags surfaced a **human trafficking** topic absent in baseline labels. All figures and tables use the wording **“human trafficking.”**

---

## Dataset
- **Source:** CFPB Consumer Complaint Database (2011–present).
- **Common fields:** `Consumer complaint narrative`, `Product/Sub-product`, `Issue/Sub-issue`, `Company`, `Date received`, `Company response`, `Consumer disputed`.
- **Prep pipeline:**
  - Text cleaning (lowercasing, punctuation normalization, basic PII scrubbing).
  - Deduping exact/near-duplicates; track counts in data profile.
  - Train/test split where supervised comparisons are run.
  - Always record the **dataset snapshot date** per run (source drift).

> Use the CFPB data under its terms. Do **not** commit raw data or PII to the repo.

---

## Methods (at a glance)
- **Embeddings:** sentence‑transformers (SBERT) and provider embeddings.
- **Clustering/Topics:** HDBSCAN, KMeans, BERTopic (exploration), LLM cluster naming (Phase 1).
- **Per‑complaint generation:** prompt libraries for **Gemini**/**LLaMA** with JSON‑like outputs.
- **Classification baselines:** GPT‑4, Gemini, LLaMA family, DeepSeek (where applicable).
- **Evaluation:** Accuracy, Precision, Recall, F1 (micro/macro), error slices by Theme and Tag.
- **Explainability:** SHAP/LIME for classical models; prompt rationales for LLMs.

---

## Repository Structure
```
.
├── notebooks/                        # EDA and Colab entry points
├── src/
│   ├── data/                         # loaders, cleaning, splits
│   ├── features/                     # embeddings, vector utils
│   ├── clustering/                   # HDBSCAN/KMeans helpers
│   ├── labeling/                     # per-complaint Theme/Tag generation
│   ├── modeling/                     # baselines, prompting, evaluation
│   ├── explain/                      # SHAP/LIME support
│   └── viz/                          # plots and dashboards
├── configs/                          # YAML/JSON configs
├── reports/                          # figures, tables, logs, data_profile.md
├── models/                           # saved artifacts (gitignored if large)
├── data/                             # raw/processed (gitignored)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Environment
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Secrets & Config
Store keys in `.env` or export:
```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...         # Gemini
COHERE_API_KEY=...         # optional
```

### 3) Minimal Workflows

**A) Embeddings → (optional) Clustering for EDA**
```bash
python -m src.data.make_splits   --input data/raw/complaints.csv   --output data/processed/ --test_size 0.2

python -m src.features.build_embeddings   --input data/processed/train.csv   --model sbert   --out data/features/train_embeddings.npy

python -m src.clustering.run   --embeddings data/features/train_embeddings.npy   --algo hdbscan   --out reports/clusters.json
```

**B) Per‑complaint Theme/Tag generation (primary)**
```bash
python -m src.labeling.generate   --provider gemini   --input data/processed/all.csv   --out reports/gemini_themes.csv   --max_retries 3 --save_every 10

python -m src.labeling.generate   --provider llama   --input data/processed/all.csv   --out reports/llama_tags.csv   --max_retries 3 --save_every 10
```

**C) Evaluation (if ground truth available)**
```bash
python -m src.modeling.evaluate   --pred reports/gemini_themes.csv   --gold data/processed/test.csv   --average macro
```

> We log run timestamps, model/provider versions, seeds, and rate‑limit events for traceability.

---

## Notebooks / Colab Entry Points
> Each notebook contains constants like `MAX_RETRIES=3` and `SAVE_EVERY=10` and saves partial results in batches.

- **UNCC: Trafficking analysis (all trafficking info)**  
  Colab: <https://colab.research.google.com/drive/1GJl8oJMkjKOeazBoSr-oMYaqe9s_JyM8>  
  Example path: `INPUT_CSV = "/content/drive/MyDrive/data/data/rv_approach2_CC_trafficking.csv"`

- **Gemini — Tag Generation**  
  Colab: <https://colab.research.google.com/drive/1LsO43J4Y11eMDICzAd9kN4PZvznQvP1n>

- **Gemini — Theme Generation**  
  Colab: <https://colab.research.google.com/drive/13iW51qChTxl-GvVpd8QMppucV8ay1DY0>  
  Example path: `FINAL_OUTPUT_CSV = "/content/drive/MyDrive/data/data/rv_approach2_Gemini_Labels_theme.csv"`

- **LLaMA — Tag Generation (POC)**  
  Colab: <https://colab.research.google.com/drive/1Sw8eNRly8HyYKf9QGEYik30_-ywEnEQN>

- **LLaMA — Theme Generation (Phase 1)**  
  `RV-Approach-2_Consumer_Complaints-Theme.ipynb`  
  Colab: <https://colab.research.google.com/drive/106vOI03t0gPdZvYD2L6ekjxyhmLbMkre>  
  Example path: `final_df = pd.read_csv('/content/drive/MyDrive/data/data/rv_approach2_LLaMA_Labels.csv')`

- **Approach 2 — Clustered Complaints (EDA)**  
  Colab: <https://colab.research.google.com/drive/1o6EcvVZCNpYpyg7nksSpyRmZaDCPS-UO>  
  Example path: `final_df = pd.read_csv('/content/drive/MyDrive/data/data/approach2-clustered_complaints.csv')`

---

## Reproducibility & Logging
- **Compute:** Colab Pro for intensive runs; local free resources were insufficient.
- **Rate limits:** Retries + backoff; checkpointing via `--save_every`.
- **Drift:** Provider behavior changes; always record **date** and **model version**.
- **Data profile:** Forthcoming `reports/data_profile.md` with NaN counts, duplicate policy, and field usage.
- **Artifacts:** Example report artifact name: `Final-2024-Themes-Tag_approach2_LLaMA.csv` (git LFS or external storage recommended).

---

## Known Limitations
- Cluster‑level naming can be unstable; **per‑complaint** labeling is primary.
- Long narratives may need chunking; maintain privacy guardrails (no PII in artifacts).
- Provider outputs may drift over time; comparisons should be date‑bounded.

---

## Roadmap
- [ ] Publish `reports/data_profile.md` with NaN/duplicate counts and rationale.
- [ ] Document HDBSCAN noise handling; add captioned figures.
- [ ] Standardize **“human trafficking”** terminology across outputs.
- [ ] Release prompt library + guardrails for Theme/Tag generation.
- [ ] Add trend and anomaly dashboards based on Tags.
- [ ] Link the final thesis PDF and defense deck in this repo.

---

## Ethical Use & Transparency
- Do not commit PII. Use CFPB data under its license/terms.
- Document prompts, parameters, and post‑processing choices.
- Provide error analysis for minority classes and edge cases.

---

## How to Cite
```
@thesis{verma2025consumercomplaints,
  author    = {Sonal Verma},
  title     = {Generative AI for Consumer Financial Complaint Understanding},
  school    = {UNC Charlotte (DSBA)},
  year      = {2025}
}
```

## Acknowledgments
Advisor: **Dr. Wlodek Zadrozny**.  
Additional thanks: **Rick Hudson**, **Bill Cronin**, peers and faculty for feedback.

## License
Choose a license before sharing (e.g., MIT, Apache‑2.0). Until then, **All Rights Reserved**.

## Contact
Questions or collaboration: **Sonal Verma** — open an issue or reach out via email/LinkedIn.
