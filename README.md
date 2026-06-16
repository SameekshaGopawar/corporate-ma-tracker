# Corporate M&A Tracker

An NLP-powered acquisition tracking system that fetches real business news from NewsAPI, extracts buyer, target, and deal information using spaCy, classifies headlines using Hugging Face zero-shot classification, filters acquisition events, stores results in CSV format, and displays them through an interactive Streamlit dashboard.

---

## Features

- **Live news ingestion** — pulls headlines from NewsAPI across 5 M&A search terms (acquired, acquisition, merger, buyout, purchased)
- **NLP extraction** — uses spaCy dependency parsing and named entity recognition to extract Buyer, Target, and Deal Value from raw headline text
- **Zero-shot AI classification** — classifies every headline as Acquisition / Partnership / Investment / Other using `facebook/bart-large-mnli` via the Hugging Face Inference API — no labelled training data required
- **Confidence filtering** — only saves headlines classified as Acquisition with >= 70% confidence
- **Data quality cleaning** — automatically rejects NLP extractions that contain deal descriptors (e.g. "55bn acquisition") instead of real company names
- **Interactive dashboard** — dark-mode Streamlit UI with KPI cards, Plotly charts, deal cards with confidence badges, and live filters
- **Secure credential management** — all API keys stored in `.env`, excluded from version control

---

## Architecture

```
+-----------------+     +-------------------+     +-------------------------+
|                 |     |                   |     |                         |
|   NewsAPI       +---->+  news_fetcher.py  +---->+     ma_tracker.py       |
|  (live news)    |     |  Fetch headlines  |     |                         |
|                 |     |  5 search terms   |     |  spaCy NLP              |
+-----------------+     +-------------------+     |  - Dependency parsing   |
                                                  |  - Named Entity Recog.  |
                                                  |  - Extract Buyer/Target |
                                                  |  - Extract Deal Value   |
                                                  |                         |
                                                  |  Hugging Face API       |
                                                  |  - Zero-shot classify   |
                                                  |  - Confidence score     |
                                                  |  - Filter >= 70%        |
                                                  |                         |
                                                  |  pandas                 |
                                                  |  - Save to CSV          |
                                                  +----------+--------------+
                                                             |
                                                             v
                                                  +----------+--------------+
                                                  |                         |
                                                  |     ma_deals.csv        |
                                                  |   (structured output)   |
                                                  |                         |
                                                  +----------+--------------+
                                                             |
                                                             v
                                                  +----------+--------------+
                                                  |                         |
                                                  |     dashboard.py        |
                                                  |   Streamlit dashboard   |
                                                  |                         |
                                                  |  - KPI cards            |
                                                  |  - Bar / Donut / Tree   |
                                                  |  - Deal cards           |
                                                  |  - Filters & Search     |
                                                  |                         |
                                                  +-------------------------+
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| spaCy (`en_core_web_lg`) | NLP — dependency parsing, NER |
| Hugging Face Inference API | Zero-shot classification (`facebook/bart-large-mnli`) |
| NewsAPI | Live business news headlines |
| pandas | Data storage and transformation |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive charts |
| python-dotenv | Secure API key management |
| requests | HTTP calls to external APIs |

---

## Project Structure

```
corporate-ma-tracker/
|
|-- ma_tracker.py          # Main pipeline: NLP extraction + HF classification
|-- news_fetcher.py        # NewsAPI integration — fetches live headlines
|-- dashboard.py           # Streamlit dashboard
|-- ner.py                 # Standalone NER demo script
|-- compare_extraction.py  # Before/after extraction comparison tool
|-- parse_debug.py         # spaCy dependency parse debugger
|-- debug_api.py           # Hugging Face API debug script
|
|-- headlines.txt          # Static test headlines (used for development)
|-- ma_deals.csv           # Pipeline output — acquisition deals (auto-generated)
|
|-- .streamlit/
|   |-- config.toml        # Streamlit server configuration
|
|-- .env                   # API keys — NOT committed to git
|-- .env.example           # Template showing required environment variables
|-- .gitignore             # Excludes .env, cache, venv
|-- requirements.txt       # Python dependencies
|-- README.md              # This file
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/corporate-ma-tracker.git
cd corporate-ma-tracker
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_lg
```

---

## Environment Variables

Create a `.env` file in the project root with the following keys:

```
HF_API_TOKEN=your_hugging_face_token_here
NEWS_API_KEY=your_newsapi_key_here
```

### How to get each key

| Key | Where to get it |
|-----|----------------|
| `HF_API_TOKEN` | https://huggingface.co → Settings → Access Tokens → New Token (Read) |
| `NEWS_API_KEY` | https://newsapi.org → Register free account → Dashboard |

**Never commit your `.env` file. It is already listed in `.gitignore`.**

---

## How to Run

### Step 1 — Run the pipeline

This fetches live headlines, runs NLP extraction and AI classification, and writes results to `ma_deals.csv`.

```bash
python ma_tracker.py
```

Expected output:
```
Fetched 85 unique headlines from NewsAPI.
Headline   : Teva Closes Acquisition of Emalex Biosciences
  Buyer      : Teva
  Target     : Emalex Biosciences
  Deal Value : Unknown
  Category   : Acquisition
  Confidence :
    Acquisition     90.0%  ##################
    ...
Rows saved : 9
Saved to   : ma_deals.csv
```

### Step 2 — Launch the dashboard

```bash
streamlit run dashboard.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## Dashboard Overview

| Section | Description |
|---------|-------------|
| KPI Cards | Total deals, largest deal value, unique buyers, avg confidence |
| Bar Chart | Acquisition volume by buyer |
| Donut Chart | Confidence score distribution (70-79%, 80-89%, 90-100%) |
| Treemap | Top buyers by deal count — larger box = more deals |
| Deal Cards | Each acquisition shown as a card with logo, buyer, target, value, confidence badge |
| Filters | Filter by buyer, target, or free-text search |
| Fetch New Data | Runs the full pipeline and refreshes the dashboard |
| Reload CSV | Re-reads the existing CSV without re-running the pipeline |

---

## Example Screenshots

> Add screenshots here after running the dashboard.
> Suggested: KPI row, charts section, deal cards section.

---

## Future Improvements

- [ ] PostgreSQL database instead of CSV for persistent storage
- [ ] Scheduled pipeline runs every hour using cron or Celery
- [ ] Email or Slack alerts for high-value deals above a threshold
- [ ] Sentiment analysis on each headline (positive / negative deal framing)
- [ ] Historical trend charts showing M&A activity over time
- [ ] Multi-language support for non-English headlines
- [ ] Deploy dashboard to cloud (Streamlit Cloud, AWS, Azure)
- [ ] Add deal source URL linking back to original article

---

## Resume Bullet Points

- Built an end-to-end NLP pipeline in Python that ingests live business news via NewsAPI, extracts M&A deal entities (buyer, target, deal value) using spaCy dependency parsing and NER, and classifies headlines using Hugging Face zero-shot classification (`facebook/bart-large-mnli`)
- Implemented a confidence-based filtering system (>= 70% threshold) to reduce noise and improve data quality across 85+ live headlines per run
- Developed an interactive dark-mode Streamlit dashboard with real-time filters, Plotly visualisations (bar, donut, treemap), and KPI metrics for business intelligence presentation
- Applied secure credential management using environment variables and `.gitignore` best practices to protect API keys

---

## Skills Demonstrated

`Python` `NLP` `spaCy` `Named Entity Recognition` `Dependency Parsing` `Hugging Face` `Zero-Shot Classification` `Large Language Models` `REST APIs` `NewsAPI` `pandas` `Streamlit` `Plotly` `Data Pipelines` `Business Intelligence` `Data Visualisation` `Environment Variable Management` `Git`
