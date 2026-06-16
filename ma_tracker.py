import os
import time
import spacy
import requests
import pandas as pd
from dotenv import load_dotenv
from news_fetcher import fetch_headlines

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "facebook/bart-large-mnli"
)

CANDIDATE_LABELS = ["Acquisition", "Partnership", "Investment", "Other"]

# ─────────────────────────────────────────────
# spaCy setup
# ─────────────────────────────────────────────

nlp = spacy.load("en_core_web_lg")

MA_VERBS           = {"acquire", "buy", "merge", "partner"}
ACQUISITION_NOUNS  = {"acquisition", "merger", "buyout", "purchase", "takeover"}
CONFIDENCE_THRESHOLD = 70.0


# ─────────────────────────────────────────────
# HELPER — multi-word noun phrases
# ─────────────────────────────────────────────

def get_noun_phrase(token):
    """
    Build a multi-word name from a token by collecting
    compound and adjective modifiers that sit to its left.
    e.g. token="America" with left compound "Nagase" -> "Nagase America"
         token="Institute" with lefts "Alexander","Technical" -> "Alexander Technical Institute"
    """
    parts = [t.text for t in token.lefts
             if t.dep_ in ("compound", "amod")] + [token.text]
    return " ".join(parts)


# ─────────────────────────────────────────────
# FUNCTION 1 — spaCy entity + dependency extraction
# ─────────────────────────────────────────────

def extract_deal(headline):
    """Return buyer, target, deal_value for one headline."""
    doc = nlp(headline)

    buyer      = "Unknown"
    target     = "Unknown"
    deal_value = "Unknown"

    # Deal value: first MONEY entity
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            deal_value = ent.text
            break

    # Find the first M&A verb
    ma_verb_token = None
    for token in doc:
        if token.lemma_.lower() in MA_VERBS:
            ma_verb_token = token
            break

    # ── Pattern A: standard verb ──────────────────────────────
    # Covers: X acquired Y  /  X buys Y  /  X acquires Y
    if ma_verb_token is not None:

        for token in ma_verb_token.children:
            if token.dep_ == "nsubj":
                buyer = get_noun_phrase(token)
                break
            if token.dep_ == "compound":
                buyer = get_noun_phrase(token)
                break

        for token in ma_verb_token.children:
            if token.dep_ == "dobj":
                if token.lemma_.lower() in ACQUISITION_NOUNS:
                    # dobj is the word "acquisition" itself ->
                    # look one level deeper for "of Y"
                    for child in token.children:
                        if child.dep_ == "prep" and child.text.lower() == "of":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    target = get_noun_phrase(grandchild)
                                    break
                else:
                    target = get_noun_phrase(token)
                break

        # "partners with X" fallback
        if target == "Unknown":
            for token in ma_verb_token.children:
                if token.dep_ == "prep" and token.text.lower() == "with":
                    for child in token.children:
                        if child.dep_ == "pobj":
                            target = get_noun_phrase(child)
                            break

    # ── Pattern B: "X to acquire Y" ──────────────────────────
    # spaCy parses "TDK to acquire Fabric8Labs" as:
    #   TDK (ROOT) <- acquire (relcl)
    # Buyer = head of the verb;  Target = dobj of the verb
    if buyer == "Unknown" or target == "Unknown":
        for token in doc:
            if token.lemma_.lower() in MA_VERBS and token.dep_ == "relcl":
                if buyer == "Unknown":
                    buyer = get_noun_phrase(token.head)
                if target == "Unknown":
                    for child in token.children:
                        if child.dep_ == "dobj":
                            target = get_noun_phrase(child)
                            break

    # ── Pattern C: acquisition noun ──────────────────────────
    # Covers: X announces acquisition of Y
    #         X completes acquisition of Y
    # No MA_VERB present; find "acquisition" as dobj,
    # then read "of Y" for target and nsubj of governing verb for buyer.
    if buyer == "Unknown" or target == "Unknown":
        for token in doc:
            if token.lemma_.lower() in ACQUISITION_NOUNS:
                if target == "Unknown":
                    for child in token.children:
                        if child.dep_ == "prep" and child.text.lower() == "of":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    target = get_noun_phrase(grandchild)
                                    break
                if buyer == "Unknown":
                    governing_verb = token.head
                    for child in governing_verb.children:
                        if child.dep_ == "nsubj":
                            buyer = get_noun_phrase(child)
                            break

    return {"buyer": buyer, "target": target, "deal_value": deal_value}


# ─────────────────────────────────────────────
# FUNCTION 2 — Hugging Face zero-shot classifier
# ─────────────────────────────────────────────

def classify_headline(headline, retries=3, wait=10):
    """
    Send one headline to the HF Inference API.
    Returns the top label and a dict of all label→score pairs.
    Retries up to `retries` times if the model is loading.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    payload = {
        "inputs": headline,
        "parameters": {"candidate_labels": CANDIDATE_LABELS}
    }

    for attempt in range(retries):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        data = response.json()

        if isinstance(data, dict) and "error" in data:
            if "loading" in data["error"].lower():
                print(f"  Model loading, retrying in {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  API error: {data['error']}")
                return "Error", {}

        # API returns a list of {"label": ..., "score": ...} dicts,
        # already sorted highest score first
        scores_dict = {
            item["label"]: round(item["score"] * 100, 1)
            for item in data
        }

        top_label = data[0]["label"]
        return top_label, scores_dict

    return "Error", {}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if not HF_API_TOKEN:
    print("ERROR: HF_API_TOKEN environment variable is not set.")
    print("Set it with:  $env:HF_API_TOKEN = 'hf_your_token_here'")
    exit(1)

headlines = fetch_headlines()

if not headlines:
    print("No headlines fetched. Check your NEWS_API_KEY in .env.")
    exit(1)

rows = []

print("=" * 60)
print("  M&A TRACKER — spaCy + HF Zero-Shot Classification")
print("=" * 60)

for headline in headlines:
    headline = headline.strip()
    if not headline:
        continue

    # Skip NewsAPI filler titles that are not real headlines
    if headline.lower() in ("[removed]", "none", ""):
        continue

    deal   = extract_deal(headline)
    label, scores = classify_headline(headline)

    safe = headline.encode("cp1252", errors="replace").decode("cp1252")
    print(f"\nHeadline   : {safe}")
    print(f"  Buyer      : {deal['buyer']}")
    print(f"  Target     : {deal['target']}")
    print(f"  Deal Value : {deal['deal_value']}")
    print(f"  Category   : {label}")
    print(f"  Confidence :")
    for lbl, pct in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "#" * int(pct / 5)
        print(f"    {lbl:<14} {pct:>5.1f}%  {bar}")

    acq_score = scores.get("Acquisition", 0)

    if label == "Acquisition" and acq_score >= CONFIDENCE_THRESHOLD:
        rows.append({
            "Buyer":       deal["buyer"],
            "Target":      deal["target"],
            "Deal_Value":  deal["deal_value"],
            "Category":    label,
            "Acq_%":       acq_score,
            "Partner_%":   scores.get("Partnership", 0),
            "Invest_%":    scores.get("Investment",  0),
            "Other_%":     scores.get("Other",       0),
        })
    elif label != "Acquisition":
        print(f"  >> Skipped (category: {label})")
    else:
        print(f"  >> Skipped (Acquisition but confidence {acq_score:.1f}% < {CONFIDENCE_THRESHOLD}%)")

df = pd.DataFrame(rows)
output_path = "ma_deals.csv"
df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print(f"  Rows saved : {len(df)}")
print(f"  Threshold  : Acquisition >= {CONFIDENCE_THRESHOLD}%")
print(f"  Saved to   : {output_path}")
print("=" * 60)
