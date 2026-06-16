"""
Runs the OLD and NEW extract_deal side by side on real headlines
so you can see exactly what improved.
"""
import spacy
nlp = spacy.load("en_core_web_lg")

MA_VERBS = {"acquire", "buy", "merge", "partner"}
ACQUISITION_NOUNS = {"acquisition", "merger", "buyout", "purchase", "takeover"}

# ─────────────────────────────────────────────
# OLD extraction logic (unchanged from before)
# ─────────────────────────────────────────────

def extract_deal_old(headline):
    doc = nlp(headline)
    buyer = target = deal_value = "Unknown"

    for ent in doc.ents:
        if ent.label_ == "MONEY":
            deal_value = ent.text
            break

    ma_verb_token = None
    for token in doc:
        if token.lemma_.lower() in MA_VERBS:
            ma_verb_token = token
            break

    if ma_verb_token is None:
        return buyer, target, deal_value

    for token in ma_verb_token.children:
        if token.dep_ == "nsubj":
            buyer = token.text
            break
        if token.dep_ == "compound":
            buyer = token.text
            break

    for token in ma_verb_token.children:
        if token.dep_ == "dobj":
            target = token.text
            break

    if target == "Unknown":
        for token in ma_verb_token.children:
            if token.dep_ == "prep" and token.text.lower() == "with":
                for child in token.children:
                    if child.dep_ == "pobj":
                        target = child.text
                        break

    return buyer, target, deal_value


# ─────────────────────────────────────────────
# NEW extraction logic
# ─────────────────────────────────────────────

def get_noun_phrase(token):
    """
    Build a multi-word name from a token by collecting
    compound and adjective modifiers that sit to its left.
    e.g. token = "America" with left compound "Nagase"
         -> returns "Nagase America"
    """
    parts = [t.text for t in token.lefts
             if t.dep_ in ("compound", "amod")] + [token.text]
    return " ".join(parts)


def extract_deal_new(headline):
    doc = nlp(headline)
    buyer = target = deal_value = "Unknown"

    # Step 1: Deal value — same as before
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            deal_value = ent.text
            break

    # Step 2: Find an M&A verb
    ma_verb_token = None
    for token in doc:
        if token.lemma_.lower() in MA_VERBS:
            ma_verb_token = token
            break

    # ── Pattern A: standard verb ──────────────────────────────
    # Covers: X acquired Y  /  X buys Y  /  X acquires Y
    if ma_verb_token is not None:

        # Buyer: subject of the verb (full noun phrase)
        for token in ma_verb_token.children:
            if token.dep_ == "nsubj":
                buyer = get_noun_phrase(token)
                break
            if token.dep_ == "compound":
                buyer = get_noun_phrase(token)
                break

        # Target: direct object of the verb (full noun phrase)
        for token in ma_verb_token.children:
            if token.dep_ == "dobj":
                if token.lemma_.lower() in ACQUISITION_NOUNS:
                    # e.g. "X closes acquisition of Y"
                    # the dobj is "acquisition" itself — look inside for "of Y"
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
    #   TDK (ROOT) ← acquire (relcl)
    # so the buyer is the HEAD of the verb, not its subject.
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
    # Here there is no MA_VERB match (announce/complete are not in MA_VERBS).
    # We find the word "acquisition" as a dobj, then look for "of Y".
    if buyer == "Unknown" or target == "Unknown":
        for token in doc:
            if token.lemma_.lower() in ACQUISITION_NOUNS:
                # Target: object of "of" attached to the noun
                if target == "Unknown":
                    for child in token.children:
                        if child.dep_ == "prep" and child.text.lower() == "of":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    target = get_noun_phrase(grandchild)
                                    break
                # Buyer: subject of whatever verb governs the noun
                if buyer == "Unknown":
                    governing_verb = token.head
                    for child in governing_verb.children:
                        if child.dep_ == "nsubj":
                            buyer = get_noun_phrase(child)
                            break

    return buyer, target, deal_value


# ─────────────────────────────────────────────
# Side-by-side comparison
# ─────────────────────────────────────────────

test_headlines = [
    "Nagase America Acquires Alexander Technical Institute",
    "Teva Closes Acquisition of Emalex Biosciences",
    "TDK to acquire US AI data center cooling firm Fabric8Labs for up to $400M",
    "Trucordia Acquires Agency",
    "X buys Y for $1 billion",
    "X announces acquisition of Y",
    "X completes acquisition of Y",
    "Google acquired Wiz for $32 billion",
    "Microsoft acquired Activision for $68.7 billion",
    "Salesforce acquired Slack for $27.7 billion",
]

print("=" * 80)
print("  EXTRACTION COMPARISON: OLD vs NEW")
print("=" * 80)

old_unknown = 0
new_unknown = 0

for h in test_headlines:
    ob, ot, ov = extract_deal_old(h)
    nb, nt, nv = extract_deal_new(h)

    old_unknown += (ob == "Unknown") + (ot == "Unknown")
    new_unknown += (nb == "Unknown") + (nt == "Unknown")

    print(f"\nHeadline : {h}")
    print(f"  OLD  ->  Buyer: {ob:<30} Target: {ot}")
    print(f"  NEW  ->  Buyer: {nb:<30} Target: {nt}")

print(f"\n{'='*80}")
print(f"  Total Unknown fields — OLD: {old_unknown}   NEW: {new_unknown}")
print(f"  Improvement: {old_unknown - new_unknown} fewer Unknown values")
print("=" * 80)
