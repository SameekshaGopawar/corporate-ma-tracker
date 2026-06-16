import spacy

nlp = spacy.load("en_core_web_lg")

with open("headlines.txt", "r") as f:
    headlines = f.readlines()

for headline in headlines:
    headline = headline.strip()
    if not headline:
        continue

    print(f"Headline: {headline}")

    doc = nlp(headline)

    found = False
    for ent in doc.ents:
        if ent.label_ in ("ORG", "MONEY"):
            print(f"  {ent.label_:<8} {ent.text}")
            found = True

    if not found:
        print("  (no ORG or MONEY entities found)")

    print()
