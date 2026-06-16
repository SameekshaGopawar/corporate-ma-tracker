import spacy
nlp = spacy.load("en_core_web_lg")

headlines = [
    "TDK to acquire US AI data center cooling firm Fabric8Labs for up to $400M",
    "Teva Closes Acquisition of Emalex Biosciences, Strengthening Late-Stage Neuroscience Pipeline",
    "Nagase America Acquires Alexander Technical Institute",
    "Trucordia Acquires Agency",
    "United CEO Refuses To Say American Airlines Merger Is Dead",
    "X buys Y for $1 billion",
    "X announces acquisition of Y",
    "X completes acquisition of Y",
]

for h in headlines:
    print(f"\n{'='*60}")
    print(f"HEADLINE: {h}")
    doc = nlp(h)
    print(f"{'TOKEN':<20} {'LEMMA':<15} {'DEP':<12} {'HEAD'}")
    for t in doc:
        print(f"  {t.text:<18} {t.lemma_:<15} {t.dep_:<12} {t.head.text}")
    ents = [(e.text, e.label_) for e in doc.ents]
    if ents:
        print(f"  ENTITIES: {ents}")
