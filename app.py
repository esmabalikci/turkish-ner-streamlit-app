import streamlit as st
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="./ner_model",
    tokenizer="./ner_model",
    aggregation_strategy="simple"
)

st.title("🧠 Türkçe NER Sistemi")

text = st.text_area("Metin gir:", "Mehmet Ankara Hacettepe Üniversitesi")

label_map = {
    "KISI": "KİŞİ",
    "YER": "YER",
    "KURUM": "KURUM"
}

if st.button("Analiz Et"):
    results = ner(text)

    st.subheader("Sonuçlar")

    for r in results:
        label = label_map.get(r["entity_group"], r["entity_group"])
        st.write(f"{r['word']} → {label}")