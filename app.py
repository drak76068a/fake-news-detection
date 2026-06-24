
import streamlit as st
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "./fake_news_mbert_model"   # adjust path as needed
MAX_LEN    = 128

@st.cache_resource
def load_model():
    tok   = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tok

def preprocess(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s\u0900-\u097F\u0B00-\u0B7F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict(text, model, tokenizer):
    clean = preprocess(text)
    enc   = tokenizer(clean, max_length=MAX_LEN, padding="max_length",
                      truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1)[0].tolist()
        pred   = int(torch.argmax(logits))
    return pred, probs

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Multilingual Fake News Detector")
st.markdown("Supports **English**, **Hindi (हिंदी)**, and **Odia (ଓଡ଼ିଆ)** news headlines.")

news_input = st.text_area("Paste a news headline or article:", height=120,
                           placeholder="Enter English, Hindi, or Odia text here...")

if st.button("🔍 Analyze"):
    if news_input.strip():
        with st.spinner("Analyzing..."):
            model, tokenizer = load_model()
            pred, probs = predict(news_input, model, tokenizer)

        if pred == 1:
            st.success(f"✅ **REAL NEWS** — Confidence: {probs[1]*100:.1f}%")
        else:
            st.error(f"❌ **FAKE NEWS** — Confidence: {probs[0]*100:.1f}%")

        col1, col2 = st.columns(2)
        col1.metric("Fake Probability",  f"{probs[0]*100:.1f}%")
        col2.metric("Real Probability",  f"{probs[1]*100:.1f}%")
    else:
        st.warning("Please enter some text first.")

st.markdown("---")
st.caption("Model: bert-base-multilingual-cased | Fine-tuned for fake news detection")
