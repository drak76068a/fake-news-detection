
import streamlit as st
import torch
import torch.nn as nn
import re
from transformers import BertTokenizer, BertModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = "./fake_news_bert_lstm"
MAX_LEN    = 128
DEVICE     = torch.device("cpu")

# ── Model definition (must match training) ───────────────────────────────────
class BertBiLSTMClassifier(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_DIR)
        self.lstm = nn.LSTM(768, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.classifier = nn.Linear(lstm_hidden * 2, 2)

    def forward(self, input_ids, attention_mask):
        bert_out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out,_ = self.lstm(bert_out.last_hidden_state)
        mask       = attention_mask.unsqueeze(-1).float()
        pooled     = (lstm_out * mask).sum(1) / mask.sum(1)
        return self.classifier(self.dropout(self.layer_norm(pooled)))

def preprocess(text):
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s\u0900-\u097F\u0B00-\u0B7F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    mdl = BertBiLSTMClassifier()
    ckpt = torch.load(f"{MODEL_DIR}/lstm_head.pt", map_location="cpu")
    mdl.lstm.load_state_dict(ckpt["lstm_state_dict"])
    mdl.classifier.load_state_dict(ckpt["classifier_state_dict"])
    mdl.layer_norm.load_state_dict(ckpt["layer_norm_state_dict"])
    mdl.eval()
    return mdl, tokenizer

def classify(text, mdl, tokenizer):
    enc = tokenizer(preprocess(text), max_length=MAX_LEN,
                    padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(enc["input_ids"], enc["attention_mask"])
        probs  = torch.softmax(logits, dim=1)[0].tolist()
        pred   = int(torch.argmax(logits))
    return pred, probs

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Multilingual Fake News Detector")
st.markdown("""
**Architecture:** mBERT + Bidirectional LSTM
**Languages:** English · Hindi (हिंदी) · Odia (ଓଡ଼ିଆ)
""")
st.divider()

news_input = st.text_area("Paste a news headline or article below:",
                           height=130,
                           placeholder="Enter text in English, Hindi, or Odia...")

if st.button("🔍 Analyze", use_container_width=True):
    if news_input.strip():
        with st.spinner("Analyzing with mBERT + BiLSTM..."):
            mdl, tok = load_model()
            pred, probs = classify(news_input, mdl, tok)
        st.divider()
        if pred == 1:
            st.success(f"✅ REAL NEWS  —  Confidence: {probs[1]*100:.1f}%")
        else:
            st.error(f"❌ FAKE NEWS  —  Confidence: {probs[0]*100:.1f}%")
        col1, col2 = st.columns(2)
        col1.metric("Fake Probability", f"{probs[0]*100:.1f}%")
        col2.metric("Real Probability", f"{probs[1]*100:.1f}%")
        st.progress(probs[1])
    else:
        st.warning("Please enter some text to analyze.")

st.divider()
st.caption("Model: bert-base-multilingual-cased + BiLSTM | NLP Fake News Detection Project")
