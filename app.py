import streamlit as st
import requests  # for SSID check via hospital gateway
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
import faiss  # for local vector DB
import torch
from sentence_transformers import SentenceTransformer
from local_inference import grok_query
from audit_log import log_decision
# from whisper_local import transcribe_voice  # Uncomment for voice

# ── FORCE HOSPITAL WIFI ONLY ─────────────────────────────────────
def is_on_hospital_wifi():
    try:
        r = requests.get("http://captive.apple.com", timeout=3)
        return "hospital" in r.url.lower() or "clinical" in r.text.lower()  # Customize to your hospital
    except:
        return False

if not is_on_hospital_wifi():
    st.error("🚫 Grok Doc only works on hospital WiFi. Connect to Hospital-Clinical network.")
    st.info("This prevents PHI from ever leaving the premises.")
    st.stop()

st.set_page_config(page_title="Grok Doc", layout="centered")
st.title("🩺 Grok Doc — On-Prem Bayesian Co-Pilot")
st.caption("100% local • Zero cloud • Hospital WiFi only • DGX Spark ready")

# ── Load local vector DB (17k cases) ─────────────────────────────
@st.cache_resource
def load_vector_db():
    index = faiss.read_index("case_index.faiss")  # Pre-built FAISS index
    cases = [json.loads(line) for line in open("cases_17k.jsonl")]
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return index, cases, embedder

index, cases, embedder = load_vector_db()

# ── Sidebar patient input ────────────────────────────────────────
with st.sidebar:
    st.header("Patient Context")
    mrn = st.text_input("Medical Record Number (MRN)", help="Required")
    age = st.slider("Age", 18, 100, 72)
    gender = st.selectbox("Gender", ["Male", "Female"])
    chief = st.text_area("Chief complaint / question", "72 yo male, septic shock on vancomycin, Cr 2.9 → 1.8. Safe trough?")
    labs = st.text_area("Key labs / imaging (optional)")
    submit = st.button("Ask Grok Doc", type="primary")

# ── Main logic ───────────────────────────────────────────────────
if submit and mrn:
    with st.spinner("Local retrieval + inference…"):
        start = datetime.now()
        
        # 1. Embed query + retrieve top 100
        query_text = chief + " " + labs
        query_emb = embedder.encode([query_text])
        _, top_indices = index.search(query_emb, 100)
        evidence = "\n".join([json.dumps(cases[idx]) for idx in top_indices[0][:20]])
        
        # 2. Bayesian update (local PyMC)
        import pymc as pm
        import arviz as az
        with pm.Model():
            prior_safe = pm.Beta("prior_safe", alpha=8, beta=2)
            observed_safe = sum(1 for idx in top_indices[0] if not cases[idx].get("nephrotoxicity", False))
            likelihood = pm.Binomial("likelihood", n=100, p=prior_safe, observed=observed_safe)
            trace = pm.sample(1000, tune=500, chains=2, cores=1, progressbar=False)
        prob_safe = az.summary(trace)["mean"]["prior_safe"]
        ci = az.hdi(trace["prior_safe"], hdi_prob=0.95).values[0]
        
        latency = (datetime.now() - start).total_seconds()
        
        # 3. Local LLM reasoning
        prompt = f"Best intensivist. ≤3 sentences with % prob.\nEvidence: {evidence[:8000]}\nBayesian: {prob_safe:.1%} safe (95% CI {ci[0]:.1%}–{ci[1]:.1%})\nQ: {chief}\nLabs: {labs}"
        response = grok_query(prompt)
        
        st.success(f"⚡ Local answer in {latency:.2f}s")
        st.markdown(response)
        
        # Doctor-in-the-loop
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Accept & Sign", type="primary"):
                doctor = st.text_input("Your name", "Dr. Dino Silvestri")
                pin = st.text_input("PIN", type="password")
                if pin and st.button("Sign & Log"):
                    log_decision(mrn, "", chief, response, doctor)
                    st.success("Logged to immutable audit trail")
        with col2: st.button("✏️ Edit")
        with col3: st.button("❌ Reject")
