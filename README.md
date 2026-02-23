# Crypto-Auditor: RAG vs PEFT Comparative Study on Cryptocurrency Whitepapers

**Evaluating Retrieval-Augmented Generation (RAG) vs Parameter-Efficient Fine-Tuning (PEFT/LoRA) under limited compute constraints**

This repository contains the code, data, and outputs for the project *"Crypto-Auditor: A Head-to-Head Comparison of RAG and PEFT on Cryptocurrency Whitepapers"*. The work demonstrates how RAG can outperform PEFT in factual QA tasks when paired with stronger generators, even with very limited resources (free Colab CPU/T4 GPU, small models, 40 manually curated examples).

### Project Overview

**Goal**: Compare RAG and PEFT on retrieval precision, hallucination control, and generation quality using four cryptocurrency whitepapers (Bitcoin, Chainlink, Solana, Uniswap).

**Key findings** (from preprint draft):
- RAG with distilgpt2: strong retrieval (Precision@3 = 0.892, MRR = 0.925), zero hallucination, ROUGE-L = 0.158
- PEFT/LoRA on distilgpt2: slightly lower lexical/semantic performance, still zero hallucination
- RAG with Phi-3-mini (8-bit): significantly higher ROUGE-L (~0.35–0.45) and semantic similarity (~0.85), maintaining grounding

**Compute constraints**:
- RAG inference: CPU (Google Colab free tier)
- PEFT/LoRA fine-tuning: T4 GPU (free Colab)
- Models: distilgpt2 (82M), Phi-3-mini-4k-instruct (3.8B, 8-bit)
- Evaluation set: 40 manually curated QA pairs (10 per whitepaper)

### Repository Contents

- `RAG.ipynb` / `RAG2.ipynb`               → Original RAG pipeline (distilgpt2 + bge-small-en-v1.5 + FAISS)
- `PEFT_final.ipynb`                        → LoRA fine-tuning on distilgpt2
- `RAGwithPHI3 (1).ipynb`                   → RAG pipeline upgraded to Phi-3-mini (8-bit)
- `crypto_rag_eval_dataset.json`            → 40 manually created QA pairs (ground truth)
- `phi_rag_outputs_final.json`              → Generated answers from Phi-3 RAG
- `rag_outputs_final.json`                  → Generated answers from distilgpt2 RAG
- `peft_outputs_final.json`                 → Generated answers from PEFT
- `*_metrics_summary.csv` / `*_metrics_detailed.csv` → Computed metrics (ROUGE-L, semantic similarity, hallucination proxy)
- `final_rag_peft_phi3_comparison.csv`      → Summary comparison table for preprint

### Reproducing the Results

1. **Requirements**  
   - Google Colab (free tier: CPU for RAG inference, T4 GPU for PEFT)
   - Python 3.10+ (Colab default)
   - Installs: transformers, sentence-transformers, faiss-gpu/cpu, bitsandbytes, accelerate, rouge-score, pandas, tabulate

2. **Quick start**  
   - Open any notebook in Colab  
   - Upload PDFs (bitcoin.pdf, chainlink.pdf, solana.pdf, uniswap.pdf) if rebuilding from scratch  
   - Run cells sequentially (most notebooks are self-contained after data loading)  
   - Expected runtime: 15–40 min per notebook on free tier

3. **Evaluation dataset**  
   - 40 manually crafted QA pairs (10 per whitepaper)  
   - Answers are verbatim or near-verbatim excerpts from the source documents  
   - Stored in `crypto_rag_eval_dataset.json`  
   - Designed for high-quality, traceable evaluation (not synthetic)

4. **Main results** (from comparison table)

| Model / Method         | ROUGE-L | Semantic Sim | Halluc Rate | Precision@3 | MRR   |
|-------------------------|---------|--------------|-------------|-------------|-------|
| RAG (distilgpt2)        | 0.158   | 0.715        | 0.000       | 0.892       | 0.925 |
| PEFT (distilgpt2 + LoRA)| 0.126   | 0.667        | 0.000       | N/A         | N/A   |
| RAG (Phi-3-mini 8-bit)  | ~0.38   | ~0.85        | ~0.000      | ~0.90       | ~0.93 |
