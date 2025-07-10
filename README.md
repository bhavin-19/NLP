# 🧠 Advanced NLP Pipelines

This repository showcases a collection of advanced NLP projects, including custom tokenization, word embedding models, neural language modeling, sentiment analysis, question answering, and multimodal understanding. All models are implemented using PyTorch and pretrained embeddings (Word2Vec, FastText, GloVe, BERT, etc.).

---

## 📌 Table of Contents

- [1. End-to-End NLP Pipeline](#1-end-to-end-nlp-pipeline)
- [2. Aspect-Based Sentiment Analysis & QA](#2-aspect-based-sentiment-analysis--qa)
- [3. Transformer Language Modeling & Multimodal Sarcasm Explanation](#3-transformer-language-modeling--multimodal-sarcasm-explanation)

---

## 1. End-to-End NLP Pipeline

- ✅ Built from scratch:
  - WordPiece Tokenizer
  - Word2Vec (CBOW) Embedding Model
  - MLP-based Neural Language Model
- 📈 Features:
  - Next-word prediction using custom embeddings
  - Multiple architectural variations
  - Loss and embedding visualization (e.g., cosine similarity)

---

## 2. Aspect-Based Sentiment Analysis & QA

- 🎯 Tasks:
  - Aspect Term Extraction
  - Aspect-Based Sentiment Analysis (ABSA)
  - Span-level Question Answering (QA) on SQuAD v2

- 🧠 Models:
  - RNN / GRU + FastText / GloVe / BERT for ABSA
  - SpanBERT & SpanBERT-CRF for QA

- 🔧 Highlights:
  - Custom BIO tagging & instance-wise preprocessing
  - Multi-task pipeline with unified training
  - Metric tracking: F1, Accuracy, Exact Match (EM)

---

## 3. Transformer Language Modeling & Multimodal Sarcasm Explanation

- 📜 Tasks:
  - Shakespearean Language Modeling (Transformer from scratch)
  - Claim Normalization using BART & T5
  - Multimodal Sarcasm Explanation

- 🖼️ Architecture:
  - Fusion of BART + Vision Transformer (ViT)
  - Custom shared encoder-decoder for text+image input

- 🧪 Evaluation:
  - F1, Accuracy, EM metrics
  - Loss-performance visualization across configurations

---

## 🛠️ Setup

```bash
git clone https://github.com/bhavin-19/NLP.git
cd NLP
pip install -r requirements.txt
