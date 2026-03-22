# 🧠 Neural Networks → LLM → RAG  
### A Mathematical & Production-Oriented Study Path

**Author:** Breno Pires  
**Goal:** Transition from classical ML to modern Deep Learning, LLMs and Retrieval-Augmented Generation systems.

---

# 📌 Overview

This repository documents my structured study path to:

- Build neural networks **from scratch**
- Understand the **mathematical foundations**
- Implement modern architectures
- Work with **LLMs in production**
- Build end-to-end **RAG systems**

The focus is:
- Strong theoretical grounding (linear algebra + calculus)
- Clean Python implementation (NumPy → PyTorch)
- Production mindset (APIs, deployment, embeddings, retrieval)

---

# 🧩 PHASE 1 — Foundations: Neural Networks as Function Approximation

## 🎯 Objective
Understand neural networks as parametric function approximators.

We treat a neural network as:

$$
f_\theta : \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

## Topics

- Linear vs Affine maps
- Logistic regression as a single neuron
- Activation functions (ReLU, Sigmoid)
- Composition of functions
- Multi-layer perceptrons (MLP)
- Vanishing / exploding gradients
- Universal Approximation Theorem (intuition)
- Backpropagation (reverse-mode autodiff)
- Gradient descent and optimization geometry

Implementation stack:
- NumPy only

---

# 🧠 PHASE 2 — Representation Learning & Optimization

## 🎯 Objective
Understand how depth creates hierarchical representations.

## Topics

- Width vs Depth
- Expressivity of neural networks
- Initialization theory (Xavier, He)
- Loss landscapes
- Implicit regularization
- Batch normalization
- Dropout

Implementation stack:
- NumPy
- PyTorch (low-level usage)

---

# 🏗 PHASE 3 — Neural Network Architectures

## 1️⃣ Feedforward Networks (MLP)

- Dense layers
- Activation stacking
- Practical training loops

Notebook:
- `1_mlp_pytorch.ipynb`

---

## 2️⃣ Convolutional Neural Networks (CNN)

- Convolution as linear operator
- Filters and feature maps
- Pooling
- Translation equivariance

Notebook:
- `2_cnn_mnist.ipynb`
56r7
---

## 3️⃣ Recurrent Networks (RNN / LSTM / GRU)
Convolutional Architectures
	Convolution operation (1D / 2D)
	Convolutional Neural Networks (CNNs)
	Residual Connections (ResNet)
	Modern Conv Architectures (DenseNet, EfficientNet, ConvNeXt)
	
Sequence Modeling Foundations
	Sequence modeling problem formulation
	Autoregressive modeling
	Recurrent Neural Networks (Vanilla RNN)
	Hidden state as a dynamical system
	Backpropagation Through Time (BPTT)
	Vanishing & Exploding Gradients
	
Gated Recurrent Architectures
	LSTM gating mechanisms
	GRU
	Bidirectional RNNs
	Seq2Seq models
	Attention (pre-Transformer attention)
	
Attention & Transformers
	Attention mechanism (dot-product attention)
	Self-attention
	Positional encoding
	Transformer encoder
	Transformer decoder
	Encoder–decoder Transformer
	Scaling laws
	Vision Transformers (ViT)


Notebook:
- `3_rnn_time_series.ipynb`
- `4_lstm_text_generation.ipynb`

---

# 🤖 PHASE 4 — Transformers & Large Language Models

## 🎯 Objective
Understand modern LLM architecture deeply.

## Topics

- Attention mechanism
- Query / Key / Value
- Self-attention as learned similarity
- Multi-head attention
- Positional encoding (Fourier intuition)
- Transformer encoder & decoder
- Pretraining objective (next token prediction)
- Fine-tuning strategies
- Scaling laws

## Notebooks

- `1_attention_from_scratch.ipynb`
- `2_mini_transformer_pytorch.ipynb`
- `3_finetuning_small_llm.ipynb`

Libraries:
- PyTorch
- HuggingFace

---

# 🔎 PHASE 5 — Retrieval-Augmented Generation (RAG)

## 🎯 Objective
Build production-ready RAG systems.

## Topics

- Embeddings as metric learning
- Cosine similarity
- Vector databases
- FAISS / Chroma
- Chunking strategies
- Hybrid search
- Prompt engineering
- Serving LLM + retriever with FastAPI

## Notebooks / Projects

- `1_embeddings_exploration.ipynb`
- `2_vector_search_faiss.ipynb`
- `3_build_rag_pipeline.ipynb`
- `4_rag_api_fastapi.ipynb`

---

# 🛠 Tools Stack

- NumPy (mathematical clarity)
- PyTorch (modeling)
- HuggingFace (LLMs)
- FAISS / ChromaDB (retrieval)
- FastAPI (serving)

---

# 📅 Suggested Timeline

| Phase | Duration |
|-------|----------|
| Foundations | 2 weeks |
| Representation Learning | 2 weeks |
| Architectures | 2 weeks |
| Transformers | 2 weeks |
| RAG Systems | 2–3 weeks |

Total: ~2 months (1h/day consistent study)

---

# 📌 Learning Philosophy

- Derive before coding
- Implement before abstracting
- Connect theory to production
- Build small experiments for every concept
- Document learnings clearly

---

# 🚀 Expected Outcome

By the end of this study path, I will be able to:

- Implement neural networks from scratch
- Understand gradient-based optimization deeply
- Train CNNs and RNNs confidently
- Explain transformer architecture rigorously
- Fine-tune LLMs
- Build and deploy RAG systems end-to-end

---

# 🗂 Repository Structure
neural-nets-llm-study/
│
├── phase_1_foundations/
├── phase_2_representation_learning/
├── phase_3_architectures/
├── phase_4_transformers/
├── phase_5_rag/
└── README.md
