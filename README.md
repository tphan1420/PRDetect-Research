# PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree

<p align="center">
  <a href="https://aclanthology.org/2025.findings-naacl.464/"><img src="https://img.shields.io/badge/NAACL_Findings-2025-blue.svg" alt="NAACL 2025"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-torch__geometric-3C2179.svg" alt="PyG">
  <img src="https://img.shields.io/badge/SpaCy-en__core__web__sm-09A3D5.svg?logo=spacy" alt="SpaCy">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
</p>

This repository contains the official implementation of the research paper:  
**"PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree"**, accepted to **Findings of the Association for Computational Linguistics: NAACL 2025** (pages 8305–8316).

---

## 📌 1. Overview & Motivation

With the rapid emergence of Large Language Models (LLMs) such as ChatGPT, GPT-4, LLaMA, and Mistral, detecting machine-generated text has become vital for academic integrity, cybersecurity, and information trustworthiness.

### The Critical Bottleneck: Fragility to Perturbations
Existing detectors (feature-based, zero-shot perplexity curvature like DetectGPT, and fine-tuned transformers like RoBERTa) fail severely when human editing or subtle word-level perturbations are applied. When merely **5%–10%** of words are replaced with synonyms, the accuracy of leading detectors drops precipitously from ~100% to near random guess (~50%).

### Core Scientific Insight of PRDetect
While surface statistics and n-gram perplexity fluctuate drastically under minor edits, **Dependency Syntax Trees remain remarkably invariant to word-level perturbations**, while exhibiting clear structural differences between human and AI writing:
- **LLM-generated sentences** tend to follow strict grammatical conventions with significantly deeper syntax trees (higher average node depth and taller root height).
- **Human-written text** demonstrates greater syntactic flexibility, concise clauses, and distinct dependency distributions.

---

## 🚀 2. Major Architecture Upgrade: Multi-Relational RGCN

To overcome the limitations of homogeneous binary adjacency matrices on complex, diverse benchmarks like **RAID (Robust AI Detection)**, PRDetect has been upgraded with **Relational Graph Convolutional Networks (RGCN)**:

```
[Input Text] ──► [SpaCy Dependency Parser] ──► [Multi-Relational Graph (edge_index + edge_type)]
      │                                                               │
      └────────► [RoBERTa Token Embedding] ──► [Node Features X]      │
                                                      │               │
                                                      ▼               ▼
                                           [RGCN Layers: W_r per relation]
                                                      │
                                                      ▼
                                           [Global Mean Pooling] ──► [Sigmoid: P(Human)]
```

### Mathematical Formulation
The hidden state of node $i$ at layer $l+1$ is updated via relation-specific transformations:

$$h_{i}^{(l+1)} = \sigma\left(\sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_{i}^{r}}\frac{1}{c_{i,r}}W_{r}^{(l)}h_{j}^{(l)} + W_{0}^{(l)}h_{i}^{(l)}\right)$$

- **$\mathcal{R}$**: Set of **60+ syntactic dependency relations** (`nsubj`, `dobj`, `amod`, `prep`, `compound`, `punct`, `ROOT`, etc.).
- **$W_{r}^{(l)}$**: Distinct trainable weight matrix dedicated to relation type $r$.
- **$W_{0}^{(l)}$**: Self-loop root weight preserving node identity.
- **$c_{i,r}$**: In-degree structural normalization factor.
- **Basis Decomposition ($B = 30$)**: $W_r = \sum_{b=1}^{B} a_{rb} V_b$ to prevent parameter explosion and regularize against overfitting across diverse domains and generators.

---

## 📂 3. Repository Structure

```plaintext
PRDetect-Research/
├── 2025.findings-naacl.464.pdf   # Official paper published at NAACL 2025
├── README.md                     # Project documentation & quickstart guide
├── DuAn.md                       # Comprehensive research report & technical analysis (Vietnamese)
├── Update.md                     # RAID benchmark analysis & architectural roadmap
├── requirements.txt              # Python package dependencies
│
├── dep_vocab.py                  # Canonical dependency relation vocabulary (60+ relations)
├── dep_vocab.json                # Predefined dependency relation mappings
├── building_graph.py             # Dependency parsing & graph dataset builder (edge_index + edge_type)
├── model/
│   ├── GCN2.py                   # Baseline 2-layer homogeneous GCN
│   └── RGCN.py                   # Relational GCN architectures (RGCN2, RGCN4, RelationalGCN)
│
├── rgcn.py                       # Training pipeline for multi-relational RGCN
├── gcn.py                        # Training script for baseline GCN
├── test_rgcn.py                  # Dedicated evaluation script for RGCN models
├── test.py                       # Unified test script supporting both GCN and RGCN (--model_type)
├── detect.py                     # Direct inference CLI for custom text inputs
├── convert_datasets.py           # Dataset converter for RAID & DetectRL benchmarks
├── drawtree.py                   # Syntax tree visualization utility
│
├── datasets/                     # Raw downloaded datasets (RAID, DetectRL, etc.)
├── original_text/                # Processed JSONL dataset files (HC3, RAID, GPT3.5-Mixed)
├── graph_data/                   # Preprocessed tensor graph datasets (.pkl)
├── model/                        # Saved model checkpoints (*.pth)
└── logs/ & output/               # TensorBoard logs and experiment outputs
```

---

## 🛠️ 4. Installation & Environment Setup

### Prerequisites
- Python 3.8 to 3.13
- CUDA-enabled GPU (optional, CPU supported with fallback)

```bash
# 1. Clone repository
git clone https://github.com/thulx18/PRDetect.git
cd PRDetect-Research

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch Geometric & SpaCy language model
pip install torch_geometric
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## ⚡ 5. Quickstart & Workflow

### Step 1: Prepare Datasets (Optional - RAID / DetectRL)
If you are evaluating on the RAID or DetectRL benchmark, convert raw samples into PRDetect's standard JSON format:
```bash
# List available datasets in datasets/
python convert_datasets.py --list

# Convert a RAID test set (e.g., 200 pairs = 400 samples)
python convert_datasets.py -i datasets/raid_dataset/raid_test_dataset_llm_type_gpt2_decoding_type_greedy_repetition_penalty_no.json -o raid_gpt2_test -n 200
```

### Step 2: Build Syntax Graphs
Extract dependency syntax trees and word embeddings with relation tags:
```bash
# Build multi-relational graphs for a specific dataset
python building_graph.py --file raid_gpt2_test

# Or build default training/validation/test sets (hc3_train, hc3_val, hc3_test)
python building_graph.py

# Optional: Add bidirectional edges with inverse relation labels
python building_graph.py --file raid_gpt2_test --bidirectional
```
*Generated graph files are stored in `graph_data/<dataset_name>.pkl`.*

### Step 3: Train Model
Train either the upgraded multi-relational RGCN or the baseline GCN:

```bash
# 1. Train Relational GCN (RGCN - Recommended)
python rgcn.py --dataset hc3 --seed 2024 --epochs 40 --lr 0.0001 --num_layers 2 --num_bases 30

# 2. Train 4-layer RGCN for longer or complex sentences
python rgcn.py --dataset hc3 --seed 2024 --epochs 40 --num_layers 4

# 3. Train Baseline GCN (Homogeneous)
python gcn.py
```
*Best checkpoints are automatically saved to `model/<dataset>_rgcn_model_<seed>.pth`.*

### Step 4: Evaluate & Test
Evaluate on test datasets to calculate Accuracy, ROC-AUC, and F1-score:

```bash
# Dedicated evaluation for RGCN:
python test_rgcn.py --file raid_gpt2_test --dataset hc3 --seed 2024 -s

# Unified evaluation using test.py:
python test.py --file raid_gpt2_test --dataset hc3 --seed 2024 --model_type rgcn -s
python test.py --file raid_gpt2_test --dataset hc3 --seed 2024 --model_type gcn -s
```

### Step 5: Direct Text Inference (CLI)
Classify any text as human-written or machine-generated:

```bash
python detect.py --text "There are many factors that contribute to the intelligence gap between humans and other organisms." --dataset hc3 --seed 2024 --model_type rgcn
```
**Output:**
```plaintext
[KẾT QUẢ DỰ ĐOÁN]
  * Mô hình: RGCN
  * Xác suất Human: 0.9842 (98.42%)
  * Nhãn dự đoán: 1 -> Human-written (Người viết)
```

---

## 📊 6. Experimental Results (NAACL 2025 Paper)

### Robustness Against Synonym Replacement Perturbations

| Model | HC3 (0% Noise) | HC3 (5%) | HC3 (10%) | HC3 (30%) | GPT-3.5 (0%) | GPT-3.5 (5%) | GPT-3.5 (10%) | GPT-3.5 (30%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RoBERTa (Fine-tuned)** | 0.9380 | 0.5800 | 0.5570 | 0.5080 | 0.8927 | 0.5055 | 0.4995 | 0.4945 |
| **DetectGPT (Zero-shot)** | 0.8350 | 0.8010 | 0.7720 | 0.6580 | 0.6060 | 0.5860 | 0.5820 | 0.5500 |
| **CoCo (Entity Graph)** | **0.9981** | 0.5432 | 0.5421 | 0.5333 | **1.0000** | 0.6995 | 0.6893 | 0.6805 |
| **PRDetect (Ours)** | 0.9878 | **0.9878** | **0.9872** | **0.9864** | 0.9656 | **0.9630** | **0.9632** | **0.9638** |

> **Key Finding:** When the perturbation ratio increases from 0% to 30%, PRDetect degrades by **less than 0.14%**, whereas traditional detectors degrade by nearly 50%.

### Computational Efficiency (Tested on NVIDIA RTX 4090)

| Model | Preprocessing | Training Time | Test / Inference Time |
| :--- | :---: | :---: | :---: |
| **RoBERTa** | — | — | 27.6s |
| **DetectGPT** | — | — | 1298.4s |
| **CoCo** | 2545.3s | 1655.0s | 28.3s |
| **PRDetect** | **1754.5s** | **629.7s** | **2.5s** |

---

## 📖 7. Citation

If you use PRDetect or this codebase in your research, please cite our NAACL 2025 paper:

```bibtex
@inproceedings{li2025prdetect,
  title={PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree},
  author={Li, Xiang and Yin, Zhiyi and Tan, Hexiang and Jing, Shaoling and Su, Du and Cheng, Yi and Shen, Huawei and Sun, Fei},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={8305--8316},
  year={2025}
}
```