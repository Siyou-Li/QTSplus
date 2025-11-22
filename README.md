<p>
  <h1>
  <img src="./assets/logo_with_glasses.svg" height=150px align="right"/>
  Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models
  </h1>
</p>

[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/abs/2511.11910)
[![PWC](https://img.shields.io/badge/%F0%9F%8C%8E%20Website-Official%20Page-blue)](https://qtsplus.github.io/)
[![PWC](https://img.shields.io/badge/HuggingFace-Demo-Green)](https://huggingface.co/collections/AlpachinoNLP/qtsplus)


## 🚀 Introduction
Despite the recent advances in the video understanding ability of multimodal large language models (MLLMs), long video understanding remains a challenge. One of the main issues is that the number of vision tokens grows linearly with video length, which causes an explosion in attention cost, memory, and latency. To solve this challenge, we present Query-aware Token Selector (QTSplus), a lightweight yet powerful visual token selection module that serves as an information gate between the vision encoder and LLMs.

Given a text query and video tokens, QTSplus dynamically selects the most important visual evidence for the input text query by (i) scoring visual tokens via cross-attention, (ii) predicting an instance-specific retention budget based on the complexity of the query, and (iii) selecting Top-n tokens. Furthermore, a small re-encoder preserves temporal order using absolute time information. Integrated into Qwen2.5-VL, QTSplus compresses the vision stream by up to 89% and reduces end-to-end latency by 28% on long videos.
```bash
QTSplus/
├── README.md, LICENSE, requirements.txt
├── assets/                     # logo and figures for the project
├── src/
│   ├── dataset/                # dataset classes & synthesis scripts
│   ├── demo/                   # interactive/demo scripts
│   ├── model/                  # vision towers, tokenizer, projectors, LLM wrappers
│   ├── preprocess/             # data‑preprocessing utilities
│   ├── train/                  # training and fine‑tuning scripts
│   └── utils/                  # misc helpers (vision preproc, dist utils, etc.)
└── test/                       # small smoke tests for models & pipelines
```

## 🫡 Acknowledgements

## ✨ Cite our work
If you find this repo useful, please consider citing: 

```bibtex

```
