# 🧠 VLM Hallucination Benchmark — POPE Evaluation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Dataset](https://img.shields.io/badge/Dataset-COCO_Val2014-lightgrey)
![Models](https://img.shields.io/badge/Models-BLIP2_|_InstructBLIP-purple)
![License](https://img.shields.io/badge/License-MIT-green)

> **Can a Vision-Language Model be trusted when it says "yes, that object is there"?**  
> This project systematically answers that — by probing two state-of-the-art VLMs across 1800 hallucination tests.

---

## 🔥 Why This Matters

Modern AI models that understand images are being deployed in healthcare, autonomous driving, and robotics. But they have a dangerous flaw — **they hallucinate objects that don't exist**, often with high confidence.

A model that says *"yes, there's a person in the road"* when there isn't one is not just wrong — it's dangerous.

This project quantifies exactly how bad (and how fixable) this problem is, using the industry-standard **POPE framework** on **COCO Val2014**.

---

## 🧪 What We Tested

Two models. Three adversarial settings. 900 probes each.

| Setting | What It Tests |
|---|---|
| **Random** | Can the model reject random absent objects? |
| **Popular** | Does the model hallucinate common COCO objects due to frequency bias? |
| **Adversarial** | Does the model hallucinate contextually plausible but absent objects? *(hardest)* |

---

## 📊 Results

| Model | Setting | Accuracy | Precision | Recall | F1 | Yes-Bias |
|---|---|---|---|---|---|---|
| BLIP-2 (opt-2.7b) | Random | 70.00% | 68.75% | 73.33% | 70.97% | 53.33% |
| BLIP-2 (opt-2.7b) | Popular | 73.67% | 73.83% | 73.33% | 73.58% | 49.67% |
| BLIP-2 (opt-2.7b) | Adversarial | 68.33% | 66.67% | 73.33% | 69.84% | 55.00% |
| **InstructBLIP (vicuna-7b)** | Random | **86.67%** | **95.83%** | 76.67% | **85.19%** | **40.00%** |
| **InstructBLIP (vicuna-7b)** | Popular | **84.33%** | **90.55%** | 76.67% | **83.03%** | 42.33% |
| **InstructBLIP (vicuna-7b)** | Adversarial | **81.33%** | **84.56%** | 76.67% | **80.42%** | 45.33% |

### 🔑 Key Findings

- **InstructBLIP outperforms BLIP-2 by ~15% accuracy** across all three settings
- **BLIP-2 has a severe confirmation bias** — 55 false positives under adversarial probing vs InstructBLIP's 21 — a **2.6× reduction**
- **Yes-Bias drops from 55% → 40%** with InstructBLIP, meaning it is far less likely to blindly say "yes" to an object query
- Even under the hardest adversarial setting, InstructBLIP holds **81.33% accuracy** — BLIP-2 collapses to 68.33%

---

## 📈 Visualizations

### Accuracy & F1 Across All Settings
![POPE Results](pope_results.png)

### Adversarial Confusion Matrix — Where Each Model Fails
![Confusion Matrix](confusion_matrix.png)

### Radar Chart — Full Metric Profile Per Model
![Radar Chart](radar_chart.png)

---

## 🛠️ Setup

```bash
# 1. Clone
git clone https://github.com/yashikasharma2004/vlm-hallucination-benchmark.git
cd vlm-hallucination-benchmark

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open notebook
jupyter notebook
```

> ⚠️ **Hardware note:** InstructBLIP (vicuna-7b) requires ~14GB VRAM.  
> Use `device_map="auto"` + `torch_dtype=float16` (already configured in notebook) to run on a single A100/T4 GPU on Kaggle/Colab.

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch 2.2 | Model inference |
| Hugging Face Transformers | BLIP-2 & InstructBLIP loading |
| accelerate | float16 + auto device mapping |
| COCO Val2014 | Evaluation image dataset |
| POPE Framework | Hallucination probing protocol |

---

## 📁 Repository Structure
vlm-hallucination-benchmark/

├── notebook53c75f753c (5).ipynb   # Main evaluation notebook

├── requirements.txt               # All dependencies

├── results_summary.csv            # Full metrics (900 probes × 2 models)

├── blip2_results.json             # Raw BLIP-2 outputs

├── final_results.json             # Combined results — both models

├── pope_results.png               # Accuracy & F1 bar charts

├── confusion_matrix.png           # Adversarial confusion matrices

└── radar_chart.png                # Multi-metric radar comparison


---

## 🚀 Future Work

- [ ] Add **LLaVA-1.5** and **Qwen-VL** as additional baselines
- [ ] Scale to full **500-image POPE standard** setting
- [ ] Integrate **CHAIR metric** for density-based hallucination scoring
- [ ] Fine-tune BLIP-2 on hard negatives to reduce Yes-Bias

---

## 📌 References

- [POPE Paper — ArXiv 2305.10355](https://arxiv.org/abs/2305.10355)
- [BLIP-2 — Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [InstructBLIP — Salesforce/instructblip-vicuna-7b](https://huggingface.co/Salesforce/instructblip-vicuna-7b)
- [COCO Dataset](https://cocodataset.org/)

---

## 📄 License

MIT License — free to use, modify, and build on.
