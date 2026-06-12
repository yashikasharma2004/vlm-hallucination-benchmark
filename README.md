# Benchmarking Object Hallucination in Vision-Language Models (VLMs) via POPE

This repository contains an end-to-head quantitative evaluation pipeline designed to benchmark and analyze **Object Hallucination** tendencies in Large Vision-Language Models (VLMs). Specifically, we perform a comparative evaluation between **BLIP-2** and **InstructBLIP** using the industry-standard **POPE (Polling-based Object Probing Evaluation)** framework across three distinct context settings on the COCO Val2014 dataset.

## 📌 Project Architecture & Domain Significance
Object Hallucination is one of the most critical vulnerabilities in modern multimodal AI systems, where a model confidently predicts the presence of visual objects that are absent in the actual image matrix. This project targets this challenge by evaluating generative capabilities across three probing categories:
1.  **Random Configuration:** Probs the presence of completely random objects missing from the target cluster.
2.  **Popular Configuration:** Probs frequently occurring dataset objects to verify if the model has a structural text-frequency bias.
3.  **Adversarial Configuration:** Probs objects that highly co-occur in similar contexts but are absent in the current instance (e.g., querying for a "sink" in a kitchen environment that doesn't feature one).

---

## 📊 Quantitative Analysis & Benchmarking Results

The complete quantitative log of our evaluation pipeline (across 900 distinct probing iterations per model) is compiled under `data/results_summary.csv`. Below is the complete empirical performance matrix:

### Performance Matrix
| Model Architecture | Evaluation Setting | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Yes-Bias (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BLIP-2** (`opt-2.7b`) | Random | 70.00% | 68.75% | 73.33% | 70.97% | 53.33% |
| **BLIP-2** (`opt-2.7b`) | Popular | 73.67% | 73.83% | 73.33% | 73.58% | 49.67% |
| **BLIP-2** (`opt-2.7b`) | Adversarial | 68.33% | 66.67% | 73.33% | 69.84% | 55.00% |
| **InstructBLIP** (`vicuna-7b`) | Random | 86.67% | 95.83% | 76.67% | 85.19% | 40.00% |
| **InstructBLIP** (`vicuna-7b`) | Popular | 84.33% | 90.55% | 76.67% | 83.03% | 42.33% |
| **InstructBLIP** (`vicuna-7b`) | Adversarial | 81.33% | 84.56% | 76.67% | 80.42% | 45.33% |

---

## 📈 Visualizations & Analytics

### 1. Unified Metric Comparison (`plots/pope_results.png`)
InstructBLIP shows a massive systemic improvement in overall accuracy and macro metrics over vanilla BLIP-2 across all three setups, maintaining stable performance even when exposed to adversarial testing.
<p align="center">
  <img src="plots/pope_results.png" alt="POPE Evaluation Results Breakdown" width="85%">
</p>

### 2. Adversarial Setting Confusion Matrix (`plots/confusion_matrix.png`)
Under target adversarial probing, the error distribution highlights a fundamental architectural shift:
* **BLIP-2** suffers heavily from a confirmation trap, flagging **55 False Positives** (asserting an object is present when it isn't).
* **InstructBLIP** actively mitigates this text-context correlation, pulling down False Positives to just **21**, demonstrating tighter cross-modal alignment.
<p align="center">
  <img src="plots/confusion_matrix.png" alt="Adversarial Setup Confusion Matrix" width="80%">
</p>

### 3. Cross-Functional Multi-Axis Boundary (`plots/radar_chart.png`)
The multi-variable radar plot captures the model's structural integrity. InstructBLIP expands the metric bounds significantly due to its highly optimized precision profiles and minimized operational bias.
<p align="center">
  <img src="plots/radar_chart.png" alt="Radar Surface Comparison" width="65%">
</p>

---

## 🛠️ Infrastructure & Execution Environment
* **Core Framework:** PyTorch & Hugging Face `transformers` API ecosystem.
* **Hardware & Memory Optimization:** Integrated `accelerate` with automatic shard distribution (`device_map="auto"`) and float16 mixed-precision tracking to run memory-intensive evaluation passes on consumer-grade execution runtimes.
* **Data Layer:** COCO Val2014 localized pipeline with customized JSON parsing interfaces (`blip2_results.json`, `final_results.json`).

---

## 🚀 Execution Pipeline Setup

1. **Clone the Project Structure:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/VLM-Hallucination-Benchmark.git](https://github.com/YOUR_USERNAME/VLM-Hallucination-Benchmark.git)
   cd VLM-Hallucination-Benchmark
