# 🧠 VM-CPU-Forecasting-Transformer

**VM-CPU-Forecasting-Transformer** is a PyTorch-based framework for **multivariate time series forecasting** of **virtual machine (VM)** resource utilization metrics such as CPU, GPU, and memory consumption.  
The model leverages a **Transformer encoder architecture** specifically adapted for spatio-temporal telemetry data and supports **JSON-based input instructions** for flexible VM-level forecasting workflows.

---

## 📘 Overview

Modern data centers and cloud-native infrastructures generate high-frequency telemetry data that capture the dynamic utilization of virtualized resources. Accurate forecasting of CPU utilization enables **proactive scheduling**, **load balancing**, and **anomaly prevention** in large-scale clusters.

This repository implements a **Forecasting Transformer** model capable of learning long-range dependencies from multivariate VM telemetry sequences, outperforming traditional autoregressive models (ARIMA) and recurrent networks (LSTM, BiLSTM).

---

## ⚙️ Features

- Transformer-based encoder architecture for multivariate forecasting  
- Learnable positional embeddings for temporal order preservation  
- JSON-driven input interface for flexible VM curve simulation  
- Model checkpoint loading and GPU/CPU auto-detection  
- Visualization of predicted vs. real CPU utilization curves  
- Modular design: `ForecastingTransformer.py` (model) and `main.py` (runner)

---

## 🧩 Architecture

The model follows the Transformer encoder paradigm with adaptations for time series:

1. **Input Projection:** Each `(T × F)` sequence (timesteps × features) is projected into an embedding space.  
2. **Positional Encoding:** A learnable positional embedding tensor is added to the sequence to encode temporal order.  
3. **Encoder Layers:** Each layer includes:
   - Pre-normalized multi-head self-attention
   - Feed-forward network with ReLU activation
   - Residual connections and dropout  
4. **Output Head:** A linear projection maps the final embedding to the forecasted values.

---

## 🧰 Installation

### Requirements
```bash
Python >= 3.12
torch >= 2.4.0
numpy
matplotlib
argparse
```

### Setup
```bash
git clone https://github.com/dannydenovi/VM-CPU-Forecasting-Transformer.git
cd VM-CPU-Forecasting-Transformer
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Forecasting Training
```bash
python3 main.py
```


### Run Forecasting Transfer Learning
```bash
python3 main.py --transfer
```

### Run Forecasting Test
```bash
python3 main.py --test
```

---

## 📦 Repository Structure

```
VM-CPU-Forecasting-Transformer/
├── ForecastingTransformer.py    # Transformer encoder model definition
├── main.py                      # Main training/inference entrypoint
├── requirements.txt             # Dependencies
├── checkpoints/                 # Model weights
├── data/                        # Telemetry datasets (Azure, Alibaba)
└── plots/                       # Output visualizations
```

---

## 🧾 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it for both academic and commercial purposes with attribution.

---

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
