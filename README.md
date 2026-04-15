# CTMAS: Proactive Threat Modeling for Intelligent Cyber-Physical Systems

## Overview
**CTMAS** is an advanced, distributed machine learning framework designed to protect critical Cyber-Physical Systems (CPS)—such as water treatment facilities—from real-time cyber-attacks. Operating in a decentralized environment using Federated Learning, the system ensures data privacy while proactively detecting anomalies. Once an anomaly is detected, the system utilizes Explainable AI (XAI) to map the mathematical deviations directly to actionable Cyber Threat Intelligence (STRIDE and MITRE ATT&CK frameworks).

This project uses the real-world **SWaT** (Secure Water Treatment) dataset to simulate an end-to-end continuous threat monitoring pipeline.

---

## 🏗️ System Architecture & Workflow

The pipeline consists of four continuous phases:

1. **Decentralized Data Preparation**: The central dataset is partitioned into isolated nodes representing distinct physical stages of a plant.
2. **Privacy-Preserving Federated Training**: The nodes collaboratively train a global anomaly detection model without ever sharing their raw sensor data.
3. **Proactive Anomaly Detection**: The trained system actively monitors incoming sensor data, computing an Early Warning Score to catch impending attacks before catastrophic failure.
4. **Threat Intelligence Mapping**: When an anomaly is detected, XAI calculates exactly which sensors are under attack and maps them to standard ICS threat matrices.

---

## 🗂️ Module Descriptions

### 1. `main.py` (The Entry Point)
Acts as the central orchestrator of the entire simulation. It starts the synchronous federated learning environment via Flower (`flwr`), initializes the 6 distinct plant stages as clients, aggregates their learnings, and triggers the post-training anomaly detection simulation on the test data.

### 2. `data_pipeline.py` (Data & Feature Engineering)
Responsible for ingesting the SWaT dataset (`normal.csv` and `attack.csv`). It maps the dataset to specific plant stages (P1 to P6). Because CPS attacks occur across time, standard tabular data is compressed into **3D Time-Series Sliding Windows** `(Batch x Sequence_Length x Features)` to capture temporal events effectively. Feature padding is applied to ensure architectural symmetry across the federated network.

### 3. `models.py` (Neural Architecture)
Defines the **1D-CNN (Convolutional) Autoencoder**. 
* **Why 1D-CNN instead of LSTM?** Recurrent neural networks (like LSTM) loop states over time, making it exceptionally difficult to compute per-sample gradients—a hard mathematical requirement for Differential Privacy. By passing a 1D Convolutional filter over the sequence axis, we efficiently capture temporal relationships while maintaining 100% compatibility with PyTorch Differential Privacy engines.

### 4. `local_training.py` (Differential Privacy Engine)
Handles the training loop that executes on isolated client devices.
* Integrates **Opacus** (Differential Privacy Engine).
* Wraps the CNN-Autoencoder to compute gradients on a per-sample basis, applies rigid gradient clipping, and injects calibrated Gaussian noise. This ensures an attacker cannot reverse-engineer raw sensor data out of the weight updates the client sends back to the server.

### 5. `server.py` (Trust-Aware FL Strategy)
Defines the `TrustAwareFedAvg` class, a custom Federated Learning strategy. It calculates the mathematical norm (size) of incoming weight updates from all clients. If a client's update deviates drastically from the mean (e.g., > 2 standard deviations due to a poisoned dataset or a hijacked node), the server automatically **drops the malicious client** and averages only the inputs of trusted nodes.

### 6. `threat_intelligence.py` (Early Warnings & MITRE Mapping)
* **Early Warning:** Instead of treating every data point in isolation, it computes an **Exponential Weighted Moving Average (EWMA)** of the model's reconstruction error. This catches slow, insidious anomalies (like a slowly draining tank).
* **Intelligence Mapping:** Intercepts the failing sensors identified by the XAI module and maps them to the **STRIDE** methodology (e.g., *Spoofing*, *Tampering*) and the **MITRE ATT&CK for ICS** framework (e.g., *T0831 Manipulation of Control*).

### 7. `xai_explainer.py` (Explainable AI)
When the model detects an attack, this module executes **SHAP (SHapley Additive exPlanations)** via `GradientExplainer`. It unpacks the black-box neural network and returns the precise features/sensors (e.g., `AIT201`, `LIT301`) that contributed the highest margin of error to the sequence, pinpointing the physical location of the hack.

### 8. `visualization.py` (Metrics & Plotting)
Plots the visual output to the `results/` folder. It maps the training loss curves alongside the expended Privacy Budget ($\epsilon$) during the FL rounds. Crucially, it plots the raw anomaly errors against the EWMA score and thresholds to visualize exactly when a cyber-attack breached the system defenses.

---

## 📈 Simulation Results & Output

When executing `main.py`, the system generates terminal telemetry and visual plots. Here is the expected output breakdown:

### Training Output
The terminal will iterate through the training of 6 isolated client stages across `N` Global Rounds. You will see the Trust-Aware aggregator accepting or rejecting clients:
```
INFO: Trust-Aware Aggregation: 5/6 clients accepted.
Round 4 completed. Aggregated metrics: {'avg_epsilon': 5.40, 'trusted_clients': 5}
```

### Anomaly Post-Simulation Output
After training, the global model tests a live data stream of known attack sequences on Stage P1.
```
[!] CRITICAL ANOMALY DETECTED at Window Index 17 (EWMA Score > 0.7 or Above Threshold)

[+] XAI Interpretability Results (SHAP):
Top Contributing Sensors/Actuators: ['AIT201', 'LIT301']

[!] Threat Intelligence Mapping:
  1. Stage: P1 | Component: AIT201
     => STRIDE Threat : Tampering
     => MITRE ATT&CK  : T0831 (Manipulation of Control)
```

**Generated Plots (Saved in `results/`):**
1. **`stage_1_anomaly_plot.png`**: Shows the blue spikes of raw reconstruction errors when the system detects an attack, trailing the smoother red line (EWMA), crossing the orange dynamic threshold limit.
2. **`federated_metrics.png`**: The global evaluation curve representing the steady lowering of network loss over time, plotted against the rise in DP epsilon limits.

---

## 🚀 Running the Project

**Prerequisites:** 
Ensure you have the required SWaT CSV datasets inside the `dataset/` directory (`normal.csv` and `attack.csv`).

**Installation:**
```bash
pip install -r requirements.txt
```

**Run the pipeline:**
```bash
python main.py
```
