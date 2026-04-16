# CTMAS: Proactive Threat Modeling for Intelligent Cyber-Physical Systems

## 🛡️ Project Philosophy
**CTMAS** (Cyber-Physical Threat Monitoring & Analysis System) is a production-grade framework designed to secure Industrial Control Systems (ICS)—specifically Water Treatment Plants—using a decentralized, privacy-preserving machine learning pipeline. 

Unlike reactive systems that trigger alerts *after* a sensor breaches a static limit, CTMAS utilizes **Proactive Threat Modeling**. It learns the "normal" behavioral fingerprint of an entire plant and detects minute statistical deviations (anomalies) that precede actual failure.

---

## 🏗️ Detailed Architecture & Workflow

The system operates in four distinct phases, ensuring security at each layer of the stack:

1.  **Data Partitioning (Decentralized Preparation)**: Raw SWaT sensor data is mapped to its physical origin (Stages P1–P6). This reflects a real-world edge computing scenario where stage controllers don't share raw data.
2.  **Federated Learning with Differential Privacy**: Nodes ship weight updates, not data. Every update is "privatized" by injecting noise, ensuring an attacker cannot reverse-engineer the plant's state from the model.
3.  **Proactive Detection (EWMA)**: An Early Warning System monitors Error reconstruction trends. If the trend exceeds a dynamic threshold, an intercept is triggered.
4.  **Intelligence Synthesis (XAI & Mapping)**: SHAP explains *which* sensor is failing, and the Threat Matrix translates that into **MITRE ATT&CK** techniques and **STRIDE** categories.

---

## 📂 Module-by-Module Analysis

### 1. `models.py`: The Neural Architect
*   **Technology**: PyTorch 1D-CNN Autoencoder.
*   **Mechanism**: A symmetric encoder-decoder that compresses time-series windows into a latent space and attempts to reconstruct them.
*   **The "Why" (CNN vs RNN)**: Standard Recurrent Neural Networks (LSTMs) are notoriously difficult to use with **Differential Privacy (DP)** because they maintain hidden states across time, making per-sample gradient computation (a requirement for Opacus) mathematically complex. We use **1D-CNNs** because they capture temporal trends via convolutional filters while remaining perfectly compatible with DP engines.

### 2. `data_pipeline.py`: Time-Series Engineering
*   **Technology**: Pandas, NumPy, Scikit-Learn.
*   **Mechanism**: Implements **Sliding Window Tokenization**. It takes raw sensor logs and creates 3D tensors `(Batch, Sequence_Length, Features)`. 
*   **Key Detail**: Features are zero-padded to `Config.NUM_FEATURES`. This ensures that even if Stage P1 has fewer sensors than P6, the neural architecture remains globally symmetric across the federated network.

### 3. `local_training.py`: The Privacy Engine
*   **Technology**: **Opacus** (by Meta).
*   **Mechanism**: Before gradients are sent to the server, this module:
    1.  Computes per-sample gradients.
    2.  Clips the gradients to a rigid norm (`MAX_GRAD_NORM`).
    3.  Adds Gaussian noise.
*   **The "Why"**: This provides a mathematical guarantee of **Differential Privacy**. It ensures that no single data pulse at the edge can be uniquely identified in the global global model.

### 4. `server.py`: Trust-Aware Aggregation
*   **Technology**: Flower (`flwr`).
*   **Mechanism**: Extends the standard `FedAvg` (Federated Averaging) strategy. It calculates the similarity of local updates.
*   **Poisoning Defense**: If a node is compromised and starts sending "junk" updates to poison the global model, the `TrustAware` logic detects the anomaly and **excludes that node** from the round aggregation.

### 5. `threat_intelligence.py`: Early Warning & Intelligence
*   **Mechanism**: 
    - **EWMA (Exponential Weighted Moving Average)**: Tracks the *trend* of reconstruction errors. This is crucial for detecting "Low-and-Slow" attacks that stay just below static thresholds for days.
    - **Intelligence Mapping**: A static knowledge base that maps sensor prefixes (e.g., `LIT`, `FIT`) to **STRIDE** and **MITRE ATT&CK for ICS** T-codes.
*   **Benefit**: Converts raw math into actionable "Combat Reports" for security operators.

### 6. `xai_explainer.py`: The Black-Box Unpacker
*   **Technology**: **SHAP** (SHapley Additive exPlanations).
*   **Mechanism**: Uses `GradientExplainer` to attribute the model's reconstruction error to specific input sensors.
*   **Outcome**: When an alert fires, SHAP tells you precisely which sensor (e.g., `AIT201`) is being spoofed.

---

## 📡 Full-Stack Visualization Layer

### `api.py`: WebSocket Backend
*   **Technology**: FastAPI + WebSockets.
*   **The "Why"**: Standard REST APIs are insufficient for live simulations. We use WebSockets to "push" events (training rounds, sensor blips, XAI results) to the frontend with zero latency.

### `frontend/`: The Command Center
*   **Technology**: React, TailwindCSS, Recharts, Lucide Icons.
*   **Dashboard Components**:
    - **Swarm View**: Shows the status of all 6 plant nodes in the federated network.
    - **Telemetry Panel**: Live graph of reconstruction error vs early warning score.
    - **Intelligence Summary**: Displays the MITRE/STRIDE mapping once an anomaly is intercepted.
    - **Neural analytics**: Renders a dynamic image of the Federated Learning loss/epsilon curve (`federated_metrics.png`).

---

## 🚀 Getting Started

### 📦 Prerequisites
- Python 3.9+
- Node.js & npm (for Web Dashboard)
- SWaT Dataset (`normal.csv` and `attack.csv` in `dataset/`)

### 🛠️ Installation
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 🏃 Running
- **Mode 1 (CLI)**: `python main.py`
- **Mode 2 (Pro Web Dashboard)**:
  1. Start backend: `python api.py`
  2. Start frontend: `cd frontend && npm run dev`
  3. Navigate to `http://localhost:5173`

---

## 🛠️ Technology Stack Summary

| Layer | Technology |
| :--- | :--- |
| **Computational Framework** | PyTorch / Python |
| **Distributed Training** | Flower (flwr) |
| **Privacy / Encryption** | Opacus (Differential Privacy) |
| **Explainable AI** | SHAP (Gradient Explainer) |
| **Web Service** | FastAPI / Uvicorn |
| **Web UI** | React / TailwindCSS / Recharts |
| **Threat Intelligence** | MITRE ATT&CK / STRIDE |
