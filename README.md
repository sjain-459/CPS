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

---

## 🖥️ Cybersecurity Command Center (Web Application)

CTMAS now includes a full-stack, real-time simulation dashboard that provides a high-fidelity visual interface for monitoring the cybersecurity pipeline. Built with **React** and **FastAPI**, it allows users to witness the system's "conscious" thought process live.

### Key Features:
*   **Live Sensor Telemetry**: WebSocket-driven real-time charts showing reconstruction errors and EWMA early warning scores.
*   **Dynamic Node Management**: Interactively add or remove federated clients to simulate swarm scaling.
*   **XAI Diagnostic Interface**: Visual breakdown of SHAP features and direct mapping to MITRE/STRIDE intelligence when an anomaly is intercepted.
*   **Privacy Observability**: Live tracking of the Privacy Budget ($\epsilon$) and Global Loss curves directly in the UI.

---

## 🗂️ Module Descriptions

### 1. `main.py` (CLI entry point)
The core simulation logic. It orchestrates the synchronous federated learning environment and post-training anomaly detection on the test data.

### 2. `api.py` (FastAPI WebSocket Backend)
A production-ready wrapper that exposes the simulation logic via WebSockets. It streams training telemetry, sensor streams, and XAI results to the frontend dashboard in real-time.

### 3. `frontend/` (React Dashboard)
A modern "Cybersecurity Command Center" built with Vite, TailwindCSS, and Lucide Icons. It manages the state of the simulation and renders complex analytics into human-readable dashboards.

### 4. `data_pipeline.py` (Data & Feature Engineering)
Responsible for ingesting the SWaT dataset. Compresses tabular data into **3D Time-Series Sliding Windows** for 1D-CNN ingestion.

*(... modules 5–8 remain the same: models, local_training, server, threat_intelligence, xai_explainer, visualization ...)*

---

## 🚀 Running the Project

### 1. Prerequisites
Ensure you have the SWaT CSV datasets inside the `dataset/` directory (`normal.csv` and `attack.csv`).

### 2. Backend Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Choose your mode:

#### **Mode A: CLI Simulation** (Terminal only)
```bash
python main.py
```

#### **Mode B: Interactive Web Dashboard** (Recommended)
1. **Start the Backend API:**
   ```bash
   python api.py # Runs on port 8001
   ```
2. **Setup and Start the Frontend:**  
   (In a separate terminal)
   ```bash
   cd frontend
   npm install
   npm run dev  # Dashboard will be live at http://localhost:5173
   ```

---

## 📈 Simulation Results & Output

### 1. Web Dashboard Analytics
*   **Swarm Monitoring**: Watch nodes pulse green during training and gray out when untrusted nodes are dropped by the strategy.
*   **Active Threat Mapping**: When the alert triggers, the UI instantly populates a red diagnostic panel showing the compromised sensor ID and the specific MITRE technique (e.g., T0831).
*   **Neural Analytics**: Automatic rendering of the `federated_metrics.png` plot in the dashboard upon session completion.

### 2. Physical Plots (Saved in `results/`)
1. **`stage_1_anomaly_plot.png`**: Visualizes reconstruction error crossing the EWMA thresholds.
2. **`federated_metrics.png`**: Tracks Global Loss vs. Differential Privacy Epsilon.

---

## 🛡️ Security Framework Mappings
*   **Data Integrity**: Differential Privacy (Opacus).
*   **Poisoning Defense**: Custom Trust-Aware FedAvg Strategy.
*   **Tactical Mapping**: MITRE ATT&CK for Industrial Control Systems (ICS).
*   **Threat Categorization**: STRIDE (Spoofing, Tampering, Repudiation, ID, Denial of Service, Elevation of Privilege).

