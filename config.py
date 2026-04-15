import os

# Project-wide constants and configurations
class Config:
    # Environment
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"
    
    # Simulation / Data specs
    NUM_STAGES = 6  # P1 to P6 in SWaT
    SEQ_LENGTH = 15 # Sliding window size for time series
    TRAIN_SAMPLES_PER_STAGE = 1000
    TEST_SAMPLES_PER_STAGE = 300
    ANOMALY_FRACTION = 0.1  # 10% of test data will be anomalous

    # Machine Learning / PyTorch
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS_PER_ROUND = 3
    NUM_FEATURES = 15  # Max Stage features padded
    
    # Federated Learning
    NUM_CLIENTS = 6
    NUM_ROUNDS = 5
    TRUST_THRESHOLD = 0.8  # Minimum cosine similarity for trust-aware aggregation

    # Differential Privacy (Opacus)
    DP_ENABLED = True
    TARGET_EPSILON = 50.0  # Moderate privacy budget for early simulation
    TARGET_DELTA = 1e-5
    MAX_GRAD_NORM = 1.0

    # Threat Intelligence / Proactive Detection
    ANOMALY_THRESHOLD_MULTIPLIER = 3.0  # mean + 3*std
    EARLY_WARNING_ALPHA = 0.3  # Exponential moving average smoothing factor
    EARLY_WARNING_CRITICAL_SCORE = 0.7  # Score triggering an alert

# Ensure directories exist
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
