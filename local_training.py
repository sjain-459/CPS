import torch
import torch.nn as nn
from opacus import PrivacyEngine
import numpy as np
from config import Config

def train_model_dp(model, train_loader, epochs=Config.EPOCHS_PER_ROUND):
    """
    Trains the autoencoder locally using Differential Privacy constraints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none') # important for Opacus individual gradients

    # Wrap model with Differential Privacy Engine
    privacy_engine = PrivacyEngine()
    
    if Config.DP_ENABLED:
        try:
            model, optimizer, dp_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=1.2, # Adds Gaussian noise
                max_grad_norm=Config.MAX_GRAD_NORM,
            )
        except Exception as e:
            print(f"[Warning] DP wrapping failed, proceeding without DP: {e}")
            dp_loader = train_loader
    else:
        dp_loader = train_loader

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, _ in dp_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            
            reconstructed = model(X_batch)
            loss_per_sample = criterion(reconstructed, X_batch).mean(dim=[1, 2])
            loss = loss_per_sample.mean()

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

    # Get consumed privacy budget
    epsilon = -1.0
    if Config.DP_ENABLED and hasattr(privacy_engine, 'get_epsilon'):
        try:
            epsilon = privacy_engine.get_epsilon(delta=Config.TARGET_DELTA)
        except:
            pass

    return model, epsilon

def evaluate_model(model, test_loader, features_list):
    """
    Evaluates reconstruction error, triggers anomalies based on threshold.
    Returns: avg_loss, raw_reconstruction_errors (batch x seq_len x features)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    criterion = nn.MSELoss(reduction='none')
    total_loss = 0.0
    all_errors = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            reconstructed = model(X_batch)
            
            # Error matrix: Batch x Seq X Features
            error = criterion(reconstructed, X_batch)
            total_loss += error.mean().item()
            all_errors.append(error.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return avg_loss, np.concatenate(all_errors, axis=0) if all_errors else np.array([])
