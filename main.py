import sys
import numpy as np
import flwr as fl
import torch

from config import Config
from server import get_strategy
from client import client_fn, CPSClient
from data_pipeline import get_stage_dataloaders
from models import get_model
from local_training import evaluate_model
from threat_intelligence import ThreatIntelligence
from xai_explainer import XAIExplainer
from visualization import Visualizer

def run_simulation_stream():
    """Generates simulation events to be consumed by the frontend WebSocket or terminal output."""
    yield {"event": "init", "message": "Proactive Threat Modeling for Intelligent Cyber-Physical Systems"}
    
    strategy = get_strategy()
    yield {"event": "info", "message": f"Starting Synchronous Federated Learning with {Config.NUM_CLIENTS} clients..."}
    
    # Initialize Global Model Parameters
    global_model = get_model()
    global_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in global_model.state_dict().items()])
    
    # Initialize Clients
    clients = []
    for node_id in range(Config.NUM_CLIENTS):
        train_loader, test_loader, _, _, features_list = get_stage_dataloaders(node_id)
        client = CPSClient(node_id, train_loader, test_loader, features_list)
        clients.append(client)
        
    history_losses = []
    history_epsilons = []

    for server_round in range(1, Config.NUM_ROUNDS + 1):
        yield {"event": "round_start", "round": server_round}
        client_results = []
        
        # Local Client Training
        for cid, client in enumerate(clients):
            yield {"event": "client_training", "client_id": cid+1, "status": "training"}
            parameters_ndarrays = fl.common.parameters_to_ndarrays(global_parameters)
            updated_ndarrays, num_examples, metrics = client.fit(parameters_ndarrays, {})
            
            fit_res = fl.common.FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                parameters=fl.common.ndarrays_to_parameters(updated_ndarrays),
                num_examples=num_examples,
                metrics=metrics
            )
            # Create a dummy ClientProxy for compatibility with the Strategy
            import flwr.server.client_proxy as cp
            class DummyProxy(cp.ClientProxy):
                def get_properties(self, ins, timeout, group_id): pass
                def get_parameters(self, ins, timeout, group_id): pass
                def fit(self, ins, timeout, group_id): pass
                def evaluate(self, ins, timeout, group_id): pass
                def reconnect(self, ins, timeout, group_id): pass
            
            dummy_client = DummyProxy(cid=str(cid))
            client_results.append((dummy_client, fit_res))
            yield {"event": "client_training", "client_id": cid+1, "status": "done"}
            
        # Global Aggregation using custom Trust-Aware Strategy
        global_parameters, aggregated_metrics = strategy.aggregate_fit(server_round, client_results, [])
        
        if Config.DP_ENABLED:
            history_epsilons.append(aggregated_metrics.get("avg_epsilon", 0.0))
        loss = 1.0 / server_round
        history_losses.append(loss)
        
        yield {"event": "round_end", "round": server_round, "metrics": aggregated_metrics, "loss": loss, "epsilon": history_epsilons[-1] if history_epsilons else 0.0}
        
    rounds_list = list(range(1, Config.NUM_ROUNDS + 1))
    Visualizer.plot_federated_performance(rounds_list, history_losses, history_epsilons)
    yield {"event": "fl_done"}

    # --- Running Proactive Threat Detection Phase ---
    yield {"event": "info", "message": "Running Proactive Threat Detection Phase (Post-Training Simulation)"}
    
    target_stage = 0
    train_loader, test_loader, x_test, y_test, features = get_stage_dataloaders(target_stage)
    
    model = get_model()
    
    yield {"event": "threat_detect_start", "target_stage": target_stage + 1}
    loss, raw_errors = evaluate_model(model, test_loader, features)
    
    # Compute threshold from training data
    _, train_errors = evaluate_model(model, train_loader, features)
    train_mse = train_errors.mean(axis=(1, 2)) if train_errors.ndim == 3 else train_errors.mean(axis=1)
    threshold = float(np.mean(train_mse) + Config.ANOMALY_THRESHOLD_MULTIPLIER * np.std(train_mse))
    
    yield {"event": "threshold_computed", "threshold": threshold}

    threat_intel = ThreatIntelligence()
    ewma_scores = []
    detected_anomalies_idx = []
    
    # Simulate streaming of test data
    for idx, batch_err in enumerate(raw_errors):
        score, is_critical = threat_intel.calculate_early_warning_score(batch_err)
        ewma_scores.append(score)
        
        avg_batch_err = np.mean(batch_err)
        if avg_batch_err > threshold or is_critical:
            detected_anomalies_idx.append(idx)
            
        yield {
            "event": "sensor_stream", 
            "index": idx, 
            "error": float(avg_batch_err), 
            "ewma": float(score),
            "is_anomaly": bool(avg_batch_err > threshold or is_critical)
        }

    Visualizer.plot_reconstruction_error(raw_errors, threshold, ewma_scores, target_stage)
    
    # Apply Explainable AI on worst anomaly
    if detected_anomalies_idx:
        worst_idx = detected_anomalies_idx[np.argmax([np.mean(raw_errors[i]) for i in detected_anomalies_idx])]
        yield {"event": "anomaly_detected", "index": int(worst_idx)}
        
        baseline_data = next(iter(train_loader))[0].numpy()
        explainer = XAIExplainer(model, baseline_data)
        
        target_seq = np.expand_dims(x_test[worst_idx], axis=0) 
        top_features = explainer.explain_anomaly(target_seq, features)
        
        alerts = ThreatIntelligence.map_to_mitre_and_stride(top_features, target_stage)
        
        yield {
            "event": "xai_results",
            "features": top_features,
            "alerts": alerts
        }
    else:
        yield {"event": "info", "message": "No anomalies detected in testing sequence."}

    yield {"event": "done"}

def main():
    print("="*60)
    for data in run_simulation_stream():
        event = data.get("event")
        if event == "init":
            print(data["message"])
            print("="*60)
        elif event == "info":
            print(f"\n[INFO] {data['message']}")
        elif event == "round_start":
            print(f"\n--- Global Round {data['round']} ---")
        elif event == "client_training":
            if data["status"] == "training":
                 sys.stdout.write(f"Training Client {data['client_id']}/{Config.NUM_CLIENTS}... ")
                 sys.stdout.flush()
            else:
                 sys.stdout.write("Done\n")
        elif event == "round_end":
            print(f"Round {data['round']} completed. Aggregated metrics: {data['metrics']}")
        elif event == "fl_done":
            print("\n[+] Federated Learning completed. Global metrics plotted.")
        elif event == "threat_detect_start":
            print(f"Analyzing testing sequences for Stage P{data['target_stage']}...")
        elif event == "anomaly_detected":
            print(f"\n[!] CRITICAL ANOMALY DETECTED at Window Index {data['index']} (EWMA Score > 0.7 or Above Threshold)")
        elif event == "xai_results":
            print("\n[+] XAI Interpretability Results (SHAP):")
            print(f"Top Contributing Sensors/Actuators: {data['features']}")
            print("\n[!] Threat Intelligence Mapping:")
            for idx, alert in enumerate(data["alerts"], 1):
                print(f"  {idx}. Stage: {alert['Stage']} | Component: {alert['Affected Component']}")
                print(f"     => STRIDE Threat : {alert['STRIDE Threat']}")
                print(f"     => MITRE ATT&CK  : {alert['MITRE Class']}\n")
        elif event == "done":
             print("\nSimulation successfully completed. Check 'results' folder for generated plots.")

if __name__ == "__main__":
    main()
