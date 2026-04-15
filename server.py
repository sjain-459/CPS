import numpy as np
import flwr as fl
from logging import INFO
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar
from typing import List, Tuple, Dict, Optional
from config import Config

class TrustAwareFedAvg(fl.server.strategy.FedAvg):
    """
    Custom strategy extending FedAvg to exclude poisoned or highly 
    anomalous client updates (Trust-Aware FL).
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results:
            return None, {}
        
        # We need to extract the raw weights from FitRes.
        # Format: results is a list of tuples (client, fit_res)
        valid_results = []
        
        # For Trust computation, we compare client updates against each other 
        # or calculate the standard deviation of weight norms to drop outliers.
        # Here we drop clients whose update norm is far from the mean norm.
        
        weight_norms = []
        for client, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            current_norm = np.sqrt(sum(np.sum(w**2) for w in weights))
            weight_norms.append((client, fit_res, current_norm))
            
        norms = np.array([w[2] for w in weight_norms])
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Accept clients within 2 standard deviations
        trusted_clients_count = 0
        for client, fit_res, norm in weight_norms:
            if std_norm == 0 or abs(norm - mean_norm) <= 2 * std_norm:
                valid_results.append((client, fit_res))
                trusted_clients_count += 1
            else:
                log(INFO, f"Server round {server_round}: Excluded untrusted client due to anomalous update norm ({norm:.2f}).")
                
        # Fallback to pure FedAvg on valid clients
        log(INFO, f"Trust-Aware Aggregation: {trusted_clients_count}/{len(results)} clients accepted.")
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, valid_results, failures)
        
        # Track epsilon and custom metrics from the round
        avg_epsilon = 0.0
        if valid_results:
            epsilons = [res.metrics.get("epsilon", 0.0) for _, res in valid_results]
            avg_epsilon = sum(epsilons) / len(epsilons)
            
        metrics_aggregated = {"avg_epsilon": avg_epsilon, "trusted_clients": trusted_clients_count}
        return aggregated_parameters, metrics_aggregated

def get_strategy():
    return TrustAwareFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=Config.NUM_CLIENTS,
        min_evaluate_clients=Config.NUM_CLIENTS,
        min_available_clients=Config.NUM_CLIENTS,
    )
