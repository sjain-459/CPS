import gc
import torch
import flwr as fl
from collections import OrderedDict

from models import get_model
from local_training import train_model_dp, evaluate_model
from config import Config

class CPSClient(fl.client.NumPyClient):
    def __init__(self, stage_id, train_loader, test_loader, features):
        self.stage_id = stage_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.features = features
        self.model = get_model()

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Apply global weights
        self.set_parameters(parameters)
        
        # Train locally (with DP)
        self.model, epsilon = train_model_dp(self.model, self.train_loader, epochs=Config.EPOCHS_PER_ROUND)
        
        # Return local weights
        num_examples = len(self.train_loader.dataset)
        metrics = {"epsilon": epsilon} if epsilon > 0 else {}
        
        # Free memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, _ = evaluate_model(self.model, self.test_loader, self.features)
        num_examples = len(self.test_loader.dataset)
        
        gc.collect()
        return float(loss), num_examples, {"val_mse": float(loss)}

def client_fn(context: fl.common.Context) -> fl.client.Client:
    """
    Flower Client Builder. Context is passed in newer flower versions.
    We retrieve the node_id to assign it as stage_id.
    """
    node_id = context.node_config['partition-id']
    from data_pipeline import get_stage_dataloaders
    train_loader, test_loader, _, _, features = get_stage_dataloaders(node_id)
    return CPSClient(node_id, train_loader, test_loader, features).to_client()
