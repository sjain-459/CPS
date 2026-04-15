import torch
import shap
import numpy as np

class XAIExplainer:
    def __init__(self, model, baseline_data):
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # We need a baseline dataset for Shapley values (e.g., normal training data)
        # Random sample of baseline to speed up computation
        if len(baseline_data) > 100:
            indices = np.random.choice(len(baseline_data), 100, replace=False)
            baseline_sample = baseline_data[indices]
        else:
            baseline_sample = baseline_data
            
        self.baseline_tensor = torch.tensor(baseline_sample, dtype=torch.float32).to(self.device).requires_grad_(True)
        
        # Wrapper to output scalar feature for SHAP
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                reconstructed = self.model(x)
                # Return the mean squared error per sequence (summing over sequence and features)
                # This explicitly yields a per-batch scalar so GradientExplainer can compute gradients.
                return ((reconstructed - x)**2).mean(dim=(1,2)).unsqueeze(-1)
                
        self.wrapped_model = ModelWrapper(self.model)
        
        # We use GradientExplainer for deep learning models
        self.explainer = shap.GradientExplainer(self.wrapped_model, self.baseline_tensor)

    def explain_anomaly(self, anomalous_sequence, features_list, top_k=2):
        """
        Explain which features contributed most to the anomaly in a specific sequence.
        anomalous_sequence: (1, Seq_Len, Features)
        """
        seq_tensor = torch.tensor(anomalous_sequence, dtype=torch.float32).to(self.device)
        
        # Shap values for 3D tensor
        shap_values = self.explainer.shap_values(seq_tensor)
        
        # Average shap values across time sequence 
        # (Assuming Shap provides values per feature across the sequence)
        
        if isinstance(shap_values, list): # happens if multiple outputs, we sum/mean them
            shap_array = np.mean(np.abs(shap_values[0]), axis=1) # Average over sequence
        else:
            shap_array = np.mean(np.abs(shap_values), axis=1)
            
        shap_flat = shap_array.flatten()
        
        top_indices = np.argsort(shap_flat)[-top_k:][::-1]
        top_features = [features_list[i] for i in top_indices]
        
        return top_features
