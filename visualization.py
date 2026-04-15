import matplotlib.pyplot as plt
import os
from config import Config

class Visualizer:
    @staticmethod
    def plot_reconstruction_error(errors, threshold, ewma_scores, stage_id):
        plt.figure(figsize=(12, 6))
        
        # Plot raw errors (averaged across features for simplicity)
        avg_errors = errors.mean(axis=(1, 2)) if errors.ndim == 3 else errors
        
        plt.plot(avg_errors, label='Reconstruction Error', alpha=0.6, color='blue')
        plt.plot(ewma_scores, label='Early Warning Score (EWMA)', color='red', linewidth=2)
        plt.axhline(y=threshold, color='orange', linestyle='--', label='Anomaly Threshold')
        plt.axhline(y=Config.EARLY_WARNING_CRITICAL_SCORE, color='darkred', linestyle='-.', label='Critical Warning Level')
        
        plt.title(f'Proactive Threat Detection - Stage P{stage_id + 1}')
        plt.xlabel('Time Windows')
        plt.ylabel('Error Score')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(Config.RESULTS_DIR, f'stage_{stage_id+1}_anomaly_plot.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    @staticmethod
    def plot_federated_performance(rounds, losses, epsilons):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Federated Round')
        ax1.set_ylabel('Global Eval Loss', color=color)
        ax1.plot(rounds, losses, marker='o', color=color, label='Eval Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        if len(epsilons) > 0 and Config.DP_ENABLED:
            ax2 = ax1.twinx()  
            color = 'tab:red'
            ax2.set_ylabel('Privacy Budget ($\epsilon$)', color=color)  
            ax2.plot(rounds, epsilons, marker='s', linestyle='--', color=color, label='Epsilon')
            ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout() 
        plt.title('Federated Learning Performance & Privacy Cost')
        save_path = os.path.join(Config.RESULTS_DIR, 'federated_metrics.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path
