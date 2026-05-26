import torch
from model import MarketLSTM

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AIPipeline:
    def __init__(self, model_path="model.pth"):
        self.device = DEVICE
        # Initialize model with input_size=1 (for Log Returns)
        self.model = MarketLSTM(input_size=1).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"AI Pipeline: Loaded model from {model_path}")
        except FileNotFoundError:
            print("AI Pipeline: WARNING - model.pth not found. Using untrained weights.")

    def predict(self, log_returns):
        """
        Takes a list of log returns and returns a probability [0.0, 1.0].
        
        Args:
            log_returns (list or np.array): The historical log return sequence.
            
        Returns:
            float: Probability of upward movement.
        """
        # 1. Convert to tensor
        # 2. Add batch dimension (0) and feature dimension (-1)
        # Resulting shape: (1, seq_len, 1)
        features = torch.tensor(log_returns, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            # Get the model output (probability)
            prediction = self.model(features)
            
        return prediction.item()

# If you still have an 'encode' function for legacy reasons, 
# you can keep it empty or remove it as it is no longer used in the new pipeline.