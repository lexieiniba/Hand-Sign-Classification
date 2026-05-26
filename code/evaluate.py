import torch
import numpy as np
import json
from model import MarketLSTM
from torch.utils.data import DataLoader, Dataset

# Ensure we use the same settings as training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvaluationDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        X = torch.tensor(item["seq"], dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(item["target"], dtype=torch.float32)
        return X, Y

def evaluate():
    # 1. Load data and model
    dataset = EvaluationDataset("training_data.json")
    loader = DataLoader(dataset, batch_size=32)
    
    model = MarketLSTM(input_size=1).to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()

    correct = 0
    total = 0

    # 2. Loop through data to check accuracy
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            
            # Get probability (0.0 to 1.0)
            probs = model(X).squeeze()
            
            # Convert to binary prediction (if > 0.5, predicted 1, else 0)
            preds = (probs > 0.5).float()
            
            # Count correct guesses
            correct += (preds == Y).sum().item()
            total += Y.size(0)

    # 3. Report
    accuracy = (correct / total) * 100
    print(f"=============================")
    print(f"DIRECTION ACCURACY: {accuracy:.2f}%")
    print(f"=============================")

if __name__ == "__main__":
    evaluate()