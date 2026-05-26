import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import MarketLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

# DATASET - Simplified to use pre-calculated log returns and binary target
class CandleDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # X: Log returns (shape: seq_len). Add dimension for feature size (seq_len, 1)
        X = torch.tensor(item["seq"], dtype=torch.float32).unsqueeze(-1)
        
        # Y: Binary target (0 or 1)
        Y = torch.tensor(item["target"], dtype=torch.float32)
        
        return X, Y

# TRAINING LOOP
def train():
    dataset = CandleDataset("training_data.json")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model (input_size=1 because we only use Log Returns)
    model = MarketLSTM(input_size=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Binary Cross Entropy loss is for Yes/No predictions
    loss_fn = nn.BCELoss()

    EPOCHS = 20
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for X, Y in loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            optimizer.zero_grad()
            # Predict
            pred = model(X).squeeze() 
            
            # Loss
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Saved model.pth")

if __name__ == "__main__":
    train()