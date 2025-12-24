"""
PyTorch Neural Network Training
================================

Train feedforward neural networks for NFL moneyline predictions.
Uses RTX 4090 GPU for accelerated training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)

from config import *
from data_loader import NFLDataLoader

print("="*120)
print("PYTORCH NEURAL NETWORK TRAINING")
print("="*120)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

# Load data
print("\n" + "="*120)
print("LOADING DATA")
print("="*120)
loader = NFLDataLoader(use_selected_features=True)
data = loader.load_and_prepare()

# Convert to PyTorch tensors (use scaled data for NN)
X_train = torch.FloatTensor(data['train']['X_scaled']).to(device)
y_train = torch.FloatTensor(data['train']['y'].values).to(device)
X_val = torch.FloatTensor(data['val']['X_scaled']).to(device)
y_val = torch.FloatTensor(data['val']['y'].values).to(device)
X_test = torch.FloatTensor(data['test']['X_scaled']).to(device)
y_test = torch.FloatTensor(data['test']['y'].values).to(device)

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=NN_PARAMS['batch_size'], shuffle=True)

print(f"\n✅ Data loaded to {device}")
print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")

# Define neural network
class NFLPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(NFLPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()

# Initialize model
input_dim = X_train.shape[1]
model = NFLPredictor(
    input_dim=input_dim,
    hidden_dims=NN_PARAMS['hidden_dims'],
    dropout=NN_PARAMS['dropout']
).to(device)

print(f"\n📊 Model Architecture:")
print(model)
print(f"\n   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=NN_PARAMS['learning_rate'])

# Training loop
print("\n" + "="*120)
print("TRAINING")
print("="*120)

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(NN_PARAMS['epochs']):
    # Training
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{NN_PARAMS['epochs']}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), MODELS_DIR / 'pytorch_nn_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= NN_PARAMS['early_stopping_patience']:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ Training complete!")
print(f"   Best validation loss: {best_val_loss:.4f}")

# Load best model for evaluation
model.load_state_dict(torch.load(MODELS_DIR / 'pytorch_nn_best.pth'))
model.eval()

# Evaluation function
def evaluate(X, y, split_name):
    with torch.no_grad():
        y_pred_proba = model(X).cpu().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_true = y.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
    
    print(f"\n{split_name} Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    return metrics

# Evaluate on all splits
print("\n" + "="*120)
print("EVALUATION")
print("="*120)
train_metrics = evaluate(X_train, y_train, "Train")
val_metrics = evaluate(X_val, y_val, "Validation")
test_metrics = evaluate(X_test, y_test, "Test")

# Save results
results = {
    'model_name': 'PyTorch_NN',
    'architecture': NN_PARAMS['hidden_dims'],
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'test_metrics': test_metrics,
    'best_val_loss': best_val_loss,
    'total_epochs': epoch + 1,
    'timestamp': datetime.now().isoformat()
}

with open(RESULTS_DIR / 'pytorch_nn_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {RESULTS_DIR / 'pytorch_nn_results.json'}")

