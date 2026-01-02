"""
Task 8A.2: Train PyTorch Neural Network with GPU Acceleration

Build and train a deep learning model for NFL prediction
"""

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*120)
print("TASK 8A.2: PYTORCH NEURAL NETWORK (GPU-ACCELERATED)")
print("="*120)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[0/8] GPU Status: {device}")
if torch.cuda.is_available():
    print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ CUDA Version: {torch.version.cuda}")
else:
    print(f"  ⚠️  No GPU available, using CPU")

# Load data
print(f"\n[1/8] Loading data...")
df = pd.read_parquet('../results/phase8_results/phase6_game_level_1999_2024.parquet')
print(f"  ✅ Loaded: {df.shape[0]:,} games × {df.shape[1]:,} columns")

# Load feature categorization
with open('../results/phase8_results/feature_categorization.json', 'r') as f:
    cat = json.load(f)

# Get pre-game features
pre_game_dict = cat['pre_game_features']
pre_game_features = []
for category, features in pre_game_dict.items():
    pre_game_features.extend(features)

# Add manually classified UNKNOWN features
unknown_pregame = [
    'OTLosses', 'losses', 'pointsAgainst', 'pointsFor', 'ties', 'winPercent', 
    'winPercentage', 'wins', 'losses_roll3', 'losses_roll5', 'losses_std',
    'winPercent_roll3', 'winPercent_roll5', 'winPercent_std',
    'wins_roll3', 'wins_roll5', 'wins_std',
    'scored_20plus', 'scored_30plus', 'streak_20plus', 'streak_30plus',
    'vsconf_OTLosses', 'vsconf_leagueWinPercent', 'vsconf_losses', 'vsconf_ties', 'vsconf_wins',
    'vsdiv_OTLosses', 'vsdiv_divisionLosses', 'vsdiv_divisionTies', 
    'vsdiv_divisionWinPercent', 'vsdiv_divisionWins', 'vsdiv_losses', 'vsdiv_ties', 'vsdiv_wins',
    'div_game', 'rest_advantage', 'opponent'
]
pre_game_features.extend(unknown_pregame)

# Create target
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# Split by season
print(f"\n[2/8] Splitting data...")
train = df[df['season'] <= 2019].copy()
val = df[(df['season'] >= 2020) & (df['season'] <= 2023)].copy()
test = df[df['season'] == 2024].copy()

print(f"  ✅ Train (1999-2019): {len(train):,} games")
print(f"  ✅ Val (2020-2023): {len(val):,} games")
print(f"  ✅ Test (2024): {len(test):,} games")

# Select pre-game features
pregame_cols = []
for feat in pre_game_features:
    home_feat = f'home_{feat}'
    away_feat = f'away_{feat}'
    if home_feat in df.columns:
        pregame_cols.append(home_feat)
    if away_feat in df.columns:
        pregame_cols.append(away_feat)

numeric_pregame = df[pregame_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"  ✅ Total pre-game features: {len(numeric_pregame)}")

# Prepare data
X_train = train[numeric_pregame].fillna(train[numeric_pregame].median())
X_val = val[numeric_pregame].fillna(train[numeric_pregame].median())
X_test = test[numeric_pregame].fillna(train[numeric_pregame].median())
y_train = train['home_win'].values
y_val = val['home_win'].values
y_test = test['home_win'].values

# Standardize features
print(f"\n[3/8] Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print(f"  ✅ Features standardized (mean=0, std=1)")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"  ✅ Data moved to {device}")

# Define Neural Network
print(f"\n[4/8] Defining neural network architecture...")

class NFLPredictor(nn.Module):
    def __init__(self, input_dim):
        super(NFLPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x

model = NFLPredictor(input_dim=len(numeric_pregame)).to(device)
print(f"  ✅ Architecture: Input({len(numeric_pregame)}) → 128 → 64 → 32 → Output(1)")
print(f"  ✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Define loss and optimizer
print(f"\n[5/8] Setting up training...")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"  ✅ Loss: Binary Cross-Entropy")
print(f"  ✅ Optimizer: Adam (lr=0.001)")
print(f"  ✅ Scheduler: ReduceLROnPlateau")

# Training loop
print(f"\n[6/8] Training neural network...")
num_epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0
training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = (outputs > 0.5).float()
        train_correct += (predictions == y_batch).sum().item()
        train_total += y_batch.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # Save history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_acc'].append(val_acc)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Train Acc={train_acc*100:.2f}%, Val Acc={val_acc*100:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'scaler': scaler
        }, '../models/pytorch_nn_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n  ⚠️  Early stopping triggered at epoch {epoch+1}")
            break

print(f"\n  ✅ Training complete!")
print(f"  ✅ Best validation loss: {best_val_loss:.4f}")

# Load best model
print(f"\n[7/8] Evaluating on test set...")
checkpoint = torch.load('../models/pytorch_nn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    test_outputs = model(X_test_tensor).squeeze()
    test_predictions = (test_outputs > 0.5).float().cpu().numpy()
    test_probs = test_outputs.cpu().numpy()

test_acc = accuracy_score(y_test, test_predictions)
print(f"  ✅ Test accuracy: {test_acc*100:.2f}%")

# Save final model and results
print(f"\n[8/8] Saving results...")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'input_features': numeric_pregame,
    'test_accuracy': test_acc
}, '../models/pytorch_nn.pth')
print(f"  ✅ Saved: ../models/pytorch_nn.pth")

# Save training history
with open('../results/phase8_results/pytorch_training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"  ✅ Saved: ../results/phase8_results/pytorch_training_history.json")

print(f"\n{'='*120}")
print("✅ PYTORCH NEURAL NETWORK TRAINING COMPLETE")
print(f"{'='*120}")
print(f"\n✅ Model trained with {len(training_history['train_loss'])} epochs")
print(f"✅ Test accuracy: {test_acc*100:.2f}%")
print(f"✅ Model saved to ../models/pytorch_nn.pth")
print(f"✅ Ready for Task 8A.3 (Ensemble Model)")
print(f"\n{'='*120}")

