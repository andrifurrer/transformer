import os
import sys
import psutil
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Add the parent directory (i.e. transformer, means parent directory of 'scripts' and 'notebooks') to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

# Import the function
from scripts.m1_functions import *
from scripts.m1_classes import *


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Load the data
df_filtered = data_loader(subject=10, action='sit')

# Initialize scalers for predictors and target
scaler_input = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

# Fit and transform predictors (pleth_4, pleth_5, pleth_6)
input_columns = ['pleth_4', 'pleth_5', 'pleth_6']
x_normalized = scaler_input.fit_transform(df_filtered[input_columns])

# Fit and transform target (ecg)
y_normalized = scaler_target.fit_transform(df_filtered[['ecg']])

# Convert to PyTorch tensors
x_data = torch.tensor(x_normalized, dtype=torch.float32)  # Shape: [samples, 3]
y_data = torch.tensor(y_normalized, dtype=torch.float32)  # Shape: [samples, 1]

# Reshape for sequence input
'''
change stepsize to 5, 10, 20?
sequences are shifted by 1 timestamp / sample per sequence! 
'''
sequence_length = 100  
sequence_step_size = 20
num_sequences = len(df_filtered) - sequence_length + 1
subset = 1

x_sequences = torch.stack([x_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 3]
y_sequences = torch.stack([y_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 1]

# Split ratio 
train_ratio = 0.8
train_size = int(train_ratio * x_sequences.size(0))  # Number of training samples
val_size = x_sequences.size(0) - train_size          # Number of validation samples

# Slicing of the ratio
X_train, X_val = x_sequences[:train_size], x_sequences[train_size:]
y_train, y_val = y_sequences[:train_size], y_sequences[train_size:]

# Print shapes for verification
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Model initialization 
d_model = 4  # Embedding dimension
input_dim = 3  # 3 PPG signals (red, green, IR)
output_dim = 1  # 1 ECG target per time step
nhead = 2  # Attention heads
num_layers = 1  # Number of transformer layers
batch_size = 8  # Batch size


model = TransformerTimeSeries(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers) 
output = model(x_sequences)

# Loss function: Mean Squared Error for regression tasks
loss_fn = nn.MSELoss()

# Optimizer: Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20  # Number of epochs to train

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    
    # Initialize running loss
    running_loss = 0.0

    # Iterate through the training data in batches
    for i in range(0, len(X_train), batch_size):
        # Get the current batch
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        predictions = model(batch_X)

        # Calculate loss (MSE between predicted ECG and actual ECG)
        loss = loss_fn(predictions, batch_y)

        # Backward pass (compute gradients)
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    # Calculate the average loss for the epoch
    avg_loss = running_loss / len(X_train)
    
    # Validation metrics (optional but useful)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_fn(val_predictions, y_val).item()
        val_rmse = torch.sqrt(torch.tensor(val_loss))
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f}")


# Save the model
torch.save(model.state_dict(), '../models/transformer_m1_ecg_model.pth')