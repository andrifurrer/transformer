import os
import sys
import psutil
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.profiler

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

#S Set all tensores to GPU by default
if device == "cuda":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

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
x_data = torch.tensor(x_normalized, dtype=torch.float32) # Shape: [samples, 3] # .to("cuda:0")
y_data = torch.tensor(y_normalized, dtype=torch.float32)  # Shape: [samples, 1] # .to("cuda:0")

# Reshape for sequence input
#change stepsize to 5, 10, 20? sequences are shifted by 1 timestamp / sample per sequence! 

sequence_length = 100  
sequence_step_size = 1
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


model = TransformerTimeSeries(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device) # .to("cuda:0") 
#output = model(x_sequences) #.to("cuda:0")

# Loss function: Mean Squared Error for regression tasks
loss_fn = nn.MSELoss()

# Optimizer: Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5  # Number of epochs to train

# Clear any residual memory before training
torch.cuda.empty_cache()

# Observe memory usage

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    
    # Initialize running loss
    running_loss = 0.0


    # Iterate through the training data in batches
    for i in range(0, len(X_train), batch_size):
        # Get the current batch
        batch_X = X_train[i:i+batch_size].to(device)
        batch_y = y_train[i:i+batch_size].to(device)
        
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
    
    # Validation metrics with batching
    model.eval()  # Set model to evaluation mode

    total_val_loss = 0
    with torch.no_grad():
        for j in range(0, len(X_val), batch_size):
            # Get the current validation batch
            batch_X_val = X_val[j:j + batch_size].to(device)
            batch_y_val = y_val[j:j + batch_size].to(device)

            # Forward pass
            val_predictions = model(batch_X_val)

            # Calculate loss for this batch
            val_loss = loss_fn(val_predictions, batch_y_val)

            # Accumulate total validation loss
            total_val_loss += val_loss.item() * batch_X_val.size(0)  # Weighted by batch size, necessary and if so why not for X_batch?

    # Average validation loss over all samples
    avg_val_loss = total_val_loss / len(X_val) # Difference to X_batch?
    val_rmse = torch.sqrt(torch.tensor(avg_val_loss))

    # Clear any residual memory before start of new epoch
    torch.cuda.empty_cache()


    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f}")


# Save the model
torch.save(model.state_dict(), '../models/transformer_m1_ecg_model_s1_still.pth')

# Reverse transform predicted ECG
val_predictions=val_predictions.squeeze(-1)
print(val_predictions.size())
predictions_original_scale = scaler_target.inverse_transform(val_predictions.numpy())
print(predictions_original_scale.shape)

# Reverse transform input if needed
y_val = y_val.squeeze(-1)
print(y_val.size())
inputs_original_scale = scaler_target.inverse_transform(y_val.numpy())
print(inputs_original_scale.shape)

# Randomly select an index from the validation data
random_index = np.random.randint(0, len(val_predictions))

# Select the corresponding actual and predicted ECG signals
actual_ecg_random = y_val[random_index].numpy()  # Actual ECG signal
predicted_ecg_random = val_predictions[random_index].numpy()  # Predicted ECG signal

# Plot the actual and predicted ECG
plt.figure(figsize=(10, 5))
plt.plot(actual_ecg_random, label='Actual ECG')
plt.plot(predicted_ecg_random, label='Predicted ECG')
plt.title(f"ECG Prediction vs Actual (Sequence {random_index})")
plt.xlabel('Time Step')
plt.ylabel('ECG Signal')
plt.legend()

plt.savefig('../results/m1_random_sequence.png')
plt.show()

# Number of random sequences to plot
num_sequences = 5

# Create a plot
plt.figure(figsize=(10, 6))

for _ in range(num_sequences):
    random_index = np.random.randint(0, len(val_predictions))
    
    # Select the corresponding actual and predicted ECG signals
    actual_ecg_random = y_val[random_index].numpy()
    predicted_ecg_random = val_predictions[random_index].numpy()
    
    # Plot both actual and predicted ECG
    plt.plot(actual_ecg_random, label=f'Actual ECG {random_index}')
    plt.plot(predicted_ecg_random, label=f'Predicted ECG {random_index}', linestyle='dashed')

plt.title("ECG Predictions vs Actual for Random Sequences")
plt.xlabel('Time Step')
plt.ylabel('ECG Signal')
plt.legend()

plt.savefig('../results/m1_multiple_sequences.png')
plt.show()

# Calculate the average ECG for both actual and predicted
average_actual_ecg = np.mean(y_val.numpy(), axis=0)  # Average over all sequences
average_predicted_ecg = np.mean(val_predictions.numpy(), axis=0)  # Average over all predictions

# Plot the average ECG
plt.figure(figsize=(10, 5))
plt.plot(average_actual_ecg, label='Average Actual ECG')
plt.plot(average_predicted_ecg, label='Average Predicted ECG')
plt.title("Average ECG Prediction vs Actual")
plt.xlabel('Time Step')
plt.ylabel('ECG Signal')
plt.legend()

plt.savefig('../results/m1_averaged_sequences.png')
plt.show()