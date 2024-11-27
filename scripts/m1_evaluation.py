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


# Load model later for inference
# Model initialization 
d_model = 4  # Embedding dimension
input_dim = 3  # 3 PPG signals (red, green, IR)
output_dim = 1  # 1 ECG target per time step
nhead = 2  # Attention heads
num_layers = 1  # Number of transformer layers
batch_size = 8  # Batch size


model = TransformerTimeSeries(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers) # .to("cuda:0") 

model.load_state_dict(torch.load('../models/transformer_m1_ecg_model_s1_still.pth'))
model.eval()  # Set to evaluation mode
with torch.no_grad():
    val_predictions = model(X_val)
    val_loss = loss_fn(val_predictions, y_val).item()
    val_rmse = torch.sqrt(torch.tensor(val_loss))
    print(f"Memory usage: {psutil.virtual_memory().percent}%")

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