import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv("/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv")
print(f"Data shape: {data.shape}")

X = data.drop('dGmix', axis=1)
y = data['dGmix']

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
if categorical_columns:
    X = X.drop(columns=categorical_columns)
    print(f"Dropped categorical columns: {categorical_columns}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):

        x = x.unsqueeze(2)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)

        out = out[:, -1, :]

        out = self.fc(out)
        return out

input_size = X_train.shape[1]
hidden_size = 8
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size).to(device)
print(f"Model created with hidden size: {hidden_size}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
losses = []

print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs.squeeze(), y_train_tensor.to(device))

    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor.to(device)).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_test.values, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.values, y_pred)
    r2 = r2_score(y_test.values, y_pred)

print(f'Test MSE: {mse:.4f}')
print(f'Test RMSE: {rmse:.4f}')
print(f'Test MAE: {mae:.4f}')
print(f'Test R²: {r2:.4f}')

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'Actual vs Predicted (MAE: {mae:.4f}, R²: {r2:.4f})')
plt.annotate(f'MAE: {mae:.4f}\nR²: {r2:.4f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
plt.grid(True, alpha=0.3)
plt.savefig('actual_vs_predicted_test.png', dpi=300)
print("Saved actual vs predicted plot to 'actual_vs_predicted_test.png'")

n_bootstrap = 100
predictions = []

print("Generating bootstrap confidence intervals...")
for i in range(n_bootstrap):

    indices = np.random.choice(len(X_test_scaled), len(X_test_scaled), replace=True)
    X_bootstrap = X_test_tensor[indices]
    
    with torch.no_grad():
        pred = model(X_bootstrap.to(device)).cpu().numpy()
        pred = y_scaler.inverse_transform(pred)
        predictions.append(pred)
    
    if (i+1) % 20 == 0:
        print(f"Bootstrap iteration {i+1}/{n_bootstrap}")

predictions_array = np.array(predictions).squeeze()

mean_prediction = np.mean(predictions_array, axis=0)
lower_bound = np.percentile(predictions_array, 5, axis=0)
upper_bound = np.percentile(predictions_array, 95, axis=0)

sorted_indices = np.argsort(y_test.values.flatten())
sorted_actual = y_test.values.flatten()[sorted_indices]
sorted_mean = mean_prediction[sorted_indices]
sorted_lower = lower_bound[sorted_indices]
sorted_upper = upper_bound[sorted_indices]

plt.figure(figsize=(12, 6))
plt.fill_between(range(len(sorted_actual)), sorted_lower, sorted_upper, 
                 alpha=0.3, label='90% Confidence Interval')
plt.plot(range(len(sorted_actual)), sorted_mean, 'b-', label='Mean Prediction')
plt.plot(range(len(sorted_actual)), sorted_actual, 'ro', markersize=3, label='Actual Values')
plt.xlabel('Sorted Test Sample Index')
plt.ylabel('dGmix Value')
plt.title(f'Prediction with 90% Confidence Intervals (MAE: {mae:.4f}, R²: {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hybrid-confidence_intervals.png', dpi=300)
print("Saved confidence intervals plot to 'hybrid-confidence_intervals.png'")

print("RNN implementation completed!")
