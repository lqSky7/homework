import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from datetime import datetime
import sys
from pathlib import Path

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"linear_nn_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('linear_nn_trainer')

# Find the root directory of the project
def find_project_root():
    current_path = Path(os.path.abspath(__file__))
    while current_path.name != 'qnn_fnl' and current_path.parent != current_path:
        current_path = current_path.parent
    if current_path.name != 'qnn_fnl':
        return None
    return current_path

# Get project root and data path
project_root = find_project_root()
if not project_root:
    logger.error("Could not find project root directory.")
    sys.exit(1)
    
data_path = project_root / "data_filtered-1.csv"

# Load the dataset
logger.info(f"Loading dataset from {data_path}")
try:
    data = pd.read_csv(data_path)
    logger.info(f"Dataset loaded successfully: {data.shape}")
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    if not os.path.exists(data_path):
        logger.error(f"File does not exist: {data_path}")
        alternative_path = os.path.join(project_root, "data_filtered-1.csv")
        logger.info(f"Trying alternative path: {alternative_path}")
        try:
            data = pd.read_csv(alternative_path)
            logger.info(f"Dataset loaded successfully from alternative path: {data.shape}")
        except:
            logger.error("Failed to load dataset from alternative path")
            sys.exit(1)

# Data preprocessing
logger.info("Preprocessing data...")

# Identify target column and features
target_col = "dGmix"
X = data.drop(columns=[target_col])
y = data[target_col]

# Identify numerical and categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

logger.info(f"Numerical columns: {len(numerical_cols)}")
logger.info(f"Categorical columns: {len(categorical_cols)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Fit and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Scale the target variable
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Convert to PyTorch tensors
# Check if X_train_processed is a sparse matrix and convert to dense array if needed
if hasattr(X_train_processed, 'toarray'):
    X_train_processed = X_train_processed.toarray()
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.FloatTensor(y_train_scaled).view(-1, 1)

# Check if X_test_processed is a sparse matrix and convert to dense array if needed
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()
X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test_scaled).view(-1, 1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Linear Neural Network model
class LinearNN(nn.Module):
    def __init__(self, input_size):
        super(LinearNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Initialize the model
input_size = X_train_processed.shape[1]
model = LinearNN(input_size)
logger.info(f"Model created with input size: {input_size}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Training the model
num_epochs = 100
train_losses = []
test_losses = []

logger.info(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate average training loss
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
    
    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    
    # Update learning rate scheduler
    scheduler.step(test_loss)
    
    # Print progress
    if (epoch+1) % 10 == 0:
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
    # Early stopping condition
    if epoch > 20 and all(test_losses[-5] < test_loss for test_loss in test_losses[-4:]):
        logger.info("Early stopping triggered - validation loss not improving")
        break

logger.info("Training completed")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy().flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_test.values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

logger.info(f'Test MSE: {mse:.4f}')
logger.info(f'Test RMSE: {rmse:.4f}')
logger.info(f'Test MAE: {mae:.4f}')
logger.info(f'Test R²: {r2:.4f}')

# Create visualizations
logger.info("Creating visualizations...")

# 1. Learning Curve Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curves_linear.png', dpi=300)
logger.info("Learning curves plot saved to 'learning_curves_linear.png'")

# 2. Actual vs Predicted Plot
plt.figure(figsize=(10, 8))
plt.scatter(y_true, y_pred, alpha=0.5)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'Actual vs Predicted (MAE: {mae:.4f}, R²: {r2:.4f})')
plt.annotate(f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted_linear.png', dpi=300)
logger.info("Actual vs predicted plot saved to 'actual_vs_predicted_linear.png'")

# 3. Error Distribution Plot
errors = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7, color='blue')
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (RMSE: {rmse:.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_distribution_linear.png', dpi=300)
logger.info("Error distribution plot saved to 'error_distribution_linear.png'")

# 4. Generate bootstrap confidence intervals
n_bootstrap = 100
logger.info(f"Generating bootstrap confidence intervals with {n_bootstrap} samples...")

predictions = []
for _ in range(n_bootstrap):
    # Sample with replacement
    indices = np.random.choice(len(y_test), len(y_test), replace=True)
    X_bootstrap = X_test_tensor[indices]
    
    with torch.no_grad():
        pred = model(X_bootstrap).numpy().flatten()
        pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        predictions.append(pred)

predictions_array = np.array(predictions)
mean_prediction = np.mean(predictions_array, axis=0)
lower_bound = np.percentile(predictions_array, 5, axis=0)
upper_bound = np.percentile(predictions_array, 95, axis=0)

# Sort by actual values for better visualization
sorted_indices = np.argsort(y_true)
sorted_actual = y_true[sorted_indices]
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
plt.title(f'Predictions with 90% Confidence Intervals (MAE: {mae:.4f}, R²: {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('confidence_intervals_linear.png', dpi=300)
logger.info("Confidence intervals plot saved to 'confidence_intervals_linear.png'")

logger.info("Linear Neural Network implementation completed!")
