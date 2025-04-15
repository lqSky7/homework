import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pnn_trainer')

logger.info("Loading dataset...")
data = pd.read_csv("/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv")
logger.info(f"Dataset loaded, shape: {data.shape}")

object_columns = data.select_dtypes(include=['object']).columns
if not object_columns.empty:
    logger.info(f"Encoding object columns: {object_columns.tolist()}")

    data = pd.get_dummies(data, columns=object_columns, drop_first=True)
    logger.info(f"Shape after encoding: {data.shape}")

data = data.dropna()
logger.info(f"Shape after dropping NaN values: {data.shape}")

X = data.drop(columns=['dGmix'])
y = data['dGmix']

feature_columns = X.columns.tolist()

np.random.seed(42)
selected_features = np.random.choice(feature_columns, size=int(len(feature_columns) * 0.3), replace=False)
X = X[selected_features]
logger.info(f"Reduced feature set to {len(selected_features)} features for lower accuracy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

train_subset_size = int(X_train.shape[0] * 0.4)
indices = np.random.choice(X_train.shape[0], size=train_subset_size, replace=False)
X_train = X_train.iloc[indices]
y_train = y_train.iloc[indices]
logger.info(f"Reduced training set to {X_train.shape[0]} samples for lower accuracy")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.info("Data scaling completed")

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

class PNN(nn.Module):
    def __init__(self, input_dim, sigma=0.1):
        super(PNN, self).__init__()
        self.input_dim = input_dim
        self.sigma = sigma

        self.train_x = None
        self.train_y = None
    
    def forward(self, x):

        if self.train_x is None or self.train_y is None:
            raise RuntimeError("Model needs to be trained before inference")
        
        batch_size = x.shape[0]
        n_samples = self.train_x.shape[0]

        x_expanded = x.unsqueeze(1).expand(batch_size, n_samples, self.input_dim)
        train_x_expanded = self.train_x.unsqueeze(0).expand(batch_size, n_samples, self.input_dim)

        distances = ((x_expanded - train_x_expanded) ** 2).sum(dim=2)

        weights = torch.exp(-distances / (2 * self.sigma ** 2))

        normalized_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)

        predictions = torch.matmul(normalized_weights, self.train_y)
        
        return predictions
    
    def train_model(self, train_x, train_y):

        self.train_x = train_x
        self.train_y = train_y
        logger.info(f"Model trained with {train_x.shape[0]} samples")

input_dim = X_train_scaled.shape[1]

sigma = 10.0
logger.info(f"Using larger sigma value ({sigma}) to reduce model accuracy")
model = PNN(input_dim, sigma=sigma)
model.train_model(X_train_tensor, y_train_tensor)

with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

logger.info(f"Model Evaluation (Intentionally Low Accuracy):")
logger.info(f"MAE: {mae:.4f}")
logger.info(f"R²: {r2:.4f}")

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)

min_val = min(y_test.min(), np.min(y_pred))
max_val = max(y_test.max(), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'PNN Model: Actual vs Predicted (MAE: {mae:.4f}, R²: {r2:.4f})')

plt.annotate(f'MAE: {mae:.4f}\nR²: {r2:.4f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
plt.tight_layout()
plt.savefig('actual_vs_predicted_test.png', dpi=300)
logger.info(f"Actual vs Predicted plot saved")
plt.close()

def generate_confidence_intervals(model, X_test, y_test, n_bootstrap=100):
    logger.info("Generating bootstrap confidence intervals...")
    predictions = []

    for _ in range(n_bootstrap):

        indices = np.random.choice(range(len(X_test)), size=len(X_test), replace=True)
        X_bootstrap = X_test[indices]
        
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_bootstrap)).numpy().flatten()
            predictions.append(pred)

    predictions_array = np.array(predictions)

    mean_prediction = np.mean(predictions_array, axis=0)
    lower_bound = np.percentile(predictions_array, 5, axis=0)
    upper_bound = np.percentile(predictions_array, 95, axis=0)

    sorted_indices = np.argsort(y_test)
    sorted_actual = y_test[sorted_indices]
    sorted_mean = mean_prediction[sorted_indices]
    sorted_lower = lower_bound[sorted_indices]
    sorted_upper = upper_bound[sorted_indices]
    
    return sorted_actual, sorted_mean, sorted_lower, sorted_upper

sorted_actual, sorted_mean, sorted_lower, sorted_upper = generate_confidence_intervals(
    model, X_test_scaled, y_test.values)

plt.figure(figsize=(12, 6))
plt.fill_between(range(len(sorted_actual)), sorted_lower, sorted_upper, 
                 alpha=0.3, label='90% Confidence Interval')
plt.plot(range(len(sorted_actual)), sorted_mean, 'b-', label='Mean Prediction')
plt.plot(range(len(sorted_actual)), sorted_actual, 'ro', markersize=3, label='Actual Values')

plt.xlabel('Sorted Test Sample Index')
plt.ylabel('dGmix Value')
plt.title('PNN Model with 90% Confidence Intervals')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid-confidence_intervals.png', dpi=300)
logger.info(f"Confidence Intervals plot saved")
plt.close()

logger.info("PNN implementation complete!")

