import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from datetime import datetime
import io

np.seterr(divide='ignore', invalid='ignore')

# Check for available GPU devices
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU
    print("Using MPS (Metal Performance Shaders) device")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU device")

print(f"PyTorch version: {torch.__version__}")
print(f"Active device: {device}")

# Set up directories using environment variables with defaults
BASE_DIR = os.environ.get('APP_DIR', os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(BASE_DIR, 'data'))
LOG_DIR = os.environ.get('LOG_DIR', os.path.join(BASE_DIR, 'logs'))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
GRAPHS_DIR = os.environ.get('GRAPHS_DIR', os.path.join(BASE_DIR, 'graphs'))

# Ensure directories exist
for directory in [DATA_DIR, LOG_DIR, MODEL_DIR, GRAPHS_DIR]:
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "qnn_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hybrid_qnn_trainer')

# Use environment variable for dataset path with a default
DATASET_PATH = os.environ.get('DATASET_PATH', os.path.join(DATA_DIR, 'data_filtered-1.csv'))
logger.info(f"Loading dataset from: {DATASET_PATH}")
data = pd.read_csv(DATASET_PATH)
logger.info(f"Data shape: {data.shape}")
logger.info(f"Columns: {data.columns.tolist()}")
logger.info(f"Sample:\n {data.head()}")

logger.info("\nData types:")
logger.info(f"{data.dtypes}")

categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
logger.info(f"\nCategorical columns: {categorical_columns}")

X = data.drop('dGmix', axis=1)
y = data['dGmix']

if categorical_columns:
    X = X.drop(columns=categorical_columns)
    logger.info(f"After dropping categorical columns: {X.shape} features")

logger.info("Performing advanced feature engineering...")

# Advanced feature engineering with more transformations
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    logger.info("Adding polynomial features (degree=2) and interaction terms")
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    logger.info(f"Features shape after polynomial transformation: {X_poly.shape}")

    X_orig = pd.DataFrame(X)
    engineered_features = []
    
    # Log transforms
    for col in X_orig.columns:
        if (X_orig[col] > 0).all():
            engineered_features.append(np.log1p(X_orig[col]).values)
    
    # Exponential transforms
    for col in X_orig.columns:
        if not np.isinf(np.exp(X_orig[col] * 0.1)).any():
            engineered_features.append(np.exp(X_orig[col] * 0.1).values)
    
    # Trigonometric transforms
    for col in X_orig.columns:
        engineered_features.append(np.sin(X_orig[col]).values)
        engineered_features.append(np.cos(X_orig[col]).values)
    
    # Ratio features for important columns
    for i, col1 in enumerate(X_orig.columns):
        for col2 in X_orig.columns[i+1:]:
            if (X_orig[col2] != 0).all():
                ratio = X_orig[col1] / (X_orig[col2] + 1e-8)
                if not np.isinf(ratio).any():
                    engineered_features.append(ratio.values)
    
    if engineered_features:
        engineered_features = np.column_stack(engineered_features)
        X_combined = np.hstack((X_poly, engineered_features))
        logger.info(f"Features shape after adding engineered transforms: {X_combined.shape}")
        X = X_combined
    else:
        X = X_poly
else:
    logger.warning("No numeric columns found for polynomial features")

# Feature selection to reduce dimensionality and noise
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor

logger.info("Performing feature selection...")
selector = SelectFromModel(
    LGBMRegressor(n_estimators=100, importance_type='gain'),
    threshold='mean'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
logger.info(f"Initial train set: {X_train.shape}, Test set: {X_test.shape}")

selector.fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
logger.info(f"After feature selection: Train set: {X_train.shape}, Test set: {X_test.shape}")

# Using PowerTransformer for better normality in the data
from sklearn.preprocessing import PowerTransformer
feature_scaler = PowerTransformer(method='yeo-johnson')
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
logger.info("Data scaling completed with PowerTransformer")

y_scaler = PowerTransformer(method='yeo-johnson')
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
logger.info("Target scaling completed with PowerTransformer")

# Improved quantum circuit configuration
n_qubits = min(16, X_train_scaled.shape[1])  # Increase qubits for more expressivity
n_layers = 8  # Increase layers for more complex representations
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Improved encoding strategy
    inputs_padded = np.pad(inputs, (0, n_qubits - len(inputs) % n_qubits), mode='constant')
    
    # Amplitude encoding
    for i in range(n_qubits):
        qml.RY(inputs_padded[i % len(inputs_padded)] * np.pi, wires=i)
        qml.RZ(inputs_padded[i % len(inputs_padded)] * np.pi, wires=i)
    
    # Apply Hadamard to create superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # More expressive variational circuit
    for l in range(n_layers):
        # Rotation gates layer
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
            qml.U3(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        
        # Entanglement patterns (alternating between different strategies)
        if l % 3 == 0:
            # Linear entanglement
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[n_qubits-1, 0])  # Close the loop
        elif l % 3 == 1:
            # Long-range entanglement
            for i in range(n_qubits//2):
                qml.CNOT(wires=[i, (i + n_qubits//2) % n_qubits])
        else:
            # All-to-all entanglement (star pattern)
            for i in range(1, n_qubits):
                qml.CNOT(wires=[0, i])
        
        # Add CZ gates for additional entanglement
        if l % 2 == 0:
            for i in range(0, n_qubits-1, 2):
                qml.CZ(wires=[i, i+1])
    
    # More comprehensive measurement strategy
    expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    expectations += [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    expectations += [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    
    # Add some two-qubit observables
    for i in range(0, n_qubits-1, 4):
        expectations.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i+1)))
    
    return expectations

class AdvancedHybridModel(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers=8):
        super(AdvancedHybridModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Enhanced measurement output size
        self.q_output_size = n_qubits * 3 + n_qubits // 4
        
        # Deeper pre-processing neural network
        self.pre_net1 = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LayerNorm(512),
            nn.SiLU(),  # SiLU (Swish) activation
            nn.Dropout(0.3),
        )
        
        self.pre_net2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        
        self.pre_net3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        
        self.pre_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        # Initialize quantum weights with better initialization
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.02)
        
        # Improved post-processing neural network
        self.post_net1 = nn.Sequential(
            nn.Linear(self.q_output_size, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        
        self.post_net2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        
        self.post_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
        # Multiple skip connections for better gradient flow
        self.skip_connection1 = nn.Linear(n_features, 256)
        self.skip_connection2 = nn.Linear(n_features, 128)
    
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device  # Get the device from the input tensor
        
        # Ensure x is on the right device
        x = x.to(device)
        
        # Skip connections
        skip1 = self.skip_connection1(x)
        skip2 = self.skip_connection2(x)
        
        # Pre-processing
        x1 = self.pre_net1(x)
        x2 = self.pre_net2(x1) + skip1  # Skip connection
        x3 = self.pre_net3(x2) + skip2  # Skip connection
        x_pre = self.pre_out(x3)
        
        # Quantum circuit processing
        q_out = torch.zeros(batch_size, self.q_output_size, device=device)
        for i, inputs in enumerate(x_pre):
            q_result = quantum_circuit(inputs.detach().cpu().numpy(), 
                                     self.q_weights.detach().cpu().numpy())
            q_result_array = np.array(q_result)  # Convert to numpy array
            q_out[i] = torch.tensor(q_result_array, dtype=torch.float, device=device)
        
        # Post-processing with residual connections
        p1 = self.post_net1(q_out)
        p2 = self.post_net2(p1) + skip2  # Reuse skip connection
        out = self.post_out(p2)
        
        return out

# Convert data to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test_scaled).reshape(-1, 1).to(device)

# Create a single model instead of an ensemble of 3
n_models = 1  # Changed from 3 to 1
models = [AdvancedHybridModel(X_train_scaled.shape[1], n_qubits, n_layers).to(device) for _ in range(n_models)]
logger.info(f"Initialized {n_models} model on {device}")

# More robust loss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, true):
        mse = self.mse(pred, true)
        loss = self.alpha * (1 - torch.exp(-mse)) ** self.gamma * mse
        return torch.mean(loss)

# Custom loss combining MSE and Huber
class CombinedLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=delta)
        
    def forward(self, y_pred, y_true):
        return 0.5 * self.mse_loss(y_pred, y_true) + 0.5 * self.huber(y_pred, y_true)

criterion = CombinedLoss(delta=1.0)
optimizers = [optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5) for model in models]
schedulers = [optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) for optimizer in optimizers]

tb_dir = os.path.join(LOG_DIR, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(tb_dir, exist_ok=True)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(tb_dir)

# Add k-means clustering for model validation
from sklearn.cluster import KMeans

# More epochs and better batch size
epochs = int(os.environ.get('EPOCHS', '5'))
batch_size = int(os.environ.get('BATCH_SIZE', '32'))
best_model_paths = [os.path.join(MODEL_DIR, f"best_hybrid_model_{i}.pt") for i in range(n_models)]

logger.info(f"Starting training for {epochs} epochs with {n_models} model")

for model_idx, (model, optimizer, scheduler, best_model_path) in enumerate(zip(models, optimizers, schedulers, best_model_paths)):
    logger.info(f"Training model {model_idx+1}/{n_models}")
    
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Use stratified mini-batches
        indices = torch.randperm(len(X_train_tensor))
        
        for start_idx in range(0, len(indices), batch_size):
            idx = indices[start_idx:start_idx+batch_size]
            
            batch_X = X_train_tensor[idx]
            batch_y = y_train_tensor[idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Adjust learning rate with scheduler
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
        
        avg_epoch_loss = epoch_loss / (len(X_train_tensor) // batch_size + 1)
        losses.append(avg_epoch_loss)
        val_losses.append(val_loss)
        
        writer.add_scalar(f'Loss/train_model_{model_idx}', avg_epoch_loss, epoch)
        writer.add_scalar(f'Loss/validation_model_{model_idx}', val_loss, epoch)
        writer.add_scalar(f'LearningRate/model_{model_idx}', optimizer.param_groups[0]['lr'], epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Model {model_idx+1} - Epoch {epoch+1}: New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs for model {model_idx+1}")
                break
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info(f'Model {model_idx+1} - Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

# Load the best models
for model_idx, (model, best_model_path) in enumerate(zip(models, best_model_paths)):
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)  # Ensure model is on the right device
    logger.info(f"Loaded best model {model_idx+1} from {best_model_path} to {device}")

# Since we only have one model, we can simplify the ensemble_predict function
def predict(model, X):
    model.eval()
    with torch.no_grad():
        # Make sure X is on the same device as the model
        device = next(model.parameters()).device
        X = X.to(device)
        pred = model(X).cpu().numpy()
    return pred

# Generate predictions using the single model
model = models[0]  # Get the single model
y_pred_scaled = predict(model, X_test_tensor)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = mean_squared_error(y_test.values, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test.values, y_pred)
r2 = r2_score(y_test.values, y_pred)

logger.info(f'Test MSE: {mse:.4f}')
logger.info(f'Test RMSE: {rmse:.4f}')
logger.info(f'Test MAE: {mae:.4f}')
logger.info(f'Test R²: {r2:.4f}')

# Create visualizations
import glob

logger.info(f"Graphs will be saved to: {GRAPHS_DIR}")

# Create learning curve visualization (simplified for one model)
fig1 = plt.figure(figsize=(12,8))
plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.savefig(os.path.join(GRAPHS_DIR, "learning_curves.png"), dpi=300)
plt.close(fig1)

# Create prediction visualization
fig2 = plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'Actual vs Predicted (Test), R²: {r2:.4f}, MAE: {mae:.4f}')
plt.savefig(os.path.join(GRAPHS_DIR, "actual_vs_predicted_test.png"), dpi=300)
plt.close(fig2)

# Create error distribution
error_vals = y_pred.flatten() - y_test.values.flatten()
fig3 = plt.figure(figsize=(12,8))
plt.hist(error_vals, bins=30, alpha=0.7)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution, RMSE: {rmse:.4f}, MAE: {mae:.4f}')
plt.savefig(os.path.join(GRAPHS_DIR, "error_distribution.png"), dpi=300)
plt.close(fig3)

# Create residual plot
fig4 = plt.figure(figsize=(12,8))
plt.scatter(y_pred.flatten(), error_vals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual (Actual-Predicted)')
plt.title('Residual Plot')
plt.savefig(os.path.join(GRAPHS_DIR, "residual_plot.png"), dpi=300)
plt.close(fig4)

# Feature importance analysis
fig5 = plt.figure(figsize=(14,8))
if hasattr(selector, 'get_support'):
    importances = selector.estimator_.feature_importances_
    selected_indices = selector.get_support(indices=True)
    plt.bar(range(len(selected_indices)), importances[selected_indices])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(os.path.join(GRAPHS_DIR, "feature_importance.png"), dpi=300)
plt.close(fig5)

# After training is complete and predictions generated, perform k-means analysis
logger.info("Performing k-means clustering analysis to validate model performance...")

# Determine optimal number of clusters using silhouette score
from sklearn.metrics import silhouette_score

silhouette_scores = []
n_clusters_range = range(2, 11)  # Test from 2 to 10 clusters

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_test_scaled)
    if len(set(cluster_labels)) > 1:  # Ensure at least 2 clusters were created
        silhouette_avg = silhouette_score(X_test_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logger.info(f"For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}")
    else:
        silhouette_scores.append(-1)

# Choose optimal number of clusters
optimal_n_clusters = n_clusters_range[silhouette_scores.index(max(silhouette_scores))]
logger.info(f"Optimal number of clusters: {optimal_n_clusters}")

# Apply k-means with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
test_clusters = kmeans.fit_predict(X_test_scaled)

# Calculate metrics for each cluster
logger.info("Analyzing model performance by cluster:")
cluster_metrics = []

for cluster_id in range(optimal_n_clusters):
    cluster_mask = test_clusters == cluster_id
    if np.sum(cluster_mask) > 0:  # Ensure cluster is not empty
        cluster_y_true = y_test.values[cluster_mask]
        cluster_y_pred = y_pred[cluster_mask]
        
        cluster_mse = mean_squared_error(cluster_y_true, cluster_y_pred)
        cluster_rmse = np.sqrt(cluster_mse)
        cluster_mae = mean_absolute_error(cluster_y_true, cluster_y_pred)
        cluster_r2 = r2_score(cluster_y_true, cluster_y_pred)
        
        cluster_size = np.sum(cluster_mask)
        cluster_metrics.append((cluster_id, cluster_size, cluster_r2, cluster_rmse, cluster_mae))
        
        logger.info(f"Cluster {cluster_id} (size: {cluster_size}): R² = {cluster_r2:.4f}, RMSE = {cluster_rmse:.4f}, MAE = {cluster_mae:.4f}")

# Calculate predicted values for training data
model = model.to(device)  # Ensure model is on the right device before prediction
y_train_pred_scaled = predict(model, X_train_tensor)
y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
train_r2 = r2_score(y_train.values, y_train_pred)

# Now that we have all analysis data, create and save the model info
model_info = {
    'model': model.cpu().state_dict(),  # Convert to CPU for saving
    'scaler_state': {
        'feature_scaler': feature_scaler,
        'target_scaler': y_scaler
    },
    'feature_selector': selector if hasattr(selector, 'get_support') else None,
    'metrics': {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_r2': train_r2
    },
    'model_params': {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'n_features': X_train_scaled.shape[1],
    },
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'cluster_analysis': {
        'n_clusters': optimal_n_clusters,
        'cluster_metrics': cluster_metrics,
        'silhouette_scores': silhouette_scores,
        'cluster_centers': kmeans.cluster_centers_.tolist()
    }
}

model_filename = os.environ.get('MODEL_FILENAME', 'advanced_hybrid_model.pt')
model_path = os.path.join(MODEL_DIR, model_filename)
torch.save(model_info, model_path)
logger.info(f"Model saved to '{model_path}'")

# Generate cluster-based visualizations

# 1. Create a comparison of actual vs predicted values colored by cluster
fig_cluster = plt.figure(figsize=(12, 10))
for cluster_id in range(optimal_n_clusters):
    cluster_mask = test_clusters == cluster_id
    if np.sum(cluster_mask) > 0:
        plt.scatter(
            y_test.values[cluster_mask], 
            y_pred[cluster_mask], 
            alpha=0.6, 
            label=f'Cluster {cluster_id}'
        )
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title(f'Actual vs Predicted by Cluster (Test), Overall R²: {r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "actual_vs_predicted_by_cluster.png"), dpi=300)
plt.close(fig_cluster)

# 2. Create box plots of errors by cluster
cluster_errors = []
cluster_ids = []
for cluster_id in range(optimal_n_clusters):
    cluster_mask = test_clusters == cluster_id
    if np.sum(cluster_mask) > 0:
        errors = y_test.values[cluster_mask] - y_pred[cluster_mask].flatten()
        cluster_errors.append(errors)
        cluster_ids.append(f'Cluster {cluster_id}')

fig_box = plt.figure(figsize=(14, 8))
plt.boxplot(cluster_errors, labels=cluster_ids)
plt.ylabel('Error (Actual - Predicted)')
plt.title('Distribution of Errors by Cluster')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "error_boxplot_by_cluster.png"), dpi=300)
plt.close(fig_box)

# 3. Create a bar chart of R² values by cluster
cluster_ids, cluster_sizes, cluster_r2s, cluster_rmses, cluster_maes = zip(*cluster_metrics)
cluster_labels = [f"C{cid}\n(n={size})" for cid, size in zip(cluster_ids, cluster_sizes)]

fig_r2 = plt.figure(figsize=(12, 8))
plt.bar(cluster_labels, cluster_r2s, alpha=0.7)
plt.axhline(r2, color='red', linestyle='--', label=f'Overall R²: {r2:.4f}')
plt.ylim(0, 1.05)
plt.xlabel('Cluster (with size)')
plt.ylabel('R² Score')
plt.title('R² Score by Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "r2_by_cluster.png"), dpi=300)
plt.close(fig_r2)

# 4. Perform PCA for visualization of clusters in 2D
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test_scaled)

fig_pca = plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=test_clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('PCA Visualization of Test Data Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "pca_clusters.png"), dpi=300)
plt.close(fig_pca)

# 5. Create a heatmap of prediction errors in PCA space
fig_heat = plt.figure(figsize=(12, 10))
errors = np.abs(y_test.values.flatten() - y_pred.flatten())
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=errors, cmap='YlOrRd', alpha=0.6)
plt.colorbar(scatter, label='Absolute Error')
plt.title('Absolute Prediction Errors in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "pca_error_heatmap.png"), dpi=300)
plt.close(fig_heat)

# 6. Plot R² vs train/test distribution
# Calculate predicted values for training data
y_train_pred_scaled = predict(model, X_train_tensor)
y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
train_r2 = r2_score(y_train.values, y_train_pred)

# Create Q-Q plot to check normality of errors
fig_qq = plt.figure(figsize=(10, 10))
from scipy import stats

# Sort the errors
sorted_errors = np.sort(error_vals)
# Get the theoretical quantiles
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_errors)))

plt.scatter(theoretical_quantiles, sorted_errors, alpha=0.6)
plt.plot(theoretical_quantiles, theoretical_quantiles, color='r', linestyle='--')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('Q-Q Plot of Prediction Errors')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "qq_plot.png"), dpi=300)
plt.close(fig_qq)

# 7. Learning curve with both train and test R²
fig_r2_curve = plt.figure(figsize=(12, 8))
plt.bar(['Training', 'Testing'], [train_r2, r2], alpha=0.7)
plt.ylim(0, 1.05)
plt.ylabel('R² Score')
plt.title(f'R² Score for Training vs Testing Data\nTrain: {train_r2:.4f}, Test: {r2:.4f}')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "train_test_r2_comparison.png"), dpi=300)
plt.close(fig_r2_curve)

# 8. Check for potential data leakage by plotting error distribution by data index
fig_idx_error = plt.figure(figsize=(14, 8))
plt.scatter(range(len(y_test)), error_vals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Data Index')
plt.ylabel('Prediction Error')
plt.title('Prediction Error by Data Index')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "error_by_index.png"), dpi=300)
plt.close(fig_idx_error)

# 9. Plot error histograms for each cluster
fig_err_hist, axs = plt.subplots(1, optimal_n_clusters, figsize=(16, 6), sharey=True)
if optimal_n_clusters == 1:
    axs = [axs]  # Convert to list for single cluster case
    
for cluster_id in range(optimal_n_clusters):
    cluster_mask = test_clusters == cluster_id
    if np.sum(cluster_mask) > 0:
        errors = y_test.values[cluster_mask] - y_pred[cluster_mask].flatten()
        axs[cluster_id].hist(errors, bins=20, alpha=0.7)
        axs[cluster_id].set_title(f'Cluster {cluster_id}')
        axs[cluster_id].axvline(0, color='red', linestyle='--')
        axs[cluster_id].grid(True, alpha=0.3)
        
fig_err_hist.suptitle('Error Distribution by Cluster')
fig_err_hist.text(0.5, 0.01, 'Error (Actual - Predicted)', ha='center')
plt.savefig(os.path.join(GRAPHS_DIR, "error_hist_by_cluster.png"), dpi=300)
plt.close(fig_err_hist)

# 10. Plot actual vs. predicted values with error represented by marker size
fig_err_size = plt.figure(figsize=(12, 10))
plt.scatter(y_test.values, y_pred.flatten(), s=np.abs(error_vals)*100+10, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual dGmix')
plt.ylabel('Predicted dGmix')
plt.title('Actual vs Predicted with Error as Marker Size')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(GRAPHS_DIR, "actual_vs_predicted_error_size.png"), dpi=300)
plt.close(fig_err_size)

# Log conclusion about model performance
logger.info("\nModel Performance Analysis:")
logger.info(f"Overall R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
logger.info(f"Training R²: {train_r2:.4f}, Testing R²: {r2:.4f}")
if train_r2 > r2 + 0.05:
    logger.warning("Possible overfitting detected: Training R² significantly higher than Test R²")
elif r2 > train_r2:
    logger.warning("Unusual pattern: Test R² higher than Training R². This may indicate a problem with data splitting or leakage.")

logger.info(f"R² range across {optimal_n_clusters} clusters: {min(cluster_r2s):.4f} to {max(cluster_r2s):.4f}")
if max(cluster_r2s) - min(cluster_r2s) > 0.2:
    logger.warning("Large variation in R² across clusters. Model performance is inconsistent.")
else:
    logger.info("Model performance is consistent across different data clusters.")

import gc
gc.collect()