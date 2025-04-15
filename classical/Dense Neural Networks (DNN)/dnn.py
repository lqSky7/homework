import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
from datetime import datetime
import seaborn as sns

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"dnn_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dnn_trainer')

# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                     "cpu")
logger.info(f"Using device: {device}")

# Find the CSV file without hardcoding the path
def find_csv_file(filename="data_filtered-1.csv"):
    """Find the CSV file by searching from the current directory up to the root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Start from current directory and go up to find the file
    while current_dir != os.path.dirname(current_dir):
        csv_path = os.path.join(current_dir, filename)
        if os.path.exists(csv_path):
            return csv_path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root
            break
        current_dir = parent_dir
    
    # If not found in parent directories, search in subdirectories
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        if filename in files:
            return os.path.join(root, filename)
    
    # Try the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    csv_path = os.path.join(project_root, filename)
    if os.path.exists(csv_path):
        return csv_path
        
    raise FileNotFoundError(f"Could not find {filename} in project directories")

# Custom Dataset for PyTorch
class ChemDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dense Neural Network
class DenseNN(nn.Module):
    def __init__(self, input_size):
        super(DenseNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load the best model
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device, y_scaler=None):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # If we scaled the targets, inverse transform to get original scale
    if y_scaler:
        all_preds = y_scaler.inverse_transform(all_preds)
        all_targets = y_scaler.inverse_transform(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    
    return all_preds, all_targets, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def create_visualizations(train_losses, val_losses, y_true, y_pred, metrics):
    # 1. Learning Curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('dnn_learning_curves.png', dpi=300)
    plt.close()
    
    # 2. Predicted vs Actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual dGmix')
    plt.ylabel('Predicted dGmix')
    plt.title(f'Actual vs Predicted (MAE: {metrics["mae"]:.4f}, R²: {metrics["r2"]:.4f})')
    plt.annotate(f'MAE: {metrics["mae"]:.4f}\nRMSE: {metrics["rmse"]:.4f}\nR²: {metrics["r2"]:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.savefig('dnn_actual_vs_predicted.png', dpi=300)
    plt.close()
    
    # 3. Error Distribution
    errors = y_pred.flatten() - y_true.flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (RMSE: {metrics["rmse"]:.4f})')
    plt.grid(True, alpha=0.3)
    plt.savefig('dnn_error_distribution.png', dpi=300)
    plt.close()
    
    # 4. Prediction Trends
    plt.figure(figsize=(12, 6))
    sorted_indices = np.argsort(y_true.flatten())
    sorted_y_true = y_true.flatten()[sorted_indices]
    sorted_y_pred = y_pred.flatten()[sorted_indices]
    
    plt.plot(range(len(sorted_y_true)), sorted_y_true, 'b-', label='Actual', alpha=0.5)
    plt.plot(range(len(sorted_y_pred)), sorted_y_pred, 'r-', label='Predicted', alpha=0.5)
    plt.xlabel('Sample Index (Sorted)')
    plt.ylabel('dGmix Value')
    plt.title('Actual vs Predicted Values by Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('dnn_prediction_trends.png', dpi=300)
    plt.close()
    
    # 5. Bootstrap Confidence Intervals
    n_bootstrap = 100
    predictions = []
    sample_indices = np.arange(len(y_true))
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        bootstrap_indices = np.random.choice(sample_indices, size=len(sample_indices), replace=True)
        predictions.append(y_pred[bootstrap_indices])
    
    predictions_array = np.array(predictions).squeeze()
    mean_prediction = np.mean(predictions_array, axis=0)
    lower_bound = np.percentile(predictions_array, 5, axis=0)
    upper_bound = np.percentile(predictions_array, 95, axis=0)
    
    # Sort by true values for better visualization
    sorted_indices = np.argsort(y_true.flatten())
    sorted_actual = y_true.flatten()[sorted_indices]
    sorted_mean = mean_prediction[sorted_indices]
    sorted_lower = lower_bound[sorted_indices]
    sorted_upper = upper_bound[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(range(len(sorted_actual)), sorted_lower, sorted_upper, 
                    alpha=0.3, label='90% Confidence Interval')
    plt.plot(range(len(sorted_actual)), sorted_mean, 'b-', label='Mean Prediction')
    plt.plot(range(len(sorted_actual)), sorted_actual, 'ro', markersize=3, label='Actual Values')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('dGmix Value')
    plt.title(f'Prediction with 90% Confidence Intervals (MAE: {metrics["mae"]:.4f}, R²: {metrics["r2"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('dnn_confidence_intervals.png', dpi=300)
    plt.close()
    
    # 6. Feature importance (correlation with target)
    logger.info("All visualizations have been saved.")

def main():
    try:
        # Find the data file
        data_path = find_csv_file()
        logger.info(f"Found data at: {data_path}")
        
        # Load the data
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded with shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        
        # Handle missing values
        data = data.dropna()
        logger.info(f"Shape after dropping NaN values: {data.shape}")
        
        # Identify categorical columns
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Categorical columns: {categorical_columns}")
        
        # Split into features and target
        X = data.drop('dGmix', axis=1)
        y = data['dGmix'].values.reshape(-1, 1)
        
        # Create preprocessing pipeline for categorical columns
        preprocessor = None
        if categorical_columns:
            # One-hot encode categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                ],
                remainder='passthrough'
            )
            X = preprocessor.fit_transform(X)
            # Handle sparse matrix if OneHotEncoder returns sparse
            if isinstance(X, (np.ndarray, np.generic)) == False:
                X = X.toarray()
            logger.info(f"One-hot encoded X shape: {X.shape}")
        else:
            # If no categorical columns, convert to numpy array if it's a DataFrame
            if isinstance(X, pd.DataFrame):
                X = X.values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Scale the features
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_val_scaled = X_scaler.transform(X_val)
        X_test_scaled = X_scaler.transform(X_test)
        
        # Scale the target
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        y_test_scaled = y_scaler.transform(y_test)
        
        # Create datasets and dataloaders
        train_dataset = ChemDataset(X_train_scaled, y_train_scaled)
        val_dataset = ChemDataset(X_val_scaled, y_val_scaled)
        test_dataset = ChemDataset(X_test_scaled, y_test_scaled)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create the model
        input_size = X_train_scaled.shape[1]
        model = DenseNN(input_size).to(device)
        logger.info(f"Model created with input size: {input_size}")
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Train the model
        logger.info("Starting model training...")
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device)
        logger.info("Model training completed.")
        
        # Evaluate the model
        logger.info("Evaluating model on test set...")
        y_pred, y_true, metrics = evaluate_model(model, test_loader, device, y_scaler)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        create_visualizations(train_losses, val_losses, y_true, y_pred, metrics)
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'X_scaler': X_scaler,
            'y_scaler': y_scaler,
            'metrics': metrics,
            'input_size': input_size,
            'categorical_encoder': preprocessor,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, 'dnn_model.pt')
        
        logger.info("Model saved to 'dnn_model.pt'")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
