import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.utils import resample

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "tdnn_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('tdnn_trainer')

def find_project_root():
    """Find the project root directory to enable path-independent operations"""
    current_path = Path(__file__).resolve().parent
    
    # Navigate up to find the project root (where data_filtered-1.csv would be)
    for _ in range(5):  # Don't go up more than 5 levels
        if (current_path / "data_filtered-1.csv").exists():
            return current_path
        current_path = current_path.parent
        
    # If we can't find it using the above method, try a common structure
    current_path = Path(__file__).resolve().parent
    potential_root = current_path.parent.parent  # Go up to /qnn_fnl/
    
    if (potential_root / "data_filtered-1.csv").exists():
        return potential_root
    
    logger.warning("Could not find project root with data file. Using relative path.")
    return Path.cwd()

class TDNN(nn.Module):
    """
    Time Delay Neural Network implementation
    
    This architecture uses 1D convolutions to capture temporal patterns across features
    with different receptive field sizes (time delays)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, kernel_sizes=[3, 5, 7], dropout_rate=0.3):
        super(TDNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input reshaping layer
        self.reshape_layer = nn.Linear(input_size, hidden_size)
        
        # Create multiple convolutional layers with different kernel sizes (time delays)
        self.conv_layers = nn.ModuleList()
        for k_size in kernel_sizes:
            # Padding to maintain the same sequence length
            padding = (k_size - 1) // 2
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=k_size, padding=padding),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(), 
                    nn.Dropout(dropout_rate)
                )
            )
        
        # Feature dimension after concatenating all conv outputs
        feature_dim = hidden_size * len(kernel_sizes)
        
        # Fully connected layers for regression
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.8)
        )
        
        self.output_layer = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        # Reshape the input for 1D convolution [batch, features] -> [batch, hidden_size, 1]
        x = self.reshape_layer(x)
        # Reshape to [batch_size, hidden_size, 1] to have hidden_size as channels dimension
        x = x.view(x.size(0), self.hidden_size, 1)
        
        # Apply each convolutional layer and collect outputs
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x)
            # Global average pooling across the sequence dimension
            pooled = torch.mean(conv_out, dim=2)
            conv_outputs.append(pooled)
        
        # Concatenate all convolution outputs
        combined = torch.cat(conv_outputs, dim=1)
        
        # Pass through fully connected layers
        x = self.fc1(combined)
        x = self.fc2(x)
        x = self.output_layer(x)
        
        return x

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset, including encoding object columns"""
    
    logger.info(f"Loading dataset from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Check for and handle NaN values
    logger.info(f"NaN values in dataset: {data.isna().sum().sum()}")
    data = data.dropna()
    logger.info(f"Dataset shape after dropping NaN: {data.shape}")
    
    # Split features and target
    target_col = "dGmix"
    if target_col not in data.columns:
        logger.error(f"Target column '{target_col}' not found in dataset.")
        logger.info(f"Available columns: {data.columns.tolist()}")
        sys.exit(1)
    
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
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols if categorical_cols else [])
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
    
    # Convert to tensors
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    X_train_tensor = torch.FloatTensor(X_train_processed)
    y_train_tensor = torch.FloatTensor(y_train_scaled).view(-1, 1)
    
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    X_test_tensor = torch.FloatTensor(X_test_processed)
    y_test_tensor = torch.FloatTensor(y_test_scaled).view(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return (train_dataset, test_dataset, X_train_processed, X_test_processed, 
            y_train, y_test, y_scaler, X_train, X_test, preprocessor)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100, early_stopping_patience=15):
    """Train the TDNN model with early stopping"""
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
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
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model from training")
    
    return model, train_losses, test_losses

def evaluate_model(model, X_test_tensor, y_test, y_scaler):
    """Evaluate the model and return predictions and metrics"""
    
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy().flatten()
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = y_test.values
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    logger.info(f'Test MSE: {mse:.4f}')
    logger.info(f'Test RMSE: {rmse:.4f}')
    logger.info(f'Test MAE: {mae:.4f}')
    logger.info(f'Test R²: {r2:.4f}')
    logger.info(f'Pearson Correlation: {pearson_corr:.4f}')
    
    return y_pred, mse, rmse, mae, r2, pearson_corr

def create_visualizations(y_test, y_pred, train_losses, test_losses, rmse, r2):
    """Create and save visualizations"""
    
    # 1. Learning Curve Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TDNN Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tdnn_learning_curve.png', dpi=300)
    logger.info("Learning curve plot saved to 'tdnn_learning_curve.png'")
    
    # 2. Actual vs Predicted Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    min_val = min(y_test.min(), np.min(y_pred))
    max_val = max(y_test.max(), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual dGmix')
    plt.ylabel('Predicted dGmix')
    plt.title(f'TDNN: Actual vs Predicted (R² = {r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tdnn_actual_vs_predicted.png', dpi=300)
    logger.info("Actual vs predicted plot saved to 'tdnn_actual_vs_predicted.png'")
    
    # 3. Error Distribution
    plt.figure(figsize=(10, 6))
    errors = y_pred - y_test.values
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'TDNN: Error Distribution (RMSE = {rmse:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tdnn_error_distribution.png', dpi=300)
    logger.info("Error distribution plot saved to 'tdnn_error_distribution.png'")
    
    # 4. Bootstrap Confidence Intervals
    plt.figure(figsize=(12, 6))
    
    # Generate bootstrap samples
    n_bootstrap = 100
    predictions = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        bootstrap_true = y_test.values[indices]
        bootstrap_pred = y_pred[indices]
        predictions.append(bootstrap_pred)
    
    predictions_array = np.array(predictions)
    lower_bound = np.percentile(predictions_array, 5, axis=0)
    upper_bound = np.percentile(predictions_array, 95, axis=0)
    
    # Sort for better visualization
    sorted_indices = np.argsort(y_test.values)
    sorted_actual = y_test.values[sorted_indices]
    sorted_pred = y_pred[sorted_indices]
    sorted_lower = lower_bound[sorted_indices]
    sorted_upper = upper_bound[sorted_indices]
    
    plt.fill_between(range(len(sorted_actual)), sorted_lower, sorted_upper, 
                     alpha=0.3, label='90% Confidence Interval')
    plt.plot(range(len(sorted_actual)), sorted_pred, 'b-', label='Predicted')
    plt.plot(range(len(sorted_actual)), sorted_actual, 'r.', markersize=2, label='Actual')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('dGmix Value')
    plt.title('TDNN Predictions with 90% Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tdnn_confidence_intervals.png', dpi=300)
    logger.info("Confidence intervals plot saved to 'tdnn_confidence_intervals.png'")
    
    # 5. Residuals Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('TDNN: Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tdnn_residuals.png', dpi=300)
    logger.info("Residuals plot saved to 'tdnn_residuals.png'")

def main():
    """Main function to run the TDNN training and evaluation"""
    try:
        logger.info("Starting TDNN implementation")
        
        # Find project root and data file path
        project_root = find_project_root()
        data_path = project_root / "data_filtered-1.csv"
        
        logger.info(f"Using data from: {data_path}")
        
        # Load and preprocess data
        (train_dataset, test_dataset, X_train_processed, X_test_processed, 
         y_train, y_test, y_scaler, X_train, X_test, preprocessor) = load_and_preprocess_data(data_path)
        
        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the model
        input_size = X_train_processed.shape[1]
        hidden_size = 128
        kernel_sizes = [3, 5, 7]
        
        model = TDNN(input_size=input_size, 
                     hidden_size=hidden_size, 
                     kernel_sizes=kernel_sizes, 
                     dropout_rate=0.3)
        
        logger.info(f"Model created with input size: {input_size}, hidden size: {hidden_size}")
        logger.info(f"Model architecture: {model}")
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5, verbose=True
        )
        
        # Train the model
        model, train_losses, test_losses = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, 
            num_epochs=150, early_stopping_patience=15
        )
        
        # Convert test data to tensor for evaluation
        if hasattr(X_test_processed, 'toarray'):
            X_test_processed = X_test_processed.toarray()
        X_test_tensor = torch.FloatTensor(X_test_processed)
        
        # Evaluate the model
        y_pred, mse, rmse, mae, r2, pearson_corr = evaluate_model(model, X_test_tensor, y_test, y_scaler)
        
        # Create visualizations
        create_visualizations(y_test, y_pred, train_losses, test_losses, rmse, r2)
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"tdnn_model_{timestamp}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'kernel_sizes': kernel_sizes,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pearson_corr': pearson_corr
            }
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        
        logger.info("TDNN implementation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in TDNN implementation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
