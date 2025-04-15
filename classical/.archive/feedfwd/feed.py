
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path):
    df = pd.read_csv(file_path)

    df = df.select_dtypes(exclude=['object'])

    if 'dGmix' not in df.columns:
        raise ValueError("Target variable 'dGmix' not found in dataset")

    X = df.drop('dGmix', axis=1)
    y = df['dGmix']
    
    return X, y

class ChemDataset(Dataset):
    def __init__(self, X, y, scaler=None):
        if scaler is not None:
            self.X = scaler.transform(X)
        else:
            self.X = X.values
        self.y = y.values.reshape(-1, 1)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 16)

        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=25):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses

def predict_with_confidence(model, test_loader, num_samples=100):
    model.eval()
    all_preds = []
    all_targets = []

    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    enable_dropout(model)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            batch_predictions = []
            for _ in range(num_samples):
                outputs = model(inputs)
                batch_predictions.append(outputs.numpy())

            batch_preds = np.stack(batch_predictions, axis=0)
            mean_preds = np.mean(batch_preds, axis=0)
            
            all_preds.extend(mean_preds)
            all_targets.extend(targets.numpy())
    
    return np.array(all_preds).flatten(), np.array(all_targets).flatten()

def plot_actual_vs_predicted(y_true, y_pred, output_path, metrics=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    if metrics:
        metrics_text = f"R² = {metrics['r2']:.4f}\nMAE = {metrics['mae']:.4f}\nMSE = {metrics['mse']:.4f}\nRMSE = {metrics['rmse']:.4f}"
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                     va='top', fontsize=10)
    
    plt.xlabel('Actual dGmix')
    plt.ylabel('Predicted dGmix')
    plt.title('Actual vs Predicted dGmix Values')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_confidence_intervals(y_true, y_pred, output_path, metrics=None):

    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    error = np.abs(y_true_sorted - y_pred_sorted)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_true_sorted)), y_true_sorted, 'b-', label='Actual')
    plt.plot(range(len(y_pred_sorted)), y_pred_sorted, 'r-', label='Predicted')
    plt.fill_between(range(len(y_pred_sorted)), 
                     y_pred_sorted - error, 
                     y_pred_sorted + error, 
                     color='gray', alpha=0.3, label='Error Margin')

    if metrics:
        metrics_text = f"R² = {metrics['r2']:.4f}\nMAE = {metrics['mae']:.4f}\nMSE = {metrics['mse']:.4f}\nRMSE = {metrics['rmse']:.4f}"
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                     va='top', fontsize=10)
    
    plt.xlabel('Sample Index (sorted)')
    plt.ylabel('dGmix')
    plt.title('Hybrid Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}

def main():
    try:

        file_path = '/Users/ca5/Desktop/qnn_fnl/data_filtered-1.csv'

        X, y = load_data(file_path)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        scaler.fit(X_train)

        train_dataset = ChemDataset(X_train, y_train, scaler)
        test_dataset = ChemDataset(X_test, y_test, scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_size = X.shape[1]
        model = FeedForwardNN(input_size)
        
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=0.05)

        print("Training model...")
        train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=25)

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('/Users/ca5/Desktop/qnn_fnl/training_loss.png')
        plt.close()

        print("Generating predictions...")
        y_pred, y_true = predict_with_confidence(model, test_loader)

        metrics = calculate_metrics(y_true, y_pred)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('/Users/ca5/Desktop/qnn_fnl/classical/feedfwd/feed_model_metrics.csv', index=False)

        plot_actual_vs_predicted(y_true, y_pred, '/Users/ca5/Desktop/qnn_fnl/classical/feedfwd/actual_vs_predicted_test.png', metrics)

        plot_confidence_intervals(y_true, y_pred, '/Users/ca5/Desktop/qnn_fnl/classical/feedfwd/hybrid-confidence_intervals.png', metrics)
        
        print("Done! Check the output directory for plots and metrics.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()