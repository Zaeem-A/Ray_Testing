import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import ray
import os

# Function to run the serial version (original code)
def run_serial():
    print("Running SERIAL version...")
    # Record start time
    start_time = time.time()
    
    # Load and preprocess data
    file_path = 'pdc_dataset_with_target.csv'
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Split features and target
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors and move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_tensor = torch.tensor(X_train).to(device)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test).to(device)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1).to(device)
    
    # DataLoaders with a larger batch size
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)
    
    # Larger Neural Network Model
    class SimpleMLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.net(x)
    
    model = SimpleMLP(X.shape[1]).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 400  # Reduced for demonstration
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Serial - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        predicted = (preds.cpu().numpy() > 0.5).astype(int)
    
    from sklearn.metrics import classification_report
    print("SERIAL RESULTS:")
    print(classification_report(y_test, predicted))
    
    # Record end time and print total time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal time taken for SERIAL execution on {device}: {total_time:.2f} seconds")
    return total_time

# Initialize Ray
ray.init()

# Create a Ray actor to handle PyTorch model training in parallel
@ray.remote(num_gpus=0.2)  # Request fraction of GPU if available
class ModelTrainer:
    def __init__(self, input_size, hidden_size=512):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = self.create_model(input_size, hidden_size)
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def create_model(self, input_size, hidden_size):
        class SimpleMLP(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.net(x)
        
        return SimpleMLP(input_size, hidden_size)
    
    def train(self, X_train, y_train, batch_size=512, num_epochs=400):
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train).to(self.device)
        y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(self.device)
        
        # Create DataLoader
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Return loss for monitoring
            return_loss = loss.item()
            print(f"Parallel Worker - Epoch [{epoch+1}/{num_epochs}], Loss: {return_loss:.4f}")
        
        return "Training completed"
    
    def evaluate(self, X_test, y_test):
        # Convert to PyTorch tensors
        X_test_tensor = torch.tensor(X_test).to(self.device)
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_test_tensor)
            predicted = (preds.cpu().numpy() > 0.5).astype(int)
        
        return predicted.flatten()


# Function to run the parallel version using Ray
def run_parallel():
    print("\nRunning PARALLEL version with Ray...")
    # Record start time
    start_time = time.time()
    
    # Load and preprocess data
    file_path = 'pdc_dataset_with_target.csv'
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Split features and target
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Determine number of parallel models to train
    num_parallel = 5  # You can adjust this based on your system
    
    # Create multiple model trainers and start training in parallel
    trainers = [ModelTrainer.remote(X.shape[1]) for _ in range(num_parallel)]
    
    # Split data for parallel training
    split_size = len(X_train) // num_parallel
    X_splits = [X_train[i*split_size:(i+1)*split_size] for i in range(num_parallel)]
    y_splits = [y_train[i*split_size:(i+1)*split_size] for i in range(num_parallel)]
    
    # Launch parallel training
    train_refs = [trainer.train.remote(X_splits[i], y_splits[i], batch_size=512, num_epochs=400) 
                  for i, trainer in enumerate(trainers)]
    
    # Wait for all training to complete
    ray.get(train_refs)
    
    # Evaluate models in parallel
    eval_refs = [trainer.evaluate.remote(X_test, y_test) for trainer in trainers]
    predictions = ray.get(eval_refs)
    
    # Combine predictions (using majority voting for this example)
    combined_preds = np.zeros_like(predictions[0])
    for pred in predictions:
        combined_preds += pred
    final_preds = (combined_preds >= (num_parallel // 2 + 1)).astype(int)
    
    # Evaluate the combined model
    from sklearn.metrics import classification_report
    print("PARALLEL RESULTS:")
    print(classification_report(y_test, final_preds))
    
    # Record end time and print total time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal time taken for PARALLEL execution with Ray: {total_time:.2f} seconds")
    return total_time

# Main function to compare serial and parallel performance
def main():
    print("=" * 80)
    print("COMPARING SERIAL VS PARALLEL EXECUTION WITH RAY")
    print("=" * 80)
    
    # Run serial version
    serial_time = run_serial()
    
    # Run parallel version
    parallel_time = run_parallel()
    
    # Print comparison
    print("\n" + "=" * 80)
    print(f"SERIAL execution time: {serial_time:.2f} seconds")
    print(f"PARALLEL execution time: {parallel_time:.2f} seconds")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
    print("=" * 80)
    
    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()
