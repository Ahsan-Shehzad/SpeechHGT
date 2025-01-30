import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging
import time

# Import the SpeechHGT model (assumes you have already defined it in hypergraph_transformer.py)
from hypergraph_transformer import SpeechHGT

# Define a custom Dataset class to load preprocessed data
class HypergraphDataset(Dataset):
    def __init__(self, features, edge_index, labels):
        """
        Args:
            features (torch.Tensor): Feature matrix (num_samples, num_features).
            edge_index (torch.Tensor): Edge index representing hypergraph connections (2, num_edges).
            labels (torch.Tensor): Labels (num_samples,).
        """
        self.features = features
        self.edge_index = edge_index
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'x': self.features[idx],
            'edge_index': self.edge_index,
            'y': self.labels[idx]
        }

# Function to compute and log performance metrics
def compute_metrics(predictions, targets):
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')
    return accuracy, precision, recall, f1

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        optimizer.zero_grad()

        # Move data to the appropriate device (GPU/CPU)
        features = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        labels = batch['y'].to(device)

        # Forward pass
        outputs = model(Data(x=features, edge_index=edge_index))
        
        # Compute loss
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Store predictions and labels for metrics calculation
        preds = torch.sigmoid(outputs).round().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Compute and log metrics
    accuracy, precision, recall, f1 = compute_metrics(np.array(all_preds), np.array(all_labels))
    return epoch_loss / len(train_loader), accuracy, precision, recall, f1

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            labels = batch['y'].to(device)

            # Forward pass
            outputs = model(Data(x=features, edge_index=edge_index))

            # Compute loss
            loss = criterion(outputs.view(-1), labels.float())
            val_loss += loss.item()

            # Store predictions and labels for metrics calculation
            preds = torch.sigmoid(outputs).round().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Compute and log metrics
    accuracy, precision, recall, f1 = compute_metrics(np.array(all_preds), np.array(all_labels))
    return val_loss / len(val_loader), accuracy, precision, recall, f1

# Early stopping and model checkpointing function
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss  # The lower the validation loss, the better
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0

    def restore_best_weights(self, model):
        model.load_state_dict(self.best_model_wts)

# Main training loop
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Load your preprocessed data (replace with actual loading process)
    features = torch.rand(1000, 32)  # Example: 1000 samples, 32 features each
    edge_index = torch.randint(0, 1000, (2, 5000))  # Example edge index
    labels = torch.randint(0, 2, (1000,))  # Example binary labels

    # Split into train and validation sets
    train_features, val_features, train_labels, val_labels, train_edge_index, val_edge_index = train_test_split(
        features, labels, edge_index.T, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = HypergraphDataset(train_features, train_edge_index.T, train_labels)
    val_dataset = HypergraphDataset(val_features, val_edge_index.T, val_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechHGT(in_channels=32, hidden_channels=64, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, 101):  # Training for 100 epochs
        # Train the model
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate the model
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device
        )

        logging.info(f"Epoch {epoch}: "
                     f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                     f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                     f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping and checkpointing
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break
        
        # Save the best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logging.info("Model checkpoint saved.")

    # Restore the best model
    early_stopping.restore_best_weights(model)
    logging.info("Training completed. Best model restored.")

    # Optionally, you can load the best model and perform final evaluations here.
    # model.load_state_dict(torch.load('best_model.pth'))

if __name__ == "__main__":
    main()
