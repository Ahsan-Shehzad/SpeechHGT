import torch
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# --- 1. Model Saving and Loading Functions ---

def save_model(model, file_path):
    """
    Saves the model state dict to the specified file path.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to be saved.
    - file_path (str): Path where the model will be saved.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path):
    """
    Loads the model state dict from the specified file path.
    
    Args:
    - model (torch.nn.Module): The model architecture to load weights into.
    - file_path (str): Path from where to load the model.
    
    Returns:
    - model (torch.nn.Module): The model with loaded weights.
    """
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {file_path}")
    return model


# --- 2. Logging Function ---

def setup_logging(log_file='training.log', level=logging.INFO):
    """
    Sets up logging for the training process.
    
    Args:
    - log_file (str): The log file where training details will be saved.
    - level (logging.LEVEL): Logging level (e.g., logging.INFO).
    """
    logging.basicConfig(filename=log_file, 
                        level=level, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(level)
    logging.getLogger().addHandler(console)
    logging.info("Logging setup complete.")


def log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_prec, val_prec, train_rec, val_rec):
    """
    Logs training and validation metrics for each epoch.
    
    Args:
    - epoch (int): The current epoch number.
    - train_loss (float): The training loss for the epoch.
    - val_loss (float): The validation loss for the epoch.
    - train_acc (float): The training accuracy for the epoch.
    - val_acc (float): The validation accuracy for the epoch.
    - train_prec (float): The training precision for the epoch.
    - val_prec (float): The validation precision for the epoch.
    - train_rec (float): The training recall for the epoch.
    - val_rec (float): The validation recall for the epoch.
    """
    logging.info(f"Epoch {epoch}")
    logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train Precision: {train_prec:.4f}, Train Recall: {train_rec:.4f}")
    logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}")


# --- 3. Plot Performance Metrics Function ---

def plot_metrics(train_metrics, val_metrics, metric_name="Accuracy"):
    """
    Plots training and validation metrics over epochs.
    
    Args:
    - train_metrics (list): A list of training metrics values (e.g., accuracy) over epochs.
    - val_metrics (list): A list of validation metrics values over epochs.
    - metric_name (str): The name of the metric (e.g., "Accuracy", "Precision").
    """
    epochs = np.arange(1, len(train_metrics) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, val_metrics, label=f'Val {metric_name}', color='red', linestyle='--', marker='o')
    
    plt.title(f'Training and Validation {metric_name} Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- 4. Metrics Calculation Function ---

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculates the accuracy, precision, and recall from true and predicted labels.
    
    Args:
    - true_labels (list/array): Ground truth labels.
    - predicted_labels (list/array): Predicted labels by the model.
    
    Returns:
    - accuracy (float): Accuracy score.
    - precision (float): Precision score.
    - recall (float): Recall score.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    
    return accuracy, precision, recall


# --- Example usage ---
if __name__ == "__main__":
    # Example model saving/loading
    model = None  # Replace with your actual model
    save_model(model, 'best_model.pth')  # Saving the model
    model = load_model(model, 'best_model.pth')  # Loading the model
    
    # Example logging
    setup_logging('training.log')
    log_metrics(epoch=1, 
                train_loss=0.1, val_loss=0.2, 
                train_acc=0.85, val_acc=0.80, 
                train_prec=0.88, val_prec=0.79, 
                train_rec=0.84, val_rec=0.76)
    
    # Example plotting metrics
    train_accs = [0.8, 0.85, 0.86, 0.87, 0.88]
    val_accs = [0.78, 0.80, 0.82, 0.83, 0.85]
    plot_metrics(train_accs, val_accs, metric_name="Accuracy")
