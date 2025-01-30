import os

# --- 1. Path Configurations ---

# Paths for datasets and preprocessed data
DATASET_PATH = '/path/to/dementiaBank/datasets'  # Modify this path based on your system
PREPROCESSED_DATA_PATH = './data/preprocessed'  # Path where preprocessed data will be stored
SEGMENTED_AUDIO_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'segmented_audio')
TRANSCRIPTIONS_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'transcriptions')

# Paths for saving models and logs
MODEL_SAVE_PATH = './models'
LOG_DIR = './logs'

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 2. Hyperparameter Configurations ---

# Training hyperparameters
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 32  # Batch size for training and validation
EPOCHS = 100  # Number of epochs to train the model
DROPOUT_RATE = 0.3  # Dropout rate for regularization
WEIGHT_DECAY = 1e-5  # L2 regularization (weight decay) for optimizer

# Optimizer settings
OPTIMIZER = 'adam'  # Optimizer type ('adam', 'sgd', etc.)
MOMENTUM = 0.9  # Momentum for the optimizer (if using SGD)

# --- 3. Model Architecture Parameters ---

# Hypergraph Transformer (SpeechHGT) parameters
NUM_CLASSES = 2  # Binary classification: AD or non-AD
HIDDEN_DIM = 256  # Hidden dimension size for the transformer layers
NUM_HEADS = 8  # Number of attention heads in the multi-head attention layers
NUM_LAYERS = 6  # Number of layers in the transformer
DROPOUT = 0.3  # Dropout rate for the transformer layers

# --- 4. Training Settings ---

# Validation and early stopping parameters
VALIDATION_SPLIT = 0.1  # Fraction of data to use for validation (10% in this case)
EARLY_STOPPING_PATIENCE = 10  # Number of epochs with no improvement before stopping training
CHECKPOINT_SAVE_FREQ = 5  # Save model checkpoint every 5 epochs
LOGGING_FREQUENCY = 10  # Log every 10 epochs

# --- 5. Miscellaneous Constants ---

# Number of features used in the model (e.g., number of linguistic + acoustic features)
NUM_FEATURES = 512  # This should match the actual number of features after extraction

# Output classes (AD vs non-AD)
CLASSES = ['non-AD', 'AD']

# Audio settings
SAMPLE_RATE = 16000  # Audio sample rate (for WAV files)
SEGMENT_LENGTH = 2  # Length of each segmented audio clip (in seconds)
FEATURE_DIM = 39  # Example feature dimension (e.g., MFCCs or acoustic features)

# --- 6. Logging and Model Evaluation Settings ---

# Metrics for model evaluation
METRICS = ['accuracy', 'precision', 'recall', 'f1-score']  # Metrics to track during training

# --- 7. Data Augmentation Settings ---

# Data augmentation configurations (for audio augmentation)
PITCH_SHIFT_RANGE = 2  # Pitch shifting in semitones (±2 semitones)
SPEED_PERTURBATION_RANGE = 0.1  # Speed perturbation range (±10%)
NOISE_ADDITION_STD = 0.005  # Standard deviation for Gaussian noise addition

# --- 8. Seed for Reproducibility ---

# Random seed for reproducibility
SEED = 42

# --- 9. Define functions to easily access config values ---

def print_config():
    """
    Function to print out the configuration settings for the project.
    """
    print("=== CONFIGURATION SETTINGS ===")
    for key, value in globals().items():
        if not key.startswith('__'):
            print(f"{key}: {value}")

# --- 10. Optional: Set the random seed for reproducibility ---

import random
import numpy as np

def set_random_seed(seed=SEED):
    """
    Function to set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    print(f"Random seed set to {seed}")

# If this script is run directly, it will print the configuration
if __name__ == "__main__":
    print_config()
