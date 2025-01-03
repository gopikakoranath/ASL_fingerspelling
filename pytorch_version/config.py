import torch

# Dataset Configuration
DATA_DIR = "./data/asl_alphabet"  # Path to the dataset directory
VALID_SPLIT = 0.2  # Proportion of the dataset used for validation

# Training Configuration
BATCH_SIZE = 32  # Batch size for training and validation
NUM_EPOCHS = 10  # Total number of training epochs
LEARNING_RATE = 0.001  # Learning rate for the optimizer

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU

# Output Configuration
PLOTS_DIR = "./plots/"  # Directory to save training and validation plots
MODEL_SAVE_PATH = "./models/saved_model.pth"  # File path to save the trained model