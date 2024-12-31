import torch
from config import DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE, PLOTS_DIR
from models.mobilenet_model import get_mobilenet_model
from utils.dataset_utils import prepare_datasets
from utils.train_utils import train_model, plot_metrics

# Prepare dataset
train_loader, valid_loader, num_classes = prepare_datasets(DATA_DIR, BATCH_SIZE)

# Initialize model, loss, and optimizer
model = get_mobilenet_model(num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train and evaluate
history = train_model(model, train_loader, valid_loader, NUM_EPOCHS, criterion, optimizer, DEVICE)

# Plot and save metrics
plot_metrics(history, PLOTS_DIR)