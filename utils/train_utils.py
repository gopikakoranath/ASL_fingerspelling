import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def train_model(model, train_loader, valid_loader, num_epochs, criterion, optimizer, device):
    """
    Train the model and evaluate on validation set.
    """
    model.to(device)
    train_loss_history, valid_loss_history = [], []
    train_acc_history, valid_acc_history = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        print(f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        train_loss /= len(train_loader)
        train_acc = train_correct.double() / len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels)

        valid_loss /= len(valid_loader)
        valid_acc = valid_correct.double() / len(valid_loader.dataset)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_acc_history.append(train_acc.item())
        valid_acc_history.append(valid_acc.item())

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    return {
        "train_loss": train_loss_history,
        "valid_loss": valid_loss_history,
        "train_acc": train_acc_history,
        "valid_acc": valid_acc_history
    }

def plot_metrics(history, plots_dir):
    """
    Plot training and validation accuracy/loss.
    """
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss Plot
    plt.figure()
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['valid_loss'], label="Valid Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "loss.png"))
    plt.close()

    # Accuracy Plot
    plt.figure()
    plt.plot(epochs, history['train_acc'], label="Train Accuracy")
    plt.plot(epochs, history['valid_acc'], label="Valid Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "accuracy.png"))
    plt.close()