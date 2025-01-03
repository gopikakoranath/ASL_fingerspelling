import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def prepare_datasets(data_dir, batch_size, valid_split=0.2):
    """
    Prepare datasets and data loaders for training and validation.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataset_size = len(dataset)
    valid_size = int(valid_split * dataset_size)
    train_size = dataset_size - valid_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, len(dataset.classes)