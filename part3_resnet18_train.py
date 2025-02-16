import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import sys

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("No GPU found")
    sys.exit(1)
print("Using Device {} ".format(device))

# Paths to the dataset
data_dir = "/data/datasets/community/deeplearning/imagenet"  # Replace with your ImageNet directory
train_dir = f"{data_dir}/train"

# Hyperparameters
batch_size = 128
learning_rate = 0.1
num_epochs = 5
num_workers = 16
log_interval = 100

# Data transformations
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Lazy Dataset for Training
class LazyImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        # Recursively list all image paths and their corresponding labels
        for class_idx, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Data loaders
train_dataset = LazyImageFolder(train_dir, transform=train_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

# Model setup
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)  # ImageNet has 1000 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)  # Decays LR by 0.1 every 30 epochs


# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )
    return running_loss / len(train_loader)


# Main training loop
for epoch in range(1, num_epochs + 1):
    train_loss = train(epoch)
    print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}%")

# Save the model
# torch.save(model.state_dict(), "resnet18_imagenet.pth")
# print("Model saved as resnet18_imagenet.pth")
