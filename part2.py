####################################################################
# CEN/CSE 524: Machine Learning Acceleration
# Instructor: Dr. Aman Arora
# Spring 2025
# TA: Kaustubh Mhatre
####################################################################


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sys

n_epochs = 3  # Number of Epochs for training TODO: Change the epochs to 1 for profiling
batch_size = 128  # Batch size for training and testing TODO: Modify this variable to change batch size
log_interval = (
    1000  # This variable manages how frequently do you want to print the training loss
)

####################################################################
# Avoid changing these parameters
learning_rate = 0.01
momentum = 0.5

random_seed = 1
# torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

####################################################################
# Train loader and test loader for the MNIST dataset

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)


####################################################################
# TODO: Define your model here


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


network = SimpleMLP()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


train_losses = []
train_counter = []
test_losses = []

####################################################################
# Train and test methods for training the model


def train(epoch):
    network.train()
    total_training_time = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_end_time = time.time()
        total_training_time += batch_end_time - batch_start_time
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
    return total_training_time


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


####################################################################
# Train the model for given epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("No GPU found")
    sys.exit(1)
print(f"Running on device: {device}")

# Send the network/model to GPU
network.to(device)

total_time = 0
# for epoch in range(1, n_epochs + 1):
#     time_per_epoch = train(epoch)
#     total_time = total_time + time_per_epoch
#     test()

print("Total Training time: {}".format(total_time))


####################################################################
# Single inference

with torch.no_grad():
    test_iterator = iter(test_loader)
    data, target = next(test_iterator)
    data, target = data.to(device), target.to(device)
    single_batch_start = time.time()
    # Run single inference for 1000 times to avoid measurement overheads
    for i in range(0, 1):
        output = network(data)
    single_batch_end = time.time()

    Single_batch_inf_time = (single_batch_end - single_batch_start) / 1000

    print(
        "Single Batch Inference time is {} for a batch size of {}".format(
            Single_batch_inf_time, test_loader.batch_size
        )
    )
