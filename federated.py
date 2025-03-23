# federated.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import numpy as np

# Define a simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Differential Privacy: Function to clip gradients and add noise
def add_dp_noise(model, clip_bound=1.0, noise_std=0.1):
    for param in model.parameters():
        if param.grad is not None:
            # Clip gradients
            param.grad.data.clamp_(-clip_bound, clip_bound)
            # Add Gaussian noise
            noise = torch.normal(0, noise_std, size=param.grad.data.size()).to(param.device)
            param.grad.data.add_(noise)

# Function for a local training step on one client
def local_train(model, train_loader, epochs=1, clip_bound=1.0, noise_std=0.1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # Apply differential privacy mechanism
            add_dp_noise(model, clip_bound, noise_std)
            optimizer.step()
    # Return the updated state dict
    return model.state_dict()

# Federated averaging: Aggregate client model updates
def federated_averaging(global_model, client_state_dicts):
    new_state = copy.deepcopy(global_model.state_dict())
    # Average each parameter over clients
    for key in new_state.keys():
        new_state[key] = torch.mean(torch.stack([client_state[key] for client_state in client_state_dicts]), dim=0)
    global_model.load_state_dict(new_state)
    return global_model
