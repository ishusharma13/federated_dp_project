# utils.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Function to split the MNIST dataset among N clients
def split_dataset(num_clients=5, batch_size=32):
    # Define transformation
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load full MNIST training set
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Determine split indices for clients
    data_len = len(train_dataset)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    split_size = data_len // num_clients
    client_loaders = []
    
    for i in range(num_clients):
        start = i * split_size
        end = (i+1) * split_size if i != num_clients - 1 else data_len
        client_subset = Subset(train_dataset, indices[start:end])
        loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
        
    return client_loaders

# Optionally: Function to load the test set
def get_test_loader(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
