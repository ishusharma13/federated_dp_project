# main.py
import torch
from federated import SimpleCNN, local_train, federated_averaging
from utils import split_dataset, get_test_loader
import copy

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    return correct / total

def main():
    # Settings
    num_clients = 5
    global_epochs = 5
    local_epochs = 1
    clip_bound = 1.0
    noise_std = 0.1
    
    # Prepare data loaders for clients and test set
    client_loaders = split_dataset(num_clients=num_clients, batch_size=32)
    test_loader = get_test_loader(batch_size=32)
    
    # Initialize global model
    global_model = SimpleCNN()
    
    # Federated learning loop
    for round in range(global_epochs):
        print(f"Global round {round+1}/{global_epochs}")
        client_state_dicts = []
        # Simulate training on each client
        for client_id, loader in enumerate(client_loaders):
            print(f" Training on client {client_id+1}")
            local_model = copy.deepcopy(global_model)
            updated_state = local_train(local_model, loader, epochs=local_epochs, 
                                        clip_bound=clip_bound, noise_std=noise_std)
            client_state_dicts.append(updated_state)
        
        # Aggregate updates to update the global model
        global_model = federated_averaging(global_model, client_state_dicts)
        # Evaluate global model on test data
        acc = test_model(global_model, test_loader)
        print(f" Test Accuracy after round {round+1}: {acc:.4f}\n")
    
    # Save final global model
    torch.save(global_model.state_dict(), "global_model.pth")
    print("Training complete. Model saved as global_model.pth.")

if __name__ == "__main__":
    main()
