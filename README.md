
# Federated Learning with Differential Privacy on MNIST

This project simulates a federated learning environment where multiple clients train a shared CNN model on the MNIST dataset. Differential privacy is applied to each client's gradients to preserve data privacy during training.

## Features
- Simulated federated learning environment with multiple clients.
- Implementation of differential privacy (gradient clipping and noise addition).
- Aggregation of client updates using federated averaging.
- Evaluation on the MNIST test set.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib (optional for plotting)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ishusharma13/Project1.git
   cd Project1