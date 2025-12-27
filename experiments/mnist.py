import torch
import torch.nn as nn
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import sys
sys.path.append('..')

from models.gsm import GSM
from models.gcn import GCN
from utils.data_utils import preprocess_mnist
from utils.train_utils import train_model


def run_mnist_gsm(device='cuda', batch_size=2000, hidden_dim=128, num_epochs=40):
    """Run GSM experiment on MNIST dataset."""
    print("="*50)
    print("MNIST Experiment: GSM")
    print("="*50)
    
    # Load dataset
    dataset = GNNBenchmarkDataset(root='../data/MNIST', name='MNIST')
    train_raw = dataset[:35000]
    test_raw = dataset[35000:]
    
    # Preprocess
    train_data = preprocess_mnist(train_raw)
    test_data = preprocess_mnist(test_raw)
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # Initialize model
    model = GSM(
        input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=10,
        edge_dim=dataset.num_edge_features
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nTraining GSM on MNIST...")
    best_acc = train_model(model, train_loader, test_loader, optimizer, 
                          criterion, device, num_epochs=num_epochs, clip_grad=1.0)
    
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    return best_acc


def run_mnist_gcn(device='cuda', batch_size=2048, hidden_dim=128, num_epochs=40):
    """Run GCN baseline experiment on MNIST dataset."""
    print("="*50)
    print("MNIST Experiment: GCN Baseline")
    print("="*50)
    
    # Load dataset
    dataset = GNNBenchmarkDataset(root='../data/MNIST', name='MNIST')
    
    # Simple preprocessing (no HAC)
    print("Preprocessing data...")
    augmented_data = []
    for i in range(len(dataset)):
        data = dataset[i].clone()
        
        if data.x.max() > 1.0:
            data.x = data.x / 255.0
        
        pos_mean = data.pos.mean(dim=0, keepdim=True)
        pos_max = data.pos.abs().max()
        if pos_max > 0:
            data.pos = (data.pos - pos_mean) / pos_max
        
        data.x = torch.cat([data.x, data.pos], dim=-1)
        augmented_data.append(data)
    
    train_dataset = augmented_data[:35000]
    test_dataset = augmented_data[35000:]
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = GCN(
        input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=10,
        num_layers=4,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nTraining GCN on MNIST...")
    best_acc = train_model(model, train_loader, test_loader, optimizer, 
                          criterion, device, num_epochs=num_epochs, clip_grad=None)
    
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    return best_acc


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Run experiments
    gsm_acc = run_mnist_gsm(device=device)
    gcn_acc = run_mnist_gcn(device=device)
    
    print("\n" + "="*50)
    print("MNIST Results Summary")
    print("="*50)
    print(f"GSM:  {gsm_acc:.4f}")
    print(f"GCN:  {gcn_acc:.4f}")
    print(f"Improvement: {(gsm_acc - gcn_acc)*100:.2f}%")
