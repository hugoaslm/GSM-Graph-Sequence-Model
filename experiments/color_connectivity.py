import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import sys
sys.path.append('..')

from models.gsm import GSM
from models.gcn import GCN
from utils.data_utils import preprocess_color_connectivity
from utils.train_utils import train_model


class PreLoadedDataset(InMemoryDataset):
    """Dataset loader for pre-processed color connectivity data."""
    
    def __init__(self, root, processed_file_name):
        self.filename = processed_file_name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [self.filename]

    def process(self):
        pass


def run_color_connectivity_gsm(device='cuda', batch_size=512, hidden_dim=128, num_epochs=20):
    """Run GSM experiment on Color Connectivity dataset."""
    print("="*50)
    print("Color Connectivity Experiment: GSM")
    print("="*50)
    
    # Load dataset
    dataset = PreLoadedDataset(
        root='../data/CC_Node',
        processed_file_name='cc_node_level_16x16_15000.pt'
    )
    
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    # Preprocess
    train_data = preprocess_color_connectivity(train_dataset)
    test_data = preprocess_color_connectivity(test_dataset)
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # Initialize model
    model = GSM(
        input_dim=4,
        hidden_dim=hidden_dim,
        output_dim=2,
        edge_dim=0
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nTraining GSM on Color Connectivity...")
    best_acc = train_model(model, train_loader, test_loader, optimizer, 
                          criterion, device, num_epochs=num_epochs, clip_grad=1.0)
    
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    return best_acc


def run_color_connectivity_gcn(device='cuda', batch_size=512, hidden_dim=128, num_epochs=20):
    """Run GCN baseline experiment on Color Connectivity dataset."""
    print("="*50)
    print("Color Connectivity Experiment: GCN Baseline")
    print("="*50)
    
    # Load dataset
    dataset = PreLoadedDataset(
        root='../data/CC_Node',
        processed_file_name='cc_node_level_16x16_15000.pt'
    )
    
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = GCN(
        input_dim=4,
        hidden_dim=hidden_dim,
        output_dim=2,
        num_layers=4,
        dropout=0.5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nTraining GCN on Color Connectivity...")
    best_acc = train_model(model, train_loader, test_loader, optimizer, 
                          criterion, device, num_epochs=num_epochs, clip_grad=None)
    
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    return best_acc


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Run experiments
    gsm_acc = run_color_connectivity_gsm(device=device)
    gcn_acc = run_color_connectivity_gcn(device=device)
    
    print("\n" + "="*50)
    print("Color Connectivity Results Summary")
    print("="*50)
    print(f"GSM:  {gsm_acc:.4f}")
    print(f"GCN:  {gcn_acc:.4f}")
    print(f"Improvement: {(gsm_acc - gcn_acc)*100:.2f}%")
