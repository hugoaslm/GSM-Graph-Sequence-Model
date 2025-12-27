import torch
import torch.nn as nn


def train_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(model, 'forward') and 'batch' in model.forward.__code__.co_varnames:
            # GCN-style forward
            out = model(data.x, data.edge_index, data.batch)
        else:
            # GSM-style forward
            out = model(data)
        
        # Handle different label formats
        if data.y.dim() > 1:
            y = data.y.view(out.size(0), -1)[:, 0]
        else:
            y = data.y
        
        loss = criterion(out, y)
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass
        if hasattr(model, 'forward') and 'batch' in model.forward.__code__.co_varnames:
            # GCN-style forward
            out = model(data.x, data.edge_index, data.batch)
        else:
            # GSM-style forward
            out = model(data)
        
        # Handle different label formats
        if data.y.dim() > 1:
            y = data.y.view(out.size(0), -1)[:, 0]
        else:
            y = data.y
        
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return correct / total


def train_model(model, train_loader, test_loader, optimizer, criterion, device, 
                num_epochs=40, clip_grad=1.0, verbose=True):
    """Complete training loop."""
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad)
        test_acc = evaluate(model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if verbose:
            print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {test_acc:.4f}")
    
    return best_acc
