"""
Main script to run all experiments.
"""
import argparse
import torch
import sys

sys.path.append('.')

from experiments.mnist_experiment import run_mnist_gsm, run_mnist_gcn
from experiments.color_connectivity_experiment import (
    run_color_connectivity_gsm, 
    run_color_connectivity_gcn
)


def main():
    parser = argparse.ArgumentParser(description='Run GSM experiments')
    parser.add_argument('--experiment', type=str, default='all',
                      choices=['all', 'mnist', 'color_connectivity'],
                      help='Which experiment to run')
    parser.add_argument('--model', type=str, default='all',
                      choices=['all', 'gsm', 'gcn'],
                      help='Which model to run')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of training epochs (overrides defaults)')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size (overrides defaults)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                      help='Hidden dimension size')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}\n")
    
    results = {}
    
    # MNIST experiments
    if args.experiment in ['all', 'mnist']:
        kwargs = {'device': device, 'hidden_dim': args.hidden_dim}
        if args.epochs:
            kwargs['num_epochs'] = args.epochs
        if args.batch_size:
            kwargs['batch_size'] = args.batch_size
            
        if args.model in ['all', 'gsm']:
            results['mnist_gsm'] = run_mnist_gsm(**kwargs)
        
        if args.model in ['all', 'gcn']:
            results['mnist_gcn'] = run_mnist_gcn(**kwargs)
    
    # Color Connectivity experiments
    if args.experiment in ['all', 'color_connectivity']:
        kwargs = {'device': device, 'hidden_dim': args.hidden_dim}
        if args.epochs:
            kwargs['num_epochs'] = args.epochs
        if args.batch_size:
            kwargs['batch_size'] = args.batch_size
            
        if args.model in ['all', 'gsm']:
            results['cc_gsm'] = run_color_connectivity_gsm(**kwargs)
        
        if args.model in ['all', 'gcn']:
            results['cc_gcn'] = run_color_connectivity_gcn(**kwargs)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    if 'mnist_gsm' in results or 'mnist_gcn' in results:
        print("\nMNIST:")
        if 'mnist_gsm' in results:
            print(f"  GSM: {results['mnist_gsm']:.4f}")
        if 'mnist_gcn' in results:
            print(f"  GCN: {results['mnist_gcn']:.4f}")
        if 'mnist_gsm' in results and 'mnist_gcn' in results:
            improvement = (results['mnist_gsm'] - results['mnist_gcn']) * 100
            print(f"  Improvement: {improvement:.2f}%")
    
    if 'cc_gsm' in results or 'cc_gcn' in results:
        print("\nColor Connectivity:")
        if 'cc_gsm' in results:
            print(f"  GSM: {results['cc_gsm']:.4f}")
        if 'cc_gcn' in results:
            print(f"  GCN: {results['cc_gcn']:.4f}")
        if 'cc_gsm' in results and 'cc_gcn' in results:
            improvement = (results['cc_gsm'] - results['cc_gcn']) * 100
            print(f"  Improvement: {improvement:.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()
