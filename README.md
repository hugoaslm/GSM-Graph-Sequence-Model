# Graph Sequence Models (GSM++)

Reproduction of experiments from the paper **"Best of Both Worlds: Advantages of Hybrid Graph Sequence Models"**. This repository implements GSM++, a hybrid architecture that combines local graph neural network (GNN) message passing with global sequence modeling using Mamba and Transformer layers to overcome the limitations of traditional Message Passing Neural Networks (MPNNs).

## Overview

Standard Message Passing Neural Networks (MPNNs) face critical limitations such as **over-smoothing** and **over-squashing**, making them inefficient at capturing global structure or long-range dependencies. This drives the need for Graph Sequence Models (GSMs) that treat graphs like sequences.

However, existing sequence models present a fundamental trade-off:
- **Transformers**: Offer robust global connectivity but lack counting expressivity (e.g., cannot distinguish graphs by cycle counts without complex positional encodings) and have quadratic complexity O(N²)
- **Recurrent Models (SSMs/Mamba)**: Excel at counting tasks with linear complexity O(N) but are highly sensitive to sequence ordering and suffer from representational collapse in deep architectures

**GSM++** resolves this trade-off by combining:
1. **Hierarchical Agglomerative Clustering (HAC)**: Structure-aware tokenization that intelligently orders nodes
2. **Hybrid Mamba-Transformer Backbone**: Leverages Mamba's efficiency for local/medium-range dependencies and Transformer's global reasoning for long-range connectivity
3. **Gated GCN Local Encoder**: Captures micro-structure with learnable edge gating

## Architecture

GSM++ operates in two distinct phases:

### Phase 1: Tokenization (HAC-DFS)
```
Input Graph → Hierarchical Agglomerative Clustering (HAC) → 
→ DFS Traversal → Structured Node Sequence
```

**HAC** organizes nodes into a hierarchical tree based on encoding similarity by iteratively merging clusters with lowest edge costs. **DFS traversal** represents each node as a path from root to leaf, encoding its hierarchical position and ensuring that neighbors in the graph are placed close together in the sequence.

### Phase 2: Hybrid Encoding
```
Tokenized Sequence → Local Encoder (Gated GCN) → 
→ Mamba Layers (2×) → Transformer Layer (1×) → Pooling → Classification
```

### Key Components:

- **Gated GCN (Local Encoder)**: Captures micro-structure with learnable edge gating that dynamically filters irrelevant edges, preferred over standard GCN for its ability to integrate edge features
- **HAC-DFS Tokenization**: Hierarchical clustering with depth-first search creates meaningful node ordering that preserves both local proximity and global structure
- **Mamba Layers (2×)**: Efficient sequence modeling with linear complexity O(N) that captures local and medium-range dependencies thanks to intelligent HAC ordering
- **Transformer Layer (1×)**: Acts as a global corrector at the end of the stack, using attention to access the entire sequence simultaneously and recover long-range dependencies that Mamba might compress or forget due to recency bias

## Project Structure

```
.
├── models/
│   ├── gsm.py              # Graph Sequence Model implementation
│   └── gcn.py              # GCN baseline implementation
├── utils/
│   ├── data_utils.py       # Data preprocessing and HAC-DFS tokenization
│   └── train_utils.py      # Training and evaluation utilities
├── experiments/
│   ├── mnist_experiment.py              # MNIST graph classification
│   └── color_connectivity_experiment.py # Color connectivity task
├── requirements.txt
└── README.md
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-sequence-models.git
cd graph-sequence-models

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

#### MNIST Graph Classification

```bash
cd experiments
python mnist_experiment.py
```

#### Color Connectivity (Long-range Dependencies)

```bash
cd experiments
python color_connectivity_experiment.py
```

## Datasets

### MNIST
- **Source**: GNNBenchmarkDataset
- **Task**: Graph-level classification (10 classes)
- **Size**: 35,000 train / 20,000 test
- **Features**: Pixel intensities + spatial coordinates

### Color Connectivity
- **Task**: Binary classification (node connectivity)
- **Size**: 15,000 graphs (12,000 train / 3,000 test)
- **Features**: 4-dimensional node features
- **Challenge**: Requires reasoning about long-range node relationships

## Experiments

### Global Structure (MNIST)
Tests the model's ability to capture global graph structure for digit classification. Traditional MPNNs struggle with this task due to over-squashing, which prevents information from distant nodes from being effectively aggregated.

### Long-range Dependencies (Color Connectivity)
Evaluates the model's capacity to reason about distant node relationships. This task specifically tests whether the model can overcome the over-squashing bottleneck that limits traditional GCNs to local neighborhoods.

## Results

### Our Reproduction vs Original Paper

| Dataset | Metric | GCN (Ours) | GSM++ (Ours) | GCN (Paper) | GSM++ (Paper) |
|---------|--------|------------|--------------|-------------|---------------|
| MNIST | Accuracy | 79.0% | **97.1%** | 90.7% | 98.5% |
| Color Connectivity | Accuracy | 59.0% | **87.1%** | 70.8% | 91.4% |

### Key Findings

**Massive gains on long-range tasks**: +28.1% improvement on Color Connectivity validates that GSM++ effectively bypasses over-squashing to capture long-range dependencies

**Superior global structure understanding**: +18.1% improvement on MNIST demonstrates the model's ability to recognize global geometric patterns

**Confirmed theoretical advantages**: Our reproduction empirically confirms that the hybrid Mamba-Transformer architecture successfully resolves the fundamental limitations of standard MPNNs

### Why the Performance Gap?

The results confirm the paper's core theoretical findings:
- **Traditional GCNs** are limited by local aggregation and over-squashing, failing to propagate information effectively across distant nodes
- **GSM++** overcomes these limitations through:
  1. HAC-DFS tokenization that preserves graph structure in sequence form
  2. Mamba layers that efficiently capture local/medium-range patterns (thanks to intelligent ordering)
  3. Transformer layer that provides global corrective attention for long-range dependencies

## Key Features

- **HAC-DFS Tokenization**: Novel graph ordering technique based on hierarchical agglomerative clustering with depth-first search traversal that ensures neighbors are placed close in the sequence
- **Hybrid Architecture**: Combines Mamba's linear-time efficiency (O(N)) with Transformer's global reasoning, achieving the "best of both worlds"
- **Gated GCN**: Local encoder with learnable edge gating for dynamic filtering of irrelevant connections
- **Overcomes MPNN Limitations**: Successfully addresses over-squashing and over-smoothing through sequence modeling
- **Efficient Training**: Gradient clipping and layer normalization for stable optimization
- **Modular Design**: Clean separation of model components, data processing, and training logic

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{gsm2024,
  title={Best of Both Worlds: Advantages of Graph Sequence Models},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Geometric team for the excellent graph learning library
- Original paper authors for the GSM architecture
