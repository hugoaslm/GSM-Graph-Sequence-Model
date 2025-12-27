import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
import math


class GatedGCNLayer(MessagePassing):
    """Gated Graph Convolutional Layer for local message passing."""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__(aggr='add')
        self.U = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)
        self.A = nn.Linear(input_dim, hidden_dim)
        self.B = nn.Linear(input_dim, hidden_dim)
        self.E = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        e_gate = torch.sigmoid(self.E(edge_attr) + self.V(x_i) + self.A(x_j))
        x_j_trans = self.U(x_j)
        return x_j_trans * e_gate

    def update(self, aggr_out, x):
        return x + self.B(aggr_out)


class LocalEncoder(nn.Module):
    """Local encoder using Gated GCN for neighborhood aggregation."""
    
    def __init__(self, input_dim, hidden_dim, edge_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        effective_edge_dim = edge_dim if edge_dim > 0 else 1
        self.edge_embedding = nn.Linear(effective_edge_dim, hidden_dim)
        self.gated_gcn = GatedGCNLayer(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), device=x.device)
        
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        x = self.embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        x_local = self.gated_gcn(x, edge_index, edge_attr)
        return self.norm(x_local)


class Mamba(nn.Module):
    """Mamba layer for efficient sequence modeling."""
    
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

        dt_init_std = self.dt_rank**-0.5 * 1.0
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -2.0)

    def forward(self, x):
        B, L, D = x.shape
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(self.d_inner, dim=-1)
        x_in = self.act(x_in)

        x_dbl = self.x_proj(x_in)
        (dt, B_ssm, C_ssm) = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())

        # Recurrence
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t = B_ssm[:, t, :].unsqueeze(1)
            C_t = C_ssm[:, t, :].unsqueeze(1)
            x_t = x_in[:, t, :].unsqueeze(-1)

            dA = torch.exp(A * dt_t)
            dB = dt_t * x_t
            h = h * dA + dB * B_t
            y.append(torch.sum(h * C_t, dim=-1))

        y = torch.stack(y, dim=1)
        y = y + x_in * self.D
        y = y * self.act(res)
        return self.out_proj(y)


class GSM(nn.Module):
    """Graph Sequence Model combining local GNN and global sequence modeling."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=0, max_seq_len=256):
        super().__init__()
        self.local_encoder = LocalEncoder(input_dim, hidden_dim, edge_dim)

        # Positional Encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Mamba layer
        self.mamba_layer = Mamba(d_model=hidden_dim, expand=2)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Local encoding
        x_local = self.local_encoder(x, edge_index, getattr(data, 'edge_attr', None))

        # Convert to sequence
        x_seq, mask = to_dense_batch(x_local, batch)

        # Add positional encoding
        seq_len_in_batch = x_seq.size(1)
        positions_indices = torch.arange(seq_len_in_batch, device=x.device)
        pos_emb = self.pos_embedding(positions_indices).unsqueeze(0)
        x_seq = x_seq + pos_emb

        # Mamba pass
        x_mamba = self.mamba_layer(x_seq)
        x_seq = self.norm1(x_seq + x_mamba)

        # Transformer pass
        src_key_padding_mask = ~mask
        x_global = self.transformer_layer(x_seq, src_key_padding_mask=src_key_padding_mask)

        # Pooling
        input_mask_expanded = mask.unsqueeze(-1).float()
        x_sum = (x_global * input_mask_expanded).sum(dim=1)
        x_count = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        x_pooled = x_sum / x_count

        return self.classifier(x_pooled)
