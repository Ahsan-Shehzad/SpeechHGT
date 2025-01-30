import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import rankdata
import torch_geometric.transforms as T

# Define the Hypergraph Attention Layer
class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, alpha=0.2):
        super(HypergraphAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.alpha = alpha

        # Linear transformation for features
        self.linear = nn.Linear(in_channels, out_channels * heads, bias=False)

        # Attention parameters
        self.attn_weight = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.attn_bias = nn.Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_uniform_(self.attn_weight)
        nn.init.zeros_(self.attn_bias)

    def forward(self, x, edge_index):
        # Perform linear transformation
        x = self.linear(x)

        # Apply attention mechanism
        # We apply attention at both the node and hyperedge levels

        # Node-level attention (pairwise interactions)
        row, col = edge_index
        edge_attr = torch.matmul(x[row], self.attn_weight) + self.attn_bias
        attn_scores = torch.sum(edge_attr * x[col], dim=-1)

        # Normalize attention scores (softmax)
        attn_scores = F.leaky_relu(attn_scores, negative_slope=self.alpha)
        attn_scores = F.softmax(attn_scores, dim=1)

        # Aggregate information using attention scores
        out = torch.zeros_like(x)
        for i in range(self.heads):
            out += torch.matmul(attn_scores, x)

        return out


# Define the Hypergraph Transformer Model
class SpeechHGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, n_layers=3):
        super(SpeechHGT, self).__init__()

        # Initialize hypergraph attention layers
        self.attn_layers = nn.ModuleList()
        self.attn_layers.append(HypergraphAttentionLayer(in_channels, hidden_channels, heads=heads))

        for _ in range(n_layers - 1):
            self.attn_layers.append(HypergraphAttentionLayer(hidden_channels, hidden_channels, heads=heads))

        # Feed-forward classification layer
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass through attention layers
        for attn_layer in self.attn_layers:
            x = attn_layer(x, edge_index)

        # Global mean pooling (Graph-level representation)
        x = global_mean_pool(x, data.batch)

        # Classification layer
        x = self.fc(x)

        return x


# Function to compute node centrality measures (betweenness, closeness) and integrate into positional encoding
def compute_positional_encoding(edge_index, num_nodes, device):
    # Convert to a graph for centrality calculations
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    row, col = edge_index
    G.add_edges_from(zip(row.cpu().numpy(), col.cpu().numpy()))

    # Compute node centrality (betweenness and closeness)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    betweenness_vals = torch.tensor([betweenness.get(i, 0) for i in range(num_nodes)], device=device)
    closeness_vals = torch.tensor([closeness.get(i, 0) for i in range(num_nodes)], device=device)

    # Combine centrality values into a single positional encoding
    positional_encoding = betweenness_vals + closeness_vals
    return positional_encoding


# Training function for SpeechHGT model
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data)
    
    # Compute loss
    loss = criterion(output, data.y)
    loss.backward()

    # Optimize
    optimizer.step()

    return loss.item()


# Example Usage
def main():
    # Hypergraph data: (features, edges, labels)
    # Generate dummy data for illustration purposes
    num_nodes = 100
    in_channels = 32  # number of input features
    hidden_channels = 64
    out_channels = 1  # Binary classification (AD vs non-AD)

    # Create synthetic feature data (node features)
    features = torch.rand((num_nodes, in_channels))

    # Create synthetic edge list (edge_index: (source, target) pairs)
    row = torch.randint(0, num_nodes, (500,))  # Random source nodes
    col = torch.randint(0, num_nodes, (500,))  # Random target nodes
    edge_index = torch.stack([row, col], dim=0)

    # Create a dummy label (0 for non-AD, 1 for AD)
    labels = torch.randint(0, 2, (num_nodes,))

    # Create PyTorch Geometric data object
    data = Data(x=features, edge_index=edge_index, y=labels)

    # Define model, optimizer, and loss function
    model = SpeechHGT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, heads=1, n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification

    # Training loop
    for epoch in range(100):
        loss = train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
