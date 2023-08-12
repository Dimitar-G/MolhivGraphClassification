from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch_geometric.transforms as T
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool
import torch
from torch.nn import Linear
from torch.nn import functional as F


class GATModel(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=6, dropout=0.5):
        super(GATModel, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.conv1 = GATv2Conv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels * self.num_heads, edge_dim=self.edge_embedding_size)
        self.linear = Linear(self.hidden_channels * self.num_heads, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        x = x.sigmoid()

        return x


# class GATModelTune(torch.nn.Module):
#     def __init__(self, input_dim, hidden_channels, edge_dim, dropout):
#         super(GATModelTune, self).__init__()
#         self.dropout = dropout
#         self.conv1 = GATv2Conv(input_dim, hidden_channels, edge_dim=edge_dim)
#         self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim)
#         self.linear = Linear(hidden_channels, 1)
#         self.atom_encoder = AtomEncoder(emb_dim=input_dim)
#         self.bond_encoder = BondEncoder(emb_dim=edge_dim)
#
#     def forward(self, x, edge_index, edge_attr, batch):
#
#         x = self.atom_encoder(x)
#         edge_attr = self.bond_encoder(edge_attr)
#
#         x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
#         x = x.relu()
#         x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
#
#         x = global_mean_pool(x, batch)
#
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.linear(x)
#         x = x.sigmoid()
#
#         return x