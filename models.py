from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch_geometric.transforms as T
from torch_geometric.nn import GATv2Conv, NNConv, GINEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.pool import SAGPooling, ASAPooling
import torch
from torch.nn import Linear
from torch.nn import functional as F


class GATModel(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=2, dropout=0.5):
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


class GATModelSAG(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=2, dropout=0.5):
        super(GATModelSAG, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.pooling = SAGPooling(in_channels=self.hidden_channels * self.num_heads)
        self.dropout = dropout

        self.conv1 = GATv2Conv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels * self.num_heads, edge_dim=self.edge_embedding_size)
        self.linear = Linear(self.hidden_channels * self.num_heads, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        x = x.sigmoid()

        return x


class GATModelPlus(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=2, dropout=0.5):
        super(GATModelPlus, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.conv1 = GATv2Conv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels * self.num_heads, edge_dim=self.edge_embedding_size)
        self.linear1 = Linear(self.hidden_channels * self.num_heads, int((self.hidden_channels * self.num_heads)/2))
        self.linear2 = Linear(int((self.hidden_channels * self.num_heads)/2), int((self.hidden_channels * self.num_heads)/2))
        self.linear3 = Linear(int((self.hidden_channels * self.num_heads)/2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.sigmoid()

        return x


class GATModelExtended(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=8, dropout=0.6):
        super(GATModelExtended, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.num_heads1 = num_heads
        self.num_heads2 = int(self.num_heads1/2)
        self.dropout = dropout

        self.conv1 = GATv2Conv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads1)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads1, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads2)
        self.linear1 = Linear(self.hidden_channels * self.num_heads2, int((self.hidden_channels * self.num_heads2) / 2))
        self.linear2 = Linear(int((self.hidden_channels * self.num_heads2) / 2), int((self.hidden_channels * self.num_heads2) / 4))
        self.linear3 = Linear(int((self.hidden_channels * self.num_heads2) / 4), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.sigmoid()

        return x


class GATModel5(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=8, dropout=0.6):
        super(GATModel5, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.num_heads1 = num_heads
        self.num_heads2 = int(self.num_heads1/2)
        self.dropout = dropout

        self.conv1 = GATv2Conv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads1)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads1, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads1)
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads1, self.hidden_channels, edge_dim=edge_embedding_size, heads=self.num_heads2)
        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads2, self.hidden_channels, edge_dim=edge_embedding_size, heads=self.num_heads2)
        self.conv5 = GATv2Conv(self.hidden_channels * self.num_heads2, self.hidden_channels, edge_dim=edge_embedding_size, heads=self.num_heads2)

        self.linear1 = Linear(self.hidden_channels * self.num_heads2, int((self.hidden_channels * self.num_heads2) / 2))
        self.linear2 = Linear(int((self.hidden_channels * self.num_heads2) / 2), int((self.hidden_channels * self.num_heads2) / 4))
        self.linear3 = Linear(int((self.hidden_channels * self.num_heads2) / 4), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv4(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv5(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.sigmoid()

        return x


class GATModel5SAG(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=8, dropout=0.6):
        super(GATModel5SAG, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.num_heads1 = num_heads
        self.num_heads2 = int(self.num_heads1/2)
        self.pooling = SAGPooling(in_channels=self.hidden_channels)
        self.dropout = dropout

        self.conv1 = GATv2Conv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads1)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads1, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads1)
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads1, self.hidden_channels, edge_dim=edge_embedding_size, heads=self.num_heads2)
        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads2, self.hidden_channels, edge_dim=edge_embedding_size, heads=self.num_heads2)
        self.conv5 = GATv2Conv(self.hidden_channels * self.num_heads2, self.hidden_channels, edge_dim=edge_embedding_size, heads=self.num_heads2)

        self.linear1 = Linear(self.hidden_channels * self.num_heads2, int((self.hidden_channels * self.num_heads2) / 2))
        self.linear2 = Linear(int((self.hidden_channels * self.num_heads2) / 2), int((self.hidden_channels * self.num_heads2) / 4))
        self.linear3 = Linear(int((self.hidden_channels * self.num_heads2) / 4), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv4(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv5(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.sigmoid()

        return x


class NNModel2(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5):
        super(NNModel2, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv1 = NNConv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.node_embedding_size * self.hidden_channels))
        self.conv2 = NNConv(self.hidden_channels, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.hidden_channels * self.hidden_channels))
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x


class NNModel2SAG(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5):
        super(NNModel2SAG, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.pooling = SAGPooling(in_channels=self.hidden_channels)
        self.dropout = dropout

        self.conv1 = NNConv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.node_embedding_size * self.hidden_channels))
        self.conv2 = NNConv(self.hidden_channels, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.hidden_channels * self.hidden_channels))
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x


class NNModel3(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5):
        super(NNModel3, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv1 = NNConv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.node_embedding_size * self.hidden_channels))
        self.conv2 = NNConv(self.hidden_channels, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.hidden_channels * self.hidden_channels))
        self.conv3 = NNConv(self.hidden_channels, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.hidden_channels * self.hidden_channels))
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x