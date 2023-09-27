from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch_geometric.transforms as T
from torch_geometric.nn import GATv2Conv, NNConv, GINEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.pool import SAGPooling, ASAPooling, TopKPooling
import torch
from torch.nn import Linear, BatchNorm1d
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

        x, _, _, batch_pool, _, _,  = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = global_mean_pool(x, batch_pool)

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


class GATModel3Pooled(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, num_heads=2, dropout=0.5, pooling_type='topk'):
        super(GATModel3Pooled, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.bn1 = BatchNorm1d(self.hidden_channels * self.num_heads)
        self.bn2 = BatchNorm1d(self.hidden_channels * self.num_heads)
        self.bn3 = BatchNorm1d(self.hidden_channels * self.num_heads)
        if pooling_type == 'topk':
            self.pooling1 = TopKPooling(self.hidden_channels * self.num_heads)
            self.pooling2 = TopKPooling(self.hidden_channels * self.num_heads)
            self.pooling3 = TopKPooling(self.hidden_channels * self.num_heads)
        elif pooling_type == 'sag':
            self.pooling1 = SAGPooling(self.hidden_channels * self.num_heads)
            self.pooling2 = SAGPooling(self.hidden_channels * self.num_heads)
            self.pooling3 = SAGPooling(self.hidden_channels * self.num_heads)
        else:
            print('Pooling type not supported.')
        self.dropout = dropout

        self.conv1 = GATv2Conv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, edge_dim=self.edge_embedding_size, heads=self.num_heads)
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels * self.num_heads, edge_dim=self.edge_embedding_size)
        self.linear1 = Linear(self.hidden_channels * self.num_heads, int((self.hidden_channels * self.num_heads)/2))
        self.linear2 = Linear(int((self.hidden_channels * self.num_heads)/2), int((self.hidden_channels * self.num_heads)/2))
        self.linear3 = Linear(int((self.hidden_channels * self.num_heads)/2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x, edge_index1, edge_attr1, batch_pool1, _, _, = self.pooling1(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = self.conv2(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
        x = self.bn2(x)
        x = x.relu()
        x, edge_index2, edge_attr2, batch_pool2, _, _, = self.pooling2(x, edge_index=edge_index1, edge_attr=edge_attr1, batch=batch_pool1)
        x = self.conv3(x=x, edge_index=edge_index2, edge_attr=edge_attr2)
        x = self.bn3(x)
        x = x.relu()
        x, _, _, batch_pool3, _, _, = self.pooling3(x, edge_index=edge_index2, edge_attr=edge_attr2, batch=batch_pool2)

        x = global_mean_pool(x, batch_pool3)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.sigmoid()

        return x

    def calculate_embedding(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x, edge_index1, edge_attr1, batch_pool1, _, _, = self.pooling1(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = self.conv2(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
        x = self.bn2(x)
        x = x.relu()
        x, edge_index2, edge_attr2, batch_pool2, _, _, = self.pooling2(x, edge_index=edge_index1, edge_attr=edge_attr1, batch=batch_pool1)
        x = self.conv3(x=x, edge_index=edge_index2, edge_attr=edge_attr2)
        x = self.bn3(x)
        x = x.relu()
        x, _, _, batch_pool3, _, _, = self.pooling3(x, edge_index=edge_index2, edge_attr=edge_attr2, batch=batch_pool2)

        x = global_mean_pool(x, batch_pool3)

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
        self.pooling = SAGPooling(in_channels=self.hidden_channels * self.num_heads2)
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

        x, _, _, batch_pool, _, _, = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = global_mean_pool(x, batch_pool)

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

        x, _, _, batch_pool, _, _, = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = global_mean_pool(x, batch_pool)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x

    def calculate_embedding(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x, _, _, batch_pool, _, _, = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = global_mean_pool(x, batch_pool)

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


class NNModel3Pooled(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5, pooling_type='topk'):
        super(NNModel3Pooled, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.bn1 = BatchNorm1d(self.hidden_channels)
        self.bn2 = BatchNorm1d(self.hidden_channels)
        self.bn3 = BatchNorm1d(self.hidden_channels)
        if pooling_type == 'topk':
            self.pooling1 = TopKPooling(self.hidden_channels)
            self.pooling2 = TopKPooling(self.hidden_channels)
            self.pooling3 = TopKPooling(self.hidden_channels)
        elif pooling_type == 'sag':
            self.pooling1 = SAGPooling(self.hidden_channels)
            self.pooling2 = SAGPooling(self.hidden_channels)
            self.pooling3 = SAGPooling(self.hidden_channels)
        else:
            print('Pooling type not supported.')

        self.conv1 = NNConv(self.node_embedding_size, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.node_embedding_size * self.hidden_channels))
        self.conv2 = NNConv(self.hidden_channels, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.hidden_channels * self.hidden_channels))
        self.conv3 = NNConv(self.hidden_channels, self.hidden_channels, edge_dim=self.edge_embedding_size, nn=Linear(self.edge_embedding_size, self.hidden_channels * self.hidden_channels))
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x, edge_index1, edge_attr1, batch_pool1, _, _, = self.pooling1(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = self.conv2(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
        x = self.bn2(x)
        x = x.relu()
        x, edge_index2, edge_attr2, batch_pool2, _, _, = self.pooling2(x, edge_index=edge_index1, edge_attr=edge_attr1, batch=batch_pool1)
        x = self.conv3(x=x, edge_index=edge_index2, edge_attr=edge_attr2)
        x = self.bn3(x)
        x = x.relu()
        x, _, _, batch_pool3, _, _, = self.pooling3(x, edge_index=edge_index2, edge_attr=edge_attr2, batch=batch_pool2)

        x = global_mean_pool(x, batch_pool3)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x


class GINEModel3(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5):
        super(GINEModel3, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.bn1 = BatchNorm1d(self.hidden_channels)
        self.bn2 = BatchNorm1d(self.hidden_channels)
        self.bn3 = BatchNorm1d(self.hidden_channels)

        self.conv1 = GINEConv(nn=Linear(self.node_embedding_size, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv2 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv3 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()

        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = x.relu()

        x = self.conv3(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = x.relu()

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x


class GINEModel3Pooled(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5, pooling_type='topk'):
        super(GINEModel3Pooled, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        if pooling_type == 'topk':
            self.pooling1 = TopKPooling(self.hidden_channels)
            self.pooling2 = TopKPooling(self.hidden_channels)
            self.pooling3 = TopKPooling(self.hidden_channels)
        elif pooling_type == 'sag':
            self.pooling1 = SAGPooling(self.hidden_channels)
            self.pooling2 = SAGPooling(self.hidden_channels)
            self.pooling3 = SAGPooling(self.hidden_channels)
        else:
            print('Pooling type not supported.')
        self.bn1 = BatchNorm1d(self.hidden_channels)
        self.bn2 = BatchNorm1d(self.hidden_channels)
        self.bn3 = BatchNorm1d(self.hidden_channels)

        self.conv1 = GINEConv(nn=Linear(self.node_embedding_size, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv2 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv3 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x, edge_index1, edge_attr1, batch_pool1, _, _, = self.pooling1(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = self.conv2(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
        x = self.bn2(x)
        x = x.relu()
        x, edge_index2, edge_attr2, batch_pool2, _, _, = self.pooling2(x, edge_index=edge_index1, edge_attr=edge_attr1, batch=batch_pool1)

        x = self.conv3(x=x, edge_index=edge_index2, edge_attr=edge_attr2)
        x = self.bn3(x)
        x = x.relu()
        x, _, _, batch_pool2, _, _, = self.pooling3(x, edge_index=edge_index2, edge_attr=edge_attr2, batch=batch_pool2)

        x = global_mean_pool(x, batch_pool2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x


class GINEModel5(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5):
        super(GINEModel5, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.bn1 = BatchNorm1d(self.hidden_channels)
        self.bn2 = BatchNorm1d(self.hidden_channels)
        self.bn3 = BatchNorm1d(self.hidden_channels)
        self.bn4 = BatchNorm1d(self.hidden_channels)
        self.bn5 = BatchNorm1d(self.hidden_channels)

        self.conv1 = GINEConv(nn=Linear(self.node_embedding_size, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv2 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv3 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv4 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv5 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()

        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = x.relu()

        x = self.conv3(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = x.relu()

        x = self.conv4(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn4(x)
        x = x.relu()

        x = self.conv5(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn5(x)
        x = x.relu()

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x


class GINEModel5Pooled(torch.nn.Module):
    def __init__(self, node_embedding_size=64, edge_embedding_size=32, hidden_channels=256, dropout=0.5, pooling_type='topk'):
        super(GINEModel5Pooled, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        if pooling_type == 'topk':
            self.pooling1 = TopKPooling(self.hidden_channels)
            self.pooling2 = TopKPooling(self.hidden_channels)
            self.pooling3 = TopKPooling(self.hidden_channels)
            self.pooling4 = TopKPooling(self.hidden_channels)
        elif pooling_type == 'sag':
            self.pooling1 = SAGPooling(self.hidden_channels)
            self.pooling2 = SAGPooling(self.hidden_channels)
            self.pooling3 = SAGPooling(self.hidden_channels)
            self.pooling4 = SAGPooling(self.hidden_channels)
        else:
            print('Pooling type not supported.')

        self.bn1 = BatchNorm1d(self.hidden_channels)
        self.bn2 = BatchNorm1d(self.hidden_channels)
        self.bn3 = BatchNorm1d(self.hidden_channels)
        self.bn4 = BatchNorm1d(self.hidden_channels)
        self.bn5 = BatchNorm1d(self.hidden_channels)

        self.conv1 = GINEConv(nn=Linear(self.node_embedding_size, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv2 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv3 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv4 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.conv5 = GINEConv(nn=Linear(self.hidden_channels, self.hidden_channels), edge_dim=self.edge_embedding_size)
        self.linear1 = Linear(self.hidden_channels, int(self.hidden_channels / 2))
        self.linear2 = Linear(int(self.hidden_channels / 2), 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x, edge_index1, edge_attr1, batch_pool1, _, _, = self.pooling1(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = self.conv2(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
        x = self.bn2(x)
        x = x.relu()
        x, edge_index2, edge_attr2, batch_pool2, _, _, = self.pooling2(x, edge_index=edge_index1, edge_attr=edge_attr1, batch=batch_pool1)

        x = self.conv3(x=x, edge_index=edge_index2, edge_attr=edge_attr2)
        x = self.bn3(x)
        x = x.relu()
        x, edge_index3, edge_attr3, batch_pool3, _, _, = self.pooling3(x, edge_index=edge_index2, edge_attr=edge_attr2, batch=batch_pool2)

        x = self.conv4(x=x, edge_index=edge_index3, edge_attr=edge_attr3)
        x = self.bn4(x)
        x = x.relu()
        x, edge_index4, edge_attr4, batch_pool4, _, _, = self.pooling4(x, edge_index=edge_index3, edge_attr=edge_attr3, batch=batch_pool3)

        x = self.conv5(x=x, edge_index=edge_index4, edge_attr=edge_attr4)
        x = self.bn5(x)
        x = x.relu()

        x = global_mean_pool(x, batch_pool4)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.sigmoid()

        return x

    def calculate_embedding(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x, edge_index1, edge_attr1, batch_pool1, _, _, = self.pooling1(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = self.conv2(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
        x = self.bn2(x)
        x = x.relu()
        x, edge_index2, edge_attr2, batch_pool2, _, _, = self.pooling2(x, edge_index=edge_index1, edge_attr=edge_attr1, batch=batch_pool1)

        x = self.conv3(x=x, edge_index=edge_index2, edge_attr=edge_attr2)
        x = self.bn3(x)
        x = x.relu()
        x, edge_index3, edge_attr3, batch_pool3, _, _, = self.pooling3(x, edge_index=edge_index2, edge_attr=edge_attr2, batch=batch_pool2)

        x = self.conv4(x=x, edge_index=edge_index3, edge_attr=edge_attr3)
        x = self.bn4(x)
        x = x.relu()
        x, edge_index4, edge_attr4, batch_pool4, _, _, = self.pooling4(x, edge_index=edge_index3, edge_attr=edge_attr3, batch=batch_pool3)

        x = self.conv5(x=x, edge_index=edge_index4, edge_attr=edge_attr4)
        x = self.bn5(x)
        x = x.relu()

        x = global_mean_pool(x, batch_pool4)

        return x


class FusionModel1(torch.nn.Module):
    def __init__(self, input_dim=3239):
        super(FusionModel1, self).__init__()
        self.fc1 = Linear(in_features=input_dim, out_features=1024)
        self.fc2 = Linear(in_features=1024, out_features=1024)
        self.fc3 = Linear(in_features=1024, out_features=1024)
        self.fc4 = Linear(in_features=1024, out_features=512)
        self.fc5 = Linear(in_features=512, out_features=256)
        self.fc6 = Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        x = x.relu()
        x = self.fc4(x)
        x = x.relu()
        x = self.fc5(x)
        x = x.relu()
        x = self.fc6(x)
        x = x.sigmoid()

        return x


class FusionModel2(torch.nn.Module):
    def __init__(self, input_dim=3239):
        super(FusionModel2, self).__init__()
        self.fc1 = Linear(in_features=input_dim, out_features=2048)
        self.fc2 = Linear(in_features=2048, out_features=1024)
        self.fc3 = Linear(in_features=1024, out_features=512)
        self.fc4 = Linear(in_features=512, out_features=256)
        self.fc5 = Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        x = x.relu()
        x = self.fc4(x)
        x = x.relu()
        x = self.fc5(x)
        x = x.sigmoid()

        return x


class FusionModel3(torch.nn.Module):
    def __init__(self, input_dim=3239):
        super(FusionModel3, self).__init__()
        self.fc1 = Linear(in_features=input_dim, out_features=2048)
        self.fc2 = Linear(in_features=2048, out_features=1024)
        self.fc3 = Linear(in_features=1024, out_features=1024)
        self.fc4 = Linear(in_features=1024, out_features=1024)
        self.fc5 = Linear(in_features=1024, out_features=512)
        self.fc6 = Linear(in_features=512, out_features=512)
        self.fc7 = Linear(in_features=512, out_features=256)
        self.fc8 = Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        x = x.relu()
        x = self.fc4(x)
        x = x.relu()
        x = self.fc5(x)
        x = x.relu()
        x = self.fc6(x)
        x = x.relu()
        x = self.fc7(x)
        x = x.relu()
        x = self.fc8(x)
        x = x.sigmoid()

        return x