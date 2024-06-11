import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATv2Conv(in_channels=2, out_channels=16, heads=8, concat=True, edge_dim=3)
        self.conv2 = GATv2Conv(in_channels=16 * 8, out_channels=16, heads=4, concat=True, edge_dim=3)
        self.conv3 = GATv2Conv(in_channels=16 * 4, out_channels=2, heads=1, concat=False, edge_dim=3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)

        return x
