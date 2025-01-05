import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.dis_GCN1 = GCNConv(args.drug_number, args.dis_hidden_1)

        self.dis_GCN2 = GCNConv(args.dis_hidden_1, args.dis_hidden_2)

        self.dis_drsnet = DRSNet(args.dis_DRSNet_output)  # 实例化深度残差神经网络
        self.dis_gat1 = GATv2Conv(args.dis_input_channels, args.dis_output, heads=args.att_head_num, concat=False, edge_dim=1, dropout=args.dropout)

        self.drug_GCN1 = GCNConv(args.disease_number, args.drug_hidden_1)

        self.drug_GCN2 = GCNConv(args.drug_hidden_1, args.drug_hidden_2)

        self.drug_drsnet = DRSNet(args.drug_DRSNet_output)  # 实例化深度残差神经网络
        self.drug_gat1 = GATv2Conv(args.drug_input_channels, args.drug_output, heads=args.att_head_num, concat=False, edge_dim=1, dropout=args.dropout)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, data):
        dis_data = data['drug_disease_matrix'].to(torch.float32).T
        drug_data = data['drug_disease_matrix'].to(torch.float32)
        dis_edge_index, drug_edge_index = data['dis_edge_index'], data['drug_edge_index']
        dis_edge_weight, drug_edge_weight = data['dis_edge_weight'], data['drug_edge_weight']

        dis_x1 = self.dis_GCN1(dis_data, dis_edge_index, dis_edge_weight)
        # dis_x1 = self.dis_ln1(dis_x1.to(torch.float32))  # 应用 LayerNorm
        dis_x1 = F.leaky_relu(dis_x1, negative_slope=0.01)
        dis_x1 = self.dropout(dis_x1)

        dis_x2 = self.dis_GCN2(dis_x1.to(torch.float32), dis_edge_index, dis_edge_weight)
        # dis_x2 = self.dis_ln2(dis_x2.to(torch.float32))  # 应用 LayerNorm
        dis_x2 = F.leaky_relu(dis_x2, negative_slope=0.01)
        dis_x2 = self.dropout(dis_x2)
        dis_x2 = dis_x2.reshape(dis_x2.shape[0], 1, dis_x2.shape[1])

        dis_x3 = self.dis_drsnet(dis_x2.to(torch.float32))
        dis_x3 = F.leaky_relu(dis_x3, negative_slope=0.01)
        dis_x3 = self.dropout(dis_x3)

        dis_x4 = self.dis_gat1((dis_x3 + dis_x1).to(torch.float32), dis_edge_index, dis_edge_weight.to(torch.float32))

        drug_x1 = self.drug_GCN1(drug_data, drug_edge_index, drug_edge_weight)
        # drug_x1 = self.drug_ln1(drug_x1.to(torch.float32))  # 应用 LayerNorm
        drug_x1 = F.leaky_relu(drug_x1, negative_slope=0.01)
        drug_x1 = self.dropout(drug_x1)

        drug_x2 = self.drug_GCN2(drug_x1.to(torch.float32), drug_edge_index, drug_edge_weight)
        # drug_x2 = self.drug_ln2(drug_x2.to(torch.float32))  # 应用 LayerNorm
        drug_x2 = F.leaky_relu(drug_x2, negative_slope=0.01)
        drug_x2 = self.dropout(drug_x2)
        drug_x2 = drug_x2.reshape(drug_x2.shape[0], 1, drug_x2.shape[1])

        drug_x3 = self.drug_drsnet(drug_x2.to(torch.float32))
        drug_x3 = F.leaky_relu(drug_x3, negative_slope=0.01)
        drug_x3 = self.dropout(drug_x3)

        drug_x4 = self.drug_gat1((drug_x3 + drug_x1).to(torch.float32), drug_edge_index, drug_edge_weight.to(torch.float32))

        output = torch.mm(drug_x4, dis_x4.t())
        return output

    def decode(self, output, edge_index):   # 点积解码器
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        link_logits = output[src_nodes, dst_nodes]  # 一维张量
        if link_logits.dim() > 1:
            link_logits = link_logits.squeeze()

        return link_logits


# ---------------------------------------------DRSNet---------------------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RSBU_CW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super(RSBU_CW, self).__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 2 if down_sample else 1
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = gap.squeeze(-1)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = threshold.unsqueeze(-1)
        # 软阈值化
        sub = x_abs - threshold
        zeros = torch.zeros_like(sub)
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)

        if self.down_sample:
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:
            input = nn.functional.pad(input, (0, 0, 0, self.out_channels - self.in_channels))

        result = x + input
        return result


class DRSNet(nn.Module):
    def __init__(self, output):
        super(DRSNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True),
            RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=False),
            RSBU_CW(in_channels=4, out_channels=8, kernel_size=3, down_sample=True),
            RSBU_CW(in_channels=8, out_channels=8, kernel_size=3, down_sample=False),
            RSBU_CW(in_channels=8, out_channels=16, kernel_size=3, down_sample=True),
            RSBU_CW(in_channels=16, out_channels=16, kernel_size=3, down_sample=False)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = Flatten()
        self.classifier = nn.Linear(16, output)

    def forward(self, input):
        x = self.conv1(input)
        x = self.blocks(x)
        x = self.global_average_pool(x)
        x = self.flatten(x)
        output = self.classifier(x)
        return output
# ------------------------------------------------------------------------------------------------------------------------

