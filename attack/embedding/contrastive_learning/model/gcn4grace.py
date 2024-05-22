import torch
import torch.nn as nn

from attack.attack_utils.from_deep_robust import normalize_adj_tensor


class GCN4GraceHetero(nn.Module):
    def __init__(self, device, data: dict, out_ft, act, dropout=0, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = {}
        self.fuse = torch.nn.Linear(len(data), 1).to(device)
        i = 0
        for k, v in data.items():
            self.layers[str(i)] = GCN4Grace(v['num_features'], out_ft, act, dropout=0).to(device)
            i += 1
    def forward(self, data):
        embeddings = []
        for k in range(len(data['adj'])):
            emb = self.layers[str(k)].forward(
                data['seq'][k], data['adj'][k], sparse=False
            )
            embeddings.append(emb)
        total_emds = torch.stack(embeddings, dim=0)
        total_emds = total_emds.T
        final_emds = self.fuse.forward(total_emds)
        return final_emds


class GCN4Grace(nn.Module):
    def __init__(self, in_ft, out_ft, act, dropout=0, bias=True):
        super(GCN4Grace, self).__init__()
        self.fc1 = nn.Linear(in_ft, 2 * out_ft, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(2 * out_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias1 = nn.Parameter(torch.FloatTensor(2 * out_ft))
            self.bias1.data.fill_(0.0)
            self.bias2 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias2.data.fill_(0.0)
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj, sparse=False):
        adj_norm = normalize_adj_tensor(adj, sparse=sparse)
        adj_norm = adj_norm.float()
        seq_fts1 = self.fc1(seq)
        if sparse:
            out1 = torch.spmm(adj_norm, seq_fts1)
        else:
            out1 = torch.mm(adj_norm, seq_fts1)
        if self.bias1 is not None:
            out1 += self.bias1
        out1 = self.act(out1)
        out1 = self.dropout(out1)

        seq_fts2 = self.fc2(out1)
        if sparse:
            out2 = torch.spmm(adj_norm, seq_fts2)
        else:
            out2 = torch.mm(adj_norm, seq_fts2)
        if self.bias2 is not None:
            out2 += self.bias2
        return self.act(out2)
