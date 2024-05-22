from attack.embedding.contrastive_learning.model.augmentation import *
from attack.embedding.contrastive_learning.model.gcn4grace import GCN4GraceHetero
from attack.embedding.contrastive_learning.model.grace import GRACE


class Metacl(torch.nn.Module):
    def __init__(self, data, device, activation, num_hidden, drop_scheme, drop_edge_rate_1, drop_edge_rate_2,
                 drop_feature_rate_1, drop_feature_rate_2, num_proj_hidden, tau, learning_rate, weight_decay,
                 num_epochs):
        super(Metacl, self).__init__()
        self.model = None
        self.optimizer = None
        self.device = device
        self.data = data
        self.drop_weights = None
        self.feature_weights = None
        self.drop_scheme = drop_scheme
        self.num_hidden = num_hidden
        self.activation = activation
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.num_proj_hidden = num_proj_hidden
        self.tau = tau
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

    def drop_edge(self, p):
        new_edge_indexs = self.drop_edge_weighted(self.data, self.drop_weights, p=p,
                                                  threshold=0.7)
        return new_edge_indexs

    def drop_edge_weighted(self, data, edge_weights_arr, p: float, threshold: float = 1.):
        edge_indexs = []
        for i in range(len(data)):
            edge_index = list(data.values())[i]['edge_index']
            edge_weights = edge_weights_arr[i]
            edge_weights = edge_weights / edge_weights.mean() * p
            edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
            sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
            edge_indexs.append(edge_index[:, sel_mask])
        return edge_indexs

    def train_gcn(self):
        self.model.train()
        self.optimizer.zero_grad()
        edge_index_1 = self.drop_edge(self.drop_edge_rate_1)
        edge_index_2 = self.drop_edge(self.drop_edge_rate_2)
        x_1 = self.drop_feature(self.data, self.drop_feature_rate_1)
        x_2 = self.drop_feature(self.data, self.drop_feature_rate_2)
        edge_sp_adj_1 = []
        edge_sp_adj_2 = []
        for i in range(len(edge_index_1)):
            edge_sp_adj_1.append(
                self.edge_index_2_spare_adj(edge_index_1[i], list(self.data.values())[i]['num_nodes'], self.device))
            edge_sp_adj_2.append(
                self.edge_index_2_spare_adj(edge_index_2[i], list(self.data.values())[i]['num_nodes'], self.device))

        if self.drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = self.drop_feature_weighted(self.data, self.feature_weights, self.drop_feature_rate_1)
            x_2 = self.drop_feature_weighted(self.data, self.feature_weights, self.drop_feature_rate_2)
        data1 = {
            'seq': x_1,
            'adj': edge_sp_adj_1
        }
        data2 = {
            'seq': x_2,
            'adj': edge_sp_adj_2
        }
        z1 = self.model(data1)
        z2 = self.model(data2)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def drop_feature_weighted(self, data, ws, p: float, threshold: float = 0.7):
        i = 0
        xs = []
        for x in [item['x'] for item in data.values()]:
            w = ws[i]
            w = w / w.mean() * p
            w = w.where(w < threshold, torch.ones_like(w) * threshold)
            drop_prob = w

            drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

            _x = x.clone()
            _x[:, drop_mask] = 0.
            xs.append(_x)
            i += 1
        return xs

    def drop_feature(self, data, drop_prob):
        feats = []
        for x in [item['x'] for item in data.values()]:
            drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
            new_x = x.clone()
            new_x[:, drop_mask] = 0
            feats.append(new_x)
        return feats

    def degree_drop_weights(self, data):
        weights_arr = []
        for edge_index in [item['edge_index'] for item in data.values()]:
            edge_index_ = to_undirected(edge_index)
            deg = degree(edge_index_[1])
            deg_col = deg[edge_index[1]].to(torch.float32)
            s_col = torch.log(deg_col)
            weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
            weights_arr.append(weights)
        return weights_arr

    def compute_drop_weights(self):
        self.drop_weights = self.degree_drop_weights(self.data)
        node_degs = []
        for edge_index_item in [item['edge_index'] for item in self.data.values()]:
            edge_index_ = to_undirected(edge_index_item)
            node_deg = degree(edge_index_[1])
            node_degs.append(node_deg)
        self.feature_weights = self.feature_drop_weights(self.data, node_c=node_degs)

    def feature_drop_weights(self, data, node_c):
        i = 0
        ss = []
        for x in [item['x'] for item in data.values()]:
            _x = x.to(torch.bool).to(torch.float32).cpu()
            w = _x.t() @ node_c[i]
            w = w.log()
            s = (w.max() - w) / (w.max() - w.mean())
            i += 1
            ss.append(s)
        return ss

    def inner_train(self):
        gcn = GCN4GraceHetero(self.device, self.data, self.num_hidden, get_activation(self.activation)).to(
            self.device)

        self.model = GRACE(
            encoder=gcn,
            num_hidden=self.num_hidden,
            num_proj_hidden=self.num_proj_hidden,
            tau=self.tau).to(
            self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.compute_drop_weights()

        for epoch in range(1, self.num_epochs + 1):
            loss = self.train_gcn()
            if epoch % 100 == 0:
                print('GRACE Model Epoch {}, training loss: {} ,'.format(epoch, loss))

    def compute_gradient(self, pe1, pe2, pf1, pf2):
        self.model.eval()
        self.compute_drop_weights()
        edge_index_1 = self.drop_edge(pe1)
        edge_index_2 = self.drop_edge(pe2)
        x_1 = drop_feature(self.data.x, pf1)
        x_2 = drop_feature(self.data.x, pf2)
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        if self.drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = self.drop_feature_weighted(self.data, self.feature_weights, pf1)
            x_2 = self.drop_feature_weighted(self.data, self.feature_weights, pf2)
        edge_adj_1 = edge_sp_adj_1.to_dense()
        edge_adj_2 = edge_sp_adj_2.to_dense()
        edge_adj_1.requires_grad = True
        edge_adj_2.requires_grad = True
        z1 = self.model(x_1, edge_adj_1, sparse=False)
        z2 = self.model(x_2, edge_adj_2, sparse=False)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        return edge_adj_1.grad, edge_adj_2.grad

    def start_train(self):
        self.inner_train()

        self.model.eval()

    def get_logits_with_grad(self, adj=None):
        self.model.eval()
        if adj is None:
            adj_sp = torch.sparse.FloatTensor(self.data.edge_index,
                                              torch.ones(self.data.edge_index.shape[1]).to(self.device),
                                              [self.data.num_nodes, self.data.num_nodes])
            adj = adj_sp.to_dense()

        feat = self.data.x
        return self.model(feat, adj, sparse=False), adj, feat

    @staticmethod
    def edge_index_2_spare_adj(edge_index, num_nodes, device):
        return torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.shape[1]).cpu(),
                                        [num_nodes, num_nodes]).to(device)

    def get_logits_no_grad(self, data):
        with torch.no_grad():
            new_data = dict(zip(
                ['seq', 'adj'],
                [[item['x'] for item in data.values()],
                 [self.edge_index_2_spare_adj(item['edge_index'], item['num_nodes'], self.device) for item in
                  data.values()]]
            ))

            self.model.eval()
            return self.model(new_data)

    def get_logits_with_adj_grad(self, adj):
        self.eval()
        return self.model(self.data.x, adj, sparse=False)
