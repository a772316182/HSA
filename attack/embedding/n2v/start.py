import os.path

import dgl
import torch

from attack.attack_utils.hg_info_extract import generate_mata_path_reachable_adjs
from attack.embedding.n2v.n2vec import DeepWalk
from openhgnn.trainerflow.node_classification import NodeClassification


def n2vec_based_embedding(flow: NodeClassification, epoch: int, path: str) -> torch.Tensor:
    print('similarity file not exist')
    hg: dgl.DGLHeteroGraph = flow.hg.cpu().clone()
    embedding_dim = 8
    category_numb_nodes = hg.num_nodes(ntype=flow.category)
    embedding = torch.zeros(size=[category_numb_nodes, embedding_dim])
    mata_path_reachable_adjs = generate_mata_path_reachable_adjs(flow, hg, numpy_format=True)

    for k, v in mata_path_reachable_adjs.items():
        meta_path_reachable = v
        model = DeepWalk()
        model.fit(adj=meta_path_reachable, embedding_dim=embedding_dim, epoch=epoch)
        embedding = embedding + torch.tensor(model.embedding)
    # 算每对节点间的余弦相似度，作为梯度信息的辅助
    torch.save(embedding, path)
    return embedding
