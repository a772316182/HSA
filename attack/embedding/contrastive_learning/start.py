import dgl
import torch

from attack.attack_utils.hg_info_extract import generate_mata_path_reachable_adjs
from attack.embedding.contrastive_learning.model.metacl import Metacl
from openhgnn.trainerflow.node_classification import NodeClassification


def cl_based_embedding(flow: NodeClassification, epoch: int, path_emb: str) -> torch.Tensor:
    example_graph: dgl.DGLHeteroGraph = flow.hg.clone()
    metapath_reachable_adjs = generate_mata_path_reachable_adjs(flow, example_graph,
                                                                numpy_format=False,
                                                                edge_index_format=True)
    data = {}
    for k, v in metapath_reachable_adjs.items():
        data_item = {}
        data_item['x'] = flow.hg.ndata['h'][flow.category]
        data_item['num_features'] = data_item['x'].shape[1]
        data_item['num_nodes'] = example_graph.num_nodes(ntype=flow.category)
        data_item['edge_index'] = torch.nonzero(v).T
        data[k] = data_item

    contrastive_model = Metacl(
        data=data,
        device=flow.device,
        drop_scheme='degree',
        drop_edge_rate_1=0.2,
        drop_edge_rate_2=0.3,
        drop_feature_rate_1=0.1,
        drop_feature_rate_2=0.2,
        num_hidden=128,
        activation='prelu',
        num_proj_hidden=128,
        tau=0.25,
        learning_rate=5e-2,
        weight_decay=1e-5,
        num_epochs=800
    ).to(flow.device)

    contrastive_model.start_train()
    embedding = contrastive_model.get_logits_no_grad(data)
    torch.save(embedding, path_emb)
    return embedding
