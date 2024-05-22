import os.path

import torch

from attack.attack_utils.hg_info_extract import get_etype_adj_dict, extract_metapath_from_dataset_for_surrogate, \
    get_transition
from attack.embedding.contrastive_learning.start import cl_based_embedding
from attack.embedding.n2v.start import n2vec_based_embedding
from openhgnn.trainerflow.node_classification import NodeClassification


def no_model_similarity(flow: NodeClassification, epoch: int, mode='n2v', return_embedding=False) -> torch.Tensor:
    assert mode in ['n2v', 'cl']
    _, metapaths = extract_metapath_from_dataset_for_surrogate(flow)
    trans_adj_list = get_transition(get_etype_adj_dict(flow.hg.clone(), 'cpu'), metapaths)

    path_emb = f'./similarity/{flow.args.dataset}_{mode}_embedding.pt'

    if not os.path.exists(path_emb):
        if mode == 'n2v':
            n2vec_based_embedding(flow, epoch=50, path=path_emb)
        elif mode == 'cl':
            cl_based_embedding(flow, epoch=800, path_emb=path_emb)
    else:
        print('load from the existing similarity file')

    embedding = torch.load(path_emb)
    similarity = torch.matmul(embedding, embedding.T).cpu()
    similarity = (flow.hyper_alpha * trans_adj_list[0] + flow.hyper_beta * trans_adj_list[1]) * similarity
    if not return_embedding:
        return similarity
    else:
        embedding = torch.load(path_emb)
        return similarity, embedding
