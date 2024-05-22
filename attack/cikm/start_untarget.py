import operator

from torch import optim
from torch.nn.functional import mse_loss, cross_entropy
from tqdm import tqdm

from attack.attack_utils.add_perturbations import *
from attack.attack_utils.attack_evaluator import *
from attack.attack_utils.grad_extract import get_grad_SHGP_dgl
from attack.attack_utils.hg_info_extract import *
from attack.cikm.model import RelationGCNs
from openhgnn import set_random_seed


def get_adj_dict_for_SHGP(g: dgl.DGLHeteroGraph, device):
    adj_dict_for_SHGP = dict(zip(g.ntypes, [{} for _ in g.ntypes]))
    for item in g.canonical_etypes:
        source = item[0]
        target = item[2]
        adj_dict_for_SHGP[source][target] = g.adj(etype=item[1]).to_dense().to_sparse().to(device)
    return adj_dict_for_SHGP


def hsa_start_cikm(flow: NodeClassification,
                   seed: int,
                   attack_rate: float,
                   level_3_query_limit: int,
                   level_1_training_epoch: int,
                   level_2_query_limit: int,
                   level_2_training_epoch: int,
                   level_3_training_epoch: int,
                   weight_decay_lv1: float,
                   weight_decay_lv3: float,
                   surrogate_lr_lv1: float,
                   surrogate_lr_lv3: float,
                   level_3_training_batch_size: int,
                   lv3_attack_rate: float,
                   weight_decay_lv2: float,
                   surrogate_lr_lv2: float,
                   level_2_training_batch_size: int
                   ):
    set_random_seed(seed)
    category_type = flow.category
    victim_nodes = flow.test_idx
    flow_test_idx = flow.test_idx
    lv1_mask = flow_test_idx
    lv2_mask = flow_test_idx

    victim_model = flow.model
    victim_model.eval()
    label = flow.labels
    device = flow.args.device
    example_graph: dgl.DGLHeteroGraph = copy.deepcopy(flow.hg.clone())
    connect_type_matrix = get_connect_type_matrix(example_graph, example_graph.ntypes)
    etype_dict = get_etype_dict(example_graph)
    h_dict = flow.hg.ndata['h']
    feat, meta_paths = extract_metapath_from_dataset_for_surrogate(flow)
    feat = feat.to(device)
    in_size = feat.shape[1]
    hidden_size = 32
    out_size = flow.num_classes

    # test is surrogate model can work well
    surrogate_model_for_norm_train_eval = RelationGCNs(
        category=category_type,
        example_graph=example_graph,
        h_dict=h_dict,
        out_size=out_size,
        in_size=in_size,
        device=device
    ).to(device)
    optimizer = torch.optim.Adam(surrogate_model_for_norm_train_eval.parameters(), lr=1e-3, weight_decay=5e-4)
    adj_dict_for_SHGP_model = get_adj_dict_for_SHGP(example_graph, device)
    for i in range(10):
        surrogate_model_for_norm_train_eval.train()
        logits = surrogate_model_for_norm_train_eval.forward(adj_dict_for_SHGP_model)
        loss = cross_entropy(logits[flow.train_idx], label[flow.train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            with torch.no_grad():
                surrogate_model_for_norm_train_eval.eval()
                logits = surrogate_model_for_norm_train_eval.forward(adj_dict_for_SHGP_model)
                logits = logits[flow.test_idx]
                labels = label[flow.test_idx]
                _, indices = torch.max(logits, dim=1)
                correct = torch.sum(indices == labels)
                acc = correct.item() * 1.0 / len(labels)
                print('test: ', i, acc)
                print('loss: ', i, loss.item())
    del surrogate_model_for_norm_train_eval
    print('=====================================')

    surrogate_model = RelationGCNs(
        category=category_type,
        example_graph=example_graph,
        h_dict=h_dict,
        in_size=in_size,
        out_size=out_size,
        device=device
    ).to(device)
    """
     lv1 training
    """

    optimizer_lv1 = optim.NAdam(surrogate_model.parameters(), lr=surrogate_lr_lv1, weight_decay=weight_decay_lv1)
    for i in range(level_1_training_epoch):
        surrogate_model.train()
        target_model_pred = get_pred_label(flow, example_graph, lv1_mask).to(device)
        surrogate_model_logits = surrogate_model.forward(get_adj_dict_for_SHGP(example_graph, device))
        surrogate_model_pred = surrogate_model_logits[lv1_mask].argmax(dim=1)
        loss_train = cross_entropy(surrogate_model_logits[lv1_mask], target_model_pred)
        optimizer_lv1.zero_grad()
        loss_train.backward()
        optimizer_lv1.step()
        if i % 10 == 0:
            print(surrogate_model_pred.sum().item(), surrogate_model_logits.sum().item())
            print(
                f"普通训练 => In epoch {i}, loss: {loss_train.item():.3f}"
            )

    """
           lv2 training
       """
    lv2_buffer = []
    for i in tqdm(range(level_2_query_limit)):
        source_netype, etype, target_netype = random.choice(example_graph.canonical_etypes)
        etype_reverse = connect_type_matrix[target_netype][source_netype]
        hg_remove_edges_by_etype = remove_all_edges_by_etype(example_graph, etype=etype, etype_reverse=etype_reverse,
                                                             device=device)
        target_model_pred = get_pred_label(flow, hg_remove_edges_by_etype, lv1_mask).to(device)
        lv2_buffer.append({
            'chosen_etype': etype,
            'hg': hg_remove_edges_by_etype.cpu(),
            'pred': target_model_pred.cpu(),
        })

    optimizer_lv2 = optim.NAdam(surrogate_model.parameters(), lr=surrogate_lr_lv2, weight_decay=weight_decay_lv2)
    for i in range(level_2_training_epoch):
        surrogate_model.train()
        batchs = random.choices(lv2_buffer, k=level_2_training_batch_size)
        loss_train = 0.0
        for item in batchs:
            hg = item['hg'].to(device)
            pred = item['pred'].to(device)
            surrogate_model_logits = surrogate_model.forward(get_adj_dict_for_SHGP(hg, device))
            surrogate_model_pred = surrogate_model_logits[lv1_mask].argmax(dim=1)
            loss_train += cross_entropy(surrogate_model_logits[lv1_mask], pred)
        optimizer_lv2.zero_grad()
        loss_train.backward()
        optimizer_lv2.step()
        print(
            f"边类型保留 => In epoch {i}, loss: {loss_train.item():.3f}"
        )

    """
        lv3 training
    """
    lv3_buffer = []
    print('=====================decision boundary exploration=====================')
    for i in tqdm(range(level_3_query_limit)):
        source_netype, etype, target_netype = random.choice(example_graph.canonical_etypes)
        hg_purb_edges_by_etype = random_flip_hg_auto_reverse(example_graph, lv3_attack_rate, etype)

        target_model_pred = get_pred_label(flow, hg_purb_edges_by_etype, lv2_mask).to(device)

        lv3_buffer.append({
            'chosen_etype': etype,
            'hg': hg_purb_edges_by_etype.cpu(),
            'pred': target_model_pred.cpu(),
        })

    print('=====================buffer statistics=====================')
    pred_mse_dicts = dict(zip(example_graph.etypes, [[] for _ in example_graph.etypes]))
    for item in lv3_buffer:
        if item['chosen_etype'] is not None:
            pred_mse_dicts[item['chosen_etype']].append(
                mse_loss(item['pred'].float(), lv3_buffer[0]['pred'].float()).item())
    for k, v in pred_mse_dicts.items():
        print(k, numpy.mean(v))

    print('=====================buffer statistics=====================')
    optimizer_lv3 = optim.NAdam(surrogate_model.parameters(), lr=surrogate_lr_lv3, weight_decay=weight_decay_lv3)
    for i in range(level_3_training_epoch):
        batchs = random.choices(lv3_buffer, k=level_3_training_batch_size)
        loss_train = 0.0
        for item in batchs:
            hg = item['hg'].to(device)
            pred = item['pred'].to(device)

            surrogate_model_logits = surrogate_model.forward(get_adj_dict_for_SHGP(hg, device))
            surrogate_model_pred = surrogate_model_logits[lv2_mask].argmax(dim=1)

            loss_train += cross_entropy(surrogate_model_logits[lv2_mask], pred)
        optimizer_lv3.zero_grad()
        loss_train.backward()
        optimizer_lv3.step()
        print(surrogate_model_pred.sum().item(), surrogate_model_logits.sum().item())
        print(
            f"边类型扰动 => In epoch {i}, loss: {loss_train.item():.3f}"
        )

    cnt = 0
    iii = 0
    filtered_etype = get_filtered_etype(example_graph, example_graph.ntypes)

    print('current %d => total %d' % (iii, len(victim_nodes)))
    print('current %d => total %d' % (iii, len(victim_nodes)))
    graph_modified = copy.deepcopy(example_graph).clone()
    graph_modified_adj_dict = get_etype_adj_dict(graph_modified, device)
    print('=== Attack Starting ===')
    purb_num = int(example_graph.num_edges() * attack_rate // 2)
    for i in range(purb_num):
        etype_grads = get_grad_SHGP_dgl(flow, graph_modified, get_adj_dict_for_SHGP(graph_modified, device),
                                        surrogate_model, flow_test_idx)
        perturbations_on_each_etype = []
        for etype in filtered_etype:
            [source_ntype, target_ntype] = etype_dict[etype]
            etype_reverse = connect_type_matrix[target_ntype][source_ntype]
            modified_adj_etype = graph_modified_adj_dict[etype]
            grad = etype_grads[etype]
            # todo check!!!
            grad = grad * (-2 * modified_adj_etype + 1)
            grad_max_value = torch.max(grad)
            torch_argmax_grad__item = torch.argmax(grad).item()
            source = torch_argmax_grad__item // grad.shape[1]
            target = torch_argmax_grad__item % grad.shape[1]

            perturbations_on_each_etype.append({
                'has_edge': (graph_modified_adj_dict[etype][source, target] == 1).item(),
                'etype': etype,
                'etype_reverse': etype_reverse,
                'source': source,
                'source_ntype': source_ntype,
                'target': target,
                'target_ntype': target_ntype,
                'grad_value': grad_max_value.item(),
                'mix_value': grad_max_value.item()
            })
        perturbations_on_each_etype.sort(key=operator.itemgetter('mix_value'), reverse=True)
        chosen_perturbation = perturbations_on_each_etype[0]
        chosen_etype = chosen_perturbation['etype']
        chosen_etype_reverse = chosen_perturbation['etype_reverse']
        chosen_source = chosen_perturbation['source']
        chosen_target = chosen_perturbation['target']
        print('=' * 100)
        print(chosen_perturbation)
        print('=' * 100)
        modified_adj_etype = graph_modified_adj_dict[chosen_etype]
        modified_adj_etype_reverse = graph_modified_adj_dict[chosen_etype_reverse]
        value = -2 * modified_adj_etype[chosen_source][chosen_target] + 1
        value_reverse = -2 * modified_adj_etype_reverse[chosen_target][chosen_source] + 1

        graph_modified_adj_dict[chosen_etype][chosen_source, chosen_target] += value
        graph_modified_adj_dict[chosen_etype_reverse][chosen_target, chosen_source] += value_reverse
        graph_modified = from_adj_dict_to_dgl_graph(example_graph=example_graph, adj_dict=graph_modified_adj_dict)

    print('example_graph.num_edges() ', example_graph.num_edges())
    print('graph_modified.num_edges() ', graph_modified.num_edges())
    print('edge add rate ', (graph_modified.num_edges() - example_graph.num_edges()) / example_graph.num_edges())

    ori_micro_f1, ori_macro_f1, ori_loss, ori_logits = eval_evasion(example_graph, flow, flow_test_idx)
    victim_micro_f1, victim_macro_f1, victim_loss, victim_logits = eval_evasion(graph_modified, flow, flow_test_idx)

    micro_down = ori_micro_f1 - victim_micro_f1
    macro_down = ori_macro_f1 - victim_macro_f1

    print(f'ori_micro => {ori_micro_f1 * 100}')
    print(f'victim_mirco => {victim_micro_f1 * 100}')
    print('===')
    print(f'ori_macro => {ori_macro_f1 * 100}')
    print(f'victim_marco => {victim_macro_f1 * 100}')
    print('===')
    print('micro_down : %s %%' % (micro_down * 100))
    print('macro_down : %s %%' % (macro_down * 100))

    return micro_down
