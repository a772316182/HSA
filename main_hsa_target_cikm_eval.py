# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn

import argparse

from attack.cikm.start_target import hsa_start_cikm
from attack.query_fullgraph_vs_subgraph.start import hsa_start_buffer_target_dgl_fullgraph_vs_etype_purb
from openhgnn.experiment import Experiment

'''
n2v target
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='HAN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    parser.add_argument('--dataset', '-d', default='imdb4GTN', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='3', type=int, help='-1 means cpu')

    parser.add_argument('--use_best_config', action='store_true', help='will load utils.best_config')
    parser.add_argument('--load_from_pretrained', action='store_true', help='load model from the checkpoint')
    args = parser.parse_args()

    if args.model == 'MHNF' and args.dataset == 'dblp4GTN':
        use_best_config = False
        load_from_pretrained = False
    else:
        use_best_config = True
        load_from_pretrained = False

    experiment = Experiment(model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu,
                            use_best_config=use_best_config, load_from_pretrained=load_from_pretrained)

    result, flow = experiment.run()
    flow.new_args = args

    hsa_start_cikm(
        flow=flow,

        seed=0,
        purb_limit_per_target=4,

        level_1_training_epoch=100,
        surrogate_lr_lv1=5e-3,
        weight_decay_lv1=1e-4,

        level_2_training_batch_size=8,
        level_2_query_limit=50,
        level_2_training_epoch=8,
        surrogate_lr_lv2=5e-4,
        weight_decay_lv2=5e-5,

        level_3_training_batch_size=8,
        level_3_query_limit=50,
        level_3_training_epoch=8,
        surrogate_lr_lv3=5e-4,
        weight_decay_lv3=5e-5,

        lv3_attack_rate=0.5,
    )
