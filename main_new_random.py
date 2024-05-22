# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao


import argparse

import numpy
from numba import jit

from attack.random_n2v.start import hsa_start_random_n2v_targeted
from openhgnn.experiment import Experiment

'''
原始主文件
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
        use_best_config = False
        load_from_pretrained = False

    experiment = Experiment(model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu,
                            use_best_config=use_best_config, load_from_pretrained=load_from_pretrained)

    result, flow = experiment.run()
    flow.new_args = args

    hsa_start_random_n2v_targeted(
        flow=flow,
        seed=0,
        purb_limit_per_target=1,
        beta=100,
        top_k_num=1000,
        use_z_score_norm=False,
        mode='cl'
    )

    # betas = numpy.arange(0., 2., 5e-2).tolist()
    # for beta in betas:
    #     hsa_start_random_n2v_targeted(
    #         flow=flow,
    #         seed=0,
    #         purb_limit_per_target=1,
    #         beta=0.1,
    #         top_k_num=100,
    #         use_z_score_norm=False,
    #         mode='cl'
    #     )
