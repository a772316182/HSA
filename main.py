# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao


import argparse

import torch

from openhgnn.experiment import Experiment

'''
原始主文件
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', default='GTN', type=str, help='name of models')
    parser.add_argument('--model', '-m', default='SHGP', type=str, help='name of models')
    # parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    parser.add_argument('--task', '-t', default='pretrain', type=str, help='name of task')
    # parser.add_argument('--dataset', '-d', default='acm4GTN', type=str, help='name of datasets')
    parser.add_argument('--dataset', '-d', default='mag', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='1', type=int, help='-1 means cpu')
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

    if args.model == 'GTN':
        torch.save(flow.model.state_dict(), f'./GTN@{args.dataset}.pth')
