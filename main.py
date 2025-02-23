import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)  # 读取的json文件参数
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)  # 传入json文件参数


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config',
                        type=str,
                        default='./exps/DAM.json',
                        help='Json file of settings.')
    '''
    "prefix": "reproduce",          前缀 
    "dataset": "cifar100",          数据集
    "memory_size": 2000,            增量学习的总样本数，当前阶段有K类时，每类有memory_size/K个样本
    "memory_per_class": 20,         每类的样本数目
    "fixed_memory": false,          前置规定储存
    "shuffle": true,                打乱顺序
    "init_cls": 10,                 初始的类
    "increment": 10,                增量
    "model_name": "finetune",       模型名称
    "convnet_type": "resnet32",     网络类型
    "device": ["0","1","2","3"],    使用设备
    "seed": [1993]                  随机种子
    '''
    return parser


if __name__ == '__main__':
    main()
