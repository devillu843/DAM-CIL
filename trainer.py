import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):  # 传入json文件参数
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    # 第一个增量阶段的类别个数，默认使用每个增量阶段类别相同
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"],
                                          init_cls, args['increment'])  # 保存地址
    # logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], 'exp', init_cls,
    #                                       args['increment'])  # 保存地址

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilenames = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        # 'exp',
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilenames + '.log', mode='a'),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )

    # 设置随机种子和使用的设备，打印参数
    _set_random()
    _set_device(args)
    print_args(args)

    # 处理数据成数据类别增量流，类中存在的属性包含：数据（训练集、测试集），类别排序，按照类别映射的标签文件
    data_manager = DataManager(args["dataset"], args["shuffle"], args["seed"],
                               args["init_cls"], args["increment"], args)
    # 实例化模型，这里选取der进行阅读。此处实例化的不是网络模型，将网络模型和其余参数封装到一起
    model = factory.get_model(args["model_name"], args)

    # TODO:CNN曲线，NME：Normalized mean error？
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    resume_one = True
    for task in range(data_manager.nb_tasks):  # 增量的次数
        # 记录参数量
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(
            count_parameters(model._network, True)))

        model.incremental_train(data_manager)

        # 验证,具体的nme那块没咋看懂
        cnn_accy, nme_accy = model.eval_task(model._cur_task)
        logging.info("eval_task 后:{}".format(torch.cuda.memory_allocated(0)))

        # 完成后冻结网络参数，更新类别数
        model.after_task()

        if nme_accy is not None:
            # 记录虚线
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            # nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            # logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
