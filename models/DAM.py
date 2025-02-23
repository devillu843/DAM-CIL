import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DAMNET
from utils.toolkit import count_parameters, target2onehot, tensor2numpy, accuracy
from utils.utils import logit_adjustment
from scipy.spatial.distance import cdist
'''
此版本增加split_sum，1、后续版本添加全量ad_tro,采用平均数划分，2、后续若根据分类器划分，需多记录每个分类器在不同类别上的准确率
'''

EPSILON = 1e-8

init_epoch = 5
init_lr = 0.001
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 5e-5

epochs = 5
lrate = 0.001
milestones = [80, 120, 150]
lrate_decay = 0.001
batch_size = 64
weight_decay = 2e-4
num_workers = 8  # 看设备够不够用了
T = 2


class DAM(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        # 传入json文件参数
        # 实例化网络
        self._network = DAMNET(args, False)  # 初始化的网络结构为空
        self.weight = [torch.tensor(1).to(self._device)]
        self.weight_best = [torch.tensor(1).to(self._device)]
        self._known_classes_curve = []
        self.adjustment = {}
        self.mode = args["mode"]
        self.args = args
        self.dataset = args["dataset"]
        self.init_cls = args["init_cls"]
        self.increment = args["increment"]
        self.model_name = args["model_name"]
        # self.model_name = 'decouple_fc_Orthogonal_bias_split_sum'
        self.convnet_type = args["convnet_type"]

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        # self._cur_task初始化为-1，记录现在是第几次增量
        self._cur_task += 1
        # 总任务量=已经学习的分类+增量的类别数
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task)

        # 加载上一轮最好的权重
        if self._cur_task > 0:
            self._network.load_state_dict(
                torch.load("./weight/{}_{}_{}_{}_{}_{}.pt".format(
                    self.dataset, self.init_cls, self.increment,
                    self.model_name, self.convnet_type, self._cur_task - 1)))

        # 更新新的分类器，继承旧分类器的参数并扩张，并更新新的辅助分类器
        self._network.update_fc(self._total_classes)
        self._known_classes_curve.append(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes,
                                                self._total_classes))

        # if self._cur_task > 0:
        #     # 非第0代数据的训练时，冻结之前网络参数，即不进行梯度回传
        #     for i in range(self._cur_task):
        #         for p in self._network.convnets[i].parameters():
        #             p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info("Trainable params: {}".format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(
            # 获取训练数据，此处返回的是一个实例化的类，调用类时对数据进行变换
            np.arange(self._known_classes, self._total_classes),  # 增量的类别索引
            source="train",
            mode="train",
            appendent=self._get_memory(),  # 这里很神奇。
        )
        # 封装数据
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       drop_last=True)
        valid_dataset = data_manager.get_dataset(
            # 获取训练数据，此处返回的是一个实例化的类，调用类时对数据进行变换
            np.arange(0, self._total_classes),  # 增量的类别索引
            source="train",
            mode="test",
        )
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=True)
        test_dataset = data_manager.get_dataset(np.arange(
            0, self._total_classes),
                                                source="test",
                                                mode="test")
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      drop_last=True)
        if self._cur_task not in self.adjustment:
            self.adjustment.update({self._cur_task: {}})
        for tro in np.arange(0, 6.25, 0.25):
            self.adjustment[self._cur_task].update({
                tro:
                logit_adjustment(self.train_loader, tro).to(self._device)
            })

        if len(self._multiple_gpus) > 1:  # 多卡玩，我玩不起
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        # 开始训练了
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class,
                                    self._cur_task)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1:
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets.train()
        # if self._cur_task >= 1:
        #     for i in range(self._cur_task):
        #         self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            # 初始训练，设置优化器和学习率策略，这里没有使用辅助分类器，和平常训练相同
            # self._network.freeze_conv()
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=init_milestones,
                gamma=init_lr_decay)
            if self.mode == 'init':
                self._init_train(train_loader, test_loader, optimizer,
                                 scheduler)
                # 保存权重
                # torch.save(
                #     self._network.state_dict(),
                #     "./weight/{}_{}_{}_{}_{}_{}.pt".format(
                #         self.args['dataset'],
                #         self.args['init_cls'],
                #         self.args['increment'],
                #         self.args['model_name'],
                #         self.args['convnet_type'],
                #         self.args['mode'],
                #     ))

            elif self.mode == 'freeze':
                self._init_train_freeze(train_loader, test_loader, optimizer,
                                        scheduler)

            else:
                # 加载权重
                self._network.load_state_dict(
                    torch.load("./weight/{}_{}_{}_{}_{}_{}.pt".format(
                        self.dataset, self.init_cls, self.increment,
                        self.model_name, self.convnet_type, self._cur_task)))

                test_acc = self._test_accuracy(self._network,
                                               self.valid_loader,
                                               self._cur_task,
                                               boost=False)

                self.weight = [np.exp(i / 100) for i in test_acc]

        else:
            # 增量训练，设置优化器和学习率策略，这里使用辅助分类器，loss为两者相加，这里参考Hinton老爷子的经典Distillation
            self._network.freeze_conv()
            self._network.fc[-1].train()
            self._network.lora_layers[-1].train()

            if self._cur_task >= 1:
                self._network.fc[-2].eval()
            if self._cur_task >= 2:
                self._network.lora_layers[-2].eval()
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                       milestones=milestones,
                                                       gamma=lrate_decay)
            self._update_representation(train_loader, test_loader, optimizer,
                                        scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes -
                                                  self._known_classes)
            else:
                self._network.weight_align(self._total_classes -
                                           self._known_classes)

    def _init_train_freeze(self,
                           train_loader,
                           test_loader,
                           optimizer,
                           scheduler,
                           curtask=0):
        prog_bar = tqdm(range(init_epoch))
        test_acc_best = 0.0
        for _, epoch in enumerate(prog_bar):
            self.train()
            self._network.convnets.eval()
            self._network_module_ptr.convnets.eval()
            losses = 0.0
            correct, total = 0, 0
            loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
            # for i, (_, inputs, targets) in enumerate(train_loader):
            for i, (_, inputs, target) in loop_train:
                targets = target2onehot(target, self._total_classes)
                inputs, targets = inputs.to(self._device), targets.to(
                    self._device)
                logits = self._network(inputs)["logits"][
                    0]  # 返回字典，包含分类器结果，辅助分类器结果和经过cat操作的backbone输出特征，修改后logits中为字典，为不同分类器的结果

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(target.to(
                    self._device).expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total,
                                  decimals=2)

            test_acc = self._test_accuracy(self._network, self.valid_loader,
                                           self._cur_task)
            # self.weight = [
            #     np.log(i / 100 / (1 - i / 100)) / len(test_acc)
            #     for i in test_acc
            # ]
            self.weight = [np.exp(i / 100) for i in test_acc]

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy :{}".format(
                self._cur_task,
                epoch + 1,
                init_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            if test_acc > test_acc_best:
                test_acc_best = test_acc
                torch.save(
                    self._network.state_dict(),
                    "./weight/{}_{}_{}_{}_{}_{}.pt".format(
                        self.args['dataset'],
                        self.args['init_cls'],
                        self.args['increment'],
                        self.args['model_name'],
                        self.args['convnet_type'],
                        self._cur_task,
                    ))
                self.weight_best = self.weight
            prog_bar.set_description(info)

        logging.info(info)

    def _init_train(self,
                    train_loader,
                    test_loader,
                    optimizer,
                    scheduler,
                    curtask=0):
        prog_bar = tqdm(range(init_epoch))
        test_acc_best = 0.0
        for _, epoch in enumerate(prog_bar):
            self.train()
            # self._network.convnets.eval()
            self._network_module_ptr.convnets.eval()
            losses = 0.0
            correct, total = 0, 0
            loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
            # for i, (_, inputs, targets) in enumerate(train_loader):
            for i, (_, inputs, target) in loop_train:
                targets = target2onehot(target, self._total_classes)
                inputs, targets = inputs.to(self._device), targets.to(
                    self._device)
                logits = self._network(inputs)["logits"][
                    0]  # 返回字典，包含分类器结果，辅助分类器结果和经过cat操作的backbone输出特征，修改后logits中为字典，为不同分类器的结果

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(target.to(
                    self._device).expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total,
                                  decimals=2)

            test_acc = self._test_accuracy(self._network, self.valid_loader,
                                           self._cur_task)
            # self.weight = [
            #     np.log(i / 100 / (1 - i / 100)) / len(test_acc)
            #     for i in test_acc
            # ]
            self.weight = [np.exp(i / 100) for i in test_acc]

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy :{}".format(
                self._cur_task,
                epoch + 1,
                init_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            if test_acc > test_acc_best:
                test_acc_best = test_acc
                torch.save(
                    self._network.state_dict(),
                    "./weight/{}_{}_{}_{}_{}_{}.pt".format(
                        self.args['dataset'],
                        self.args['init_cls'],
                        self.args['increment'],
                        self.args['model_name'],
                        self.args['convnet_type'],
                        self._cur_task,
                    ))
                self.weight_best = self.weight
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self,
                               train_loader,
                               test_loader,
                               optimizer,
                               scheduler,
                               curtask=1):
        prog_bar = tqdm(range(epochs))

        test_acc = self._test_accuracy(self._network,
                                       self.valid_loader,
                                       self._cur_task,
                                       boost=False)
        # self.weight = [
        #     np.log(i / 100 / (1 - i / 100) / len(test_acc)) for i in test_acc
        # ]
        self.weight = [np.exp(i / 100) for i in test_acc]

        for _, epoch in enumerate(prog_bar):
            self.train()
            self._network.convnets.eval()
            self._network_module_ptr.convnets.eval()
            losses = 0.0
            loss_clf = 0
            losses_clf = 0.0
            losses_aux = 0.0
            losses_clf_weight = 0.0
            loss_clf_last = 0.0
            total = 0
            correct = 0
            test_acc_best = 0
            correct_fc = [0 for _ in range(self._cur_task + 1)]
            # for i, (_, inputs, targets) in enumerate(train_loader):
            loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (_, inputs, target) in loop_train:
                targets = target2onehot(target, self._total_classes)
                inputs, targets = inputs.to(self._device), targets.to(
                    self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]

                # 计算每个fc的交叉熵
                # loss_split_fc = []
                # logit_fill = []
                # for i,logit in enumerate(logits):
                #     logit_fill.append(logit.clone())
                #     logit_fill[i] = self.onehot_fill0(logit_fill[i], self._total_classes)
                # loss_clf = loss_clf + F.cross_entropy(logit_fill[i], targets)
                #     if i == 0:
                #         loss_split_fc.append(F.cross_entropy(logit_fill[i], targets))
                #     else:
                #         loss_split_fc.append(loss_split_fc[-1]+F.cross_entropy(logit_fill[i], targets))

                # loss_clf = loss_split_fc[-1].clone()

                # 单独计算最后一个fc的交叉熵，没什么用
                loss_clf_last = F.cross_entropy(logits[-1], targets)
                aux_targets = target.clone()

                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1, 0)
                aux_targets = target2onehot(
                    aux_targets, self._total_classes - self._known_classes +
                    1).to(self._device)
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                loss = loss_clf_last + loss_aux

                # 需要分别记录每个fc的准确率和boost的准确率，方便后续计算权重
                output = []
                for i, j in enumerate(logits):
                    output.append(
                        self.onehot_fill0(j, self._known_classes_curve[i]) -
                        self.adjustment[i][1.0])
                # 分别计算正确个数
                for i, j in enumerate(output):
                    predicts = torch.max(j, dim=1)[1]
                    correct_fc[i] += (predicts.cpu() == target).sum()
                total += len(targets)

                # 加权和累计
                # weight_output = torch.zeros_like(logits[-1]).to(self._device)
                # for i, j in enumerate(output):
                #     weight_output += j * self.weight[i]

                # self.weight = [
                #     np.abs(np.exp(i.cpu().detach().numpy() / len(output)))
                #     for i in output
                # ]
                # split_sum
                weight_output = self.weight_sum(logits, self.weight,
                                                self._known_classes_curve)

                # loss_clf_weight = F.cross_entropy(weight_output, targets)

                # loss = loss_clf_weight + loss_aux
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf_last.item()
                # losses_clf_weight += loss_clf_weight.item()

                _, preds = torch.max(weight_output, dim=1)
                # correct += preds.eq(target.expand_as(preds)).cpu().sum()

                correct += preds.eq(target.to(
                    self._device).expand_as(preds)).cpu().sum()

            correct = torch.from_numpy(np.array(correct))
            correct = np.around(tensor2numpy(correct) * 100 / total,
                                decimals=2)

            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)
            '''
            这里注释了原有的测试代码，下面需要对权重进行更新
            '''

            test_acc = self._test_accuracy(self._network, self.valid_loader,
                                           self._cur_task)
            # self.weight = [
            #     np.log(i / 100 / (1 - i / 100)) / len(test_acc)
            #     for i in test_acc
            # ]
            self.weight = [np.exp(i / 100) for i in test_acc]

            test_acc = self._test_accuracy(self._network, test_loader,
                                           self._cur_task)

            if test_acc[-1] > test_acc_best:
                test_acc_best = test_acc[-1]
                torch.save(
                    self._network.state_dict(),
                    "./weight/{}_{}_{}_{}_{}_{}.pt".format(
                        self.args['dataset'], self.args['init_cls'],
                        self.args['increment'], self.args['model_name'],
                        self.args['convnet_type'], self._cur_task))
                self.weight_best = self.weight
            test_acc = self._compute_accuracy(self._network, test_loader,
                                              self._cur_task)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, correct_fc {}".format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader),
                losses_clf / len(train_loader), losses_aux / len(train_loader),
                train_acc, test_acc, correct_fc)
            prog_bar.set_description(info)
            logging.info(info)

        # for tro in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        for tro in np.arange(0, 6.25, 0.25):
            test_acc = self._compute_accuracy(self._network, test_loader,
                                              self._cur_task)

            cnn_accy, nme_accy = self.eval_task_logit(self._cur_task, tro)
            logging.info(
                "The tro is {}: the post logit adjustment acc: {}".format(
                    tro, test_acc))
            if nme_accy is not None:
                # 记录虚线
                logging.info("logit adjustment CNN: {}".format(
                    cnn_accy["grouped"]))
                logging.info("logit adjustment NME: {}".format(
                    nme_accy["grouped"]))

                logging.info("logit adjustment CNN top1 curve: {}".format(
                    cnn_accy["top1"]))
                # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
                logging.info("logit adjustment NME top1 curve: {}".format(
                    nme_accy["top1"]))
                # logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
            else:
                logging.info("No NME accuracy.")
                logging.info("logit adjustment CNN: {}".format(
                    cnn_accy["grouped"]))
                logging.info("logit adjustment CNN top1 curve: {}".format(
                    cnn_accy["top1"]))
                # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
    def weight_sum(self, outputs, weight, know_classes):
        '''
        样本增量计算：50， 10，10
        样本增量计算：m,n,n
        weight采用np.exp(acc)计算而来
        self.know_classes: [50, 60, 70]
        self.know_classes: [m, m+n, m+2n]


        fc_correct: [acc0, acc1, acc2]
        outputs: [output0, output1, output2],默认补过0
        weight: [w0, w1, w2]    
        '''
        weight = [torch.tensor(x).to(self._device) for x in weight]
        out_tem = torch.zeros_like(outputs[0])
        for i, number in enumerate(know_classes):
            out_tem = self.onehot_fill0(out_tem, outputs[i].shape[1])
            if i == 0:
                for j, temp in enumerate(outputs[i:]):
                    temp = temp
                    out_tem += temp[:, :know_classes[i]] * weight[j] / (
                        len(outputs) - i)
            else:
                for j, temp in enumerate(outputs[i:]):
                    out_tem += self.onehot_fill0_before(
                        temp[:, know_classes[i - 1]:know_classes[i]],
                        outputs[i].shape[1]) * weight[j + i] / (len(outputs) -
                                                                i)
        return out_tem

    def onehot_fill0(self, one_hot_encoded, desired_length):
        if one_hot_encoded.shape[1] < desired_length:
            num_zeros_to_add = desired_length - one_hot_encoded.shape[1]
            zeros_to_add = torch.zeros(
                (one_hot_encoded.shape[0], num_zeros_to_add),
                dtype=one_hot_encoded.dtype).to(self._device)
            one_hot_encoded = torch.cat((one_hot_encoded, zeros_to_add), dim=1)
        return one_hot_encoded

    def onehot_fill0_before(self, one_hot_encoded, desired_length):
        if one_hot_encoded.shape[1] < desired_length:
            num_zeros_to_add = desired_length - one_hot_encoded.shape[1]
            zeros_to_add = torch.zeros(
                (one_hot_encoded.shape[0], num_zeros_to_add),
                dtype=one_hot_encoded.dtype).to(self._device)
            one_hot_encoded = torch.cat((zeros_to_add, one_hot_encoded), dim=1)
        return one_hot_encoded

    # TODO:这里没有进行修改，仿照上面进行修改
    def eval_task_logit(self, cur_task, tro):
        y_pred, y_true = self._eval_cnn_logit(self.test_loader, cur_task, tro)
        cnn_accy = self._evaluate_logit(y_pred, y_true)

        nme_accy = None
        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme_logit(self.test_loader,
                                                  self._class_means, cur_task)
            nme_accy = self._evaluate_logit(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn_logit(self, loader, cur_task, tro):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            output = []
            with torch.no_grad():
                outputs = self._network(inputs, cur_task)["logits"]
                for i, j in enumerate(outputs):
                    output.append(
                        self.onehot_fill0(j, self._known_classes_curve[i]) -
                        self.adjustment[i][tro])
                    # weight_output = torch.ones_like(output[-1]).to(
                    #     self._device)
            # for i, j in enumerate(output):
            #     weight_output += j * self.weight[i]
            weight_output = self.weight_sum(output, self.weight_best,
                                            self._known_classes_curve)
            predicts = torch.topk(weight_output,
                                  k=self.topk,
                                  dim=1,
                                  largest=True,
                                  sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _evaluate_logit(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true,
                                 (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def _eval_nme_logit(self, loader, class_means, cur_task):
        self._network.eval()
        # 提取向量
        vectors, y_true = self._extract_vectors(loader, cur_task)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _compute_accuracy(self, model, loader, cur_task):
        model.eval()
        total = 0
        correct = 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            output = []
            with torch.no_grad():
                # 存储每个fc的结果
                outputs = model(inputs, cur_task)["logits"]
                for i, j in enumerate(outputs):
                    output.append(j - self.adjustment[i][1.0])

                # weight_output = torch.ones_like(output[-1]).to(self._device)
                # for i, j in enumerate(output):
                #     weight_output += j * self.weight[i]
                weight_output = self.weight_sum(output, self.weight_best,
                                                self._known_classes_curve)

            predicts = torch.max(weight_output, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        if isinstance(correct, int):
            correct = torch.from_numpy(np.array(correct))
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _test_accuracy(self, model, loader, cur_task, tro=1.0, boost=True):
        model.eval()
        total = 0
        correct = [0 for _ in range(cur_task + 1)]
        weight_correct = 0
        for _, (_, inputs, targets) in enumerate(loader):
            output = []
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # 存储每个fc的结果
                outputs = model(inputs, cur_task)["logits"]
                for i, j in enumerate(outputs):
                    output.append(j - self.adjustment[i][tro])
            # 分别计算正确个数
            for i, j in enumerate(output):
                predicts = torch.max(j, dim=1)[1]
                correct[i] += (predicts.cpu() == targets).sum()
            total += len(targets)

            # boost个数
            if boost:
                weight_output = self.weight_sum(output, self.weight,
                                                self._known_classes_curve)
                predicts = torch.max(weight_output, dim=1)[1]
                weight_correct += (predicts.cpu() == targets).sum()

        correct = torch.from_numpy(np.array(correct))
        correct = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

        if boost:
            weight_correct = torch.from_numpy(np.array(weight_correct))
            weight_correct = np.around(tensor2numpy(weight_correct) * 100 /
                                       total,
                                       decimals=2)

        logging.info(
            'Test Accuracy of the fc: {}, Test Accuracy of the boost: {}'.
            format(correct, weight_correct))

        return correct

    #TODO:重构部分，和网络对应
    def eval_task(self, cur_task):
        # 获取最终的判断结果
        y_pred, y_true = self._eval_cnn(
            self.test_loader,
            cur_task,
        )
        cnn_accy = self._evaluate(y_pred, y_true)

        nme_accy = None
        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader,
                                            self._class_means, cur_task)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader, cur_task):
        # TODO：这里需要进行修改，在最终推理的时候需要进行boost。
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            output = []
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, cur_task)["logits"]
            # 存储每个fc的结果
            # outputs = self._network(inputs, cur_task)["logits"]
            for i, j in enumerate(outputs):
                output.append(j - self.adjustment[i][1.0])
            # weight_output = torch.ones_like(output[-1]).to(self._device)
            # for i, j in enumerate(output):
            #     weight_output += j * self.weight[i]
            weight_output = self.weight_sum(output, self.weight,
                                            self._known_classes_curve)

            predicts = torch.topk(weight_output,
                                  k=self.topk,
                                  dim=1,
                                  largest=True,
                                  sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(
            y_true)  # [N, topk] ,所有的预测结果。

    def _eval_nme(self, loader, class_means, cur_task):
        self._network.eval()
        # 提取向量
        vectors, y_true = self._extract_vectors(loader, cur_task)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, :self.topk], y_true  # [N, topk]

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true,
                                 (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret
