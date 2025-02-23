import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8  # 像不像一个超参数
batch_size = 16  # 不解释了，写在这个地方真的牛


# 增量学习的基础类，所有的模型都继承他
class BaseLearner(object):

    def __init__(self, args):
        self._cur_task = -1  # 记录现在是第几个增量
        self._known_classes = 0  # 已知类别数目
        self._total_classes = 0  # 学习的总类别
        self._network = None  # 新网络
        self._old_network = None  # 旧网络
        self._data_memory, self._targets_memory = np.array([]), np.array(
            [])  # 存储数据和标签
        self.topk = 5  # 计算topk的正确率
        # json文件中参数
        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)  # 默认为False
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim  # 添加Backbone数目*最后的channel

    def build_rehearsal_memory(self, data_manager, per_class, cur_task):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class, cur_task)
        else:
            logging.info('start')
            self._reduce_exemplar(data_manager, per_class,
                                  self._cur_task)  # 初始化任务时，没有在这里分配内存空间。
            self._construct_exemplar(data_manager, per_class, self._cur_task)
            logging.info('end')

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

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

    def eval_task(self, cur_task):
        y_pred, y_true = self._eval_cnn(self.test_loader, cur_task)
        cnn_accy = self._evaluate(y_pred, y_true)

        nme_accy = None
        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader,
                                            self._class_means, cur_task)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader, cur_task, adjustment = 0):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, cur_task)["logits"]-adjustment
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        if isinstance(correct, int):
            correct = torch.from_numpy(np.array(correct))
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader, cur_task):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, cur_task)["logits"]
            predicts = torch.topk(outputs,
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

    def _extract_vectors(self, loader, cur_task):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:  # 这里读取loader花费时间不短，推理时间不长
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(
                        _inputs.to(self._device), cur_task))
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device),
                                                 cur_task)  # 这里进行网络正向推理
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m, cur_task):
        # 传入的是实例化的包含数据类的普通类，m每类的样本数量
        logging.info("Reducing exemplars...({} per classes)".format(m))
        # 初始为空；拷贝旧数据
        dummy_data, dummy_targets = copy.deepcopy(
            self._data_memory), copy.deepcopy(self._targets_memory)
        # 传入的是实例化的包含数据类的普通类，m每类的样本数量
        self._class_means = np.zeros(
            (self._total_classes,
             self.feature_dim))  # self.feature_dim:最后一层输出channel*backbone数目
        # 清空原有
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        # 在已知类中搜索，初始化时已知类为0，跳过
        # 好了，现在是增量了，没啥说的了
        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]  # 初始为空


            m_temp = len(mask) if m >len(mask) else m

            dd, dt = dummy_data[mask][:m_temp], dummy_targets[mask][:m_temp]
            self._data_memory = (np.concatenate(
                (self._data_memory,
                 dd)) if len(self._data_memory) != 0 else dd)
            self._targets_memory = (np.concatenate(
                (self._targets_memory,
                 dt)) if len(self._targets_memory) != 0 else dt)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([],
                                                   source="train",
                                                   mode="test",
                                                   appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader, cur_task)
            vectors = (vectors.T /
                       (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m, cur_task):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(idx_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
            m_temp = len(idx_loader) if m > len(idx_loader) else m
            vectors, _ = self._extract_vectors(idx_loader, cur_task)
            vectors = (vectors.T /
                       (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            # 这里挑选样本，没啥说的
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m_temp + 1):
                S = np.sum(
                    exemplar_vectors,
                    axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S +
                        EPSILON) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p)**2, axis=1)))
                selected_exemplars.append(np.array(
                    data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(
                    vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i,
                    axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i,
                    axis=0)  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m_temp, class_idx)
            # 这里保存了数据，原始为空则使用挑选数据，不为空则合并
            self._data_memory = (np.concatenate(
                (self._data_memory, selected_exemplars)) if len(
                    self._data_memory) != 0 else selected_exemplars)
            self._targets_memory = (np.concatenate(
                (self._targets_memory, exemplar_targets)) if len(
                    self._targets_memory) != 0 else exemplar_targets)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(idx_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader, cur_task)
            vectors = (vectors.T /
                       (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            # 记录每一类的平均
            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m, cur_task):
        # m自己算算
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(
                m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network 原版注释，自己翻译去
        # 初始跳过
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset([],
                                                  source="train",
                                                  mode="test",
                                                  appendent=(class_data,
                                                             class_targets))
            class_loader = DataLoader(class_dset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=4)
            vectors, _ = self._extract_vectors(class_loader, cur_task)

            vectors = (vectors.T /
                        (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
 
            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(class_dset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=4)

            vectors, _ = self._extract_vectors(class_loader, cur_task)
            vectors = (vectors.T /
                       (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors,
                    axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p)**2, axis=1)))

                selected_exemplars.append(np.array(
                    data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(
                    vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i,
                    axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i,
                    axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (np.concatenate(
                (self._data_memory, selected_exemplars)) if len(
                    self._data_memory) != 0 else selected_exemplars)
            self._targets_memory = (np.concatenate(
                (self._targets_memory, exemplar_targets)) if len(
                    self._targets_memory) != 0 else exemplar_targets)

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(exemplar_dset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader, cur_task)
            vectors = (vectors.T /
                        (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / (np.linalg.norm(mean) + EPSILON)


            _class_means[class_idx, :] = mean

        self._class_means = _class_means
