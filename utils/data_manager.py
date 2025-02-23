import logging
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iImageNettiny, icub, iImageNeta, iImageNetr


class DataManager(object):

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.dataset_name = dataset_name  # json文件中dataset属性，数据集名称
        self._setup_data(dataset_name, shuffle, seed,
                         args)  # 设置数据，类别排序，按照类别排序的标签
        assert init_cls <= len(
            self._class_order), "No enough classes."  # 确保初始类别小于总类别
        self._increments = [init_cls]  # 类别增量
        while sum(self._increments) + increment < len(
                self._class_order):  # 扩充增量数目
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)  # 最后一次增加过量的数目
        # SNN训练时总是报错，所以关掉了
        if offset > 0:
            self._increments.append(offset)  # 添加在末尾

    @property  # 创造只读属性
    def nb_tasks(self):  # 返回增量
        return len(self._increments)

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        # 获取一次增量数据
        self,
        indices,
        source,
        mode,
        appendent=None,
        ret_data=False,
        m_rate=None
        # 传入数据：indices（需要提取的类别索引）
        # source，mode：数据类型
        # appendent：初始传入为空，DER中调用get_memory分配内存空间
        # Der未用到m_rate、ret_data参数
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "valid":
            x, y = self._valid_data, self._valid_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose([
                *self._test_trsf,
                transforms.RandomHorizontalFlip(p=1.0),
                *self._common_trsf,
            ])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            # 根据类别索引挑选所有数据
            if m_rate is None:
                class_data, class_targets = self._select(
                    # x数据，y标签
                    x,
                    y,
                    low_range=idx,
                    high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x,
                                                             y,
                                                             low_range=idx,
                                                             high_range=idx +
                                                             1,
                                                             m_rate=m_rate)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            # 这个写法真牛逼，在增量时调用get_memory,在计算样本平均时直接传入样本数据。不传入原始数据，可以将训练数据与样本数据分开。
            # 第一次时appendent为None
            # 传入训练数据为空，只传入appenddent即可在此中进行处理。
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        # 去除列表形式，模型axis=0
        data, targets = np.concatenate(data), np.concatenate(targets)

        # 源代码中使用的公共数据集，在调用时即是图片这里使用的路径，转为图片,修改
        if isinstance(data[0], str):
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf, True)
            else:
                # return DummyDataset(data, targets, trsf, self.use_path)
                return DummyDataset(data, targets, trsf, True)
        else:
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf,
                                                   self.use_path)
            else:
                return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self,
                               indices,
                               source,
                               mode,
                               appendent=None,
                               val_samples_per_class=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "valid":
            x, y = self._valid_data, self._valid_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        test_data, test_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x,
                                                     y,
                                                     low_range=idx,
                                                     high_range=idx + 1)
            val_indx = np.random.choice(len(class_data),
                                        val_samples_per_class,
                                        replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))

            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(appendent_data,
                                                           appendent_targets,
                                                           low_range=idx,
                                                           high_range=idx + 1)
                val_indx = np.random.choice(len(append_data),
                                            val_samples_per_class,
                                            replace=False)
                train_indx = list(
                    set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(
            val_targets)

        # return DummyDataset(
        #     train_data, train_targets, trsf, self.use_path
        # ), DummyDataset(val_data, val_targets, trsf, self.use_path)
        return DummyDataset(train_data, train_targets, trsf,
                            True), DummyDataset(val_data, val_targets, trsf,
                                                True)

    def _setup_data(self, dataset_name, shuffle, seed, args):
        idata = _get_idata(dataset_name)  # 根据pytorch提供的数据集进行实例化

        # Data，调取类中的属性：数据、标签、变换
        # 使用公共数据集时会报错
        if isinstance(idata, tuple):
            self._train_data, self._train_targets, self._valid_data, self._valid_targets, self._test_data, self._test_targets = idata[
                0], idata[1], idata[2], idata[3], idata[4], idata[5]
        else:
            idata.download_data()  # 下载数据》
            self._train_data, self._train_targets = idata.train_data, idata.train_targets
            # self._valid_data, self._valid_targets = idata.valid_data, idata.valid_targets
            self._test_data, self._test_targets = idata.test_data, idata.test_targets
            self.use_path = idata.use_path  # 存储地址

        # self._train_data, self._train_targets = idata.train_data, idata.train_targets
        # # self._valid_data, self._valid_targets = idata.valid_data, idata.valid_targets
        # self._test_data, self._test_targets = idata.test_data, idata.test_targets
        # self.use_path = idata.use_path  # 存储地址

        # Transforms
        self._train_trsf = [
            # transforms.Resize((32,32)),
            transforms.Resize((args['img_size'], args['img_size'])),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
        ]
        self._test_trsf = [
            # transforms.Resize((32,32)),
            transforms.Resize((args['img_size'], args['img_size'])),
            # transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
        self._common_trsf = [
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]

        # Order
        # 对数据进行排序并去除重复元素，构建了1-总类别的列表
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            # 随机排序
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            # 顺序排列
            order = idata.class_order
        # 类别序列
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        # 这里传入的是标签文件和类别排序，将所有的标签文件按照类别映射
        # 只映射了标签，没有排列原始数据
        # self._train_targets = _map_new_class_index(
        #     self._train_targets, self._class_order
        # )
        # logging.info(self._train_targets)
        # self._train_data = _map_new_class_index(
        #     self._train_data, self._class_order
        # )
        # self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        # self._test_data = _map_new_class_index(self._test_data, self._class_order)

    def _select(self, x, y, low_range, high_range):
        # x数据，y标签，low：index， high：index+1
        # 挑选数据
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # logging.info(np.array(x)[idxes])
        # logging.info(np.array(y)[idxes])
        return np.array(x)[idxes], np.array(y)[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(0,
                                               len(idxes),
                                               size=int(
                                                   (1 - m_rate) * len(idxes)))
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y
                                                < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):

    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    # 对所有的y和order使用map进行映射
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    if os.path.isdir(dataset_name):
        with open('txt/{}/train.txt'.format(dataset_name), 'r') as file:
            train_path = file.read().splitlines()
            train_images_path, train_images_label = [], []
            for path in train_path:
                train_images_path.append(path.split(' ')[0])
                train_images_label.append(int(path.split(' ')[1]))
        with open('txt/{}/test.txt'.format(dataset_name), 'r') as file:
            valid_path = file.read().splitlines()
            valid_images_path, valid_images_label = [], []
            for path in valid_path:
                valid_images_path.append(path.split(' ')[0])
                valid_images_label.append(int(path.split(' ')[1]))
        with open('txt/{}/test.txt'.format(dataset_name), 'r') as file:
            test_path = file.read().splitlines()
            test_images_path, test_images_label = [], []
            for path in test_path:
                test_images_path.append(path.split(' ')[0])
                test_images_label.append(int(path.split(' ')[1]))
        return train_images_path, train_images_label, valid_images_path, valid_images_label, test_images_path, test_images_label

    else:
        name = dataset_name.lower()
        if name == "cifar10":
            return iCIFAR10()
        elif name == "cifar100":
            return iCIFAR100()
        elif name == "imagenet1000":
            return iImageNet1000()
        elif name == "imagenet100":
            return iImageNet100()
        elif name == "imagenet-a":
            return iImageNeta()
        elif name == "imagenet-r":
            return iImageNetr()
        elif name == "cub":
            return icub()
        elif name == "tinyimagenet":
            return iImageNettiny()
        elif name == "imagenettiny":
            return iImageNettiny()
        else:
            raise NotImplementedError(
                "Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
