import copy
import logging
import torch
import numpy as np
import timm
from torch import nn
from torchvision import transforms

from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear

# from backbone.vit_mos import Block
from functools import partial
# from convs.s_resnet import spiking_resnet18, spiking_resnet34, spiking_resnet50, spiking_resnet101, spiking_wide_resnet50_2, spiking_resnet152, spiking_resnext101_32x8d, spiking_resnext50_32x4d, spiking_wide_resnet101_2


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()  # 大写转小写
    if name == 'vit_base_patch16_224':
        model = timm.create_model("vit_base_patch16_224",
                                  pretrained=True,
                                  num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == 'vit_base_patch16_224_in21k':
        model = timm.create_model("vit_base_patch16_224_in21k",
                                  pretrained=True,
                                  num_classes=0)
        model.out_dim = 768
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):

    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        # LwF网络模型的最低类，获取定义好的模型
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x, cur_task):
        return self.convnet(x)["features"]

    def forward(self, x, cur_task=0):
        x = self.convnet(x)
        out = self.fc(x)
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class DAMNET(nn.Module):

    def __init__(self, args, pretrained):
        super().__init__()
        self.convnet_type = args["convnet_type"]  # 使用的基础网络
        self.convnets = nn.ModuleList()  # 初始化空的网络
        self.pretrained = pretrained  # 预训练模型，默认是False
        self.out_dim = None  # 最后一层网络输出特征的channnel
        self.fc = nn.ModuleList()  # 分类器
        self.lora_layers = nn.ModuleList()
        self.aux_fc = None  # 辅助分类器
        self.task_sizes = []  # 任务大小，记录每次增量训练的类别数目
        self.args = args  # json参数

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        # 添加Backbone数目*最后的channel，貌似是由于冻结之前网络层导致的
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x, curtask=0):
        features = [convnet(x) for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x, curtask=0):
        features = [convnet(x) for convnet in self.convnets]  # 正向推理
        features = torch.cat(features, 1)  # 在第一个维度上进行特征的拼接，拼接每一层网络的输出特征

        # {logics: self.fc(features)}  # 分类器输出，返回字典，key值为'logits'
        # out = self.fc(features)
        out = {"logits": []}
        # 添加辅助分类器的输出特征
        aux_logits = self.aux_fc(features[:,
                                          -self.out_dim:])["logits"]  # 辅助分类器
        out.update({"aux_logits": aux_logits, "features": features})

        features_clone = []
        for i, fc in enumerate(self.fc):
            if i == 0:
                features_clone.append(torch.clone(features))
            else:
                features_clone.append(self.lora_layers[i -
                                                       1](features)["logits"])
                # features = self.lora_layers[i-1](features)["logits"]
            out["logits"].append(self.fc[i](features_clone[-1])["logits"])
        return out
        """
    输出out用于计算loss，特征不是很重要，TODO:后续修改loss计算方式
        {
            'features': features
            'logits': [logits1, logits2, logits3]
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        # nb_classes:当前的总类别数
        # 跟新分类器（类别增加）
        if len(self.convnets) == 0:
            # 刚实例化时，网络为空，根据json参数更新，即backbone网络
            self.convnets.append(get_convnet(self.args, pretrained=True))
        # else:
        #     # 不为空时，先添加Backbone网络，最后一个网络复制倒数第二层的参数（冻结网络？？？）
        #     self.convnets.append(get_convnet(self.args))
        #     self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())
        # self.output_weight = torch.ones_like(output[0]).to(self._device)
        new_task_size = nb_classes - sum(self.task_sizes)  # 每次增加的类别数目
        self.task_sizes.append(new_task_size)  # 记录每次增加的类别数目
        if self.out_dim is None:
            # 设为Backbone的最后层输出的channel
            self.out_dim = self.convnets[-1].out_dim
        # 产生新的分类器，这个分类器绝了，使用nn.linear，初始化了w和b，省去了手动flatten的操作

        fc = self.generate_fc(self.feature_dim, nb_classes)
        # 继承旧分类器数据
        if self.fc:
            nb_output = self.fc[-1].out_features  # 分类器的输出类别
            weight = copy.deepcopy(self.fc[-1].weight.data)  # 权重
            bias = copy.deepcopy(self.fc[-1].bias.data)  # 偏置
            fc.weight.data[:nb_output, :self.
                           feature_dim] = weight  # 新分类器继承旧分类器的权重
            fc.bias.data[:nb_output] = bias  # 新分类器继承旧分类器的偏置
            self.lora_layers.append(
                self.generate_fc(self.feature_dim, self.feature_dim))
        self.aux_fc = self.generate_fc(self.out_dim,
                                       new_task_size + 1)  # 辅助分类器，只负责新增加类别的分类

        self.fc.append(fc)

    def generate_fc(self, in_dim, out_dim):
        # 传入的是channel数目*backbone数目，总类别数目
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc[-1].weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc[-1].weight.data[-increment:, :] *= gamma
