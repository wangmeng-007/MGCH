import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from MGCH.utils.utils import *

from torch.nn import Parameter
from scipy.io import loadmat


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc_encode = nn.Linear(2048, code_len)
        self.alpha = 1.0

    def forward(self, x):

        feat = self.fc1(x)
        feat = F.relu(self.fc2(feat))
        code = torch.tanh(self.alpha * self.fc_encode(feat))


        return feat, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(300, 4096).to("cuda:0")
        self.fc2 = nn.Linear(4096, 2048).to("cuda:0")
        self.fc_encode = nn.Linear(2048, code_len).to("cuda:0")
        self.alpha = 1.0

    def forward(self, x):
        feat = self.fc1(x)
        feat = F.relu(self.fc2(feat))
        code = torch.tanh(self.alpha * self.fc_encode(feat))
        return feat, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)



class JNet(nn.Module):   # 特征融合模块
    def __init__(self, code_len):
        super(JNet, self).__init__()
        self.fc_encode = nn.Linear(4096, code_len).to("cuda:0")
        self.alpha = 1.0

    def forward(self, x):
        code = torch.tanh(self.alpha * self.fc_encode(x.cuda()))
        return code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)




class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCNLI(nn.Module):
    def __init__(self, code_len):

        super(GCNLI, self).__init__()

        # self.num_classes = num_classes
        self.gcn1 = GCN(2048, 2048).to("cuda:0")
        self.gcn2 = GCN(2048, 2048).to("cuda:0")
        self.gcn3 = GCN(2048, 2048).to("cuda:0")
        self.relu = nn.LeakyReLU(0.2).to("cuda:0")
        self.hypo = nn.Linear(3 * 2048, 2048).to("cuda:0")
        self.fc_encode = nn.Linear(2048, code_len).to("cuda:0")
        self.alpha = 1.0
    def forward(self, input, adj):
        layers = []

        x = self.gcn1(input, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn2(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn3(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = torch.cat(layers, -1)  # 将多个特征x按最后一个维度进行拼接
        x = self.hypo(x)  # 将拼接后的特征x通过线性层hypo进行映射
        code = torch.tanh(self.alpha * self.fc_encode(x))

        return x, code
    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNLT(nn.Module):
    def __init__(self, code_len):
        super(GCNLT, self).__init__()

        # self.num_classes = num_classes
        self.gcn1 = GCN(2048, 2048).to("cuda:0")
        self.gcn2 = GCN(2048, 2048).to("cuda:0")  # 创建第二层超图卷积层SuperGCN，输入通道数为minus_one_dim，输出通道数为minus_one_dim
        self.gcn3 = GCN(2048, 2048).to("cuda:0")
        self.relu = nn.LeakyReLU(0.2).to("cuda:0")
        self.hypo = nn.Linear(3 * 2048, 2048).to("cuda:0")
        self.fc_encode = nn.Linear(2048, code_len).to("cuda:0")
        self.alpha = 1.0
    def forward(self, input, adj):
        layers = []

        x = self.gcn1(input, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn2(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn3(x, adj)
        x = self.relu(x)
        layers.append(x)
        x = torch.cat(layers, -1)  # 将多个特征x按最后一个维度进行拼接
        x = self.hypo(x)  # 将拼接后的特征x通过线性层hypo进行映射
        code = torch.tanh(self.alpha * self.fc_encode(x))

        return x, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class GCNL(nn.Module):
    def __init__(self, minus_one_dim=2048, num_classes=10, in_channel=300, t=0,
    adj_file='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/mirflickr/adj.mat', inp='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/mirflickr/mirflickr-inp-glove6B.mat'):
    #mrflickl : adj_file='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/mirflickr/adj.mat', inp='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/mirflickr/mirflickr-inp-glove6B.mat'
    #NUSWIDE : adj_file='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/NUSWIDE/adj.mat', inp='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/NUSWIDE/NUS-WIDE-TC21-inp-glove6B.mat'
    #MSCOCO : adj_file='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/MSCOCO/adj.mat', inp='/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/MSCOCO/MS-COCO-inp-glove6B.mat'

        super(GCNL, self).__init__()
        inp = loadmat(inp)['inp']
        inp = torch.FloatTensor(inp)
        #self.num_classes = num_classes
        self.gcn1 = GCN(in_channel, minus_one_dim)
        self.gcn2 = GCN(minus_one_dim, minus_one_dim)
        self.gcn3 = GCN(minus_one_dim, minus_one_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.hypo = nn.Linear(3 * minus_one_dim, minus_one_dim)

        _adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))

        self.adj = Parameter(gen_adj(_adj), requires_grad=False)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))

        # image normalization
        # self.image_normalization_mean = [0.485, 0.456, 0.406]
        # self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = feature_img
        view2_feature = feature_text

        layers = []

        x = self.gcn1(self.inp, self.adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn2(x, self.adj)
        x = self.relu(x)
        layers.append(x)
        x = self.gcn3(x, self.adj)
        x = self.relu(x)
        layers.append(x)
        x = torch.cat(layers, -1)  # 将多个特征x按最后一个维度进行拼接
        x = self.hypo(x)  # 将拼接后的特征x通过线性层hypo进行映射




        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(x.cuda(), dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x.cuda(), dim=1)[None, :] + 1e-6
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x.cuda())
        y_text = torch.matmul(view2_feature, x.cuda())
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        return  y_img, y_text,  x.transpose(0, 1)




def calc_loss(view1_predict, view2_predict, labels_1, labels_2):
    loss = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

    return loss



