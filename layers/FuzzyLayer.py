import torch
import torch.nn as nn


class FuzzyLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FuzzyLayer, self).__init__()
        # 给创建的对象赋值输入与输出，后面不需要再传入对象，即可读写对象的输入、输出。

        self.input_dim = input_dim
        self.output_dim = output_dim
        # 输入的向量被torch.Tensor拼成一个权重矩阵
        fuzzy_degree_weights = torch.Tensor(self.input_dim, self.output_dim)
        # 使用nn.Parameter包装，使得fuzzy_degree_weights变为可学习参数，可以在训练期间得到优化。该参数将成为模块参数的一部分，可以在训练期间访问和更新。
        self.fuzzy_degree = nn.Parameter(fuzzy_degree_weights)
        # sigma_weights同理，应用这种方式作为可学习参数参与到训练中
        sigma_weights = torch.Tensor(self.input_dim, self.output_dim)
        self.sigma = nn.Parameter(sigma_weights)

        # initialize fuzzy degree and sigma parameters
        # 使用Xavier均匀初始化方法初始化模糊度权重，Xavier 初始化是一种流行的权重初始化技术，旨在维持整个网络层的激活和梯度的方差。它有助于缓解训练期间梯度消失/爆炸问题。
        nn.init.xavier_uniform_(self.fuzzy_degree)  # fuzzy degree init
        # 取决于神经网络的具体要求。
        nn.init.ones_(self.sigma)  # sigma init

    def forward(self, input):
        # 初始化空列表，存储输入运算的输出
        fuzzy_out = []
        # 对input进来的张量进行迭代
        for variable in input:
            # 计算每个变量的模糊输出。
            # 1.计算输入与目前迭代的权重之间的差
            # 2.计算此时的sigma矩阵中所有数据的2次方
            # 3.求平方根求和，目的是均方误差，并统计本次计算的全部误差
            # 4.用欧拉数e应用于标量，自然底数
            #输出是计算后的模糊隶属值，公式可以有改动
            fuzzy_out_i = torch.exp(-torch.sum((variable - self.fuzzy_degree)**2 / (self.sigma ** 2)))

            # 检查计算结果是不是nan，如果不是nan，就获取模糊隶属度计算结果，如果是nan，就将输入值记录到模糊隶属度中。
            if torch.isnan(fuzzy_out_i):
                fuzzy_out.append(variable)
            else:
                fuzzy_out.append(fuzzy_out_i)
        #         计算的模糊隶属度作为张量输出，类型是float。
        return torch.tensor(fuzzy_out, dtype=torch.float)

