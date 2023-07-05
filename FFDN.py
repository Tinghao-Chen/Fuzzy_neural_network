import torch
import torch.nn as nn

from layers.FuzzyLayer import FuzzyLayer

# 网络定义
class FusedFuzzyDeepNet(nn.Module):
    def __init__(self, input_vector_size, fuzz_vector_size, num_class, fuzzy_layer_input_dim=1,
                 fuzzy_layer_output_dim=1,
                 dropout_rate=0.5):
        # 输入了普通向量数据，模糊向量数据，num_class是输出的节点数量,模糊层输入，模糊层输出，dropout率。
        super(FusedFuzzyDeepNet, self).__init__()
        # 初始化了所有输入
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_vector_size = input_vector_size
        self.fuzz_vector_size = fuzz_vector_size
        self.num_class = num_class
        self.fuzzy_layer_input_dim = fuzzy_layer_input_dim
        self.fuzzy_layer_output_dim = fuzzy_layer_output_dim

        self.dropout_rate = dropout_rate
        # 初始化模糊线性层
        # Linear创建线性层，指定了普通输入和模糊输入的维度，并执行了线性变换
        self.fuzz_init_linear_layer = nn.Linear(self.input_vector_size, self.fuzz_vector_size)
        # 初始化模糊规则层
        fuzzy_rule_layers = []
        # 根据输入模糊向量的数量，逐个模糊
        for i in range(self.fuzz_vector_size):
            # 每个模糊向量在模糊层中计算模糊隶属度后返回，添加到模糊规则的待计算列表中。
            fuzzy_rule_layers.append(FuzzyLayer(fuzzy_layer_input_dim, fuzzy_layer_output_dim))
        #     将模糊规则层的输入数据用容器ModuleList保存
        self.fuzzy_rule_layers = nn.ModuleList(fuzzy_rule_layers)
        # 创建用输入数据创建线形层1，与输入向量大小n一致的，大小为n*n的线性层。
        self.dl_linear_1 = nn.Linear(self.input_vector_size, self.input_vector_size)
        # 创建线性映射层2，与输入向量大小n一致的，大小为n*n的线性层。
        self.dl_linear_2 = nn.Linear(self.input_vector_size, self.input_vector_size)
        # 创建具有dropout率的dropout层
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        # 创建数据融合层，输入是两倍的线性向量大小，输出是一倍的向量数据。
        self.fusion_layer = nn.Linear(self.input_vector_size * 2, self.input_vector_size)
        #输出层
        self.output_layer = nn.Linear(self.input_vector_size, self.num_class)

        # 此行沿输入张量的第二维(dim=1)创建一个log softmax 激活层(),以获得类概率的对数。
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        # 输入模糊层，并对输入执行线性变换，将其映射到模糊输入空间（神经元）。
        fuzz_input = self.fuzz_init_linear_layer(input)
        # 初始化与输入张量大小相同的张量，用0填充，
        fuzz_output = torch.zeros(input.size(), dtype=torch.float, device=self.device)
        #  函数迭代列输入特征，fuzz_input.size()[1]表示张量的第二个维度大小，在线形层映射后，获取了输入的列数
        for col_idx in range(fuzz_input.size()[1]):
            # 从当前列索引处选择一个列向量
            col_vector = fuzz_input[:, col_idx:col_idx + 1]
            fuzz_col_vector = self.fuzzy_rule_layers[col_idx](col_vector).unsqueeze(0).view(-1, 1)
            # 输出的某行就是这个操作后的统一化向量
            fuzz_output[:, col_idx:col_idx + 1] = fuzz_col_vector

        # self.dl_linear_1(input)将线形层应用于输入后，对结果进行激活
        dl_layer_1_output = torch.sigmoid(self.dl_linear_1(input))
        # 对层1的结果输入应用激活，输出层2结果，并输入到dropout层中。
        dl_layer_2_output = torch.sigmoid(self.dl_linear_2(dl_layer_1_output))
        dl_layer_2_output = self.dropout_layer(dl_layer_2_output)

        #沿第二个维度（列向量）连接，融合计算模糊输出和全连接层输出。
        cat_fuzz_dl_output = torch.cat([fuzz_output, dl_layer_2_output], dim=1)

        # 融合后激活
        fused_output = torch.sigmoid(self.fusion_layer(cat_fuzz_dl_output))
        fused_output = torch.relu(fused_output)
        # 沿第二维应用logsoftmax激活函数
        output = self.log_softmax(self.output_layer(fused_output))
        # 输出张量
        return output
