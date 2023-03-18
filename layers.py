import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


# Multi-Layer Perceptron(MLP) layer
class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    # 参数分别是输入数据维度、中间隐层维度、输出数据维度，dropout控制dropout概率（也就是通过丢失结点增加鲁棒性）、偏置是增加可能会影响变化的因素
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()、
        # 输入层
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        # 输出层
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        # 批量归一化 对输出标准化 消除输出差异较大的影响  
        self.norm = nn.BatchNorm1d(output_dim)
        
        # 该论文称他们为 MLP 的输出添加了 Batch Normalization，如第 4.2 节所示
        
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    # 初始化步骤
    def init_weights(self):
        #对所有子模块进行遍历
        for m in self.modules():
            # 判断m是否是nn.Linear这个类 也就是是否为线性变换层
            if isinstance(m, nn.Linear):
                # 采用Xavier 正态分布 
                nn.init.xavier_normal_(m.weight.data) # 初始化权重 
                m.bias.data.fill_(0.1) # 初始化bias为0.1
                
            # 判断m是否是nn.batchnormld 批标准化层 
            elif isinstance(m, nn.BatchNorm1d):
                # 将scale参数设置为1，将bias参数设置为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
                
    # 用于将输入的tensor进行标准化，其中self.norm是一个BatchNorm1d层，用于对每个特征维度进行标准化。
    # 在 MLP 中，经过每个全连接层（即 nn.Linear），都会接上一个 batch normalization 层，将每一层的输出进行规范化，以使得网络更易于训练。
    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:                              # 如果输入inputs是一个形状为(batch_size, sequence_length, feature_dim)的3D张量，
            x = inputs.view(inputs.size(0) * inputs.size(1), -1) # 将把它reshape成形状为(batch_size * sequence_length, feature_dim)的2D张量x，
            x = self.norm(x)                                     # 并对它进行Batch Normalization。
            return x.view(inputs.size(0), inputs.size(1), -1)    # 将x再reshape回形状为(batch_size, sequence_length, feature_dim)的3D张量，并返回。
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))             # fc1全连接层，输出经过ReLu函数。
        x = F.dropout(x, self.dropout, training=self.training)  # training 参数设置为self.training 仅在训练期间使用
        x = F.relu(self.fc2(x))     # fc2是输出层 输出再次经过relu函数
        return self.batch_norm(x)   # 进行批量归一化之后再输出


class EraseAddGate(nn.Module):
    """
    擦除和添加门模块
    注意：这个擦除和添加门与 DKVMN 中的有点不同。
    关于Erase & Add gate的更多信息，请参考论文《Dynamic Key-Value Memory Networks for Knowledge Tracing》
    作用是将输入的特征矩阵中的旧信息删除并添加新信息，以更新模型的状态。
    
    """
    
    # feature_dim 和concept_num两个参数 特征维度默认固定为32 概念数量为另一个参数；
    def __init__(self, feature_dim, concept_num, bias=True):
        super(EraseAddGate, self).__init__()
        
        # weight
        # torch.rand（x） x用于指定生成随机张量的形状 为（x,）
        # nn.Parameter(torch.rand(concept_num))生成一个由随机数构成的可学习参数张量，其大小为(concept_num,)。
        
        self.weight = nn.Parameter(torch.rand(concept_num))
        self.reset_parameters() #在reset_parameters函数中，它会被初始化为均匀分布的随机值，以进行训练和优化。
        
        
        # erase gate
        # 它将输入特征矩阵中的每个特征映射到一个标量，其值域在0到1之间，表示需要从输入特征中删除的信息量。
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias) 
        
        
        
        # add gate
        # 表示需要从输入特征中增加的信息量。
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))  # 计算了一个标准差stdv，其值为1除以可学习参数self.weight的第一维的平方根。
        self.weight.data.uniform_(-stdv, stdv) 
        # 这里的uniform_函数用于对张量进行均匀分布的初始化，其中下划线表示在原地修改该张量。
        # self.weight.data相当于访问weight中的data数据 对其进行后续函数操作。
        # uniform 随机分布初始化，范围在-stdv到stdv之间
        
        
        
    def forward(self, x):
        r"""
        参数：
             x：输入特征矩阵
         形状：
             x: [batch_size, concept_num, feature_dim]
             res: [batch_size, concept_num, feature_dim]
         返回：
             res：返回特征矩阵，删除旧信息并添加新信息
         GKT 论文没有提供关于这个擦除-添加门的详细解释。 由于 GKT 中的擦加门只有一个输入参数，
         这个门与 DKVMN 的不同。 我们使用输入矩阵来构建擦除和添加门，而不是 DKVMN 中的 $\mathbf{v}_{t}$ 向量。
        """
        
        
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, concept_num, feature_dim]
        
        # self.weight.unsqueeze(dim=1) 扩展维度 维度变为2 然后形状是 shape: [concept_num, 1] 意思是每一维度上有concept_num和1个元素 第一维度有concept_num个 第二维度有1个
        
        
        # 计算出擦除门，然后用原来的x减去擦除门得到最后的结果
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x  
           
       
        # [concept_num , 1] 会扩展成为 [1,concept_num,1]然后进行逐元素相乘
        
        # 增加门
        add_feat = torch.tanh(self.add(x))  # [batch_size, concept_num, feature_dim]
     
        
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        
        
        return res


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    NOTE: Stole and modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    """

    def __init__(self, temperature, attn_dropout=0.):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, mask=None):
        r"""
        Parameters:
            q: multi-head query matrix
            k: multi-head key matrix
            mask: mask matrix
        Shape:
            q: [n_head, mask_num, embedding_dim]
            k: [n_head, concept_num, embedding_dim]
        Return: attention score of all queries
        """
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))  # [n_head, mask_number, concept_num]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # pay attention to add training=self.training!
        attn = F.dropout(F.softmax(attn, dim=0), self.dropout, training=self.training)  # pay attention that dim=-1 is not as good as dim=0!
        return attn


class MLPEncoder(nn.Module):
    """
    MLP encoder module.
    NOTE: Stole and modify the code from https://github.com/ethanfetaya/NRI/blob/master/modules.py
    """
    def __init__(self, input_dim, hidden_dim, output_dim, factor=True, dropout=0., bias=True):
        super(MLPEncoder, self).__init__()
        self.factor = factor
        self.mlp = MLP(input_dim * 2, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.mlp2 = MLP(hidden_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        if self.factor:
            self.mlp3 = MLP(hidden_dim * 3, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        else:
            self.mlp3 = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, sp_send, sp_rec):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(sp_rec, x)
        senders = torch.matmul(sp_send, x)
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, sp_send_t, sp_rec_t):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(sp_rec_t, x)
        return incoming

    def forward(self, inputs, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            inputs: input concept embedding matrix
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            inputs: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            output: [edge_num, edge_type_num]
        """
        x = self.node2edge(inputs, sp_send, sp_rec)  # [edge_num, 2 * embedding_dim]
        x = self.mlp(x)  # [edge_num, hidden_num]
        x_skip = x

        if self.factor:
            x = self.edge2node(x, sp_send_t, sp_rec_t)  # [concept_num, hidden_num]
            x = self.mlp2(x)  # [concept_num, hidden_num]
            x = self.node2edge(x, sp_send, sp_rec)  # [edge_num, 2 * hidden_num]
            x = torch.cat((x, x_skip), dim=1)  # Skip connection  shape: [edge_num, 3 * hidden_num]
            x = self.mlp3(x)  # [edge_num, hidden_num]
        else:
            x = self.mlp2(x)  # [edge_num, hidden_num]
            x = torch.cat((x, x_skip), dim=1)  # Skip connection  shape: [edge_num, 2 * hidden_num]
            x = self.mlp3(x)  # [edge_num, hidden_num]
        output = self.fc_out(x)  # [edge_num, output_dim]
        return output


class MLPDecoder(nn.Module):
    """
    MLP decoder module.
    NOTE: Stole and modify the code from https://github.com/ethanfetaya/NRI/blob/master/modules.py
    """

    def __init__(self, input_dim, msg_hidden_dim, msg_output_dim, hidden_dim, edge_type_num, dropout=0., bias=True):
        super(MLPDecoder, self).__init__()
        self.msg_out_dim = msg_output_dim
        self.edge_type_num = edge_type_num
        self.dropout = dropout

        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * input_dim, msg_hidden_dim, bias=bias) for _ in range(edge_type_num)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hidden_dim, msg_output_dim, bias=bias) for _ in range(edge_type_num)])
        self.out_fc1 = nn.Linear(msg_output_dim, hidden_dim, bias=bias)
        self.out_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_fc3 = nn.Linear(hidden_dim, input_dim, bias=bias)

    def node2edge(self, x, sp_send, sp_rec):
        receivers = torch.matmul(sp_rec, x)  # [edge_num, embedding_dim]
        senders = torch.matmul(sp_send, x)  # [edge_num, embedding_dim]
        edges = torch.cat([senders, receivers], dim=-1)  # [edge_num, 2 * embedding_dim]
        return edges

    def edge2node(self, x, sp_send_t, sp_rec_t):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(sp_rec_t, x)
        return incoming

    def forward(self, inputs, rel_type, sp_send, sp_rec, sp_send_t, sp_rec_t):
        r"""
        Parameters:
            inputs: input concept embedding matrix
            rel_type: inferred edge weights for all edge types from MLPEncoder
            sp_send: one-hot encoded send-node index(sparse tensor)
            sp_rec: one-hot encoded receive-node index(sparse tensor)
            sp_send_t: one-hot encoded send-node index(sparse tensor, transpose)
            sp_rec_t: one-hot encoded receive-node index(sparse tensor, transpose)
        Shape:
            inputs: [concept_num, embedding_dim]
            sp_send: [edge_num, concept_num]
            sp_rec: [edge_num, concept_num]
            sp_send_t: [concept_num, edge_num]
            sp_rec_t: [concept_num, edge_num]
        Return:
            output: [edge_num, edge_type_num]
        """
        # NOTE: Assumes that we have the same graph across all samples.
        # Node2edge
        pre_msg = self.node2edge(inputs, sp_send, sp_rec)
        all_msgs = Variable(torch.zeros(pre_msg.size(0), self.msg_out_dim, device=inputs.device))  # [edge_num, msg_out_dim]
        for i in range(self.edge_type_num):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, self.dropout, training=self.training)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = self.edge2node(all_msgs, sp_send_t, sp_rec_t)  # [concept_num, msg_out_dim]
        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(agg_msgs)), self.dropout, training=self.training)  # [concept_num, hidden_dim]
        pred = F.dropout(F.relu(self.out_fc2(pred)), self.dropout, training=self.training)  # [concept_num, hidden_dim]
        pred = self.out_fc3(pred)  # [concept_num, embedding_dim]
        return pred
