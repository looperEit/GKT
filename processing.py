import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class KTDataset(Dataset): #Torch中通过重写Dataset类来处理和存储数据
    #①__init__：传入数据，或者像下面一样直接在函数里加载数据
    def __init__(self, features, questions, answers):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers
        
        
    #③__getitem__:返回一条训练数据，并将其转换成tensor
    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]
    
    #②__len__：返回这个数据集一共有多少个item
    def __len__(self):
        return len(self.features)


def pad_collate(batch): #进行数据批处理
    (features, questions, answers) = zip(*batch)#表示解包batch，让它的每行都都压缩成一个新的元组
    
    features = [torch.LongTensor(feat) for feat in features] #将features列表中的每一个元素（每一个都是一个用户的特征列表）转化成张量
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    
    #pad_sequence(sequences,batch_first,padding_value)
    #pad_sequence 函数会将不同长度的序列填充到相同的长度，使得它们可以一起进行批处理操作
    #sequernces可变长度序列列表
    #batch_first=false 按列填充0 true按行扩充0
    #padding_value=xxx 按照xxx填充

    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad


def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
    r"""
    Parameters:
        file_path：知识溯源数据的输入文件路径
         batch_size：学生批次的大小
         graph_type：概念图的类型
         shuffle：是否打乱数据集
         use_cuda：是否使用GPU加速训练速度
    Return:
        concept_num：所有概念（或问题）的数量
         graph: static graph is graph type is in ['Dense', 'Transition', 'DKT'], 否则graph为None
         train_data_loader：训练数据集的数据加载器
         valid_data_loader：验证数据集的数据加载器
         test_data_loader：测试数据集的数据加载器
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    
    df = pd.read_csv(file_path)
    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {file_path}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {file_path}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {file_path}")

    # if not (df['correct'].isin([0, 1])).all():
    #     raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill 处理skill为空的行 直接删除
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer  移除只有一条回答的记录
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id           给skill重新编号，将skill_id排序后用0、1…编号
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises

    # Step 3 - Cross skill id with answer to form a synthetic feature 交叉技能id与答案形成新的综合特征
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    if use_binary:
        df['skill_with_answer'] = df['skill'] * 2 + df['correct']
    else:
        df['skill_with_answer'] = df['skill'] * res_len + df['correct'] - 1
    
    

    

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []

    #那么其实最后用到的数据，只有特征中放入的skill_with_answer、skill、correct、
    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['correct'].eq(1).astype('int').tolist())
        seq_len_list.append(series['correct'].shape[0])#每个学生的答案列表长度转换为int存入 也就是每个学生的答题次数

    df.groupby('user_id').apply(get_data)
    max_seq_len = np.max(seq_len_list)#最大答题次数
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)#学生数量
    print('student num: ', student_num)
    feature_dim = int(df['skill_with_answer'].max() + 1)#特征维度
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)#问题维度=问题的知识点类别数量 不同的知识点对应不同的问题
    print('question_dim: ', question_dim)
    concept_num = question_dim #概念的数量就是问题的维度

    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        if use_cuda and graph_type in ['Dense', 'Transition', 'DKT']:
            graph = graph.cuda()
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader


def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph
