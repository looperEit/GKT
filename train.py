import numpy as np
import time
import random
import argparse
import pickle
import os
import gc
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models import GKT, MultiHeadAttention, VAE, DKT
from metrics import KTLoss, VAELoss
from processing import load_dataset

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

#argparse 模块是 Python 内置的一个用于命令项选项与参数解析的模块，argparse 模块可以让人轻松编写用户友好的命令行接口。
#----------------------设置命令行的提示参数--------------------------
#相当于安装LL的时候的哪些提示的显示
#1、创建一个解析器——创建 ArgumentParser() 对象
parser = argparse.ArgumentParser()
#2、添加参数——调用 add_argument() 方法添加参数
parser.add_argument('--no-cuda', action='store_false', default=True, help='Disables CUDA training.')#禁用 CUDA 训练
parser.add_argument('--seed', type=int, default=42, help='Random seed.')#随机种子
#用于加载输入数据的数据目录。
parser.add_argument('--data-dir', type=str, default='data', help='Data dir for loading input data.')
#输入数据文件的名称。
parser.add_argument('--data-file', type=str, default='assistment_test15.csv', help='Name of input data file.')
#保存训练模型的位置，留空不保存任何东西。
parser.add_argument('--save-dir', type=str, default='logs', help='Where to save the trained model, leave empty to not save anything.')
#保存概念图的目录。
parser.add_argument('-graph-save-dir', type=str, default='graphs', help='Dir for saving concept graphs.')
#微调时加载训练模型的位置。' + '留空从头开始训练
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
#在哪里加载预训练的 dkt 图
parser.add_argument('--dkt-graph-dir', type=str, default='dkt-graph', help='Where to load the pretrained dkt graph.')
#DKT图数据文件名。
parser.add_argument('--dkt-graph', type=str, default='dkt_graph.txt', help='DKT graph data file name.')
#要使用的模型类型，支持 GKT 和 DKT。
parser.add_argument('--model', type=str, default='GKT', help='Model type to use, support GKT and DKT.')
#隐藏知识状态的维度。
parser.add_argument('--hid-dim', type=int, default=32, help='Dimension of hidden knowledge states.')
#概念嵌入的维度。
parser.add_argument('--emb-dim', type=int, default=32, help='Dimension of concept embedding.')
#多头注意力层的维度。
parser.add_argument('--attn-dim', type=int, default=32, help='Dimension of multi-head attention layers.')
#vae 编码器中隐藏层的维度。
parser.add_argument('--vae-encoder-dim', type=int, default=32, help='Dimension of hidden layers in vae encoder.')
#vae 解码器中隐藏层的维度。
parser.add_argument('--vae-decoder-dim', type=int, default=32, help='Dimension of hidden layers in vae decoder.')
#要推断的边类型的数量。
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
#潜在概念图的类型。
parser.add_argument('--graph-type', type=str, default='Dense', help='The type of latent concept graph.')
#
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
#是否为神经网络层添加偏置。
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
#结果是否只使用0/1。
parser.add_argument('--binary', type=bool, default=True, help='Whether only use 0/1 for results.')
#使用多个结果时的结果类型数。
parser.add_argument('--result-type', type=int, default=12, help='Number of results types when multiple results are used.')
#Gumbel softmax 的温度。
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
#在前向传播训练中使用离散样本。
parser.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
#禁用因子图模型。
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
#是否使用稀疏先验;
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
#输出方差。
parser.add_argument('--var', type=float, default=1, help='Output variance.')
#要训练的epoch数。
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
#每批样品数。
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
#数据集中训练样本的比例。
parser.add_argument('--train-ratio', type=float, default=0.6, help='The ratio of training samples in a dataset.')
#数据集中验证样本的比例。
parser.add_argument('--val-ratio', type=float, default=0.2, help='The ratio of validation samples in a dataset.')
#是否打乱数据集。
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
#初始学习率。
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
#在怎样的epochs后通过伽马因子来进行LR衰减
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
#LR 衰减因子。
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
#是否测试现有模型。
parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
#现有模型文件目录
parser.add_argument('--test-model-dir', type=str, default='logs/expDKT', help='Existed model file dir.')



args = parser.parse_args()
#3、ArgumentParser.parse_args()方法运行解析器并将提取的数据放在一个argparse.Namespace对象中：
#命名空间是当前定义的符号名称以及每个名称引用的对象的信息的集合。
#命名空间视为字典，其中键是对象名称，值是对象本身。每个键值对将一个名称映射到其对应的对象。

args.cuda = not args.no_cuda and torch.cuda.is_available() #args.no_cuda 表示设没设置禁用cuda 并且在返回指示 CUDA 当前是否可用。
args.factor = not args.no_factor
print(args)



random.seed(args.seed)#随机数种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)#设置生成随机数的种子。返回一个 torch.Generator对象。
if args.cuda:
    torch.cuda.manual_seed(args.seed)#使用随机值初始化张量（例如模型的学习权重）很常见，但有时（尤其是在研究环境中）您需要确保结果的可重复性。
    torch.cuda.manual_seed_all(args.seed)
    #cuDNN 是英伟达专门为深度神经网络所开发出来的 GPU 加速库，针对卷积、池化等等常见操作做了非常多的底层优化，比一般的 GPU 程序要快很多。
    #设置这个 flag 为 True，我们就可以在 PyTorch 中对模型里的卷积层进行预先的优化，也就是在每一个卷积层中测试 cuDNN 提供的所有卷积实现算法，然后选择最快的那个。
    #使用它可能会大大增加运行时间
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #将这个 flag 置为 True 的话，每次返回的卷积算法将是确定的，即默认算法。
    #！！！如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。！！！

res_len = 2 if args.binary else args.result_type 

# 保存模型和元数据。 始终保存在新的子文件夹中。
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")

# 加载数据集
dataset_path = os.path.join(args.data_dir, args.data_file)
dkt_graph_path = os.path.join(args.dkt_graph_dir, args.dkt_graph)
if not os.path.exists(dkt_graph_path):
    dkt_graph_path = None
concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(dataset_path, args.batch_size, args.graph_type, dkt_graph_path=dkt_graph_path,
                                                                           train_ratio=args.train_ratio, val_ratio=args.val_ratio, shuffle=args.shuffle,
                                                                           model_type=args.model, use_cuda=args.cuda)

# 构建模型
graph_model = None #结点特征传播学习方法，用于学习图结构。有三种 静态、多头注意力机制、VAE。
if args.model == 'GKT':
    if args.graph_type == 'MHA':#定义将多头注意机制，在训练过程中无需预先定义边缘权值即可实现学习。
        graph_model = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim, args.attn_dim, dropout=args.dropout)
    elif args.graph_type == 'VAE':#变分自编码器(VAE)以无监督方式学习潜在图结构。
        graph_model = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types, args.vae_decoder_dim, args.vae_decoder_dim, concept_num,
                          edge_type_num=args.edge_types, tau=args.temp, factor=args.factor, dropout=args.dropout, bias=args.bias)
        vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior, var=args.var)
        if args.cuda:
            vae_loss = vae_loss.cuda()
    if args.cuda and args.graph_type in ['MHA', 'VAE']:
        graph_model = graph_model.cuda()
        
#上面部分是学习图结构。
#上面部分是传统GNN所需要做的，MHA解决周围部分结点对目标节点的影响，VAE学习潜在的图结构

#但是并没法解决模拟熟练程度的时间过度。（没有办法扩展这些边缘特征学习机制）

#所以已下定义GKT模型

    model = GKT(concept_num, args.hid_dim, args.emb_dim, args.edge_types, args.graph_type, graph=graph, graph_model=graph_model,
                dropout=args.dropout, bias=args.bias, has_cuda=args.cuda)
elif args.model == 'DKT':
    model = DKT(res_len * concept_num, args.emb_dim, concept_num, dropout=args.dropout, bias=args.bias)
else:
    raise NotImplementedError(args.model + ' model is not implemented!')
kt_loss = KTLoss() #计算损失 预测值和真实值的差距 


# 创建optimizer（最简单的优化算法是SGD梯度下降）
#计算得出loss之后，通常会使用Optimizer对所构造的数学模型/网络模型进行参数优化，
#通常情况下，优化的最终目的是使得loss趋向于最小。

optimizer = optim.Adam(model.parameters(), lr=args.lr)#创建的是Adam的优化器 和梯度下降一样

#optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
#optimizer （Optimizer）：要更改学习率的优化器；
#step_size（int）：每训练step_size个epoch，更新一次参数；
#gamma（float）：更新lr的乘法因子；
#last_epoch （int）：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始。

#scheduler 是用来更新学习率的。提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。
#每过step_size个epoch，做一次更新。
#

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)


# load model/optimizer/scheduler params
# 加载模型/优化器/学习率调整参数

if args.load_dir:
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    model_file = os.path.join(args.load_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    scheduler.load_state_dict(torch.load(scheduler_file))
    args.save_dir = False



#为什么创建了两次？
#再次创建优化器和优化安排器
# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

#
#前面是创建模型 
#后面是用GKT 并采用稀疏（相对更大的范围/集合中，具有较少的响应，可以理解为只存在较少的非零值。）先验（先验知识）的内容。
#先跳过 这里可能涉及训练方式不同？？？？？？？

if args.model == 'GKT' and args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    model = model.cuda()
    kt_loss = KTLoss()


def train(epoch, best_val_loss):
    
    t = time.time()
    loss_train = []
    kt_train = []  #kt模型的损失值
    vae_train = [] #这个部分是变分自编码器的训练结果
    auc_train = [] #训练AUC(Area under Curve)：Roc曲线下的面积，介于0.1和1之间。Auc作为数值可以直观的评价分类器的好坏，值越大越好。
    acc_train = [] #测试集的准确率的训练结果
    if graph_model is not None:
        graph_model.train()
    model.train() #model.train是用来设置训练模式的
    
    
#batch_idx 代表要进行多少次batch_size的迭代，十框石头一次挑一框，batch_idx即为10。
#batch_size ： 代表每次从所有数据中取出一小筐子数据进行训练，类似挑十框石头，每次挑一筐，此时的batch_size=1。这个参数是由于深度学习中尝使用SGD（随机梯度下降）产生。

#适当增加batch_size能够增加训练速度和训练精度（因为梯度下降时震动较小），过小会导致模型收敛困难。

#epoch ： 把所有的训练数据全部迭代遍历一遍（单次epoch），在上面的例子是把train_loader的50000个数据遍历一遍，举例的话是将十框石头全部搬一遍称为一个epoch。
#数据个数/长度 = 1 epoch = batch_size * batch_idx

    for batch_idx, (features, questions, answers) in enumerate(train_loader):
    #for i ,data in enumerate(dataset) i为索引 data部分是每一个索引下所对应的数据
     
        t1 = time.time()
        if args.cuda:
            features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
        ec_list, rec_list, z_prob_list = None, None, None
        if args.model == 'GKT':
            pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
        elif args.model == 'DKT':
            pred_res = model(features, questions)
        else:
            raise NotImplementedError(args.model + ' model is not implemented!')
        loss_kt, auc, acc = kt_loss(pred_res, answers)
        kt_train.append(float(loss_kt.cpu().detach().numpy()))
        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)

#VAE可以定义先验分布
        if args.model == 'GKT' and args.graph_type == 'VAE':
            if args.prior:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
            else:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                vae_train.append(float(loss_vae.cpu().detach().numpy()))
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'loss vae: ', loss_vae.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
            loss = loss_kt + loss_vae
        else:
            loss = loss_kt
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
        loss_train.append(float(loss.cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        del loss
        print('cost time: ', str(time.time() - t1))

    loss_val = []
    kt_val = []
    vae_val = []
    auc_val = []
    acc_val = []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(valid_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')
            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt = float(loss_kt.cpu().detach().numpy())
            kt_val.append(loss_kt)
            if auc != -1 and acc != -1:
                auc_val.append(auc)
                acc_val.append(acc)

            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_val.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_val.append(loss)
            del loss
    if args.model == 'GKT' and args.graph_type == 'VAE':
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'kt_train: {:.10f}'.format(np.mean(kt_train)),
              'vae_train: {:.10f}'.format(np.mean(vae_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'kt_val: {:.10f}'.format(np.mean(kt_val)),
              'vae_val: {:.10f}'.format(np.mean(vae_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
        
#对比损失值选出最佳模型。
    if args.save_dir and np.mean(loss_val) < best_val_loss:
        print('Best model so far, saving...')
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        torch.save(scheduler.state_dict(), scheduler_file)
        if args.model == 'GKT' and args.graph_type == 'VAE':
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'kt_train: {:.10f}'.format(np.mean(kt_train)),
                  'vae_train: {:.10f}'.format(np.mean(vae_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'kt_val: {:.10f}'.format(np.mean(kt_val)),
                  'vae_val: {:.10f}'.format(np.mean(vae_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            del kt_train
            del vae_train
            del kt_val
            del vae_val
        else:
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    res = np.mean(loss_val)
    del loss_train
    del auc_train
    del acc_train
    del loss_val
    del auc_val
    del acc_val
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res


def test():
    loss_test = []
    kt_test = []
    vae_test = []
    auc_test = []
    acc_test = []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        for batch_idx, (features, questions, answers) in enumerate(test_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')
            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt = float(loss_kt.cpu().detach().numpy())
            if auc != -1 and acc != -1:
                auc_test.append(auc)
                acc_test.append(acc)
            kt_test.append(loss_kt)
            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_test.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_test.append(loss)
            del loss
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    if args.model == 'GKT' and args.graph_type == 'VAE':
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'kt_test: {:.10f}'.format(np.mean(kt_test)),
              'vae_test: {:.10f}'.format(np.mean(vae_test)),
              'auc_test: {:.10f}'.format(np.mean(auc_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
    else:
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'auc_test: {:.10f}'.format(np.mean(auc_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
    if args.save_dir:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        if args.model == 'GKT' and args.graph_type == 'VAE':
            print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                  'kt_test: {:.10f}'.format(np.mean(kt_test)),
                  'vae_test: {:.10f}'.format(np.mean(vae_test)),
                  'auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
            del kt_test
            del vae_test
        else:
            print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                  'auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
        log.flush()
    del loss_test
    del auc_test
    del acc_test
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()

if args.test is False:
    # Train model
    print('start training!')
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_dir:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

test()
if log is not None:
    print(save_dir)
    log.close()
