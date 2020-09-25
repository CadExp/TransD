import torch
import numpy as np
from utils import *
from Config import Config
from run import *

# torch.cuda.set_device(0)
if __name__ == '__main__':

    e, r_num, time_unit = get_total_number(Config.data, 'stat.txt')  # 得到知识图谱的实体数目，关系数目，时间单元（默认是1）

    train_TKG, last_time = load_TKG(Config.data, 'train_quadruples.txt')  # 那个字典接收知识图谱，然后记录最后的时间戳  #动态知识图谱
    valid_TKG, last_time = load_TKG(Config.data, 'valid_quadruples.txt')
    test_TKG, last_time = load_TKG(Config.data, 'test_quadruples.txt')
    # 时序知识图谱词典，TKG[time]为在time时的（头实体，关系，尾实体）的list; last_time为最后时间

    # 完全不考虑时间，就是三元组的静态知识图谱
    train_SKG = load_static_graph(Config.data, 'train_quadruples.txt', last_time, 0)
    test_SKG = load_static_graph(Config.data, 'test_quadruples.txt', last_time, 0)
    valid_SKG = load_static_graph(Config.data, 'valid_quadruples.txt', last_time, 0)
    # 将TKG转化为静态知识图谱SKG，格式为（头实体，关系，尾实体）

    head_pre_result = []  # 头存储实体预测结果
    tail_pre_result = []  # 尾存储实体预测结果
    relation_pre_result = []  # 存储关系预测结果
    test_num = 0  # 测试的事实的数目

    entity2id = load_enorrel2id(Config.data, 'ent2id.txt')
    relation2id = load_enorrel2id(Config.data, 'rel2id.txt')
    Run = run(1)

    Run.train(train_SKG, -1, entity2id, relation2id,valid_SKG)  # 将模型在静态知识图谱上预训练,得到特征和初始参数。注，有些简单模型像TransE只有特征，没有参数；也有一些模型只调整参数，不改变输入特征
    s_evaluation_head, s_evaluation_tail = Run.evaluate(test_SKG,train_SKG + valid_SKG + test_SKG,relation2id, entity2id, -1)

    print(s_evaluation_head)
    print(s_evaluation_tail)
    # result 测试结果：hits@1，hits@3, hits@10, mrr       'HITS@1' 'HITS@3' 'HITS@10' 'MRR'
    # 先在SKG上评价，作为baseline

    for time in range(0, last_time):
        Run.train(train_TKG[time], time, entity2id, relation2id,valid_TKG[time])  # 将模型拟合于第time个时间戳，即根据第time个时间戳调整模型参数和特征X

        evaluation_head, evaluation_tail = Run.evaluate(test_TKG[time],train_TKG[time] + valid_TKG[time] +test_TKG[time], relation2id, entity2id,time)

        result = [evaluation_head['HITS@1'], evaluation_head['HITS@3'], evaluation_head['HITS@10'],
                  evaluation_head['MRR']]
        for count in range(0, len(test_TKG[time])):
            head_pre_result.append(result)

        result = [evaluation_tail['HITS@1'], evaluation_tail['HITS@3'], evaluation_tail['HITS@10'],
                  evaluation_tail['MRR']]
        for count in range(0, len(test_TKG[time])):
            tail_pre_result.append(result)

        #result = [evaluation_relation['HITS@1'], evaluation_relation['HITS@3'], evaluation_relation['HITS@10'],
        #          evaluation_relation['MRR']]
        #relation_pre_result.append(elm * len(test_TKG) for elm in result)

        # head_pre_result.append(elm*len (test_TKG) for elm in evaluation_head)#实体预测结果，evaluate_entity()应返回[hits@1，hits@3, hits@10, mrr]
        # tail_pre_result.append(elm * len(test_TKG) for elm in evaluation_tail)
        # relation_pre_result.append(elm*len (test_TKG) for elm in evaluation_relation)
        test_num += len(test_TKG[time])

    print(np.sum(head_pre_result, axis=0) / test_num)  # 每个时间戳评价的加权平均值
    print(np.sum(tail_pre_result, axis=0) / test_num)
    #print(np.sum(relation_pre_result, axis=0) / test_num)
    # 整个TKG的评价结果

