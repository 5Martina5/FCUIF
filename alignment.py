import torch
import torch.nn.functional as F
import numpy as np
import models


def row2one(P):
    P_sum = P.sum(dim=1, keepdim=True)
    one = torch.ones(1, P.shape[1]).to(P.device)
    return P - (P_sum - 1).mm(one) / P.shape[1]

def col2one(P):
    P_sum = P.sum(dim=0, keepdim=True)
    one = torch.ones(P.shape[0], 1).to(P.device)
    return P - (one).mm(P_sum - 1) / P.shape[0]     

def P_init(D):
    P = torch.zeros_like(D)
    D_rowmin = D.clone()
    max_d = D.max()
    min_ind = torch.argmin(D_rowmin, dim=0)
    D_rowmin[:, :] = max_d   # 让D_rowmin全都扩大
    # 这段代码的作用是将D[min_ind, torch.arange(D.shape[1]).long()]的值按照min_ind的索引更新到D_rowmin张量的第0维上。
    # 1*128 这个值的意思是其中包含了D张量中第min_ind行的所有元素。torch.arange(D.shape[1]).long()创建了一个从0到D张量的列数的整数张量，然后使用这个整数张量作为索引来获取D张量中指定行的元素。
    D_rowmin = D_rowmin.scatter(0, min_ind.unsqueeze(0), D[min_ind, torch.arange(D.shape[1]).long()].unsqueeze(0))


    _, idx_max = torch.min(D_rowmin, dim=1)

    P[torch.arange(D.shape[0]).long(), idx_max.long()] = 1.0    # 对于每一列的最小值就是P对应的值

    return P

def alignment(D, tau_1=30, tau_2=10, lr=0.1):
    # 按照距离度量得到的P
    P = P_init(D)

    # 下面就是让P不再是单纯的0和1，而是
    d = [torch.zeros_like(D) for _ in range(3)] 
    
    for i in range(tau_1):               
        P = P - lr * D 
        for j in range(tau_2):
                P_0 = P.clone()       
                P = P + d[0]
                Y = row2one(P)
                d[0] = P - Y

                P = Y + d[1]
                Y = col2one(P)
                d[1] = P - Y

                P = Y + d[2]
                Y = F.relu(P)
                d[2] = P - Y

                P = Y
                if (P - P_0).norm().item() == 0:
                    break

    return P

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def myAlignment(H1, H2):
    inner_product = np.dot(H1, H2.T)

    # 获取矩阵的行数和列数
    d_k = inner_product.shape[-1]

    # 计算公式中的除法和开平方
    scaled_inner_product = inner_product / np.sqrt(20)

    # 使用 softmax 函数计算最终的 P 矩阵
    P = np.exp(scaled_inner_product - np.max(scaled_inner_product, axis=-1, keepdims=True))
    P /= np.sum(P, axis=-1, keepdims=True)
    # P = target_distribution(P)


    return P