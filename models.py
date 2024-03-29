import copy

import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z))

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return q

class ShuffleLayer(nn.Module):

    def __init__(self, size):
        super(ShuffleLayer, self).__init__()
        self.size = size
        self.shuffle = Parameter(torch.Tensor(self.size, self.size))

    def forward(self, H1, H2,device):
        inner_product = torch.mm(H1, H2.T)

        d_k = inner_product.shape[-1]
        d_k = torch.tensor(d_k).cuda().to(device).to(torch.float32)

        scaled_inner_product = inner_product / torch.sqrt(d_k)
        max_values, _ = torch.max(scaled_inner_product, axis=-1, keepdims=True)

        P1 = torch.exp(scaled_inner_product - max_values)

        sum_P = torch.sum(P1, axis=-1, keepdims=True)
        P = P1 / sum_P
        self.shuffle.data = P
        return P

class Concat(nn.Module):

    def __init__(self, batch_size, feature_dim):
        super(Concat, self).__init__()
        self.z = Parameter(torch.Tensor(batch_size, feature_dim))

    def forward(self, x):

        return x

class Concat1(nn.Module):

    def __init__(self,id):
        super(Concat1, self).__init__()
        self.z = Parameter(torch.Tensor(id))

    def forward(self, x):

        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )
        for m in self.encoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
        for m in self.decoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.decoder(x)


def make_qp(x, centroids):
    q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - torch.tensor(centroids).cuda(), 2), 2))
    q = (q.t() / torch.sum(q, 1)).t()
    p = target_distribution(q)
    return q, p


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def euclidean_dist(x, y, root=False):
    m, n = x.size(0), y.size(0)

    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy

    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class WKLDiv(torch.nn.Module):
    def __init__(self):
        super(WKLDiv, self).__init__()

    def forward(self, q_logit, p):
        p_logit=torch.log(p + 1e-12)
        kl = torch.sum(p * (p_logit - q_logit), 1)
        return torch.mean(kl)


class myServer(nn.Module):
    def __init__(self,feature_dim):
        super(myServer, self).__init__()

        self.feature_dim = feature_dim

        self.feature_module = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim)
        )
        for m in self.feature_module:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.feature_module(x)
        return z,

class myServers(nn.Module):
    def __init__(self,config_dict,feature_dim):
        super(myServers, self).__init__()

        self.viewNumber = config_dict['view']
        self.feature_dim = feature_dim

        servers = []

        for viewIndex in range(self.viewNumber):
            servers.append(myServer(feature_dim))

        self.servers = nn.ModuleList(servers)
        self.shuffleLayer = ShuffleLayer(256)

    def forward(self, x):
        outputs = []
        for viewIndex in range(self.viewNumber):
            outputs.append(self.servers[viewIndex](x[:, viewIndex * self.feature_dim: (viewIndex + 1) * self.feature_dim]))
        return outputs

# client
class my(nn.Module):
    def __init__(self, config_dict, n_clusters,client_id,input_size,input_num,pretrain):
        super(my, self).__init__()

        self.feature_dim = 10
        self.encoder = Encoder(input_size, self.feature_dim)

        self.decoder = Decoder(input_size, self.feature_dim)

        self.n_clusters = n_clusters
        self.pretrain = pretrain

        # Setup arguments
        self.instance_num = config_dict['instance_num']
        self.imputation = False

        self.p = Concat(self.instance_num, n_clusters)
        self.q = Concat(input_num, n_clusters)
        self.client_id = Concat1(1)
        self.globalZ = Concat(self.instance_num, self.feature_dim * config_dict['view'])
        self.localZ = Concat(input_num, self.feature_dim)
        self.globalClusteringLayer = ClusteringLayer(n_clusters , self.feature_dim * config_dict['view'])
        self.localClusteringLayer = ClusteringLayer(n_clusters, self.feature_dim)


    def forward(self, x):
        z = self.encoder(x)
        x_bar = self.decoder(z)
        q = self.localClusteringLayer(z)

        return z, x_bar, q,

    def Match(self, y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        new_y = np.zeros(y_true.shape[0])

        matrix = np.zeros((D, D), dtype=np.int64)
        matrix[row_ind, col_ind] = 1
        for i in range(y_pred.size):
            for j in row_ind:
                if y_true[i] == col_ind[j]:
                    new_y[i] = row_ind[j]
        return new_y, row_ind, col_ind, matrix

    def global_clustering(self, Z, rnd):
        with torch.no_grad():
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
            Z = Z.cpu().detach().data.numpy()
            kmeans.fit_predict(Z)
            centers = kmeans.cluster_centers_
            q_f, p_f = make_qp(torch.tensor(Z).cuda(), centers)
            p_f = p_f.cpu().detach().data.numpy()
            q_f = q_f.cpu().detach().data.numpy()

            print("rnd", rnd)
            if rnd > 1:
                p_f_before = copy.deepcopy(self.p.z.data).cpu().detach().data.numpy()
            else:
                p_f_before = copy.deepcopy(p_f)
            new_y, row_ind, col_ind, matrix = self.Match(np.argmax(p_f, axis=1), np.argmax(p_f_before, axis=1))
            p_f = np.dot(p_f, matrix)
            q_f = np.dot(q_f,matrix)

            self.globalClusteringLayer.centroids.data = torch.tensor(centers).cuda()
            self.p.z.data = torch.tensor(p_f).cuda()

        return q_f

