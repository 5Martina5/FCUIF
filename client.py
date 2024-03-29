import argparse
import os
import gc
import traceback
import warnings
from collections import OrderedDict
import copy

import flwr as fl
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
from torch.optim import Adam

import Load_data
import models
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


warnings.filterwarnings("ignore", category=UserWarning)
cudnn.deterministic = True
cudnn.benchmark = False
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


def my_train(net, trainloader, testloader, n_clusters, client_id, path, initial,config_dict, epochs, lr, device):

    if net.pretrain:
        net.train()
        optimizer = Adam(net.parameters(), lr=lr)
        tot_loss = 0
        # pretrain autoencoder
        for epoch in range(500):
            for batch_idx, (x, _, _) in enumerate(trainloader):
                x = x.cuda()
                x = x.to(torch.float32)
                # x = x.reshape(x.shape[0], -1)
                # x = x.reshape(x.shape[0], -1, 28, 28)
                output = net(x)
                loss = F.mse_loss(output[1], x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()


        for batch_idx, (x, y, _) in enumerate(testloader):
            x = x.cuda()
            x = x.to(torch.float32)
            # x = x.reshape(x.shape[0], -1)
            # x = x.reshape(x.shape[0], -1, 28, 28)
            output = net(x)
        kmeans = KMeans(n_clusters=n_clusters, n_init=100)
        z_v = output[0]
        kmeans.fit_predict(z_v.cpu().detach().data.numpy())

        centers = kmeans.cluster_centers_
        net.localClusteringLayer.centroids.data = torch.tensor(centers).cuda()  # 更新net的参数
        net.localZ.z.data = torch.tensor(z_v).cuda()
        net.client_id.z.data = torch.tensor(client_id).cuda()
        # print("client_id",net.client_id.z.data)
        net.eval()
        torch.save(net.state_dict(), path)
        return
    print(config_dict["n_initial"][client_id],client_id)
    p_f = copy.deepcopy(net.p.z.data).cpu().detach().data.numpy()
    # p
    local_bsize = config_dict["local_bsize"]
    missingPath = config_dict["data_dir"] + config_dict["dataset"] + 'missing_'+str(client_id)+'.csv'
    missing_index = np.loadtxt(missingPath, delimiter=',', dtype=np.float32)
    missing_index = missing_index.astype(np.int64)
    index = int(config_dict['instance_num']*config_dict['miss_rate'])
    ali_index = int(index * config_dict['align_rate'])
    size = index - ali_index
    p = p_f[:index,:]
    # imp = copy.deepcopy(net.imputation)
    if net.imputation == True:
        for i in range(index,len(missing_index)-1):
            p = np.concatenate((p, p_f[missing_index[i],:].reshape(1,-1)), axis=0)
        print("Imputation---------")
    else:
        print("Abandon imputation---------")

    p = torch.tensor(p, device=device).cuda()
    # # abandon alignment
    # p = p[size:ali_index, :]

    if not initial:
        pt = torch.load(path)
        pt.pop("client_id.z")
        net.load_state_dict(pt,strict=False)
    else:
        localCenters = copy.deepcopy(net.localClusteringLayer.centroids.data)
        pt = torch.load(path)
        pt.pop("client_id.z")
        net.load_state_dict(pt, strict=False)
        net.localClusteringLayer.centroids.data = localCenters
    # print("p:", p)
    kl_div = models.WKLDiv()
    # loss_function = nn.KLDivLoss(reduction='mean')
    net.train()
    optimizer = Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        # The outputs correspond to z, x_bar, and q, respectively.
        for batch_idx, (x, y, _) in enumerate(trainloader):
            x = x.cuda()
            x = x.to(torch.float32)
            # x = x.reshape(x.shape[0], -1)
            # x = x.reshape(x.shape[0], -1, 28, 28)
            output = net(x)
            z_v = output[0]
            q = output[2]
            # loss
            mseloss = torch.nn.functional.mse_loss(output[1][:index,:], x[:index,:])
            batch_p = p[batch_idx*config_dict['local_bsize']:(batch_idx+1)*config_dict['local_bsize'],:]
            kl_loss = kl_div(torch.log(q[:batch_p.shape[0],:]+1e-12), batch_p)
            # kl_loss = kl_div(torch.log(q[:p.shape[0], :] + 1e-12), p)
            alpha = 0.1
            loss = mseloss + alpha * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



def test(net, testloader, client_id, path,config_dict):
    net.eval()
    ii = int(config_dict['instance_num'] * config_dict['miss_rate'])
    print("\n")
    # print("net",net.state_dict())
    with torch.no_grad():
        # output--z, x_bar, q
        for batch_idx, (x, y, _) in enumerate(testloader):
            x = x.cuda()
            x = x.to(torch.float32)
            # x = x.reshape(x.shape[0], -1)
            # x = x.reshape(x.shape[0], -1, 28, 28)
            output = net(x)
        z_v = output[0]
        q = models.target_distribution(output[2])
        missingPath = config_dict["data_dir"] + config_dict["dataset"] + 'missing_' + str(client_id) + '.csv'
        missing_index = np.loadtxt(missingPath, delimiter=',', dtype=np.float32)
        missing_index = missing_index.astype(np.int64)
        y_pred = (np.argmax(q[:len(missing_index),:].cpu().detach().data.numpy(), axis=1))
        # print("y_pred", y_pred)
        # print("q", q, q.shape)
        for batch_idx, (x, y, index) in enumerate(testloader):
            y = y.data.cpu().numpy()[:len(missing_index)]
            # y = y.data.cpu().numpy()[:ii]
            index = index.data.cpu().numpy()
        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        ari = ari_score(y, y_pred)
        print('client', client_id, ':Acc {:.4f}'.format(acc),
              ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

        config_dict["client_acc"][client_id] = acc

        net.localZ.z.data = z_v.cuda()
        net.client_id.z.data = torch.tensor(client_id).cuda()
        net.q.z.data = q
        net.eval()
        torch.save(net.state_dict(), path)

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    res = []
    for i in range(ind[0].size):
        res.append(w[ind[0][i], ind[1][i]])
    # print(res)
    return sum(res) * 1.0 / y_pred.size

def make_client(cid, device=None, stateless=True, config_dict=None):
    try:
        gc.collect()
        torch.cuda.empty_cache()
        client_id = int(cid)  # cid is of type str when using simulation

        if device is None:
            print("Client {} CUDA_VISIBLE_DEVICES: {}".format(cid, os.environ["CUDA_VISIBLE_DEVICES"]))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_save_path = config_dict["save_dir"] + "/saved_models/" + config_dict["dataset"] + "_client_" + str(
            client_id) + ".pth"

        ##### Create model
        n_clusters = config_dict["n_clusters"]

        ##### Load data
        trainloader = Load_data.load_data(config_dict, client_id=client_id,
                                                             n_clients=config_dict['num_clients'],
                                                             alpha=config_dict['alpha'],
                                                             bsize=config_dict["local_bsize"])
        testloader = Load_data.load_data(config_dict, client_id=client_id,
                                          n_clients=config_dict['num_clients'],
                                          alpha=config_dict['alpha'],
                                          bsize=config_dict["instance_num"])
        input_num, input_size =0, 0
        for batch_idx, (x, y, _) in enumerate(testloader):
            # x = x.reshape(x.shape[0], -1)
            # x = x.reshape(x.shape[0], -1, 28, 28)
            input_num = x.shape[0]
            input_size = x.shape[1]
        # Define model
        if (config_dict["train_mode"] == "my"):
            net = models.my(config_dict, n_clusters, client_id,
                            input_size,input_num,config_dict["n_pretrain"][client_id] )
            print(input_size,"------------------")
            net = net.to(device)


        ##### Flower client
        class flclient(fl.client.NumPyClient):
            def get_parameters(self):
                return [val.cpu().numpy() for _, val in net.state_dict().items()]

            def set_parameters(self, parameters):
                # print(parameters)
                params_dict = zip(net.state_dict().keys(), parameters)
                state_dict = OrderedDict(
                    {k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict if
                     ('encoder' not in k and 'decoder' not in k and 'q' not in k and 'localZ' not in k )})
                # print(state_dict)
                net.load_state_dict(state_dict, strict=False)

            def fit(self, parameters, config):
                try:
                    path = f'{config_dict["save_dir"]}/saved_models/{client_id}_client_cache.pkl'
                    self.set_parameters(parameters)
                    # print("client global",net.globalClusteringLayer.centroids.data)

                    # Supervised training
                    if (config_dict["train_mode"] == "my"):

                        initial = False
                        if config_dict["n_initial"][client_id] and not net.pretrain :
                            local_bsize = config_dict["local_bsize"]
                            globalCentroids = net.globalClusteringLayer.centroids.data.cpu().detach().data.numpy()
                            dim = net.feature_dim * (client_id + 1)
                            dim1 = net.feature_dim * client_id

                            net.localClusteringLayer.centroids.data = torch.tensor(
                                globalCentroids[:, dim1:dim]).cuda()

                            initial = True

                        my_train(net, trainloader,testloader, n_clusters, client_id,path,initial,config_dict=config_dict,epochs=config_dict["local_epochs"], lr=config_dict["local_lr"],
                                  device=device)
                        test(net, testloader, client_id, path, config_dict)


                    return [val.cpu().numpy() for _, val in net.state_dict().items()], len(trainloader), {}
                except Exception as e:
                    print(f"Client {cid} - Exception in client fit {e}")
                    print(f"Client {cid}", traceback.format_exc())

            def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                if (config_dict["train_mode"] == "my"):
                    accuracy = 0
                    return float(0), len(trainloader), {"accuracy": float(accuracy)}

            def save_net(self):
                ##### Save local model
                state = {'net': net.state_dict()}
                torch.save(state, model_save_path)
                print(f"Client: {client_id} Saving network to {model_save_path}")

        gc.collect()
        torch.cuda.empty_cache()
        return flclient()
    except Exception as e:
        print(f"Client {cid} - Exception in make_client {e}")
        print(f"Client {cid}", traceback.format_exc())


##### Federation of the pipeline with Flower
def main(config_dict):
    """Create model, load data, define Flower client, start Flower client."""
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client_id", type=int, default=0)
    args = parser.parse_args()

    # device = torch.device(config_dict["main_device"] if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + str(args.client_id % 8) if torch.cuda.is_available() else "cpu")

    local_client = make_client(args.client_id, device=device, stateless=True, config_dict=config_dict)

    ##### Start client
    fl.client.start_numpy_client("[::]:9081", client=local_client)

    local_client.save_net()
