import copy
import random
import os
from typing import Callable, Dict, List, Optional, Tuple
from flwr.common import Parameters, Scalar, Weights, parameters_to_weights, weights_to_parameters, FitRes
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from torch.optim import Adam
import torch.nn.functional as F
import datasets

import models
import Load_data
import alignment as ali
import client as c
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from collections import OrderedDict
import torch


class MyFedAvg(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        eta: float = 1e0,
        config_dict=None
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        self.current_weights = parameters_to_weights(initial_parameters)[-7:]
        self.eta = eta
        self.config_dict = config_dict

    def __repr__(self) -> str:
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep


    def __repr__(self) -> str:
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep


    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        config_dict = self.config_dict
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}


        weights_results = []
        for client, fit_res in results:
            p_t_w = parameters_to_weights(fit_res.parameters)[-6:]  # 取最后一个参数，就是local_centerid
            weights_results.append((p_t_w, fit_res.num_examples))
        metrics_aggregated = {}

        # Collect representations
        if (config_dict['train_mode'] == 'my'):
            print("\nRetrieving Representations... ")
            device = torch.device(config_dict['main_device'] if torch.cuda.is_available() else "cpu")

            client_center = dict()
            client_z = dict()
            client_ypred = dict()
            client_q = dict()
            Z1 = np.array(weights_results[0][0][-1])  # -1 is local centroids
            client_center[weights_results[0][0][-5].min()] = np.array(weights_results[0][0][-1])
            client_z[weights_results[0][0][-5].min()] = np.array(weights_results[0][0][-3])
            q = np.array(weights_results[0][0][-6])
            client_q[weights_results[0][0][-5].min()] = q
            y_pred = (np.argmax(q, axis=1))
            client_ypred[weights_results[0][0][-5].min()] = y_pred
            for client_data in weights_results[1:]:
                Z1 = np.concatenate((Z1, client_data[0][-1]), axis=0)  # -1 is local centroids
                client_center[client_data[0][-5].min()] = np.array(client_data[0][-1])
                client_z[client_data[0][-5].min()] = np.array(client_data[0][-3])
                q = np.array(client_data[0][-6])
                client_q[client_data[0][-5].min()] = q
                y_pred = (np.argmax(q, axis=1))
                client_ypred[client_data[0][-5].min()] = y_pred

            n_client = config_dict["num_clients"]
            # y
            client_y = dict()
            for client_id in range(n_client):
                testloader = Load_data.load_data(config_dict, client_id=client_id,
                                                 n_clients=config_dict['num_clients'],
                                                 alpha=config_dict['alpha'],
                                                 bsize=config_dict["instance_num"])
                for batch_idx, (x, y, _) in enumerate(testloader):
                    y = y.data.cpu().numpy()
                client_y[client_id] = y

            index = int(config_dict['instance_num'] * config_dict['miss_rate'])
            ali_index = int(index*config_dict['align_rate'])

            # alignment input
            ali_in = []   # complete part
            from sklearn.preprocessing import Normalizer
            for client_id in range(n_client):
                client_z[client_id] = Normalizer(norm='l1').fit_transform(client_z[client_id])
                if client_id == 0:
                    ali_in = client_z[client_id][:index, :]
                else:
                    ali_in = np.concatenate((ali_in, client_z[client_id][:index, :]), axis=1)
            ali_in_ali = copy.deepcopy(ali_in[:ali_index,:])
            # ali_in = torch.tensor(ali_in).cuda().to(device).to(torch.float32)
            from sklearn.cluster import KMeans
            n_clusters = config_dict["n_clusters"]
            kmeans = KMeans(n_clusters=n_clusters, n_init=100)
            kmeans.fit_predict(ali_in_ali)
            ali_c = kmeans.cluster_centers_

            # Make sure the trained H is in the same space
            ali_in_qc = []
            for client_id in range(n_client):
                if client_id == 0:
                    ali_in_qc = np.dot(client_q[client_id], ali_c)
                else:
                    temp = np.dot(client_q[client_id], ali_c)
                    if len(ali_in_qc) < len(temp):
                        size = len(ali_in_qc)
                        ali_in_qc = np.concatenate((ali_in_qc, temp[:size,:]), axis=1)
                    else:
                        size = len(temp)
                        ali_in_qc = np.concatenate((ali_in_qc[:size, :], temp), axis=1)
            ali_in_qc = torch.tensor(ali_in_qc).cuda().to(device).to(torch.float32)
            ali_in_qc_train = copy.deepcopy(ali_in_qc[:ali_index, :])

            # Alignment pretrain
            feature_dim = 10*n_client
            alignmentNet = models.myServers(config_dict=config_dict,feature_dim=feature_dim).to(device)
            optimizer = Adam(alignmentNet.parameters(), lr=config_dict['server_lr'])

            anchor = config_dict["anchor"]
            print("anchor-------",anchor)
            ali_in_shuffle = copy.deepcopy(ali_in_qc_train)
            print("ali_in_shuffle", ali_in_shuffle.shape)
            train_dataset = datasets.TrainDataset(ali_in_shuffle,config_dict['view'])
            batch_sampler = datasets.Data_Sampler(train_dataset, shuffle=True, batch_size=256, drop_last=False)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
            kl_div = models.WKLDiv()
            for epoch in range(200):
                loss = 0.
                loss1 = 0.
                loss2 = 0.
                for i, (batch_X_qc, client_p) in enumerate(train_loader):
                    batch_X_qc = batch_X_qc.reshape(batch_X_qc.shape[1], -1)
                    # print("batch_X_qc", batch_X_qc.shape)
                    output = alignmentNet(batch_X_qc)
                    # if output == 1:
                    #     continue
                    # print("output", output[0][0].shape)
                    for viewIndex in range(config_dict['view']):
                        if viewIndex == anchor:
                            continue
                        P_pred = alignmentNet.shuffleLayer(output[anchor][0],output[viewIndex][0],device)
                        ali_out = torch.mm(P_pred, output[viewIndex][0])
                        batch_P = torch.tensor(client_p[viewIndex]).cuda().to(device).to(torch.float32)
                        loss1 = F.mse_loss(output[anchor][0].to(torch.float32), ali_out)
                        constraint_term = 0.1 * torch.norm(P_pred - batch_P, p=2)
                        loss = loss + loss1 + constraint_term
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            alignmentNet.eval()

            random.seed(0)
            np.random.seed(0)
            # alignment module
            client_z_ali = dict()
            client_q_ali = dict()
            client_z_ali[anchor] = ali_in[:, anchor*10: (anchor+1)*10]
            client_q_ali[anchor] = client_q[anchor][:index,:]
            for viewIndex in range(config_dict['view']):
                if viewIndex == anchor:
                    continue
                size = len(ali_in)-ali_index
                client_z_ali[viewIndex] = ali_in[:, viewIndex * 10: (viewIndex + 1) * 10]
                client_q_ali[viewIndex] = client_q[viewIndex][:index,:]
                ## aligned--ablation
                P_index = random.sample(range(size), size)
                P_gt = np.eye(size).astype('float32')
                P_gt = P_gt[:, P_index]
                ali_in_temp = ali_in_qc[:size, viewIndex * feature_dim: (viewIndex + 1) * feature_dim][P_index]
                ali_in_qc[:size, viewIndex * feature_dim: (viewIndex + 1) * feature_dim] = ali_in_temp
                temp1 = ali_in[:size, viewIndex * 10: (viewIndex + 1) * 10][P_index]
                ali_in[:size, viewIndex * 10: (viewIndex + 1) * 10] = temp1
                temp_q = client_q[viewIndex][:size,:][P_index]
                client_q_ali[viewIndex][:size,:] = temp_q

            output = alignmentNet(ali_in_qc)
            ali_in = torch.tensor(ali_in).cuda().to(device).to(torch.float32)
            for viewIndex in range(config_dict['view']):
                if viewIndex == anchor:
                    continue
                P_pred = ali.myAlignment(output[anchor][0].cpu().detach().data.numpy(),
                                         output[viewIndex][0].cpu().detach().data.numpy())
                # ablation WH
                # P_pred = ali.myAlignment(ali_in_qc[:, anchor * feature_dim: (anchor + 1) * feature_dim].cpu().detach().data.numpy(),
                #                          ali_in_qc[:, viewIndex * feature_dim: (viewIndex + 1) * feature_dim].cpu().detach().data.numpy())

                client_q_ali[viewIndex][:size, :] = np.dot(P_pred[:size,:size],client_q_ali[viewIndex][:size, :])
                P_pred = torch.tensor(P_pred).cuda().to(device).to(torch.float32)
                client_z_ali[viewIndex][:size,:] = \
                    torch.mm(P_pred[:size,:size],
                             ali_in[:size, viewIndex * 10: (viewIndex + 1) * 10]).cpu().detach().data.numpy()


                # ablation alignment
                # client_z_ali[viewIndex][:size, :] = ali_in[:size, viewIndex * 10: (viewIndex + 1) * 10].cpu().detach().data.numpy()


            client_z_temp = dict()
            client_q_temp = dict()
            q_sum = np.zeros((index, client_q[0].shape[1]))
            for client_id in range(n_client):
                client_z_temp[client_id] = np.zeros((config_dict['instance_num'], client_z[client_id].shape[1]))
                q_sum = np.add(q_sum, client_q[client_id][:index, :])
            # global Z
            w = [1] * n_client
            globalZ = []
            missing_index = []
            qList = []
            p_temp = []
            for client_id in range(n_client):
                missingPath = config_dict["data_dir"] + config_dict["dataset"] + 'missing_'+str(client_id)+'.csv'
                temp = np.loadtxt(missingPath, delimiter=',', dtype=np.float32)
                temp = temp.astype(np.int64)
                missing_index.append(temp)
                qList.append(client_q[client_id][:index, :])
                if client_id == 0:
                    globalZ = client_z_ali[client_id]
                    p_temp = client_q_ali[client_id]
                    client_z_temp[client_id][:index, :] = client_z_ali[client_id]
                else:
                    globalZ = np.concatenate((globalZ, client_z_ali[client_id]), axis=1)
                    p_temp = np.add(p_temp, client_q_ali[client_id])
                    client_z_temp[client_id][:index, :] = client_z_ali[client_id]
                client_q_temp[client_id] = np.subtract(q_sum,client_q[client_id][:index, :])/(sum(w)-w[client_id])
            p_temp /= sum(w)
            label = client_y[0][:index]


            net = models.my(config_dict, n_clusters, -1,
                            2500, 2500, False).to(device)

            # Load parameters
            params_dict = zip(net.state_dict().keys(), self.current_weights)
            state_dict = OrderedDict(
                {k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict if
                    ('encoder' not in k and 'decoder' not in k)})
            net.load_state_dict(state_dict, strict=False)
            # global centers
            if rnd > 1:
                init_centers = copy.deepcopy(net.globalClusteringLayer.centroids.data).cpu().detach().data.numpy()
                kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=10)
            else:
                kmeans = KMeans(n_clusters=n_clusters, n_init=100)
            kmeans.fit_predict(globalZ)
            centers = kmeans.cluster_centers_
            #   QC
            globalZ_temp = np.zeros(((config_dict['instance_num'] - index), centers.shape[1]))
            for i in range(index, config_dict['instance_num']):
                w_sum = 0
                q_temp = []
                missing = []
                y_temp = []
                for v in range(n_client):
                    if i in missing_index[v]:
                        w_sum += 1
                        row = np.where(missing_index[v] == i)
                        # print(v,row,i)
                        z_temp = client_z[v][row, :].reshape(1, -1)
                        globalZ_temp[i - index, v * 10: (v + 1) * 10] = z_temp
                        if len(q_temp) == 0:
                            q_temp = client_q[v][row, :].reshape(1, -1)
                            y_temp = client_y[v][row]
                            client_z_temp[v][i, :] = client_z[v][row, :].reshape(1, -1)
                        else:
                            q_temp = np.add(q_temp, client_q[v][row, :].reshape(1, -1))
                            client_z_temp[v][i, :] = client_z[v][row, :].reshape(1, -1)
                    else:
                        missing.append(v)
                q_temp = q_temp / w_sum
                for id in missing:
                    globalZ_temp[i - index, id * 10: (id + 1) * 10] = np.dot(q_temp, centers[:, id * 10: (id + 1) * 10])

                p_temp = np.concatenate((p_temp, q_temp), axis=0)
                label = np.concatenate((label, y_temp), axis=0)

            z = np.concatenate((globalZ, globalZ_temp), axis=0)  # 增加行
            z = torch.tensor(z).cuda().to(device)

            # server network
            netServer = models.myServers(config_dict=config_dict,feature_dim=10).to(device)
            optimizer = Adam(netServer.parameters(), lr=config_dict['server_lr'])
            x = []
            for client_id in range(n_client):
                x_temp = np.dot(client_q_temp[client_id], centers[:, client_id * 10: (client_id + 1) * 10])
                if len(x) == 0:
                    x = x_temp
                else:
                    x = np.concatenate((x, x_temp), axis=1)
            x = torch.tensor(x).cuda().to(device).to(torch.float32)

            # MLP
            for epoch in range(300):
                loss = 0.
                output = netServer(x)
                for viewIndex in range(config_dict['view']):
                    zz = []
                    zz.append(z[: index, viewIndex * 10: (viewIndex + 1) * 10])
                    loss = loss + F.mse_loss(output[viewIndex][0].to(torch.float32), zz[0].to(torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            netServer.eval()

            globalZ_temp = np.zeros(((config_dict['instance_num']-index),centers.shape[1]))
            for i in range(index,config_dict['instance_num']):
                w_sum = 0
                q_temp = []
                missing = []
                y_temp = []
                for v in range(n_client):
                    if i in missing_index[v]:
                        w_sum += 1
                        row = np.where(missing_index[v] == i)
                        z_temp = client_z[v][row, :].reshape(1, -1)
                        globalZ_temp[i-index, v*10 : (v+1)*10] = z_temp
                        if len(q_temp) == 0:
                            q_temp = client_q[v][row, :].reshape(1, -1)
                            y_temp = client_y[v][row]
                        else:
                            q_temp = np.add(q_temp, client_q[v][row, :].reshape(1, -1))
                    else:
                        missing.append(v)
                q_temp = q_temp/w_sum
                for id in missing:
                    q_temp = p_temp[i,:]
                    x = torch.tensor(np.dot(q_temp, centers)).cuda().to(device).reshape(1,-1)
                    output = netServer(x)
                    globalZ_temp[i - index, id * 10: (id + 1) * 10] = output[id][0].cpu().detach().data.numpy()

                    # ablation WI
                    # globalZ_temp[i - index, id * 10: (id + 1) * 10] = x[:,id * 10: (id + 1) * 10].cpu().detach().data.numpy()

            globalZ = np.concatenate((globalZ, globalZ_temp), axis=0)

            y_pred = (np.argmax(p_temp, axis=1))
            acc1 = c.cluster_acc(label, y_pred)
            nmi = nmi_score(label, y_pred)
            ari = ari_score(label, y_pred)
            print('server:Acc {:.4f}'.format(acc1),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            globalZ = torch.tensor(globalZ, device=device).cuda()
            q_f = net.global_clustering(globalZ, rnd=rnd)

            net.client_id.z.data = torch.tensor(-1).cuda()
            p_f = copy.deepcopy(net.p.z.data).cpu().detach().data.numpy()
            y_pred = (np.argmax(p_f, axis=1))
            y_pred1 = (np.argmax(q_f, axis=1))
            testloader = Load_data.load_data(config_dict, client_id=0,
                                              n_clients=config_dict['num_clients'],
                                              alpha=config_dict['alpha'],
                                              bsize=config_dict["local_bsize"])
            for batch_idx, (x, y, _) in enumerate(testloader):
                y = y.data.cpu().numpy()
                # y = y[:index]
            acc = c.cluster_acc(label, y_pred)
            acc2 = c.cluster_acc(label, y_pred1)
            nmi = nmi_score(label, y_pred)
            ari = ari_score(label, y_pred)
            # print('server2:Acc {:.4f}'.format(acc),
            #       ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
            # print(y_pred,y_pred.shape)
            print("q_f acc", acc2)

            # adaptive imputation
            if rnd ==1:
                net.imputation = False
                print("Abandon imputation---------")
            else:
                if acc1 > acc2 :
                    net.imputation = False
                    net.globalClusteringLayer.centroids.data = torch.tensor(ali_c).cuda()
                    # net.p.z.data = torch.tensor(p_temp).cuda()
                    print("Abandon imputation---------")
                else:
                    net.imputation = True
                    print("Imputation---------")


            # AR
            viewNumber = config_dict["view"]
            view_wise_Y = np.zeros((viewNumber, index))
            for viewIndex in range(viewNumber):
                view_wise_result = qList[viewIndex]
                y_pred = (np.argmax(view_wise_result, axis=1))
                view_wise_Y[viewIndex, :] = y_pred
            view_wise_P = view_wise_Y - np.tile(y_pred, (viewNumber, 1))
            view_wise_P = np.abs(view_wise_P)
            view_wise_P = np.sum(view_wise_P, axis=0)
            mask = view_wise_P > 0
            view_wise_P[mask] = 1
            biaInstances = np.sum(view_wise_P)
            AR = 1.0 - 1.0 * biaInstances / index
            print("---------------AR:", AR)


            # Retrieve trained parameters and update fedavg output
            self.current_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]

            # Delete network to free memory; it's not needed anymore
            del net
            del alignmentNet
            del netServer
            if config_dict["n_pretrain"] == [True] * n_client:
                config_dict["n_pretrain"] = [False]* n_client
                return weights_to_parameters(self.current_weights), metrics_aggregated
            config_dict["n_initial"] = [False]* n_client

        return weights_to_parameters(self.current_weights), metrics_aggregated