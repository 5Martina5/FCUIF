

import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
import pickle as pkl
import os, shutil
import Load_data

parser = argparse.ArgumentParser(description="Sample data for clients")
parser.add_argument("--dataset", default="Scene", choices=["BDGP","Scene","REU","HW","Fashion","MNIST_USPS"])
parser.add_argument("--instanceNumber", default=4485, choices=[2500, 4485, 1200, 2000, 10000])
parser.add_argument("--view", default=3, choices=[2, 3, 5, 6, 6])
parser.add_argument("--n_clients", type=int, default=3)
parser.add_argument("--alpha", type=float, default=1e2, choices=[1e-2, 1e0, 1e2])
parser.add_argument("--data_dir", default="./data/")
parser.add_argument("--missing", default=0.5)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}', exist_ok=True)
os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/train', exist_ok=True)

##### Print setup to confirm things are correct
print("\nSampling configuration:")
print("\tDataset:", args.dataset)
print("\tNumber of clients:", args.n_clients)
print("\tMissing rate:", args.missing)
print("\tWriting data at this location: ", args.data_dir + "/" + str(args.n_clients))
print("\tAlpha for Dirichlet distribution:", args.alpha)

n_clients = args.n_clients
instance_number = args.instanceNumber
view = args.view
slice = instance_number * view // n_clients

x, y = Load_data.load_data_conv(args.dataset)
X, Y, index = Load_data.Form_Incomplete_Data(missrate=args.missing, X=x, Y=y)
# X, Y, index = Load_data.generate_non_iid_missing_data(x,y,args.missing,args.alpha)


##### Save IDs
# Train
i =0
index1 = 0
index2 = slice
client_ids = {client_num: {} for client_num in range(n_clients)}
for client_num in range(n_clients):
    client_ids[client_num] = [X[client_num], Y[client_num], index[client_num]]
    missingPath = './data/' + args.dataset + 'missing_' + str(client_num) + '.csv'
    np.savetxt(missingPath, index[client_num], delimiter=',')


# print("\nDistribution over classes:")
for client_num in range(n_clients):
    with open(f"{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/train/"+args.dataset+"_"+str(client_num)+'.pkl', 'wb') as f:
        pkl.dump(client_ids[client_num], f)
    # print(client_ids[client_num].shape)