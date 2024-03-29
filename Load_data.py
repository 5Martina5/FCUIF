
import scipy.io as scio
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import numpy as np

np.random.seed(0)
cudnn.deterministic = True
cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")

path = './data'

def Scene_15():
    mat = scio.loadmat(path + "/Scene-15.mat")
    X = mat['X'][0]

    x1 = X[1]
    x2 = X[0]
    x3 = X[2]
    xx1 = np.copy(x1)
    xx2 = np.copy(x2)
    xx3 = np.copy(x3)
    Y = np.copy(mat['Y'].T[0])
    index = [i for i in range(4485)]
    np.random.seed(4485)
    np.random.shuffle(index)
    for i in range(4485):
        xx1[i] = x1[index[i]]
        xx2[i] = x2[index[i]]
        xx3[i] = x3[index[i]]
        Y[i] = mat['Y'].T[0][index[i]]

    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    xx1 = min_max_scaler.fit_transform(xx1)
    xx2 = min_max_scaler.fit_transform(xx2)
    xx3 = min_max_scaler.fit_transform(xx3)

    return [xx1, xx2, xx3], [Y, Y, Y]




def Form_Incomplete_Data(missrate=0.5, X = [], Y = []):
    np.random.seed(0)
    size = len(Y[0])
    view_num = len(X)
    t = np.linspace(0, size - 1, size, dtype=int)
    import random
    random.shuffle(t)
    Xtmp = []
    Ytmp = []
    for i in range(view_num):
        xtmp = np.copy(X[i])
        Xtmp.append(xtmp)
        ytmp = np.copy(Y[i])
        Ytmp.append(ytmp)
    for v in range(view_num):
        for i in range(size):
            Xtmp[v][i] = X[v][t[i]]
            Ytmp[v][i] = Y[v][t[i]]
    X = Xtmp
    Y = Ytmp

    # complete data index
    index0 = np.linspace(0, (1 - missrate) * size - 1, num=int((1 - missrate) * size), dtype=int)
    missindex = np.ones((int(missrate * size), view_num))
    # print(missindex.shape)
    # incomplete data index
    index = []
    for i in range(missindex.shape[0]):
        missdata = np.random.randint(0, high=view_num, size=view_num - 1)
        # print(missdata)
        missindex[i, missdata] = 0
    # print(missindex)
    for i in range(view_num):
        index.append([])
    miss_begain = (1 - missrate) * size
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index[j].append(int(miss_begain + i))
    # print(index)
    maxmissview = 0
    for j in range(view_num):
        if maxmissview < len(index[j]):
            # print(len(index[j]))
            maxmissview = len(index[j])
    # to form complete and incomplete views' data
    for j in range(view_num):
        index[j] = list(index0) + index[j]
        X[j] = X[j][index[j]]
        print(X[j].shape)
        Y[j] = Y[j][index[j]]
        print(Y[j].shape)
    print("----------------generate incomplete multi-view data-----------------------")
    return X, Y, index



def Form_Incomplete_Data_Block(missrate=0.5, X = [], Y = []):
    np.random.seed(0)
    size = len(Y[0])
    view_num = len(X)
    t = np.linspace(0, size - 1, size, dtype=int)
    import random
    random.shuffle(t)
    Xtmp = []
    Ytmp = []
    for i in range(view_num):
        xtmp = np.copy(X[i])
        Xtmp.append(xtmp)
        ytmp = np.copy(Y[i])
        Ytmp.append(ytmp)
    for v in range(view_num):
        for i in range(size):
            Xtmp[v][i] = X[v][t[i]]
            Ytmp[v][i] = Y[v][t[i]]
    X = Xtmp
    Y = Ytmp

    # complete data index
    index0 = np.linspace(0, (1 - missrate) * size - 1, num=int((1 - missrate) * size), dtype=int)
    missindex = np.ones((int(missrate * size), view_num))
    # print(missindex.shape)
    # incomplete data index
    index = []
    missdata = np.random.randint(0, high=view_num, size=view_num - 1)
    block = np.random.randint(10)
    for i in range(missindex.shape[0]):
        if block == 0:
            missdata = np.random.randint(0, high=view_num, size=view_num - 1)
            block = np.random.randint(10)
        else:
            block = block-1
        # print(missdata)
        missindex[i, missdata] = 0
    # print(missindex)
    for i in range(view_num):
        index.append([])
    miss_begain = (1 - missrate) * size
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index[j].append(int(miss_begain + i))
    maxmissview = 0
    for j in range(view_num):
        if maxmissview < len(index[j]):
            # print(len(index[j]))
            maxmissview = len(index[j])
    # to form complete and incomplete views' data
    for j in range(view_num):
        index[j] = list(index0) + index[j]
        X[j] = X[j][index[j]]
        print(X[j].shape)
        Y[j] = Y[j][index[j]]
        print(Y[j].shape)
    print("----------------generate incomplete multi-view data-----------------------")
    return X, Y, index



def generate_non_iid_missing_data(X, Y, missrate, alpha):
    size = len(Y[0])
    view_num = len(X)

    # complete data index
    index0 = np.linspace(0, (1 - missrate) * size - 1, num=int((1 - missrate) * size), dtype=int)
    missindex = np.ones((int(missrate * size), view_num))
    print(missindex.shape)

    # incomplete data index
    index = []
    for i in range(view_num):
        index.append([])

    miss_begain = (1 - missrate) * size

    dirichlet_dist = np.random.dirichlet(np.repeat(alpha, view_num), size=1)[0]
    dirichlet_dist /= dirichlet_dist.sum()
    missing_categories = np.where(dirichlet_dist >= np.sort(dirichlet_dist)[-2])[0]
    for i in range(missindex.shape[0]):
        missdata = np.random.randint(0, high=view_num, size=view_num-1)
        flag  = 0
        for j in range(view_num):
            rnd = np.random.uniform(0, 1, 1)
            if flag == view_num-1:
                index[j].append(int(miss_begain + i))
            else:
                if dirichlet_dist[j] > rnd:
                    missindex[i, j] = 0
                    flag += 1
                else:
                    if j in missdata:
                        missindex[i, j] = 0
                        flag += 1
                    else:
                        index[j].append(int(miss_begain + i))


    maxmissview = 0
    for j in range(view_num):
        if maxmissview < len(index[j]):
            print(len(index[j]))
            maxmissview = len(index[j])
    print(maxmissview)
    # to form complete and incomplete views' data
    for j in range(view_num):
        index[j] = list(index0) + index[j]
        X[j] = X[j][index[j]]
        Y[j] = Y[j][index[j]]
    print("----------------generate incomplete multi-view data-----------------------")
    return X, Y, index

def load_data_conv(dataset):
    print("load:", dataset)
    if dataset == 'Scene':
        return Scene_15()
    else:
        raise ValueError('Not defined for loading %s' % dataset)

######### Client Dataset class #########
class clientDataset(Dataset):
    def __init__(self, dataset, y, missing):
        self.dataset = dataset
        self.labels = y
        self.missing = missing

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_tensor = torch.from_numpy(self.dataset[index].astype(float))
        label = self.labels[index]
        label_tensor = torch.tensor(label)
        index_tensor = torch.tensor(self.missing[index])

        return data_tensor, label_tensor, index_tensor

######### Server Dataset class #########
class serverDataset(Dataset):

    def __init__(self, dataset, y):
        self.viewNumber = len(dataset)
        self.data=[]
        for viewIndex in range(self.viewNumber):
            self.data.append(dataset[viewIndex])
        self.labels = y


    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)

######### Dataloaders #########
def load_data(config_dict, client_id=-1, n_clients=50, alpha=1e0, bsize=2500):
    dataset_name = config_dict["dataset"]
    data_dir = config_dict['data_dir']

    x, y = load_data_conv(dataset_name)

    # Dataloaders for given client
    if (client_id > -1):
        with open(
                f'{data_dir}/{n_clients}/{alpha}/{dataset_name}/train/' + dataset_name + "_" + str(client_id) + ".pkl",
                "rb") as f:
            train_ids = pkl.load(f)

        trainloader = DataLoader(clientDataset(train_ids[0], train_ids[1], train_ids[2]), batch_size=bsize
                                 , drop_last=False, shuffle=False)

    else:  # client_id == -1 implies server
        trainloader = DataLoader(serverDataset(x, y[0]), batch_size=bsize,  drop_last=False,shuffle=False)
        # Sanity check
        print("\nTrain set size: {}".format(len(trainloader.dataset)))

    return trainloader