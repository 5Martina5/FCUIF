######### Shared config #########
config_dict = {

    #### Data ####
    'dataset': 'Scene', # choices: "Scene","REU","HW",Fashion,MNIST_USPS
    'view': 3,
    'instance_num': 4484,    # 4484, 1200, 2000, 6772, 10000
    'n_clusters': 15,    # 15,6,10,10
    'seed': 1, # set random seed
    'num_clients': 3, # choices: Int
    'alpha': 1e2,  # choices= 1e-2, 1e0, 1e2
    'data_dir': './data/', # where data is located
    'save_dir': '.', # where model directory is to be created
    'force_restart_training': False, # set False if you want to restart federated training from the last global server model
    'force_restart_hparam': True,
    'n_pretrain' : [True] * 3, # Solve the first round of training for each client without using the global p
    'n_initial' : [True] * 3, # Solve client initialization with global p
    'AR' : 0.95, # alignment rate
    'miss_rate': 0.5, # overlapping rate
    'align_rate': 0.5, # alignment rate
    'anchor': 0,
    'client_acc': [0.0] * 3,

    #### GPU and virtualization ####
    'main_device': 'cuda:0', # choices: 0--7
    'CUDA_VISIBLE_DEVICES': '0,1', # choices: 0--7
    'virtualize': True, # choices: True, False
    'client_vram': 3000, # vram for client in MB

    #### Training config ####
    # Train/eval technique
    'train_mode': 'my',
    'da_method': 'my',
    'div_aware_update': False,
    'stateful_client': False,

    # Fit fractions
    'fraction_fit': 1 , # choices: Float <= 1
    'fraction_eval': 1.,


    # Hyperparameters
    'num_rounds': 1, # number of communication rounds
    'local_bsize': 256, #
    'local_epochs': 300, # number of local epochs for client training caltech-200 bdgp-30
    'local_lr': 0.001, # Learning rate for client training
    'server_lr': 0.001,
}



def get_config_dict():
    return config_dict



