
import argparse
import config
import torch
import os
import pprint
import torch.backends.cudnn as cudnn
import server
import time

cudnn.deterministic = True
cudnn.benchmark = False

def update_configs(args, config_dict):
    config_dict_update = eval(args.config_dict)
    config_dict.update(config_dict_update)

    if(config_dict["train_mode"]=="my"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["seed"]}_seed'
    else:
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["seed"]}_seed'
    return config_dict


def get_parser():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--config_dict", type=str, default="{}")
    return parser


def run(config_dict):
    print("========================")
    print("Run Configurations:")
    print("config_dict:")
    print(config_dict)
    print("========================")

    os.makedirs(f'{config_dict["save_dir"]}/configs', exist_ok=True)
    pp = pprint.PrettyPrinter()
    with open(f'{config_dict["save_dir"]}/configs/config_dict.txt', 'w') as f:
        f.write(pp.pformat(config_dict))

    # Federated training
    server.server_run(config_dict)

    # Save results
    if (config_dict["train_mode"] == "my"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["seed"]}_seed'
    else:
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["seed"]}_seed'
    eval_dest = f'{config_dict["save_dir"]}/saved_models/eval_{config_dict["dataset"]}_{config_dict["alpha"]}_alpha_' + model_name + '.txt'


if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    config_dict = config.get_config_dict()
    config_dict = update_configs(args, config_dict)
    torch.manual_seed(config_dict['seed'])
    run(config_dict)
    end_time = time.time()
    run_time = end_time - start_time

    print("time：", run_time, "s")