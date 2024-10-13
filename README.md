## A Novel Federated Multi-View Clustering Method for Unaligned and Incomplete Data Fusion

Code for the paper "A Novel Federated Multi-View Clustering Method for Unaligned and Incomplete Data Fusion". (Information Fusion)
Code for the paper "Federated deep multi-view clustering with global self-supervision". (ACM MM 2023)

## Requirements

The code requires:

* Python 3.6 or higher

* Pytorch 1.9 or higher

We use the [Flower](https://flower.dev) federated learing framework for all client-server implementation. Flower and other dependencies can be installed via following command:

```setup
pip install -r requirements.txt
```

## Example execution 

First use the following command to setup the dataset of your choice (e.g., Scene) for any number of clients (e.g., 3):

```sampler
python sampler.py --dataset="Scene" --n_clients=3
python sampler.py --dataset="Scene" --n_clients=3 --missing=0.5
```

Then, to train a new model, run:

```execution
python main.py 
```

Further settings for the dataset, number of clients, overlapping rate, align_rate, and other parameters can be configured in config.py.

## Citation 
If you find our code useful, please cite:

```latex
@article{ren2024novel,
  title={A novel federated multi-view clustering method for unaligned and incomplete data fusion},
  author={Ren, Yazhou and Chen, Xinyue and Xu, Jie and Pu, Jingyu and Huang, Yonghao and Pu, Xiaorong and Zhu, Ce and Zhu, Xiaofeng and Hao, Zhifeng and He, Lifang},
  journal={Information Fusion},
  volume={108},
  pages={1-10},
  year={2024}
}

@inproceedings{chen2023federated,
  title={Federated Deep Multi-View Clustering with Global Self-Supervision},
  author={Chen, Xinyue and Xu, Jie and Ren, Yazhou and Pu, Xiaorong and Zhu, Ce and Zhu, Xiaofeng and Hao, Zhifeng and He, Lifang},
  booktitle={ACM MM},
  pages={3498--3506},
  year={2023}
}
```

Thanks. Any problem can contact Xinyue Chen (martinachen2580@gmail.com).
