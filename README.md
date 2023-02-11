# USER: Unsupervised Structural Entropy-based Robust Graph Neural Network (AAAI-2023)

Yifei Wang, Yupan Wang, Zeyu Zhang, Song Yang, Kaiqi Zhao, Jiamou Liu

The University of Auckland, Auckland, New Zealand

{wany107, ywan980, zzha669, syan382}@aucklanduni.ac.nz, {kaiqi.zhao, jiamou.liu}@auckland.ac.nz
<!--#### -->

## Introduction
Unsupervised/self-supervised graph neural networks (GNN) are vulnerable to inherent randomness in the input graph data which greatly affects the performance of the model in downstream tasks. 

![image](images/Unsupervised_RGRL.png "The aim of unsupervised robust graph representation learning")

In this paper, we alleviate the interference of graph randomness and learn appropriate representations of nodes without label information. To this end, we propose USER, an unsupervised robust version of graph neural networks that is based on structural entropy. We analyze the property of intrinsic connectivity and define intrinsic connectivity graph. We also identify the rank of the adjacency matrix as a crucial factor in revealing a graph that provides the same embeddings as the intrinsic connectivity graph. We then introduce structural entropy in the objective function to capture such a graph. 
Extensive experiments conducted on clustering and link prediction tasks under random-noises and meta-attack over three datasets show USER outperforms benchmarks and is robust to heavier randomness.

![image](images/USER_framework.png "The proposed USER framework")

## Datasets

* [cora](https://github.com/kimiyoung/planetoid/tree/master/data)

* [LastFM](https://grouplens.org/datasets/)

* [Yelp2018](https://www.yelp.com/dataset/challenge)

* [MovieLens](https://grouplens.org/datasets/movielens/)

## Requirements

* python >= 3.9

* torch>=1.7.0

* dgl>=0.7.0

* scikit-learn>=0.24.0






### Command and configurations

#### on Amazon-book
```bash
python -u main.py --model_type baseline  --dataset amazon-book --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### on LastFM
```bash
python -u main.py --model_type baseline --dataset last-fm --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### on Yelp2018
```bash
python -u main.py --model_type baseline --dataset yelp2018 --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### on MovieLens
```bash
python -u main.py --model_type baseline --dataset movie-lens --gpu_id 0 --ue_lambda 0.4 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### General flags

```{txt}
optional arguments:
  --dataset                       dataset                               
  --idf_sampling                  negative entity number
  --layer_size                    size of each layer
  --embed_size                    dimension of embedding vector 
  --epoch                         max epochs before stop
  --pretrain                      use pretrain or not
  --batch_size                    batch size
```
