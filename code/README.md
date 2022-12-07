#USER and baselines


##Requirements
```shell
python == 3.7
pytorch ==1.8
networkx == 2.5
deeprobust == 0.2.4
torch-geometric == 2.0.1
```


Dataset - 
we use public raw data from:
```shell
https://github.com/kimiyoung/planetoid/tree/master/data
```
##Node clustering

Prepare dataset
```shell
cd node_clustering
mkdir data/
```
Download raw data into it

Generate random noises
```shell
python random_attack.py  
```
Generate meta-attack result
```shell
python meta_attack.py  
```
Run USER
```shell
python attack_main.py
```

##Link prediction

Prepare dataset
```shell
cd link_prediction
mkdir data/
```
Download raw data into it

Generate random noises
```shell
python lp_random_attack.py  
```

Run USER
```shell
python lp_attack_main.py
```