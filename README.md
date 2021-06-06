# Source-inference-FL
This repository contains the source code for evaluating source inference attack in federated learning (FL). 

# Requirement
* torch==1.8.1
* numpy==1.18.1
* torchvision=0.9.1

# Key factors
We use the FedAvg algorithm (https://arxiv.org/abs/1602.05629) which is the first and perhaps the most widely used FL algorithm to train the FL models. There are several factors in FedAvg influencing the source inference attack performance.

* Data distribution (α): FL often assumes the training data across different parties are non-iid. We use a Dirichlet distribution to divide non-iid training data to each local party. The level of non-iid is controlled by a hyperparameter α of the Dirichlet distribution. The smaller α, the higher level of non-iid.
* Number of parties (K): FL allows multiple parties to train the joint model collaboratively. Intuitively, it is more difficult to identify where a training example comes from when the number of parties increases.
* Number of local epochs (E): In each training round, the local party will train the local model for several epochs and then uploads the updated model to the central server. The more local epochs performed by the local party, the local model better remembers its local training data.

# Implementation
You can run the following code to implement the source inference attacks. The datasets provided in this rep are `Synthetic` and `MNIST` datasets. You can try different `--alpha` (data distribution), `--number_users`(number of parties), `--local_ep` (number of local epochs) to see how the attack performance changes. For `Synthetic` dataset, we use `--model=mlp`. For `MNIST` dataset, we use `--model=cnn`.
```python
python main_fed.py --dataset=Synthetic --model=mlp --alpha=1 --number_users=10 --local_ep=5
```
