# Source-inference-FL
This repository contains the source code of the paper "Source Inference Attacks in Federated Learning" for evaluating source inference attacks in federated learning (FL). 

# Requirement 
* torch==1.8.1
* numpy==1.18.1
* torchvision==0.9.1
* sklearn==0.22.1

# Key factors
We use the FedAvg algorithm (https://arxiv.org/abs/1602.05629) which is the first and perhaps the most widely used FL algorithm to train the FL models. We investigate two factors in FedAvg influencing the source inference attack performance.

* Data distribution (α): FL often assumes the training data across different parties are non-iid. We use a Dirichlet distribution to divide non-iid training data to each local party. The level of non-iid is controlled by a hyperparameter α of the Dirichlet distribution. The smaller α, the higher level of non-iid.
* Number of local epochs (E): In each training round, the local party will train the local model for several epochs and then uploads the updated model to the central server. The more local epochs performed by the local party, the local model better remembers its local training data.

# Implementation
You can run the following code to implement the source inference attacks. The datasets provided in this rep are `Synthetic` and `MNIST` datasets. For the `MNIST` dataset, it will automatically be downloaded. For `Synthetic` dataset, please first run the following code to generate it:
```python
python generate_synthetic.py
```

You can try different `--alpha` (data distribution), `--num_users`(number of parties), `--local_ep` (number of local epochs) to see how the attack performance changes. For `Synthetic` dataset, we set `--model=mlp`. For `MNIST` dataset, we set `--model=cnn`.
```python
python main_fed.py --dataset=Synthetic --model=mlp --alpha=1 --num_users=10 --local_ep=5
```
