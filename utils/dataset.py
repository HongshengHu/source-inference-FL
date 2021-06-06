import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from utils.sampling import sample_dirichlet_train_data, synthetic_iid


def get_dataset(args):
    # mnist dataset: 10 classes, 60000 training examples, 10000 testing examples.
    # synthetic dataset: 10 classes, 100,000 examples.

    if args.dataset == 'MNIST':
        data_dir = './data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)

    elif args.dataset == 'Synthetic' and args.iid == True:
        data_dir = './data/synthetic/synthetic_x_0.npz'
        synt_0 = np.load(data_dir)
        X = synt_0['x'].astype(np.float64)
        Y = synt_0['y'].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))
        # sample iid data
        dict_party_user, dict_sample_user = synthetic_iid(train_dataset, args.num_users, args.num_samples)

    elif args.dataset == 'Synthetic' and args.iid == False:

        data_dir = './data/synthetic/synthetic_x_0.npz'
        synt_0 = np.load(data_dir)
        X = synt_0['x'].astype(np.float64)
        Y = synt_0['y'].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))
        # sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
                                                                        args.alpha)

    else:
        train_dataset = []
        test_dataset = []
        dict_party_user, dict_sample_user = {}, {}
        print('+' * 10 + 'Error: unrecognized dataset' + '+' * 10)
    return train_dataset, test_dataset, dict_party_user, dict_sample_user


def exp_details(args):
    print('\nExperimental details:')
    print(f'Model     : {args.model}')
    print(f'Optimizer : sgd')
    print(f'Learning rate: {args.lr}')
    print(f'Global Rounds: {args.epochs}\n')

    print('Federated parameters:')

    print('{} dataset, '.format(args.dataset) + f' has {args.num_classes} classes')
    if args.iid == False:
        print(f'Level of non-iid data distribution: \u03B1 = {args.alpha}')
    else:
        print('The training data are iid across parties')
    print(f'Number of users    : {args.num_users}')
    print(f'Local Batch size   : {args.local_bs}')
    print(f'Local Epochs       : {args.local_ep}\n')
    return
