import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from .datasets import CIFAR10Pair, CIFAR100Pair, FEMNISTPair
from torch.utils.data import DataLoader



def load_cifar10_data(datadir):

    
    transform = transforms.Compose([transforms.ToTensor()])
    
    cifar10_train_ds = CIFAR10Pair(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10Pair(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()
    
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):

    
    transform = transforms.Compose([transforms.ToTensor()])
    
    cifar100_train_ds = CIFAR100Pair(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100Pair(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()
    
    return (X_train, y_train, X_test, y_test)

def load_femnist_data(datadir):
    
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNISTPair(datadir, train=True, transform=transform, download=False)
    mnist_test_ds = FEMNISTPair(datadir, train=False, transform=transform, download=False)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, dataidxs_test=None):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'):
        
        if dataset == 'cifar10':
            dl_obj = CIFAR10Pair
            
            s = 1  
            color_jitter = transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
            )
            transform_train = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        transforms.RandomApply([color_jitter], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]
                )
            transform_test = transforms.Compose([
                transforms.ToTensor(), 
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100Pair
            
            s = 1  
            color_jitter = transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
            )
            transform_train = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        transforms.RandomApply([color_jitter], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]
                )
            transform_test = transforms.Compose([
                transforms.ToTensor(), 
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif dataset == 'femnist':
            dl_obj = FEMNISTPair
            
            
            transform_train = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(28),
                        transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        transforms.ToTensor(),
                    ]
                )
            transform_test = transforms.Compose([
                transforms.ToTensor(), 
            ])
        
        else :
            train_ds = None
            test_ds = None
            train_dl = None
            test_dl = None

            return train_dl, test_dl, train_ds, test_ds
            
            

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
        test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
        
    return train_dl, test_dl, train_ds, test_ds

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts


def shared_data(dataset, datadir, nclasses, nsamples_shared):
    
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    if dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
    
    idxs_test = np.arange(len(X_test))
    labels_test = np.array(y_test)
    # Sort Labels Train 
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    labels_test = idxs_labels_test[1, :]

    # breakpoint()
    
    idxs_test_shared = []
    N = nsamples_shared//nclasses
    ind = 0
    for k in range(nclasses): 
        ind = max(np.where(labels_test==k)[0])
        idxs_test_shared.extend(idxs_test[(ind - N):(ind)])
    
    test_targets = y_test
    for i in range(nclasses):
        print(f'Shared data has label: {i}, {len(np.where(test_targets[idxs_test_shared[i*N:(i+1)*N]]==i)[0])} samples')

    return idxs_test_shared

def partition_data(dataset, datadir, partition, client_num, beta=10, test_data = True):
    
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    if dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)

    elif partition == "default" and dataset == "femnist":
            num_user = u_train.shape[0]
            user = np.zeros(num_user+1,dtype=np.int32)
            for i in range(1,num_user+1):
                user[i] = user[i-1] + u_train[i-1]
            no = np.random.permutation(num_user)
            batch_idxs = np.array_split(no, client_num)
            net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(client_num)}
            for i in range(client_num):
                for j in batch_idxs[i]:
                    net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))
    
    if partition == "noniid-label#2" and dataset == 'cifar10':
        labels_test = y_train
        idxs_test = np.arange(len(y_train))
        idxs_labels_train = np.vstack((idxs_test, labels_test))
        idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
        idxs_train = idxs_labels_train[0, :]
        labels_train = idxs_labels_train[1, :]
        
        net_dataidx_map = {i: [] for i in range(client_num)}
        sid = 0
        eid = 0
        cid = 0
        for k in range(10):
            if k in (0,1):
                cid = 0 
            elif k in (8,9):
                cid = 1
            elif k in (2,3):
                cid = 2
            elif k in (4,5):
                cid = 3
            else :
                cid = 4
            num = len(np.where(labels_train == k)[0])
            sid = eid
            eid += num
            net_dataidx_map[cid].extend(idxs_train[sid:eid])

    if partition == "noniid-label#3" and client_num == 6:
        labels_test = y_train
        idxs_test = np.arange(len(y_train))
        idxs_labels_train = np.vstack((idxs_test, labels_test))
        idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
        idxs_train = idxs_labels_train[0, :]
        labels_train = idxs_labels_train[1, :]
        
        net_dataidx_map = {i: [] for i in range(6)}
        sid = 0
        eid = 0
        for k in range(10):
            num = len(np.where(labels_train == k)[0])
            if k in (2,3,4):
                per = (int)(num/2)
                sid = eid
                eid += per
                net_dataidx_map[0].extend(idxs_train[sid:eid])
                sid = eid
                eid += per
                net_dataidx_map[1].extend(idxs_train[sid:eid])
            if k in (5,6,7):
                per = (int)(num/2)
                sid = eid
                eid += per
                net_dataidx_map[2].extend(idxs_train[sid:eid])
                sid = eid
                eid += per
                net_dataidx_map[3].extend(idxs_train[sid:eid])
            if k in (0,1,8,9):
                per = (int)(num/2)
                sid = eid
                eid += per
                net_dataidx_map[4].extend(idxs_train[sid:eid])
                sid = eid
                eid += per
                net_dataidx_map[5].extend(idxs_train[sid:eid])
    
    if partition == "iid":
        n_train = y_train.shape[0]
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, client_num)
        net_dataidx_map = {i: batch_idxs[i] for i in range(client_num)}
    
    elif partition == "noniid-labeldir":

        min_size = 0
        min_require_size = 10
        class_num = 10
        N = y_train.shape[0]
        
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(client_num)]
            for k in range(class_num):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, client_num))
                proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if class_num == 2 and client_num <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(client_num):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    
    print(f'partition: {partition}')
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    print(f'Data statistics Train: {str(traindata_cls_counts)}')
        
    if test_data:
        net_dataidx_map_test = {i: [] for i in range(client_num)}
        for k_id, stat in traindata_cls_counts.items():
            labels = list(stat.keys())
            for l in labels:
                idx_k = np.where(y_test==l)[0]
                net_dataidx_map_test[k_id].extend(idx_k.tolist())

        testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test)
        print('Data statistics Test:\n %s \n' % str(testdata_cls_counts))
    else: 
        net_dataidx_map_test = None 
        testdata_cls_counts = None 

    return  net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts

def expavg_times(t, min_alpha=2, max_alpha=50, global_epochs=1000):
    a = min_alpha
    b = np.log(max_alpha / a) / global_epochs
    return np.round(a * np.exp(b * t)).astype(int)

def linear_growth(t, min_alpha=2, max_alpha=50, global_epochs=1000):
    a = min_alpha  # 初始值
    b = (max_alpha - min_alpha) / global_epochs  # 指数增长速率，确保函数在 t=1000 时达到50
    return np.round(a + t * b).astype(int)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def model_similarity(model1, model2):
    
    model1_params = model1.flatten()
    model2_params = model2.flatten()
    
    similarity = cosine_similarity(model1_params, model2_params)
    return similarity








