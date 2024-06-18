import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from utils.training import *

def cos_similarity(A1, A2):
    client_sim = [0 for i in range(len(A1))]
    for i in range(len(A1)):
        A1_norm = torch.norm(A1[i].type(torch.cuda.FloatTensor), 'fro')
        A2_norm = torch.norm(A2[i].type(torch.cuda.FloatTensor), 'fro')
        A1_A2 = A1_norm * A2_norm
        client_sim[i] = ((A1[i]*A2[i]).sum() / A1_A2).item()

    return client_sim

def KLdivergence(P, Q, reduction="batchmean", device = "cuda"):
    '''
    P: target，目標分佈P
    Q: input，待度量分佈Q
    '''
    return F.kl_div(Q.view(-1).softmax(-1).log(), P.view(-1).softmax(-1), reduction=reduction).detach()

def similarity_mat(save_path, epoch, matidx, clients_idxs, clients, shared_data_loader, nclasses=10, nsamples=2500):
    #clients_idxs = np.arange(10)
    
    nclients = len(clients_idxs)
    #nclasses = 10
    #nsamples = 2500
    
    clients_correct_pred_per_label = {idx: {i: 0 for i in range(nclasses)} for idx in clients_idxs}
    clients_pred_per_label = {idx: [] for idx in clients_idxs}
    
    with torch.no_grad():
        for batch_idx, (data, _, target) in enumerate(shared_data_loader):
            data, target = data.cuda(), target.cuda()
            for idx in clients_idxs: 
                

                # net = copy.deepcopy(clients[idx]) for supervised
                net = copy.deepcopy(nn.Sequential(*list(clients[idx].online_encoder.children())[:-1]))
                net.cuda()
                net.eval()
                # clients[idx].eval()

                output = net(data)
                pred = output.data
                
    
                clients_pred_per_label[idx].append(pred)


    A = {idx: torch.stack(clients_pred_per_label[idx]).view(nsamples, nclasses) for idx in clients_idxs}
    clients_similarity = {idx: [] for idx in clients_idxs}
    # clusters = []

    # 計算cosine similarity
    for idx1 in clients_idxs:
        for idx2 in clients_idxs:
            A1_norm = torch.norm(A[idx1].type(torch.cuda.FloatTensor), 'fro')
            A2_norm = torch.norm(A[idx2].type(torch.cuda.FloatTensor), 'fro')
            A1_A2 = A1_norm * A2_norm
            if A1_A2 == 0:
                print("There's a zero")
            sim = ((A[idx1]*A[idx2]).sum() / A1_A2).item()
            clients_similarity[idx1].append(sim)
    # 計算KL相似度
    # for idx1 in clients_idxs:
    #     for idx2 in clients_idxs:
    #         sim = KLdivergence(A[idx1],A[idx2])
    #         clients_similarity[idx1].append(sim.cpu().numpy())
            
    mat_sim = np.zeros([nclients,nclients])
    for i in range(nclients):
        mat_sim[i, :] = np.array(clients_similarity[clients_idxs[i]])
        # breakpoint()
        # mat_sim[i, :] = clients_similarity[clients_idxs[i]]

    sns.heatmap(mat_sim,square=True,annot=True,linecolor='white',cmap='Blues',vmin=0.5)
    # sns.heatmap(mat_sim,square=True,annot=True,linecolor='white',cmap='Blues_r',norm=LogNorm())
    if not os.path.exists(save_path + f"/similarity_mat/{epoch}_epoch"):
        os.makedirs(save_path + f"/similarity_mat/{epoch}_epoch")
    plt.savefig(save_path + f"/similarity_mat/{epoch}_epoch/heatmap_{matidx}.png") 
    plt.clf()

    
    return clients_similarity, mat_sim, A