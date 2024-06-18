import os
from functools import wraps
from collections import defaultdict
from tqdm import tqdm

import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
import copy
import random
import time
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from argparse import ArgumentParser
from torchvision import transforms as tt
from PIL import Image
from utils import AverageMeter

def save_checkpoint(state, checkpoint, filename= 'checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    os.makedirs(checkpoint, exist_ok=True)
    torch.save(state, filepath)
    # print(f'global epoch {state["glepoch"]} saved')

def next_data_batch(train_iter, train_loader):
    try:
        img1, img2, labels = next(train_iter)
        
    except StopIteration:
        train_iter = iter(train_loader)
        img1, img2, labels= next(train_iter)
        
    return img1, img2, labels

def optimizer_zero_grads(optimizer_server, optimizer_clients): 
    optimizer_server.zero_grad()
    
    for i in range(len(optimizer_clients)):
        optimizer_clients[i].zero_grad()


def train_server(online_proj_one, online_proj_two, target_proj_one, target_proj_two, server_model):
    
    # print("shape", online_proj_one.shape)
    # print("online_proj_one", online_proj_one)
    
    
    online_proj_one.requires_grad = True
    online_proj_two.requires_grad = True
    
    online_proj_one.retain_grad()
    online_proj_two.retain_grad()

    server_model.train()
    
    # forward prop
    loss = server_model(online_proj_one, online_proj_two, target_proj_one, target_proj_two)
    
    if online_proj_one.grad is not None:
        online_proj_one.grad.zero_()
        
    if online_proj_two.grad is not None:
        online_proj_two.grad.zero_()
            
    # backward prop
    loss.backward()
    online_proj_one_grad, online_proj_two_grad = online_proj_one.grad.detach().clone(), online_proj_two.grad.detach().clone()
    # print("online_proj_one_grad", online_proj_one_grad.shape)
    # print("online_proj_two_grad", online_proj_two_grad.shape)
    
    return online_proj_one_grad, online_proj_two_grad, loss
'''
def training(global_model, client_models, server_model, optimizer_server, optimizer_clients, rounds, batch_size, lr, avg_freq, training_loader, epoch, save_path, client_num):
   
    # training loss
    train_loss = []
    test_loss = []
    test_accuracy = []
    best_accuracy = 0
    avg_times = 0
    # measure time
    start = time.time()
    client_weights = [1/5 for i in range(client_num)]
    
    writer = SummaryWriter(f'./logs/SplitFSSL_BYOL32_DifAvgtimes/resnet18Maxpooling_cifar10_{batch_size}_{avg_freq}_{client_num}')
    global_step = 0
    for curr_round in range(epoch, rounds):
        metrics = defaultdict(list)
        print(f"Global Round:", curr_round)
        w, local_loss = [], []
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
       
        # client forward
        # select 5 client to join training
        s_clients = []
        s_clients = random.sample(range(client_num), 5)
        for i, client_id in enumerate(s_clients):
            num_batch = len(training_loader[client_id])
            train_iter = iter(training_loader[client_id])
            p_bar = tqdm(range(num_batch))
            for batch in range(num_batch):
                optimizer_zero_grads(optimizer_server, optimizer_clients)
                # Compute a local update
                img1, img2 = next_data_batch(train_iter)
                
                img1 = img1.cuda()
                img2 = img2.cuda()
                
                data_time.update(time.time() - start)
                
                # pass to client model
                client_models[client_id].train()
                online_proj_one, online_proj_two, target_proj_one, target_proj_two = client_models[client_id](img1, img2)
                
               
                online_proj_one_grad, online_proj_two_grad, loss = train_server(online_proj_one.detach(), online_proj_two.detach(), target_proj_one, target_proj_two, server_model)
                
                optimizer_server.step()
              
                online_proj_one.backward(online_proj_one_grad)
                online_proj_two.backward(online_proj_two_grad)
                optimizer_clients[client_id].step()
                
                # calculate EMA after every batch
                client_models[client_id].update_moving_average()

                batch_time.update(time.time() - start)
                start = time.time()
                #=======================================set p_bar description=======================================================
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss {loss:.4f}.".format(
                        epoch=curr_round,
                        epochs=rounds+1,
                        batch=batch + 1,
                        iter=num_batch,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=loss.item()))
                p_bar.update()
                #=======================================set p_bar description=======================================================
            p_bar.close()
            local_loss.append((loss.item()))
            metrics["Loss/train"].append(loss.item())
        
        global_step += 1
        
        if (curr_round+1) == rounds or ((curr_round+1) % avg_freq == 0):
            print("aggregate")
            with torch.no_grad():
                # aggregate client models
                for key in global_model.state_dict().keys():
                    # num_batches_tracked is a non trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    if "running" in key or "num_batches" in key:
                        continue
                    # elif 'target' in key:
                    #     continue
                    else:
                        temp = torch.zeros_like(global_model.state_dict()[key]).to('cuda')
                        for client_idx in s_clients:
                            temp += client_weights[client_idx] * client_models[client_idx].state_dict()[key]                        
                        global_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(len(client_models)):
                            client_models[client_idx].state_dict()[key].data.copy_(global_model.state_dict()[key])
        
        # scheduler_server.step()
        for k, v in metrics.items():
            writer.add_scalar(k, np.array(v).mean(), curr_round)


        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        
        if curr_round % 5 == 0:
            optimizer_dict = []
            optimizer_dict.append(optimizer_server.state_dict())
            for client_idx in range(client_num):
                optimizer_dict.append(optimizer_clients[client_idx].state_dict())
            state_dict = global_model.online_encoder.cpu().state_dict()
            save_checkpoint({
                'glepoch': curr_round+1,
                'state_dict': state_dict,
                'optimizer': optimizer_dict,
            }, save_path)
        if curr_round % 100 == 0 and curr_round != 0:
            torch.save(global_model.online_encoder.cpu().state_dict(), save_path + f"_{curr_round}_epoch.pt")
        
        
        print(f"Global round: {curr_round} | Average loss: {loss_avg}")
        # print('best_accuracy:', best_accuracy, '---Round:', curr_round, '---lr', lr, '----localEpocs--', E)

    end = time.time()
   
    print("Training Done!")
    print("Total time taken to Train: {}".format(end - start))
    print(f"Total average times : {avg_times}")

    return global_model, client_models, train_loss
'''