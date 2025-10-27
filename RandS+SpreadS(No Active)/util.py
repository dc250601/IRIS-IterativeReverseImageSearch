import os
import shutil
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision import datasets
import PIL
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import einops
import wandb
import PIL
from sklearn.utils import shuffle
import gc
import random
import h5py

class vic_args:
  arch = "resnet50"
  mlp = "8192-8192-8192"

class bt_args:
  projector = "8192-8192-8192"


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class LinkImageDataset(Dataset):
    "Given a list of Image Links all the images will be loaded along with their associated labels"
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = pil_loader(img_path)
        label = (self.img_list[idx].split("/")[-1]).split("_")[0]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

def nearest_neighbours_begin(query_feature,no_neighbours,feature_list):
    dist = []
    for i in range(feature_list.shape[0]):
        dist.append(nn.functional.mse_loss(query_feature[0],
                                           feature_list[i,:])
                   )
    feature_space = np.array([ np.arange(feature_list.shape[0]),
                               np.array(dist)]).T
    df = pd.DataFrame(feature_space)
    df = df.sort_values(by = 1)
    indexes = df.iloc[:no_neighbours,0]
    indexes = indexes.tolist()
    indexes = list(map(lambda x: int(x), indexes))
    return indexes


def nearest_neighbours_fast(query_index,forbidden_index,no_neighbours,feature_list,device = "cuda:1",batch_size = 30000):
    dist = []
    query_index.sort() #HDF5 requirement
    query_list = torch.tensor(feature_list[query_index],device = device)
    batch_size = batch_size
    iters = int(feature_list.shape[0]/batch_size)+1
    for i in range(iters):
        batch = torch.tensor(feature_list[i*batch_size:(i+1)*batch_size,:],device = device)
        dist_batch = []
        for j in range(len(query_index)):
            query_feature_expanded = query_list[j,:].repeat(batch.shape[0],1)
            dist1 = torch.square((query_feature_expanded - batch))
            dist1 = torch.sum(dist1,axis=1)
            dist1 = dist1/batch.shape[-1]
            dist_batch.append(dist1.cpu().numpy())
        dist.append(dist_batch)
    dist = np.concatenate(dist,axis=-1)
    
    dist = dist.min(axis=0)
    feature_space = np.array([ np.arange(feature_list.shape[0]),
                               np.array(dist)]).T
    df = pd.DataFrame(feature_space)
    df = df.sort_values(by = 1)
        
    i = 0
    c = 0
    new_index = []
    while(c<no_neighbours):
        if int(df.iloc[i,0]) not in forbidden_index:
            c = c+1
            new_index.append(int(df.iloc[i,0]))
        i = i+1
    return new_index

def labeller(query_index,label_list,label = "n01534433"):
    recieved = len(query_index)
    check = label_list[query_index] == label
    query_index_ = np.where(check)
    index_list = np.array(query_index)[query_index_[0].tolist()].tolist()
    detected = len(index_list)
    p = (detected/recieved)*100
    return index_list, p

def get_forbidden_indices(index):
    forbidden = set([])
    for i in range (len(index)):
        kappa = set(index[i])
        if len(forbidden.intersection(kappa)) !=0:
            print("Warning, Forbidden index found",len(forbidden.intersection(kappa)))
        forbidden = forbidden.union(kappa)
    return list(forbidden) 


def head_train(head,model,steps,positive_list,negative_list, path_list,device):
    """
    Sometime the no of positive samples can be two low to train the head in that case training of the head is skipped. Less than 4 positive samples causes this step to be skipped
    """
    
    len_p = len(positive_list)
    len_q = len(negative_list)
    p_batch_size = 0
    if min(len_p,len_q)<4:
        print("Very less no of positive/negative samples skiping active learning")
        return 0
    elif min(len_p,len_q)>4 and min(len_p,len_q)<9:
        p_batch_size = 4
    elif min(len_p,len_q)>8 and min(len_p,len_q)<17:
        p_batch_size = 8
    else:
        p_batch_size = 16
        
    transform = transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor()
                           ])
    
    
    positive_dataset = LinkImageDataset(path_list[positive_list],
                           transform=transform)
    negative_dataset = LinkImageDataset(path_list[negative_list],
                                             transform=transform
                                            )
    
    positive_dataloader = torch.utils.data.DataLoader(positive_dataset,
                                                          batch_size=p_batch_size,
                                                          shuffle=True,
                                                          drop_last = True,
                                                          num_workers=8,
                                                          pin_memory = True,
                                                          persistent_workers = True

                            )
    negative_dataloader = torch.utils.data.DataLoader(negative_dataset,
                                                      batch_size=p_batch_size,
                                                      shuffle=True,
                                                      drop_last = True,
                                                      num_workers=8,
                                                      pin_memory = True,
                                                     persistent_workers = True)
    
    positive_dataloader_iter = iter(positive_dataloader)
    negative_dataloader_iter = iter(negative_dataloader)
    
    optimizer = torch.optim.AdamW(head.parameters(), lr = 0.0005, weight_decay = 0.05)
    criterion = nn.BCEWithLogitsLoss()
    
    for i in range(steps):
        head.train()
        model.eval()
        try:
            positive_batch = next(positive_dataloader_iter)[0]
        except StopIteration:
            positive_dataloader_iter = iter(positive_dataloader)
            positive_batch = next(positive_dataloader_iter)[0]

        try:
            negative_batch = next(negative_dataloader_iter)[0]
        except StopIteration:
            negative_dataloader_iter = iter(negative_dataloader)
            negative_batch = next(negative_dataloader_iter)[0]

        positive_label = torch.tensor([1]*positive_batch.shape[0])
        negative_label = torch.tensor([0]*positive_batch.shape[0])
        batch = torch.cat([positive_batch,negative_batch],axis = 0)
        label = torch.cat([positive_label,negative_label],axis = 0)
        batch,label = shuffle(batch,label)
        label = label.unsqueeze(1)
        batch = batch.to(device)
        label = label.to(device)

        for param in head.parameters():
            param.grad = None
        with torch.no_grad(): 
            outputs = model.backbone(batch)
        outputs = head(outputs)
        loss = criterion(outputs, label.float())
        loss.backward()
        optimizer.step()
    gc.collect()
    
def exclude_bias_and_norm(p):
    return p.ndim == 1
    
def setup(algo):
    print("In setup")
    
    if algo == "vicreg":
        args = vic_args()
        import vicreg
        from vicreg.main import exclude_bias_and_norm
        file_feature = h5py.File("../VICReg/space/feature_list.h5","r")
        feature_list = file_feature["data"]
        # feature_list = np.load("../VICReg/space/feature_list.npy")
        label_list = np.load("../VICReg/space/label_list.npy")
        path_list = np.load("../VICReg/space/path_list.npy")
        
        file_emb = h5py.File("../VICReg/space/embedding_list.h5","r")
        embedded_space = file_emb["data"]
        
        # embedded_space = np.load("../VICReg/space/embedding_list.npy")
        component_space = np.load("../VICReg/space/feature16.npy")
        model = vicreg.main.VICReg(args=args)
        checkpoint = torch.load("../VICReg/resnet50_fullckpt.pth")
        weights = {}
        for k in checkpoint["model"].keys():
            weights[k[7:]] = checkpoint["model"][k]
        model.load_state_dict(weights)
    
    elif algo == "bt":
        args = bt_args()
        import barlow_and_twin
        file_feature = h5py.File("../BAT/space/feature_list.h5","r")
        feature_list = file_feature["data"]
        # feature_list = np.load("../BAT/space/feature_list.npy")
        label_list = np.load("../BAT/space/label_list.npy")
        path_list = np.load("../BAT/space/path_list.npy")
        
        file_emb = h5py.File("../BAT/space/embedding_list.h5","r")
        embedded_space = file_emb["data"]
        
        # embedded_space = np.load("../BAT/space/embedding_list.npy")
        component_space = np.load("../BAT/space/feature16.npy")
        model = barlow_and_twin.main.BarlowTwins(args=args)
        checkpoint = torch.load("../BAT/checkpoint.pth")
        weights = {}
        for k in checkpoint["model"].keys():
            weights[k[7:]] = checkpoint["model"][k]
        model.load_state_dict(weights)
    else:
        print(f"{algo} is not supported yet")
    print("Out of setup")
        
    return feature_list, label_list, path_list,embedded_space,component_space, model

def sigmoid_to_softmax(sigmoid_outputs):
    # Convert sigmoid output to logits
    logits = np.log(sigmoid_outputs / (1 - sigmoid_outputs))
    
    # Create logits for the second class (negative of the original logits)
    neg_logits = -logits
    
    # Stack logits and neg_logits to form [n, 2] logits array
    logits_2d = np.hstack((neg_logits, logits))
    
    # Apply softmax to logits_2d
    exp_logits = np.exp(logits_2d)
    softmax_outputs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return softmax_outputs