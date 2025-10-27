import numpy as np
import h5py
import os
import torch

from tqdm.auto import tqdm

import sys
import gc

if __name__ == "__main__":
    
    idx_ = int(sys.argv[1])
    device = int(sys.argv[2])
    
    file_feature=h5py.File("../../VICReg/space/feature_list.h5","r")
    feature_list=file_feature["data"]
    
    batch_size = 2**18
    device = f"cuda:{device}"
    
    i = 0
    j = 0

    if feature_list.shape[0]%batch_size !=0:
        No_iters = feature_list.shape[0]//batch_size + 1
    else:
        No_iters = feature_list.shape[0]//batch_size
        
    feature_list = torch.tensor(feature_list[:],device = device,dtype = torch.float32) # Using Lower precision
    
    classes = [
    "n04296562","n03461385","n02804610","n02091134","n04311174",
    "n03930313","n03188531","n03710637","n03782006","n02123159",
    "n03970156","n02088364","n01558993","n02687172","n01443537",
    "n04254680","n04252077","n07615774","n03028079","n02727426",
    "n02342885","n07718747","n03658185","n03467068","n04118538",
    "n15075141","n04347754","n02013706","n01855032","n04493381",
    "n04037443","n02106030","n03445924","n03877472","n02138441",
    "n02930766","n03759954","n04579432","n03950228","n04201297",
    "n03871628","n02086079","n03344393","n03891332","n02280649",
    "n02395406","n04251144","n04008634","n01860187","n02971356"
    ]
    
    label_list = np.load("../../VICReg/space/label_list.npy")
    
    c = classes[idx_]
    class_idx = np.where(label_list == c)[0].tolist()
    
    dist = np.zeros((len(class_idx),feature_list.shape[0]),dtype=np.float32)
    order = class_idx.copy()
    
    
    for idx in tqdm(range(len(class_idx))):
        for j in range(No_iters):
            batchi = feature_list[[idx],:]
            batchj = feature_list[batch_size*j:batch_size*(j+1),:]
            batchi = batchi[:,None,:].repeat(1,batchj.shape[0],1)
            batchj = batchj[None,:,:].repeat(batchi.shape[0],1,1)
            mse = torch.square(batchi-batchj).mean(-1)
            dist[idx,batch_size*j:batch_size*(j+1)] = mse[0,:].cpu().numpy()

    np.save(f"../../VICReg/space/lookup/dist_{c}.npy",dist)
    np.save(f"../../VICReg/space/lookup/order_{c}.npy",order)
    
    gc.collect()