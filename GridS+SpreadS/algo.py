#Import Blocks
#---------------------------------------------------------
print("Python script started")
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
import util
from sklearn.utils import shuffle
from util import head_train
import gc
import random

import sys
sys.path.append("../VICReg/")
sys.path.append("../BAT/")
import util
import vicreg
from vicreg.main import *


def algo_main(config=None):
    
    # wandb.login(<USER KEY>)

    Total_No_epochs = 10 #Fixed but can be played around with
    steps_per_epoch = 100 #Fixed but can be played around with
    device_ = "cuda:0" #Fixed by setting visible device
    batch_size = 4096 # This depends from system to system

    Class_id = config.class_id
    No_seeds = config.no_seeds
    zeta = config.zeta

    Run_Name = f"{Class_id}_{No_seeds}"

    # feature_list, label_list, path_list,embedded_space, model = util.setup(config.algo)
    if config.algo !="vicreg":
        sys.exit(1)
        
    feature_list, label_list, path_list,embedded_space,component_space, model = util.setup(config.algo)

    order_array = np.load(f"../VICReg/space/lookup/order_{Class_id}.npy")
    distance_array = np.load(f"../VICReg/space/lookup/dist_{Class_id}.npy")

    model = model.to(device_)
    model.eval()
    print()



    class_id = Class_id
    label_list_indx = np.where(label_list == class_id)[0].tolist()
    random.shuffle(label_list_indx)

    query_indices = label_list_indx[:No_seeds] 

    # feature_list = torch.tensor(feature_list)


    total_samples = len(label_list[label_list == class_id])


    lambda_list = []
    nearest_neighbour_list = [] #-------
    head = nn.Sequential(nn.Linear(2048,1))
    head = head.to(device_)
    #----------------------------------------------------------------
    a = config.a
    k_samples = util.nearest_neighbours_fast(query_index=query_indices,
                                             forbidden_index= [],
                                             no_neighbours=a,
                                             feature_list=feature_list,
                                             device = device_)


    b = 50
    no_labellings = a+b
    random_indices = np.array(list(set(np.arange(0,len(feature_list)).tolist()) - set(k_samples)))
    np.random.shuffle(random_indices)
    delta_samples = random_indices[:b].tolist()


    gamma_samples = k_samples + delta_samples

    labelled_list,_ = util.labeller(query_index=gamma_samples,
                  label_list=label_list,
                  label = class_id,
                 )

    labels = list(map(lambda x: True if x in labelled_list else False,
                      gamma_samples))

    lambda_list.extend(np.array(gamma_samples)[labels])
    print("Debug:len(lambda_list)",len(lambda_list))
    gamma_negative_list = list(set(gamma_samples) - set(lambda_list))

    head_train(model = model,
               head = head,
               steps=steps_per_epoch,
               positive_list=lambda_list,
               negative_list=gamma_negative_list,
               path_list = path_list,
               device = device_
              )

    print("-------------------------------------------")
    print("Zero Pass completed")
    print("No of samples labelled",(a+b))
    print("No of samples discovered",(len(lambda_list)))
    print("-------------------------------------------")
    #----------------------------------------------------------------
    a = config.a
    l = 50
    # alpha = 0.5
    alpha = 1
    iterations = Total_No_epochs

    nearest_neighbour_list.append(k_samples)

    last_lambda = np.array(gamma_samples)[labels] #### This is where the NN is fixed

    NUMBER_GRID = 10
    NUMBER_COMP = 3
    component_space = component_space[:,:NUMBER_COMP]
    # Normalising the space
    ##########################################################################################
    ## Finding the IQR 
    percentile25 = np.percentile(component_space,25,axis = 0)
    percentile50 = np.percentile(component_space,50,axis = 0)
    percentile75 = np.percentile(component_space,75,axis = 0)
    IQR = percentile75 - percentile25

    thres = 1.5
    for dim in range(component_space.shape[-1]):
        component_space[:,dim] = np.clip(component_space[:,dim],
                                    a_min=(percentile25[dim]-thres*IQR[dim]),
                                    a_max=(percentile75[dim]+thres*IQR[dim]))
    ##########################################################################################


    space = int(100/NUMBER_GRID)
    PERCENTILE_ZONE = list(range(0,100+space,space))
    percentile = np.percentile(component_space,PERCENTILE_ZONE,axis = 0)
    number_of_borders,number_of_components = percentile.shape

    ##########################################################################################
    # Grid Calculation
    ##########################################################################################
    print("Starting grid calculation")
    eps = 1e-6
    grid = []

    if NUMBER_COMP == 3:
        for border_1 in range(number_of_borders-1):
            l1 = percentile[border_1,0]
            if border_1 == 0:
                l1 = l1 - eps
            h1 = percentile[border_1+1,0]        
            idx1 = (component_space[:,0] <= h1)*(component_space[:,0] > l1)
            for border_2 in range(number_of_borders-1):
                l2 = percentile[border_2,1]
                if border_2 == 0:
                    l2 = l2 - eps
                h2 = percentile[border_2+1,1]
                idx2 = (component_space[:,1] <= h2)*(component_space[:,1] > l2)
                for border_3 in range(number_of_borders-1):
                    l3 = percentile[border_3,2]
                    if border_3 == 0:
                        l3 = l3 - eps
                    h3 = percentile[border_3+1,2]
                    idx3 = (component_space[:,2] <= h3)*(component_space[:,2] > l3)

                    idx = idx1*idx2*idx3
                    idx = list(set(np.where(idx)[0].tolist()) - set(lambda_list))
                    grid.append(idx)
    print("Grid calculation completed")
    ##########################################################################################




    for epochs in range(iterations):

        b = len(lambda_list)
        k_samples = util.nearest_neighbours_fast(query_index = last_lambda, ### <--------------- NN fixed here 
                                        forbidden_index = lambda_list,
                                        no_neighbours = a,
                                        feature_list = feature_list,
                                        device = device_,
                                        batch_size = 30000)


        confidence = []
        nearest_neighbour_list.append(k_samples)

        #-----------------------------------------------------------------------------------------------
        iters = int(embedded_space.shape[0]/batch_size)+1
        for i in range(iters):
            feature = torch.tensor(embedded_space[i*batch_size:(i+1)*batch_size,:],
                                   device = device_)
            head.eval()
            with torch.no_grad():
                confidence.extend(torch.sigmoid(head(feature)).cpu().numpy()[:,0].tolist())
        #-------------------------------------------------------------------------------------------------

        ##########################################################################################
        #Delta Calculation
        ##########################################################################################
        # This will be done after every epoch
        grid_stat = []
        grid_most_confident = []

        confidence_np = np.array(confidence) ## The first metric
        confidence_metric = (confidence_np-confidence_np.min())/(confidence_np.max()-confidence_np.min())


        dist_array_idx_list = []
        for elem in lambda_list:
            dist_array_idx_list.append(np.where(order_array == elem)[0][0])
        distance_subset = distance_array[dist_array_idx_list]

        distance_np = np.min(distance_subset,0) ## The second metric
        distance_metric = (distance_np-distance_np.min())/(distance_np.max()-distance_np.min())

        master_metric = zeta*confidence_metric + (1-zeta)*distance_metric

        for i in range(len(grid)):
            grid[i] = list(set(grid[i]) - set(lambda_list)) # To remove any new discovery from the grid
            idx = grid[i]
            grid_confidence = master_metric[idx]
            grid_most_confident_ = idx[np.argmax(grid_confidence)]
            grid_most_confident.append(grid_most_confident_)
            grid_stat.append(len(idx)) # This wil be used for debugging

        assert np.array(grid_stat).sum()+len(lambda_list) == component_space.shape[0], "Grid contamination error"

        grid_most_confident_score = master_metric[grid_most_confident]
        top_idx = np.argpartition(grid_most_confident_score, -b)[-b:] # Chooses the Top b sample
        delta_samples = (np.array(grid_most_confident)[top_idx]).tolist()
        ##########################################################################################

        gamma_samples = list(set(k_samples).union(set(delta_samples)))
        labelled_list,_ = util.labeller(query_index=gamma_samples,
                      label_list=label_list,
                      label = class_id,
                     )
        labels = list(map(lambda x: True if x in labelled_list else False,
                          gamma_samples))
        lambda_list.extend(np.array(gamma_samples)[labels])
        gamma_negative_list = list(set(gamma_samples) - set(lambda_list))

        last_lambda = np.array(gamma_samples)[labels] #### This is where the NN is fixed



        head_train(model = model,
                   head = head,
                   steps=steps_per_epoch,
                   positive_list=lambda_list,
                   negative_list=gamma_negative_list,
                   path_list = path_list,
                   device = device_
                  )

        no_labellings = no_labellings + len(gamma_samples)

#         print("-------------------------------------------")
#         print(f"{epochs}th Pass completed")
#         print("No of samples labelled in this step",len(gamma_samples))
#         print("No of samples discovered in this step",
#               len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])+
#               len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]))
#         print("The no of negative samples",len(gamma_negative_list))
#         print("The no of positive samples",len(lambda_list))
#         print("No of k samples",len(k_samples))
#         print("No of positive k samples",len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0]))
#         print("No of delta samples",len(delta_samples))
#         print("No of positive delta samples",len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]))
#         print("No of samples labelled till now",no_labellings)
#         print("Total No of samples discovered",len(lambda_list))
#         print("-------------------------------------------")


#         wandb.log({"Epoch": epochs,
#                   "Samples labelled in the step": len(gamma_samples),
#                   "Samples discovered in this step": len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])+
#                    len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]),
#                   "No of negative samples": len(gamma_negative_list),
#                   "No of positive samples": len(lambda_list),
#                   "No of k samples":len(k_samples),
#                   "No of positive k samples":len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0]),
#                   "No of delta samples":len(delta_samples),
#                   "No of positive delta samples":len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]),
#                   "No of samples labelled till now":no_labellings,
#                   "Total No of samples discovered":len(lambda_list),
#                   "Sampling Efficiency":(len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])+
#                    len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]))/(b+l),
#                   "Random Sampling Efficiency":(len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0])/len(delta_samples)),
#                   "Nearest Neighbour Efficiency":(len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])/len(k_samples)),
#                   "Cumulative Sampling Efficiency":(len(lambda_list)/no_labellings),
#                   "Discovery":(len(lambda_list)/total_samples)
#                   }
#                  )

    #----------------------------------------------------------------
    confidence = []
    #-----------------------------------------------------------------------------------------------
    iters = int(embedded_space.shape[0]/batch_size)+1
    for i in range(iters):
        feature = torch.tensor(embedded_space[i*batch_size:(i+1)*batch_size,:],
                               device = device_)
        head.eval()
        with torch.no_grad():
            confidence.extend(torch.sigmoid(head(feature)).cpu().numpy()[:,0].tolist())
    #-------------------------------------------------------------------------------------------------
    alpha = config.alpha2
    m = np.array(confidence).mean()
    s = np.array(confidence).std()
    final = np.where(np.array(confidence) > m+2*alpha*s)[0]
    labelled_list,p = util.labeller(query_index=final,
                  label_list=label_list,
                  label = class_id
                 )

    lambda_list,p = util.labeller(query_index=lambda_list, #Labelling it but this time correctly
                  label_list=label_list,
                  label = class_id,
                 )

    final_discovery = len(list(set(labelled_list).union(set(lambda_list))))

    print("Percentage Discovery",(final_discovery/total_samples))
    print("Labelling effeciency final",final_discovery/(no_labellings+len(final)))

    wandb.log({"Final Percentage Discovery":(final_discovery/total_samples),
               "Final Labelling Efficiency":final_discovery/(no_labellings+len(final))})
    wandb.finish()
    gc.collect()