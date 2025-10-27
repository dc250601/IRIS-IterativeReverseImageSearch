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



def ssald(config=None):

    # wandb.login(<USER KEY>)

    Total_No_epochs = 10 #Fixed but can be played around with
    steps_per_epoch = 100 #Fixed but can be played around with
    device_ = "cuda:0" #Fixed by setting visible device
    batch_size = 4096

    Class_id = config.class_id
    No_seeds = config.no_seeds
    
    
    Run_Name = f"{Class_id}_{No_seeds}"

    feature_list, label_list, path_list,embedded_space,component_space, model = util.setup(config.algo)
    
    model = model.to(device_)
    model.eval()

    class_id = Class_id
    label_list_indx = np.where(label_list == class_id)[0].tolist()
    random.shuffle(label_list_indx)

    query_indices = label_list_indx[:No_seeds] 


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

    # delta_samples = util.nearest_neighbours_fast(query_index=l_points,
    #                                              forbidden_index=[],
    #                                              no_neighbours=b ,
    #                                              feature_list=feature_list,
    #                                              device = device_
    #                                             )

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
    alpha = config.alpha1
    iterations = Total_No_epochs

    nearest_neighbour_list.append(k_samples)
    
    last_lambda = np.array(gamma_samples)[labels] #### This is where the NN is fixed


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
        for i in tqdm(range(iters)):
            feature = torch.tensor(embedded_space[i*batch_size:(i+1)*batch_size,:],
                                   device = device_)
            head.eval()
            with torch.no_grad():
                confidence.extend(torch.sigmoid(head(feature)).cpu().numpy()[:,0].tolist())
        #-------------------------------------------------------------------------------------------------

        m = np.array(confidence).mean()
        s = np.array(confidence).std()
        random_indices =np.array(list(set(np.where(np.array(confidence) > (m+alpha*s))[0]) - set(k_samples)))
        np.random.shuffle(random_indices)
        
        delta_samples = random_indices[0:b].tolist()

        # delta_samples = util.nearest_neighbours_fast(query_index=l_points,
        #                                              forbidden_index=lambda_list,
        #                                              no_neighbours=b ,
        #                                              feature_list=feature_list,
        #                                              device = device_
        #                                             )

        gamma_samples = k_samples + delta_samples
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

        # print("-------------------------------------------")
        # print(f"{epochs}th Pass completed")
        # print("No of samples labelled in this step",len(gamma_samples))
        # print("No of samples discovered in this step",
        #       len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])+
        #       len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]))
        # print("The no of negative samples",len(gamma_negative_list))
        # print("The no of positive samples",len(lambda_list))
        # print("No of k samples",len(k_samples))
        # print("No of positive k samples",len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0]))
        # print("No of delta samples",len(delta_samples))
        # print("No of positive delta samples",len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]))
        # print("No of samples labelled till now",no_labellings)
        # print("Total No of samples discovered",len(lambda_list))
        # print("-------------------------------------------")


        # wandb.log({"Epoch": epochs,
        #           "Samples labelled in the step": len(gamma_samples),
        #           "Samples discovered in this step": len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])+
        #            len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]),
        #           "No of negative samples": len(gamma_negative_list),
        #           "No of positive samples": len(lambda_list),
        #           "No of k samples":len(k_samples),
        #           "No of positive k samples":len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0]),
        #           "No of delta samples":len(delta_samples),
        #           "No of positive delta samples":len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]),
        #           "No of samples labelled till now":no_labellings,
        #           "Total No of samples discovered":len(lambda_list),
        #           "Sampling Efficiency":(len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])+
        #            len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0]))/(b+l),
        #           "Random Sampling Efficiency":(len(util.labeller(query_index=delta_samples,label_list=label_list,label = class_id)[0])/len(delta_samples)),
        #           "Nearest Neighbour Efficiency":(len(util.labeller(query_index=k_samples,label_list=label_list,label = class_id)[0])/len(k_samples)),
        #           "Cumulative Sampling Efficiency":(len(lambda_list)/no_labellings),
        #           "Discovery":(len(lambda_list)/total_samples)
        #           }
        #          )

    #----------------------------------------------------------------
    confidence = []
    #-----------------------------------------------------------------------------------------------
    iters = int(embedded_space.shape[0]/batch_size)+1
    for i in tqdm(range(iters)):
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