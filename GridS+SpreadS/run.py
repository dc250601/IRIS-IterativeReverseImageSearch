import wandb
import algo
import gc
import time

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

def runner(config = None):
    print("Inside runnner")
    gc.collect()
    wandb.init(config = config)
    config = wandb.config

    algo.algo_main(config)