import wandb
import sys
import os

def exclude_bias_and_norm(p): 
    return p.ndim == 1

if __name__ == "__main__":
    device = sys.argv[1]
    sweep_id = 'mq3gulvf'
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    wandb.login(<USER KEY>)
    print("Login successful")
    from run import runner 
    from run import *
    
    wandb.agent(sweep_id,runner,entity = <USER ID>, project="ELSA_no_ac_fixed")
    
    
