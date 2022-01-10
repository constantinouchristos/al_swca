from scipy import stats
import pandas as pd
import torch
import numpy as np


ALL_AQUISITIONS = ['max_entropy','random','bald','var_ratios','mean_std']

def aquisition_function(logits=None,n_query=10,type_='max_entropy'):
    
    
    if type_ == "max_entropy":
        
        logits_stack = np.stack(logits)
        all_logits_tens = torch.tensor(logits_stack)
        softed = torch.softmax(all_logits_tens,dim=2)
        
        pc = softed.mean(axis=0).numpy()
        
        acquisition = (-pc * np.log(pc + 1e-10)).sum(axis=-1)  # To avoid division with zero, add 1e-10
        
        idx = (-acquisition).argsort()[:n_query]
        
        return idx
    
    elif type_ == 'random':
        
        logits_stack = np.stack(logits)
        all_idxs = list(range(logits_stack.shape[1]))
        
        random_choice = np.random.choice(all_idxs,replace=False,size=n_query)
        
        return random_choice
    
    elif type_ == 'bald':
        
        logits_stack = np.stack(logits)
        all_logits_tens = torch.tensor(logits_stack)
        
        softed = torch.softmax(all_logits_tens,dim=2)
        pc = softed.mean(axis=0).numpy()
        
        H = (-pc * np.log(pc + 1e-10)).sum(axis=-1)
        
        out_numpy = softed.numpy()
        
        E_H = -np.mean(np.sum(out_numpy * np.log(out_numpy + 1e-10), axis=-1), axis=0)
        
        acquisition = H - E_H
        
        idx = (-acquisition).argsort()[:n_query]
        
        return idx
    
    elif type_ == 'var_ratios':
        
        logits_stack = np.stack(logits)
        all_logits_tens = torch.tensor(logits_stack)
        softed = torch.softmax(all_logits_tens,dim=2)
        
        softed_nump = softed.numpy()
        
        preds = np.argmax(softed_nump, axis=2)
        
        _, count = stats.mode(preds, axis=0)
        
        acquisition = (1 - count / preds.shape[1]).reshape((-1,))
        
        idx = (-acquisition).argsort()[:n_query]
        
        return idx
    
    elif type_ == 'mean_std':
        
        logits_stack = np.stack(logits)
        all_logits_tens = torch.tensor(logits_stack)
        softed = torch.softmax(all_logits_tens,dim=2)
        
        softed_nump = softed.numpy()
        
        sigma_c = np.std(softed_nump, axis=0)
        acquisition = np.mean(sigma_c, axis=-1)
        
        idx = (-acquisition).argsort()[:n_query]
        
        return idx
    
    else:
        
        raise NotImplementedError
        
        
        