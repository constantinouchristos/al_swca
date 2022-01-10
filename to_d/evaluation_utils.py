import torch
import os
import re

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
from transformers import BertConfig,BertModel

# from bert_lstm_model import Bert_lstm,Bert_lstm_adv
# from swag_modeling import SWAG,SWAG_adversarial
# from bert_modeling_custom import BertModel_c
from tqdm.auto import tqdm


def evaluate_model(model=None,
                   dataset=None,
                   my_trainer=None,
                   swag=False
                  ):
    
    """
    Args:
        model                          (torch.nn.module)  : model to evaluate
        my_trainer                     (Trainer class)    : dataframe of filtered input features (from constraints)
        swag                           (bool)             : wether to use swag or normal model


    Returns:
        dict:                          (dictionary)       : Dictionary of evaluated metrics
        
    """
    
    ignore_keys = None
    prediction_loss_only = None

    if not swag:
        model.to('cuda')
        
    test_dataloader = my_trainer.get_test_dataloader(dataset)    
    
    # lists to store predictions and groud trtuh values
    all_labels_true = []
    all_preds = []
    all_logits = []


    # progress bar
    prediction_bar = tqdm(total=len(test_dataloader))

    model.eval()

    for step, inputs in enumerate(test_dataloader):

        if my_trainer.use_adversarial:
            loss, logits, labels = my_trainer.prediction_step_bidir_adversarial(model, 
                                                                                inputs, 
                                                                                prediction_loss_only, 
                                                                                ignore_keys=ignore_keys)
        else:
            loss, logits, labels = my_trainer.prediction_step_bidir(model, 
                                                                    inputs, 
                                                                    prediction_loss_only, 
                                                                    ignore_keys=ignore_keys)

        all_preds.extend(logits.argmax(axis=1).cpu().numpy().tolist())
        all_labels_true.extend(labels.cpu().numpy().tolist())
        all_logits.append(logits.cpu())

        prediction_bar.update(1)
        
        
    array_true = np.array(all_labels_true)
    array_pred = np.array(all_preds)
    
    # accuracy score
    ac = accuracy_score(array_true,array_pred)
    # weighted f1 score
    f1 = f1_score(array_true,array_pred,average='weighted')
    
    return ac,f1,all_logits
    