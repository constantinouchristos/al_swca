import numpy as np
from transformers import (
    BertForSequenceClassification,
    BertConfig)

from bert_modeling_custom import BertForSequenceClassification_c
from bert_classifier_model import Bert_classifier_adversarial
from swag_modeling import SWAG_adversarial

from bert_trainer_utils import Trainer_custom
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score

from tqdm.auto import tqdm
import random
import torch
import importlib

_torch_available = importlib.util.find_spec("torch") is not None


def is_torch_available():
    return _torch_available

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

    
    
class Active_learner:
    
    
    def __init__(self,
                 data_handler=None,
                 data_args=None,
                 model_args=None,
                 training_args=None,
                 compute_metrics=None,
                 tokenizer=None,
                 data_collator=None,
                 criterion=None,
                 aquisiiton_f=None,
                 num_labels=None,
                 bays_aversarial=False,
                 iters=10):
        
        # data handler
        self.data_handler = data_handler

        # aquisition function wrapper
        self.aquisiiton_f = aquisiiton_f
        
        # iterations for active learning 
        self.n_iters = iters
        
        # for model initialization
        self.data_args = data_args
        self.model_args = model_args
        self.num_labels = num_labels
        self.training_args = training_args
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.criterion = criterion
        
        # debuging testing variables
        #self.last_logits = None
        
        # function to evaluate model
        #self.evaluate_func = evaluate_func
        
        # original datasets
        self.original_train = self.data_handler.original_train
        self.original_validation = self.data_handler.original_validation
        self.original_test = self.data_handler.original_test
        
        #wether to use the baysian adversarial apporoch or not
        self.bays_aversarial = bays_aversarial
        
        # set seed
        set_seed(73)
        
        
    def initialize_model_trainer(self,c_train_data=None):
        """
        initialize models and add them with the chosen data to the trainer
        """
        
        
        if c_train_data is None:
            c_train_data = self.original_train
        
        config = BertConfig.from_pretrained(    
            pretrained_model_name_or_path=self.model_args.model_name_or_path,
            num_labels=self.num_labels
        )
        


        if self.bays_aversarial:
            # BERT model
            model = BertForSequenceClassification_c.from_pretrained(
                pretrained_model_name_or_path=self.model_args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=None
            )


            clasifier_adversarial = Bert_classifier_adversarial(
                bert_seq_class=model

            )

            # swag adversarial
            SWAG_adversarial_m = SWAG_adversarial(
                BertForSequenceClassification_c,
                no_cov_mat=not True,
                max_num_models=20,
                num_classes=self.num_labels,
                config=config,
                model_args=self.model_args

            )

            self.my_trainer = Trainer_custom(
                model=clasifier_adversarial,
                args=self.training_args,
                train_dataset=c_train_data if self.training_args.do_train else None,
                eval_dataset=self.original_validation if self.training_args.do_eval else None,
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                swag_model=SWAG_adversarial_m,
                criterion=self.criterion
                                       )

            # setting up adversarial parameters
            self.my_trainer.use_adversarial = self.data_args.use_adversarial
            self.my_trainer.adversarial_epsilon = self.data_args.adversarial_epsilon
            self.my_trainer.adv_attk_tpe = self.data_args.adv_attk_tpe
            self.my_trainer.lambdaa = self.data_args.lambdaa
            self.my_trainer.temperature_contrastive = self.data_args.temperature_contrastive
            
            # setting up swag params
            self.my_trainer.swag_per_start = self.data_args.swag_per_start


        else:
            # BERT model
            classifier = BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.model_args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=None
            )
            
            
            self.my_trainer = Trainer_custom(
                model=classifier,
                args=self.training_args,
                train_dataset=c_train_data if self.training_args.do_train else None,
                eval_dataset=self.original_validation if self.training_args.do_eval else None,
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                swag_model=None,
                criterion=self.criterion
                                       )
            
            self.my_trainer.use_adversarial = False


            
    def evaluate_model(self,
                       dataset=None,
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
        
        if self.bays_aversarial:
            
            if swag:
                model = self.my_trainer.swag_model
                model.sample(0.0)

            else:
                model = self.my_trainer.model
                model.to('cuda')
        else:
            model = self.my_trainer.model
            model.to('cuda')


        test_dataloader = self.my_trainer.get_test_dataloader(dataset)    

        # lists to store predictions and groud trtuh values
        all_labels_true = []
        all_preds = []
        all_logits = []


        # progress bar
        prediction_bar = tqdm(total=len(test_dataloader))

        model.eval()

        for step, inputs in enumerate(test_dataloader):


            loss, logits, labels = self.my_trainer.prediction_step_bidir_adversarial(model, 
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
        #balanced accuracy score
        b_acc = balanced_accuracy_score(array_true,array_pred)

        return ac,f1,b_acc,all_logits
    
    



    def model_predict(self,
                      dataset=None,
                      swag=False,
                      T=50,
                      training=False,
                      mc_dropout=True,
                      low_rank=False
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


            test_dataloader = self.my_trainer.get_test_dataloader(dataset)   

            # lists to store predictions and groud trtuh values
            all_trus = []
            all_preds = []
            all_logits = []


            # progress bar
            prediction_bar = tqdm(total=len(test_dataloader) * T)

            if mc_dropout:
                if self.bays_aversarial:
                    if swag:

                        model = self.my_trainer.swag_model
                        model.sample(0.0)

                    else:
                        
                        raise NotImplementedError
                else:

                    model = self.my_trainer.model
                    model.to('cuda')


                if training:
                    model.train()
                else:
                    model.eval()


            for t in range(T):

                if not mc_dropout:
                    assert swag == True, 'incorect parameters chosen'

                    model = self.my_trainer.swag_model
                    
                    if low_rank:
                        model.sample(np.random.uniform(0,50),True)
                    else:
                        model.sample(np.random.uniform(0,50))

                    if training:
                        model.train()
                    else:
                        model.eval()


                temp_preds = []
                temp_logits = []
                temp_true = []

                for step, inputs in enumerate(test_dataloader):


                    loss, logits, labels = self.my_trainer.prediction_step_bidir_adversarial(model, 
                                                                            inputs, 
                                                                            prediction_loss_only, 
                                                                            ignore_keys=ignore_keys)



                    temp_logits.append(logits.cpu())
                    temp_preds.append(logits.argmax(axis=1).cpu().numpy())
                    temp_true.append(labels.cpu().numpy())

                    prediction_bar.update(1)


                all_preds.append(np.concatenate(temp_preds,axis=0))
                all_logits.append(np.concatenate(temp_logits,axis=0))
                all_trus.append(np.concatenate(temp_true,axis=0))


            return all_preds,all_trus,all_logits

 
        
        
        
    def active_train(self,
                     initial_n_train=100,
                     n_sample_infer=250,
                     T_aq=50,
                     n_query=10,
                     aquisition_strategy='random',
                     use_swag=False,
                     r_s=None,
                     initial_train_seed=None,
                     mc_dropout=True,
                     train_mode=True,
                     low_rank=None
                     
                    ):
        
        
        test_performance_ac = []
        test_performance_f1 = []
        test_performance_b_ac = []
      
        
        number_of_train_data = []
        
        # original test data
        test_data_for_inference = self.data_handler.original_test
        
        
        # initialize train data and pool data
        self.data_handler.initialize_train_pool(n_init_train=initial_n_train,r_seed=initial_train_seed)
        
        
        print('data info:')
        print(f'pool data size: {len(self.data_handler.indexes_pool)}')
        print(f'train data size: {len(self.data_handler.indexes_train)}')
        print(f'train indexes: {self.data_handler.indexes_train}')
        
        

        for iterr in range(self.n_iters):
            
            # get train data for training
            c_train_data = self.data_handler.get_train_data()
            
            if iterr == 0:
                print('initial_data:')
                print(c_train_data.to_pandas().text)
                
            
            # initialize models and trainer
            self.initialize_model_trainer(c_train_data=c_train_data)
            
            # number of points used for training
            number_of_points_used_in_training = c_train_data.shape[0]
            
            #train trainer with chosen data
            self.my_trainer.train()
            
            
            #evaluate trained model
            ac,f1,b_acc,all_logits = self.evaluate_model(
                dataset=test_data_for_inference,
                swag=use_swag
                      )
            
            print(f'iterr: {iterr}, f1: {f1}')
            
            # store the performance values and the number of train data used
            test_performance_ac.append(ac)
            test_performance_f1.append(f1)
            test_performance_b_ac.append(b_acc)
            number_of_train_data.append(number_of_points_used_in_training)
            
            np.random.seed(r_s)
            random.seed(r_s)
            
            # we sample from train data pool for further labeling
            sample_for_inference = self.data_handler.sample_pool(n_sample=n_sample_infer)
            
            
            # no need to do prediciton on last iteration
            if iterr == self.n_iters -1 :
                continue
            
            # perform inference on sampled data
            all_preds,all_trus,all_logits = self.model_predict(
                dataset=sample_for_inference,
                swag=use_swag,
                T=T_aq,
                training=train_mode,
                mc_dropout=mc_dropout,
                low_rank=low_rank
                     )
            
            # use aquisition function on infered sample data from pool
            chosen_indexes = self.aquisiiton_f(logits=all_logits,
                                               n_query=n_query,
                                               type_=aquisition_strategy
                                              )
             

            # update the pool and train data in the data handler
            self.data_handler.update_pool_train_indexes(chosen_indexes)

            
            print('data info:')
            if self.data_handler.clustered_idx_pool:

                print(f'pool data size: {sum([len(self.data_handler.clustered_idx_pool[io]) for io in  self.data_handler.clustered_idx_pool])}')
                
            else:
                print(f'pool data size: {len(self.data_handler.indexes_pool)}')
            print(f'train data size: {len(self.data_handler.indexes_train)}')
            
            
            self.data_handler.sanity_check()
            print('SANITY CHECK passed')
            
            
        
        
        self.test_performance_ac = test_performance_ac
        self.test_performance_b_ac = test_performance_b_ac
        self.test_performance_f1 = test_performance_f1
        self.number_of_train_data = number_of_train_data
        
        
        