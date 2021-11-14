

params_active_l_n = {
    "trials_strategy": 5,
    "n_epochs_train": 20,
    "T_aq_uncert": 50,
    "n_query_pick": 10,
    "bays_adv_approach": False,
    "active_learning_iters": 10,
    "initial_train_data_size": 10,
    "n_sample_size_pool_inference": 1000,
    "use_swag": False,
    "name": 'normal.json'
    
}

params_active_l_b = {
    "trials_strategy": 5,
    "n_epochs_train": 20,
    "T_aq_uncert": 50,
    "n_query_pick": 10,
    "bays_adv_approach": True,
    "active_learning_iters": 10,
    "initial_train_data_size": 10,
    "n_sample_size_pool_inference": 1000,
    "use_swag": True,
    "name": 'bays_advers.json'
    
}