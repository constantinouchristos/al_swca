



# # custom data 2 cyclic
arguments= {"resume_from_checkpoint": False,
            "task_name":None,
             "model_name_or_path": 'bert-base-uncased',
             "output_dir": "./experiment_run_classification/",
             "dataset_name": None,
            "dataset_config_name":None,
             "do_eval" : True,
             "do_train" : True,
             "max_seq_length": 128,
             "version_2_with_negative": True,
             "overwrite_output_dir": True,
             "num_train_epochs": 20,
             "doc_stride": 128,
             "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
             "save_steps": 500,
             "logging_steps": 100,
             "save_total_limit": 7,
             "gradient_accumulation_steps":2,
             "fp16": True,
             "seed": 42,
            "pad_to_max_length":True,
            "main_data_dir":'./available_datasets/',
            "dataset_name_source":'cola',
             "train_file":'./available_datasets/train.csv',
            "validation_file":'./available_datasets/dev.csv',
            "test_file":'./available_datasets/test.csv',
            "use_adversarial": True,
            "adversarial_epsilon": 0.001,
            "adv_attk_tpe": 'l2',
            "lambdaa": 0.5,
            "swag_per_start": 0.5,
            "temperature_contrastive": 0.7,
            
 
            
            
}