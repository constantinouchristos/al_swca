import numpy as np

# class Data_Handler:
    
#     def __init__(self,
#                  train_dataset,
#                  eval_dataset,
#                  predict_dataset
                 
#                 ):
        
#         self.original_train = train_dataset
#         self.original_validation = eval_dataset
#         self.original_test = predict_dataset
        
#         self.indexes_pool = []
#         self.indexes_train = []
        
        
        
#     def initialize_train_pool(self,n_init_train=100,r_seed=73):
        
#         # instances to sample from each class
#         num_each_class = int(n_init_train/2)
        
#         # dataset to pandas
#         pd_data = self.original_train.to_pandas()
#         # separate into 2 classes
#         pos_labesl = pd_data[pd_data.labels == 1].copy()
#         neg_labesl = pd_data[pd_data.labels == 0].copy()
        
#         # sample
#         samples_index_pos = pos_labesl.sample(n=num_each_class,replace=False,random_state=r_seed).index.values
#         samples_index_neg = neg_labesl.sample(n=num_each_class,replace=False,random_state=r_seed).index.values
        
#         # join and sort all indexes
#         all_initial_indexes_train = np.sort(np.concatenate([samples_index_pos,samples_index_neg],axis=0))
#         self.indexes_train = all_initial_indexes_train
        
#         # indexes for pool
#         self.indexes_pool = np.array([i for i in pd_data.index.values if i not in self.indexes_train])
        
        
#     def get_train_data(self):
        
#         return self.original_train.select(self.indexes_train)
    
    
#     def get_pool_data(self):
        
#         return self.original_train.select(self.indexes_pool)
    
    
#     def sample_pool(self,n_sample=2000):
        
#         # make sure that sample size is not greater than pool size
#         if len(self.indexes_pool) < n_sample:
#             n_sample = len(self.indexes_pool)
            
            
#         self.random_indexes_from_pool = np.random.choice(self.indexes_pool,size=n_sample,replace=False)
        
#         return self.original_train.select(self.random_indexes_from_pool)
    
#     def update_pool_train_indexes(self,chosen_indexes=None):
        
#         # map chosen indexes to indexes from sample
#         actual_indexes_pool = self.random_indexes_from_pool[chosen_indexes]
        
#         # add chosen pool indexes to train 
#         self.indexes_train = np.sort(np.concatenate([self.indexes_train,actual_indexes_pool]),axis=0)
        
#         # remove chosen pool indexes from pool
#         self.indexes_pool = np.sort(np.array([i for i in self.indexes_pool if i not in actual_indexes_pool]))


        
class Data_Handler:
    
    def __init__(self,
                 train_dataset=None,
                 eval_dataset=None,
                 predict_dataset=None,
                 cluster_pool_idx=None
                 
                ):
        
        self.original_train = train_dataset
        self.original_validation = eval_dataset
        self.original_test = predict_dataset
        
        self.indexes_pool = []
        self.indexes_train = []
        
        self.clustered_idx_pool = cluster_pool_idx
        
        self.iter = 0
        
        self.all_random_ccc = []
        
    def initialize_train_pool(self,n_init_train=100,r_seed=73):
        
        # dataset to pandas
        pd_data = self.original_train.to_pandas()
        
        # number of classes
        num_classes_all = pd_data.labels.nunique()
        
        # instances to sample from each class
        num_each_class = int(n_init_train/num_classes_all)
        
        
        samples_all_classes = []

        for class_t in pd_data.labels.unique():

            # instances only for class class_t
            temp_df = pd_data[pd_data.labels==class_t].copy()

            # sample
            if num_each_class > len(temp_df):
                samples_index = temp_df.sample(n=len(temp_df),replace=False,random_state=r_seed).index.values
            else:
                samples_index = temp_df.sample(n=num_each_class,replace=False,random_state=r_seed).index.values

            samples_all_classes.append(samples_index)

        # join and sort all indexes
        all_initial_indexes_train = np.sort(np.concatenate(samples_all_classes,axis=0))
        
        
#         if num_classes_all == 2:
            
#             # separate into 2 classes
#             pos_labesl = pd_data[pd_data.labels == 1].copy()
#             neg_labesl = pd_data[pd_data.labels == 0].copy()

#             # sample
#             samples_index_pos = pos_labesl.sample(n=num_each_class,replace=False,random_state=r_seed).index.values
#             samples_index_neg = neg_labesl.sample(n=num_each_class,replace=False,random_state=r_seed).index.values

#             # join and sort all indexes
#             all_initial_indexes_train = np.sort(np.concatenate([samples_index_pos,samples_index_neg],axis=0))
            
#         else:
            
        
        self.indexes_train = all_initial_indexes_train
        
#         print('initial train data:',len(self.indexes_train))
        
        # indexes for pool
        self.indexes_pool = np.array([i for i in pd_data.index.values if i not in self.indexes_train])
        
        if self.clustered_idx_pool is not None:
            for clust in self.clustered_idx_pool:
                # remove indexes drom the cluster pool
                self.clustered_idx_pool[clust] = np.sort(np.array([i for i in self.clustered_idx_pool[clust] if i not in self.indexes_train]))
            
#         print('initial pool dict size:',sum([len(self.clustered_idx_pool[i]) for i in self.clustered_idx_pool]))
              
#         print('initial pool size:',len(self.indexes_pool))
              
              
              
        
    def get_train_data(self):
        
        return self.original_train.select(self.indexes_train)
    
    
    def get_pool_data(self):
        
        return self.original_train.select(self.indexes_pool)
    
    
    def sample_pool(self,n_sample=2000):
        
        
        # make sure that sample size is not greater than pool size
        if len(self.indexes_pool) < n_sample:
            n_sample = len(self.indexes_pool)
        
        if self.clustered_idx_pool is not None:
            
            number_of_clusters = len(self.clustered_idx_pool)
            
            max_budget_to_try_per_cluster = int(n_sample/number_of_clusters)
            #print('max_budget_to_try_per_cluster:',max_budget_to_try_per_cluster)
            
            total_budget_left = n_sample
            
            all_sampled_indexes_from_all_clusters = []
            
            for clust in self.clustered_idx_pool:
                
                cluster_indexes_to_sample_from = self.clustered_idx_pool[clust]
                
                
                
                if len(cluster_indexes_to_sample_from) < max_budget_to_try_per_cluster:
                    t_max_budget_to_try_per_cluster = len(cluster_indexes_to_sample_from)
                    
                    sample_temp_cluster = np.random.choice(cluster_indexes_to_sample_from,
                                                           size=t_max_budget_to_try_per_cluster,
                                                           replace=False)
                    
                
                else:
                    sample_temp_cluster = np.random.choice(cluster_indexes_to_sample_from,
                                                           size=max_budget_to_try_per_cluster,
                                                           replace=False)
                
                total_budget_left -= len(sample_temp_cluster)
                all_sampled_indexes_from_all_clusters.append(sample_temp_cluster)
                #print(f'sampled: {len(sample_temp_cluster)} from cluster {clust}')
            
            # puting all the sampled cluster indexes together
            all_sampled_indexes_from_all_clusters = np.hstack(all_sampled_indexes_from_all_clusters)
            #print('all_sampled_indexes_from_all_clusters:',len(all_sampled_indexes_from_all_clusters))
            #print('total_budget_left:',total_budget_left)
            
            # sample remaining indexes if there is budget left
            if total_budget_left != 0:
                
            
                # remaining indexes 
                index_remain = [i for i in self.indexes_pool if i not in all_sampled_indexes_from_all_clusters]
                
                # extra choice to reach the n_sample sample
                extra_choice = np.random.choice(index_remain,
                                              size=total_budget_left,
                                              replace=False)
                
                
                all_sampled_indexes_from_all_clusters = np.hstack([all_sampled_indexes_from_all_clusters,
                                                                   extra_choice
                                                                  ])
            
            assert len(all_sampled_indexes_from_all_clusters) == n_sample, f'something is wrong size of sampled indexes: {len(all_sampled_indexes_from_all_clusters)} != n_sample: {n_sample}'
            
            
            self.random_indexes_from_pool = all_sampled_indexes_from_all_clusters
            
            self.all_random_ccc.append(self.random_indexes_from_pool)
        
            return self.original_train.select(self.random_indexes_from_pool)
            
        else:
            
            
            self.random_indexes_from_pool = np.random.choice(self.indexes_pool,
                                                             size=n_sample,
                                                             replace=False,
                                                            )
            
            self.all_random_ccc.append(self.random_indexes_from_pool)

            return self.original_train.select(self.random_indexes_from_pool)
    
    
    
    
    def update_pool_train_indexes(self,chosen_indexes=None):
        
        # map chosen indexes to indexes from sample
        actual_indexes_pool = self.random_indexes_from_pool[chosen_indexes]
        
        
        # add chosen pool indexes to train 
        self.indexes_train = np.sort(np.concatenate([self.indexes_train,actual_indexes_pool]),axis=0)
        
        
        if self.clustered_idx_pool is not None:
            
            for clust in self.clustered_idx_pool:
                # remove indexes drom the cluster pool
                self.clustered_idx_pool[clust] = np.sort(np.array([i for i in self.clustered_idx_pool[clust] if i not in actual_indexes_pool]))
            
        else:
        
            # remove chosen pool indexes from pool
            self.indexes_pool = np.sort(np.array([i for i in self.indexes_pool if i not in actual_indexes_pool]))
        
        self.iter += 1
#         print(f'pool dict size: {sum([len(self.clustered_idx_pool[i]) for i in self.clustered_idx_pool])}, iter: {self.iter}')
#         print(f'train size: {len(self.indexes_train)}, iter: {self.iter}')


    def sanity_check(self,verbose=False):
        
        if self.clustered_idx_pool is not None:
            
            all_indexes_in_pool = np.sort(np.hstack([self.clustered_idx_pool[j] for j in self.clustered_idx_pool]))
            
            k = [i for i in self.indexes_train if i in all_indexes_in_pool]
            
            assert len(k) == 0 ,f'train data and pool data have similar data. {len(k)}'
            
            if verbose:
                print(f'pool_data size: {len(all_indexes_in_pool)}')
                print(f'train data size: {len(self.indexes_train)}')
                print(f'original data size: {len(self.original_train.to_pandas())}')
            
        else:
            
            k = [i for i in self.indexes_train if i in self.indexes_pool]
            
            assert len(k) == 0 ,f'train data and pool data have similar data. {len(k)}'
            
            if verbose:
                print(f'pool_data size: {len(self.indexes_pool)}')
                print(f'train data size: {len(self.indexes_train)}')
                print(f'original data size: {len(self.original_train.to_pandas())}')
            
        
        
        
        