from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def get_gradient(x1,x2):
    
    return (abs(x2[1] - x1[1] ))/(abs(x2[0] - x1[0] ))


def percentage_reduction(x1,x2):
    
    return abs(x2-x1)






def get_optimal_k_val(data=None,
                      k_to_test=12,
                      show=False,
                      rand_state=None
                     ):


    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    K = range(1,k_to_test) 

    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k,random_state=rand_state).fit(data)     

        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 
                          'euclidean'),axis=1)) / data.shape[0]) 
        inertias.append(kmeanModel.inertia_) 

        mapping1[k] = sum(np.min(cdist(data, kmeanModel.cluster_centers_, 
                     'euclidean'),axis=1)) / data.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 



    points_disto = [[i,j] for i,j in zip(K, distortions)]
    gradsss = [get_gradient(points_disto[i],points_disto[i+1]) for i in range(len(points_disto)-1)]
    gradsss_red = [percentage_reduction(gradsss[i],gradsss[i+1]) for i in range(len(gradsss)-1)]
    optimal_k_disto = gradsss_red.index(min(gradsss_red)) +1

    points_iner = [[i,j] for i,j in zip(K, inertias)]
    gradsss_in = [get_gradient(points_iner[i],points_iner[i+1]) for i in range(len(points_iner)-1)]
    gradsss_red_in = [percentage_reduction(gradsss_in[i],gradsss_in[i+1]) for i in range(len(gradsss_in)-1)]
    optimal_k_inertia = gradsss_red.index(min(gradsss_red)) +1

    opt_k = min(optimal_k_inertia,optimal_k_disto)
    
    if show:
        
        plt.plot(K, distortions, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Distortion') 
        plt.title('The Elbow Method using Distortion') 
        plt.show()
        print()
        
        plt.plot(K, inertias, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Inertia') 
        plt.title('The Elbow Method using Inertia') 
        plt.show() 
    
    
    return opt_k


def get_clustered_indexes_dict(raw_datasets_init=None,
                               rand_state=None,
                               cl_m='normal'
                              ):
    
    # acces raw data
    initi_train = raw_datasets_init['train']
    
    # acesss text
    examples = initi_train.to_pandas().text.values
    
    # get init labels
    initial_labels = initi_train.to_pandas().labels.values
    
    
    # define sentece transformer model
    model_sent = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    
    # encode sentences
    embeddings = model_sent.encode(examples)
    
    # find optimal k value for kmeans with elbow method
    opt_k = get_optimal_k_val(embeddings,
                              k_to_test=12,
                              show=False,
                              rand_state=rand_state)
    
    if cl_m == 'normal':
        # fit kmeans model with optimal k
        kmeans = KMeans(n_clusters=opt_k, random_state=rand_state).fit(embeddings)
    else:
        kmeans = KMeans(n_clusters=100, random_state=rand_state).fit(embeddings)
        
    
    # preictiong labels of clustered data
    preds = kmeans.predict(embeddings)
    
    # put data into data frame
    all_data = pd.DataFrame(data=embeddings,
                            columns=['feat_'+str(i) for i in range(embeddings.shape[1])]
                           )
    
    # adding predictions to data frame
    all_data['target'] = preds
    
    # initila labels
    all_data['labels'] = initial_labels
    
    # get indexes per cluster
    indexs_per_cluster = {i:all_data[all_data.target == i].copy().index.values for i in  all_data.target.unique()}
    
    return indexs_per_cluster
    