#!/bin/sh

# DATA_ROOT=./data



echo "=== running al experiments ==="
echo "---"

# trial set


#mc
# python3 run_compare_clust.py -ts=2 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=16 -cl_s=16 -aq_s=bald -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_16_norm_bald_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high

# #swag sampling
# python3 run_compare_clust.py -ts=2 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=16 -cl_s=16 -aq_s=bald -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_16_swca_bald_swag_sampling_high_clust -data=cola -clus -cl_m=high -badv -swag


# python3 run_compare_clust.py -ts=2 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=16 -cl_s=16 -aq_s=bald -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_16_swca_bald_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high -badv -swag










###################################### MAIN EXPERIMENTS ###################################### 


# so experiments a (we need to use mc dropout for all aquisitions with normal model)

# random method : already done from clustering experiemnt
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=16 -cl_s=16 -aq_s=random -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_16_norm_random_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high

#  bald method : already done from clustering experiemnts
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=31 -cl_s=16 -aq_s=bald -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_31_norm_bald_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high


#  var_rations method : 
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=29 -cl_s=16 -aq_s=var_ratios -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_29_norm_var_ratios_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high




#so experiments b (we need to use mc dropout for all aquisitions with swag adversarial model)

# random 
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=16 -cl_s=16 -aq_s=random -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_16_swca_random_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high -badv -swag


# bald :

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=31 -cl_s=16 -aq_s=bald -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_31_swca_bald_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high -badv -swag

# var ratios
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=29 -cl_s=16 -aq_s=var_ratios -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_29_swca_var_ratios_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high -badv -swag




# so experiments c (we need to use swag sampling for all aquisitions with swag model model)


# random:
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=16 -cl_s=16 -aq_s=random -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_16_swca_random_swag_sampling_high_clust -data=cola -clus -cl_m=high -badv -swag


# bald :

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=31 -cl_s=16 -aq_s=bald -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_31_swca_bald_swag_sampling_high_clust -data=cola -clus -cl_m=high -badv -swag

# var ratios
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=13 -init_ts=20 -pss=0.05 -its=29 -cl_s=16 -aq_s=var_ratios -path_s=final_publish_paper_al_experiments_cola -exp_n=cola_ts_10_e_20_taq_50_nq_10_init_ts_20_pss_05_its_29_swca_var_ratios_swag_sampling_high_clust -data=cola -clus -cl_m=high -badv -swag




echo "=== clustering al experiments complete ==="







