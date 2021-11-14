

#!/bin/sh

### descritpion
# -mc_d is chosen we always use mcd_dropout and we need to use -tr_m with it
# -mc_d is not chosen we always use swag sampling
# -badv is cosen we use the bay adversarial so -swag needs to be chosen as well
# -clust should always be used for now


#all aquisitions = ['bald','var_ratios','random']



echo "=== running al experiments ==="
echo "---"

# so experiments a (we need to use mc dropout for all aquisitions with normal model)

# random method : already done from clustering experiemnt
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_norm_random_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high

#  bald method : already done from clustering experiemnts
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_norm_bald_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high


#  var_rations method : 
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=var_ratios -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_norm_var_ratios_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high




#so experiments b (we need to use mc dropout for all aquisitions with swag adversarial model)

# random 
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_random_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high -badv -swag


# bald :

# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_bald_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high -badv -swag

# var ratios
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=var_ratios -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_var_ratios_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high -badv -swag




# so experiments c (we need to use swag sampling for all aquisitions with swag model model) all done


# random:
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_random_swag_sampling_high_clust -data=cola -clus -cl_m=high -badv -swag


# bald :

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_bald_swag_sampling_high_clust -data=cola -clus -cl_m=high -badv -swag

# var ratios
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=var_ratios -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_var_ratios_swag_sampling_high_clust -data=cola -clus -cl_m=high -badv -swag


############################################# new experiments #############################################






# so experiments d (we need to use swag sampling for all aquisitions with swag model model and also mc)


# random: not done   we are only test the bald methods for now do this later
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_random_swag_sampling_plus_MC_high_clust -data=cola -clus -cl_m=high -badv -swag -tr_m


# bald :

# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_bald_swag_sampling_plus_MC_high_clust -data=cola -clus -cl_m=high -badv -swag -tr_m

# var ratios
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=var_ratios -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_var_ratios_swag_sampling_plus_MC_high_clust -data=cola -clus -cl_m=high -badv -swag -tr_m









# so experiments e (we need to use swag sampling but with low rank s for all aquisitions with swag model model)


# random: not done   we are only test the bald methods for now do this later
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_random_swag_sampling_lowrank_high_clust -data=cola -clus -cl_m=high -badv -swag -low_r


# bald :

# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_bald_swag_sampling_lowrank_high_clust -data=cola -clus -cl_m=high -badv -swag -low_r

# var ratios
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=var_ratios -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_var_ratios_swag_sampling_lowrank_high_clust -data=cola -clus -cl_m=high -badv -swag -low_r








# so experiments f (we need to use swag sampling but with low rank s for all aquisitions with swag model model plus mc)


# random: not done   we are only test the bald methods for now do this later
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_random_swag_sampling_lowrank_MC_high_clust -data=cola -clus -cl_m=high -badv -swag -tr_m -low_r


# bald :

# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_bald_swag_sampling_plus_lowrank_MC_high_clust -data=cola -clus -cl_m=high -badv -swag -tr_m -low_r

# var ratios
# python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=13 -cl_s=13 -aq_s=var_ratios -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_13_swca_var_ratios_swag_sampling_lowrank_MC_high_clust -data=cola -clus -cl_m=high -badv -swag -tr_m -low_r





echo "=== clustering experiments complete ==="




