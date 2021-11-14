

#experiments to compare clustering 
## initial train size 50 final 350 like papeer online so sample size 50


### clustering cola

# random without clustering      progress: running terminal 2
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_random_mc_no_clust -data=cola -mc_d -tr_m

# random with normal clustering    progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_random_mc_normal_clust -data=cola -mc_d -tr_m -clus -cl_m=normal

# random with high clustering      progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_random_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high

### clustering cola bald

# random without clustering           progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_bald_mc_no_clust -data=cola -mc_d -tr_m

# random with normal clustering       progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_bald_mc_normal_clust -data=cola -mc_d -tr_m -clus -cl_m=normal

# random with high clustering       progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=cola_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_bald_mc_high_clust -data=cola -mc_d -tr_m -clus -cl_m=high


### clustering ag news







so experiments a (we need to use mc dropout for all aquisitions with normal model)

#bald      progres: 
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=bald -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_bald_mc_clust -clus -mc_d -tr_m

#random      progres: 
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=random -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_random_mc_clust -clus -mc_d -tr_m


# var ratios       progres: 
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=var_ratios -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_var_mc_clust -clus -mc_d -tr_m






