
# using clustering bay adversarial and swag
python3 run_compare_clust.py -ts=20 -e=20 -taq=50 -nq=10 -al_it=10 -init_ts=20 -pss=400 -aq_s=bald -path_s=final_experiments -exp_n=bays_clust_20_trials_bald -badv -swag -clus


# using clustering without using adversarial and swag
python3 run_compare_clust.py -ts=20 -e=20 -taq=50 -nq=10 -al_it=10 -init_ts=20 -pss=400 -aq_s=bald -path_s=final_experiments -exp_n=clust_20_trials_bald -clus


# not using clustering without using adversarial and swag
python3 run_compare_clust.py -ts=20 -e=20 -taq=50 -nq=10 -al_it=10 -init_ts=20 -pss=400 -its=77 -aq_s=bald -path_s=final_experiments -exp_n=no_clust_20_trials_bald 




python3 run_compare_clust.py -ts=20 -e=10 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -aq_s=bald -path_s=final_experiments -exp_n=no_clust_20_trials_bald_correct 

python3 run_compare_clust.py -ts=20 -e=10 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -aq_s=bald -path_s=final_experiments -exp_n=clust_20_trials_bald_correct -clus




# test case
python3 run_compare_clust.py -ts=2 -e=5 -taq=5 -nq=10 -al_it=3 -init_ts=20 -pss=50, -its=77 -aq_s=bald -path_s=final_experiments -exp_n=no_clust_test_trials_bald 


python3 run_compare_clust.py -ts=2 -e=5 -taq=5 -nq=10 -al_it=3 -init_ts=20 -pss=50 -aq_s=bald -path_s=final_experiments -exp_n=clust_test_trials_bald -clus



# experiments with clustering with bays and without

## test cases
python3 run_compare_clust.py -ts=2 -e=20 -taq=5 -nq=10 -al_it=2 -init_ts=20 -pss=50 -its=77 -aq_s=bald -path_s=final_experiments -exp_n=testing_delete_me_bays -badv -swag -clus

python3 run_compare_clust.py -ts=2 -e=20 -taq=5 -nq=10 -al_it=2 -init_ts=20 -pss=50 -its=77 -aq_s=bald -path_s=final_experiments -exp_n=testing_delete_me_norm -clus


python3 run_compare_clust.py -ts=2 -e=20 -taq=5 -nq=10 -al_it=3 -init_ts=20 -pss=50 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_experiments -exp_n=testing_delete_me_norm_var -clus

python3 run_compare_clust.py -ts=2 -e=20 -taq=5 -nq=10 -al_it=3 -init_ts=20 -pss=50 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_experiments -exp_n=testing_delete_me_bays_var -badv -swag -clus


experiments


python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_norm_var -clus

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_var -badv -swag -clus



python3 run_compare_clust.py -ts=2 -e=20 -taq=5 -nq=10 -al_it=2 -init_ts=20 -pss=50 -its=33 -aq_s=bald -path_s=final_experiments -exp_n=testing_delete_me_bays_sampling_and_train -badv -swag -clus -tr_m

python3 run_compare_clust.py -ts=2 -e=20 -taq=5 -nq=10 -al_it=2 -init_ts=20 -pss=50 -its=33 -aq_s=bald -path_s=final_experiments -exp_n=testing_delete_me_bays_sampling_not_train -badv -swag -clus 




python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_var_sampling_and_swag -badv -swag -clus -tr_m

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_var_no_mc -badv -swag -clus


python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=bald -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_bald_no_mc -badv -swag -clus

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=bald -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_bald_swag_sampling_mc -badv -swag -clus -tr_m



python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=bald -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_bald_swag_mc -badv -swag -clus -tr_m -mc_d

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=bald -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_bald_norm_mc -clus -tr_m -mc_d



python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=random -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_random_swag_sampling -badv -swag -clus

python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=random -path_s=final_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_bays_random_norm_mc -clus -tr_m -mc_d















### descritpion
-mc_d is chosen we always use mcd_dropout and we need to use -tr_m with it
-mc_d is not chosen we always use swag sampling
-badv is cosen we use the bay adversarial so -swag needs to be chosen as well
-clust should always be used for now

all aquisitions = ['bald','var_ratios','random']

so experiments a (we need to use mc dropout for all aquisitions with normal model)

#bald      progres: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=bald -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_norm_bald_mc -clus -mc_d -tr_m

#random      progres: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=random -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_norm_random_mc -clus -mc_d -tr_m

# var ratios       progres: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_norm_var_mc -clus -mc_d -tr_m



so experiments b (we need to use mc dropout for all aquisitions with swag adversarial model)

#bald      progres: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=bald -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_advswag_bald_mc -clus -badv -swag -mc_d -tr_m

#random      progres: running
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=random -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_advswag_random_mc -clus -badv -swag -mc_d -tr_m

# var ratios       progres: not done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_advswag_var_mc -clus -badv -swag -mc_d -tr_m


so experiments c (we need to use swag sampling for all aquisitions with swag model model)

#bald      progres: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=bald -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_advswag_bald_swag_sampling -clus -badv -swag

#random      progres: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=random -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_advswag_random_swag_sampling -clus -badv -swag 

# var ratios       progres: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=10 -al_it=9 -init_ts=20 -pss=300 -its=33 -cl_s=35 -aq_s=var_ratios -path_s=final_paper_experiments -exp_n=ts_10_e_20_taq_50_pss_300_its_33_advswag_var_swag_sampling -clus -badv -swag








