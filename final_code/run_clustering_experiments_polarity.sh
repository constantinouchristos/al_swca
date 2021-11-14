#!/bin/sh

# DATA_ROOT=./data


echo "=== running clustering experiments ==="
echo "---"

### clustering cola

# random without clustering      progress: done
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=polarity_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_random_mc_no_clust -data=polarity -mc_d -tr_m

# random with normal clustering    progress: running terminal 1
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=polarity_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_random_mc_normal_clust -data=polarity -mc_d -tr_m -clus -cl_m=normal

# random with high clustering      progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=random -path_s=final_publish_paper_experiments -exp_n=polarity_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_random_mc_high_clust -data=polarity -mc_d -tr_m -clus -cl_m=high

### clustering cola bald

# bald without clustering           progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=polarity_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_bald_mc_no_clust -data=polarity -mc_d -tr_m

# bald with normal clustering       progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=polarity_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_bald_mc_normal_clust -data=polarity -mc_d -tr_m -clus -cl_m=normal

# bald with high clustering       progress:
python3 run_compare_clust.py -ts=10 -e=20 -taq=50 -nq=50 -al_it=7 -init_ts=50 -pss=400 -its=35 -cl_s=35 -aq_s=bald -path_s=final_publish_paper_experiments -exp_n=polarity_ts_10_e_20_taq_50_nq_50_init_ts_50_pss_400_its_35_norm_bald_mc_high_clust -data=polarity -mc_d -tr_m -clus -cl_m=high

echo "=== clustering experiments complete ==="



