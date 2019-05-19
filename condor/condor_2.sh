universe = vanilla
Executable = /lusr/bin/bash
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "NN Experiment"
Requirements = TARGET.GPUSlot && InMastodon
getenv = True
request_GPUs = 1
+GPUJob = true
Notification = complete
Notify_user = ankgarg@cs.utexas.edu
Initialdir = /scratch/cluster/ankgarg/gqa/GraphQA/condor/

# Bottom Up 5p
Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_bua.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_bua.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_bua.out
Arguments = run_2.sh --expt_name=5p_gold_bua --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=0 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=128
Queue 1

# Bottom Up 5p mix
Log = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_bua.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_bua.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_bua.out
Arguments = run_2.sh --expt_name=5p_mix_gold_bua --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p_mix/ --test_dirname test_set_mix --gcn_depth=0 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=128
Queue 1

# GCN Gold RelWords
Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw.out
Arguments = run_2.sh --expt_name=5p_gold_g1_relw --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw.out
Arguments = run_2.sh --expt_name=5p_mix_gold_g1_relw --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p_mix/ --test_dirname test_set_mix --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

# GCN Gold RelWords SAN
Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_san2.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_san2.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_san2.out
Arguments = run_2.sh --expt_name=5p_gold_g1_relw_san2 --use_san --n_attn_layers 2 --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_san3.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_san3.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_san3.out
Arguments = run_2.sh --expt_name=5p_gold_g1_relw_san3 --use_san --n_attn_layers 3 --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw_san2.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw_san2.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw_san2.out
Arguments = run_2.sh --expt_name=5p_mix_gold_g1_relw_san2 --use_san --n_attn_layers 2 --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p_mix/ --test_dirname test_set_mix --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw_san3.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_miLog = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw_san3.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw_san3.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_relw_san3.out
Arguments = run_2.sh --expt_name=5p_mix_gold_g1_relw_san3 --use_san --n_attn_layers 3 --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p_mix/ --test_dirname test_set_mix --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16
Queue 1

# GCN Gold RelImage
Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_rel_emb.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_rel_emb.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_rel_emb.out
Arguments = run_2.sh --expt_name=5p_gold_g1_rel_emb --use_rel_emb --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_rel_emb.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_rel_emb.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_mix_gold_g1_rel_emb.out
Arguments = run_2.sh --expt_name=5p_mix_gold_g1_rel_emb --use_rel_emb --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p_mix/ --test_dirname test_set_mix --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16
Queue 1