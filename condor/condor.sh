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
Notify_user = abhinav@cs.utexas.edu
Initialdir = /u/abhinav/Projects/GraphQA/condor/

Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g0_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g0_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g0_relw.out
Arguments = run_temp.sh --expt_name=1p_gold_g0_relw --gcn_depth=0 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g1_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g1_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g1_relw.out
Arguments = run_temp.sh --expt_name=1p_gold_g1_relw --gcn_depth=1 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g2_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g2_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g2_relw.out
Arguments = run_temp.sh --expt_name=1p_gold_g2_relw --gcn_depth=2 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g0_relw_blind.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g0_relw_blind.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g0_relw_blind.out
Arguments = run.sh --expt_name=5p_gold_g0_relw_blind --gcn_depth=0 --use_rel_words --use_blind --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_blind.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_blind.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g1_relw_blind.out
Arguments = run.sh --expt_name=5p_gold_g1_relw_blind --gcn_depth=1 --use_rel_words --use_blind --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g2_relw_blind.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g2_relw_blind.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g2_relw_blind.out
Arguments = run.sh --expt_name=5p_gold_g2_relw_blind --gcn_depth=2 --use_rel_words --use_blind --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g0_rel_new.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g0_rel_new.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g0_rel_new.out
Arguments = run.sh --expt_name=5p_gold_g0_rel_new --gcn_depth=0 --use_rel_emb --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g1_rel_new.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g1_rel_new.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g1_rel_new.out
Arguments = run.sh --expt_name=5p_gold_g1_rel_new --gcn_depth=1 --use_rel_emb --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g2_rel_new.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g2_rel_new.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g2_rel_new.out
Arguments = run.sh --expt_name=5p_gold_g2_rel_new --gcn_depth=2 --use_rel_emb --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

# Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_bce.log
# Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_bce.err
# Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_bce.out
# Arguments = run.sh --expt_name=5p_gold_g5_bce --gcn_depth=5  --critierion=bce --bsz=128
# Queue 1

# Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_bce_tanh.log
# Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_bce_tanh.err
# Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_bce_tanh.out
# Arguments = run.sh --expt_name=5p_gold_g5_bce_tanh --gcn_depth=5  --critierion=bce --nl=gated_tanh --bsz=128
# Queue 1

# Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_img.log
# Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_img.err
# Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_img.out
# Arguments = run.sh --expt_name=5p_gold_g5_img --gcn_depth=5  --use_img_feats --bsz=128
# Queue 1
