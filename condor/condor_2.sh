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

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g3_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g3_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g3_relw.out
Arguments = run_2.sh --expt_name=5p_gold_g3_relw --gcn_depth=3 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g4_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g4_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g4_relw.out
Arguments = run_2.sh --expt_name=5p_gold_g4_relw --gcn_depth=4 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_relw.out
Arguments = run_2.sh --expt_name=5p_gold_g5_relw --gcn_depth=5 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_256_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_256_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_256_relw.out
Arguments = run_2.sh --expt_name=5p_gold_g5_relw_256 --gcn_depth=5 --use_rel_words --n_attn=256 --n_ans_gate=256 --n_qi_gate=256 --n_ques_emb=256 --bsz=32
Queue 1

# Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_drop0.log
# Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_drop0.err
# Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g5_drop0.out
# Arguments = run_2.sh --expt_name=5p_gold_g5_drop0 --gcn_depth=5 --bsz=128 --drop_prob=0.0
# Queue 1

# Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g3.log
# Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g3.err
# Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g3.out
# Arguments = run_2.sh --expt_name=1p_gold_g3 --gcn_depth=3 --bsz=128
# Queue 1

# Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g4.log
# Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g4.err
# Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g4.out
# Arguments = run_2.sh --expt_name=1p_gold_g4 --gcn_depth=4 --bsz=128
# Queue 1

# Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g5.log
# Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g5.err
# Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g5.out
# Arguments = run_2.sh --expt_name=1p_gold_g5 --gcn_depth=5 --bsz=128
# Queue 1

# Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g3_rel.log
# Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g3_rel.err
# Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g3_rel.out
# Arguments = run_2.sh --expt_name=1p_gold_g3_rel --gcn_depth=3 --use_rel_emb --bsz=4
# Queue 1

# Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g4_rel.log
# Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g4_rel.err
# Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g4_rel.out
# Arguments = run_2.sh --expt_name=1p_gold_g4_rel --gcn_depth=4 --use_rel_emb --bsz=4
# Queue 1

# Log = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g5_rel.log
# Error = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g5_rel.err
# Output = /scratch/cluster/ankgarg/gqa/logs/1p_gold_g5_rel.out
# Arguments = run_2.sh --expt_name=1p_gold_g5_rel --gcn_depth=5 --use_rel_emb --bsz=4
# Queue 1