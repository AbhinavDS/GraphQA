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

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs_sum.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs_sum.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs_sum.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g2_rel_probs_sum --use_rel_probs_sum --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=2 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=8 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g1_rel_probs --use_rel_probs --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=8 --gen_mode=pred_cls
Queue 1