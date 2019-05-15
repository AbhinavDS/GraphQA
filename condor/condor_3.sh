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

# GCN Gold Depth Expts
Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g0_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g0_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g0_relw.out
Arguments = run_2.sh --expt_name=5p_gold_g0_relw --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=0 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=64
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g2_relw.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g2_relw.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_gold_g2_relw.out
Arguments = run_2.sh --expt_name=5p_gold_g2_relw --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=2 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32
Queue 1

# GCN PredCls Probs
Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_probs.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_probs.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_probs.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g0_rel_probs --use_rel_probs --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=0 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g1_rel_probs --use_rel_probs --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g2_rel_probs --use_rel_probs --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=2 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

# GCN RelProbSum
Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_probs_sum.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_probs_sum.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_probs_sum.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g0_rel_probs_sum --use_rel_probs_sum --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=0 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs_sum.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs_sum.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_probs_sum.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g1_rel_probs_sum --use_rel_probs_sum --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs_sum.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs_sum.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_probs_sum.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g2_rel_probs_sum --use_rel_probs_sum --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=2 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

# GCN PredCls RelWords
Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_words.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_words.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g0_rel_words.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g0_rel_words --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=0 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_words.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_words.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_words.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g1_rel_words --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_words.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_words.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g2_rel_words.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g2_rel_words --use_rel_words --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=2 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1

Log = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_emb.log
Error = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_emb.err
Output = /scratch/cluster/ankgarg/gqa/logs/5p_pred_cls_g1_rel_emb.out
Arguments = run_2.sh --expt_name=5p_pred_cls_g1_rel_emb --use_rel_emb --reduce_img_feats --expt_data_dir=/scratch/cluster/ankgarg/gqa/test_data/5p/ --test_dirname test_set --gcn_depth=1 --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=16 --gen_mode=pred_cls
Queue 1