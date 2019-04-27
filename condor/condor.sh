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

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g3.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g3.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g3.out
Arguments = run.sh --expt_name=5p_gold_g3 --gcn_depth=3 --bsz=128
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g4.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g4.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g4.out
Arguments = run.sh --expt_name=5p_gold_g4 --gcn_depth=4 --bsz=128
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5.out
Arguments = run.sh --expt_name=5p_gold_g5 --gcn_depth=5 --bsz=128
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g3_rel.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g3_rel.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g3_rel.out
Arguments = run.sh --expt_name=5p_gold_g3_rel --gcn_depth=3 --use_rel_emb --bsz=4
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g4_rel.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g4_rel.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g4_rel.out
Arguments = run.sh --expt_name=5p_gold_g4_rel --gcn_depth=4 --use_rel_emb --bsz=4
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_rel.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_rel.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g5_rel.out
Arguments = run.sh --expt_name=5p_gold_g5_rel --gcn_depth=5 --use_rel_emb --bsz=4
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
