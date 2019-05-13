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

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g0_met.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g0_met.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g0_met.out
Arguments = run.sh --expt_name=5p_gold_g0_met --gcn_depth=0 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32 --opt_met
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g1_met.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g1_met.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g1_met.out
Arguments = run.sh --expt_name=5p_gold_g1_met --gcn_depth=1 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32 --opt_met
Queue 1

Log = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g2_met.log
Error = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g2_met.err
Output = /u/abhinav/Projects/condor_gnlp/logs/5p_gold_g2_met.out
Arguments = run.sh --expt_name=5p_gold_g2_met --gcn_depth=2 --use_rel_words --n_attn=512 --n_ans_gate=512 --n_qi_gate=512 --n_ques_emb=512 --bsz=32 --opt_met
Queue 1
