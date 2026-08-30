# Success-transition bootstrap audit

Control success is a true terminal: sampled done=1, so the target is the immediate reward. Fixed-horizon success before step 200 is nonterminal: sampled done=0, so SAC bootstraps. Horizon truncation is masked as timeout and bootstraps in both arms. Fixed horizon also repeats reward 1 while success persists; the treatment therefore changes both terminal bootstrapping and post-success state/reward occupancy, although the reward function itself is identical.
