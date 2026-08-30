#!/usr/bin/env python3
"""Validate and document (but do not launch) the fixed-horizon 100k->300k continuation."""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC

ROOT=Path('/home/brainlab/FaRL'); BASE=ROOT/'results/vanilla_rl_baseline_v1'
SRC=BASE/'sac_seed0_100k_fixed_horizon'; OUT=BASE/'termination_ab_v2'
MODEL=SRC/'checkpoints/last_model.zip'; REPLAY=SRC/'checkpoints/replay_buffer.pkl'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True))
def main():
 before={str(p):sha(p) for p in [MODEL,REPLAY]}; model=SAC.load(MODEL,device='cpu'); model.load_replay_buffer(REPLAY)
 rb=model.replay_buffer; optimizer_states={n:len(getattr(model,n).optimizer.state) for n in ['actor','critic']}; optimizer_states['entropy']=len(model.ent_coef_optimizer.state)
 with zipfile.ZipFile(MODEL) as z:members=sorted(z.namelist())
 valid=model.num_timesteps==100000 and rb.size()==100000 and all(v>0 for v in optimizer_states.values()) and model.log_ent_coef is not None
 after={str(p):sha(p) for p in [MODEL,REPLAY]}; unchanged=before==after
 audit={'SOURCE_MODEL':str(MODEL),'SOURCE_REPLAY':str(REPLAY),'SOURCE_NUM_TIMESTEPS':model.num_timesteps,'REPLAY_BUFFER_SIZE':rb.size(),'REPLAY_DONE_FRACTION':float(np.asarray(rb.dones[:rb.size()]).mean()),'REPLAY_TIMEOUT_FRACTION':float(np.asarray(rb.timeouts[:rb.size()]).mean()),'OPTIMIZER_STATE_ENTRIES':optimizer_states,'ENTROPY_STATE_RESTORED':model.log_ent_coef is not None,'MODEL_ZIP_MEMBERS':members,'SOURCE_SHA256_BEFORE':before,'SOURCE_SHA256_AFTER':after,'SOURCE_ARTIFACTS_UNCHANGED':unchanged,'TRUE_CONTINUATION_READY':'YES' if valid and unchanged else 'NO','RNG_LIMITATION':'Python/NumPy/environment RNG streams are not separately checkpointed; model/replay/optimizers/entropy/timestep are restored.'};dump('fixed_horizon_continuation_audit.json',audit)
 plan={'status':'PREPARED_NOT_STARTED','parent_run':SRC.name,'parent_step':100000,'target_total_steps':300000,'additional_steps':200000,'seed':0,'termination_mode':'fixed_horizon','reward_mode':'control','horizon':200,'evaluation_episodes':20,'evaluation_frequency':10000,'requested_checkpoint_steps':[120000,140000,160000,180000,200000,225000,250000,275000,300000],'final_paired_evaluation_episodes':100,'final_seeds':'900000..900099','launch_command':'./scripts/run_fixed_horizon_300k_continuation.sh','monitor_command':'./scripts/watch_vanilla_sac.sh sac_seed0_300k_fixed_horizon_from_100k','expected_wall_time_based_on_100k_run':'approximately 12-15 hours including evaluation','long_run_started':False};dump('fixed_horizon_300k_continuation_plan.json',plan)
 print(json.dumps({'TRUE_CONTINUATION_READY':audit['TRUE_CONTINUATION_READY'],'launch_command':plan['launch_command'],'monitor_command':plan['monitor_command']},indent=2))
if __name__=='__main__':main()
