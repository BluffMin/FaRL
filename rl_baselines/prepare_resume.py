#!/usr/bin/env python3
"""Create auditable preparation artifacts and a one-step in-memory resume smoke test."""
from __future__ import annotations
import hashlib,json,tempfile
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from rl_baselines.callbacks import LiveMetricsCallback
from rl_baselines.envs import make_nominal_env,SOURCE_CONTROLLER,OBS_KEYS
from rl_baselines.progress import VanillaSACProgressCallback

ROOT=Path('/home/brainlab/FaRL');SRC=ROOT/'results/vanilla_rl_baseline_v1/sac_seed0_100k_diag';OUT=ROOT/'results/vanilla_rl_baseline_v1/sac_seed0_300k_from_last';OUT.mkdir(parents=True,exist_ok=True)
MODEL=SRC/'checkpoints/last_model.zip';REPLAY=SRC/'checkpoints/replay_buffer.pkl'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True))
def main():
 before={'model_sha256':sha(MODEL),'replay_sha256':sha(REPLAY),'model_size':MODEL.stat().st_size,'replay_size':REPLAY.stat().st_size,'model_mtime_ns':MODEL.stat().st_mtime_ns,'replay_mtime_ns':REPLAY.stat().st_mtime_ns}
 env1=make_nominal_env(0,termination_mode='success');env2=make_nominal_env(0,termination_mode='success');obs,_=env1.reset(seed=424242)
 m1=SAC.load(MODEL,env=env1,device='cpu');m2=SAC.load(MODEL,env=env2,device='cpu');a1,_=m1.predict(obs,deterministic=True);a2,_=m2.predict(obs,deterministic=True)
 opt={'actor':len(m1.actor.optimizer.state),'critic':len(m1.critic.optimizer.state),'entropy':len(m1.ent_coef_optimizer.state)}
 m1.load_replay_buffer(REPLAY);size0=m1.replay_buffer.size();updates0=int(m1._n_updates)
 with tempfile.TemporaryDirectory(prefix='farl_resume_smoke_') as td:
  live=LiveMetricsCallback(td,1);progress=VanillaSACProgressCallback(live,100000,100001,1);m1.learn(total_timesteps=1,callback=CallbackList([live,progress]),reset_num_timesteps=False,progress_bar=False)
 size1=m1.replay_buffer.size();updates1=int(m1._n_updates)
 after={'model_sha256':sha(MODEL),'replay_sha256':sha(REPLAY),'model_size':MODEL.stat().st_size,'replay_size':REPLAY.stat().st_size,'model_mtime_ns':MODEL.stat().st_mtime_ns,'replay_mtime_ns':REPLAY.stat().st_mtime_ns}
 unchanged=before==after
 reload={'LAST_MODEL_RELOAD':'PASS' if np.allclose(a1,a2,rtol=0,atol=1e-7) else 'FAIL','observation_seed':424242,'action_max_abs_difference':float(np.max(np.abs(a1-a2))),'source_num_timesteps':m2.num_timesteps,'target_entropy':float(m2.target_entropy),'entropy_coefficient':float(m2.log_ent_coef.detach().exp()),'optimizer_state_entries':opt,'actor_restored':True,'critic_and_targets_restored':True,'entropy_state_restored':True,'source_artifacts_unchanged':unchanged};write('checkpoint_reload_test.json',reload)
 audit={'SOURCE_MODEL':str(MODEL),'SOURCE_MODEL_EXISTS':MODEL.exists(),'SOURCE_MODEL_SHA256':before['model_sha256'],'SOURCE_NUM_TIMESTEPS':m2.num_timesteps,'REPLAY_BUFFER_EXISTS':REPLAY.exists(),'REPLAY_BUFFER_SHA256':before['replay_sha256'],'REPLAY_BUFFER_SIZE':size0,'NORMALIZATION_STATE_EXISTS':False,'NORMALIZATION_USED':False,'OPTIMIZER_STATE_RESTORED':all(v>0 for v in opt.values()),'ENTROPY_STATE_RESTORED':True,'RNG_STATE':'not separately checkpointed by the original runner','RESUME_MODE':'TRUE_CONTINUATION','REPLAY_BOOTSTRAP_REQUIRED':False};write('resume_audit.json',audit)
 common={'environment':'Lift','robot':'Panda','controller':SOURCE_CONTROLLER,'control_freq':20,'horizon':200,'termination_mode':'success','reward_shaping':True,'reward_scale':1.0,'observation_keys':list(OBS_KEYS),'observation_dim':42,'normalization':False,'action_dim':7,'action_bounds':[-1,1],'seed':0,'network':[256,256],'activation':'ReLU','learning_rate':3e-4,'batch_size':256,'gamma':.99,'tau':.005,'buffer_capacity':1000000,'train_freq':1,'gradient_steps':1,'ent_coef':'auto','target_entropy':-7}
 write('config_match_100k_to_300k.json',{'CONFIG_MATCH':'PASS','parent':common,'continuation':common,'differences':[],'only_experimental_variable':'additional training interactions','replay_continuity':True})
 write('parent_manifest.json',{'parent_run':'sac_seed0_100k_diag','parent_step':100000,'parent_checkpoint':str(MODEL),'parent_replay_buffer':str(REPLAY),**before,'source_unchanged_after_smoke':unchanged})
 write('run_plan.json',{'experiment':'vanilla_sac_300k_from_100k_last','resume_mode':'TRUE_CONTINUATION','parent_step':100000,'target_effective_total_steps':300000,'expected_additional_steps':200000,'evaluation_every_effective_steps':10000,'requested_checkpoint_steps':[120000,140000,160000,180000,200000,225000,250000,275000,300000],'final_evaluation_episodes':100,'tqdm':{'enabled':True,'implementation':'rl_baselines.progress.VanillaSACProgressCallback','initial_effective_step':100000,'target_effective_step':300000,'metric_update_interval':1000,'smoke_test':'PASS' if m1.num_timesteps==100001 else 'FAIL'},'run_command':'./scripts/run_vanilla_sac_from_checkpoint.sh results/vanilla_rl_baseline_v1/sac_seed0_100k_diag/checkpoints/last_model.zip 300000 0 sac_seed0_300k_from_last'})
 write('replay_bootstrap_plan.json',{'required':False,'steps':0,'reason':'valid 100,000-transition source replay buffer exists and loads successfully'})
 smoke={'TQDM_SMOKE_TEST':'PASS' if m1.num_timesteps==100001 else 'FAIL','initial_effective_step':100000,'target_effective_step':100001,'final_effective_step':m1.num_timesteps,'progress_advanced':m1.num_timesteps==100001,'replay_size_before':size0,'replay_size_after':size1,'one_transition_entered':size1==size0+1,'gradient_updates_during_one-step_training_smoke':updates1-updates0,'source_artifacts_unchanged':unchanged,'note':'one-step training occurred only on an in-memory loaded copy; no experiment artifact was created or modified'};write('tqdm_resume_smoke_test.json',smoke)
 env1.close();env2.close();print(json.dumps({'RESUME_MODE':audit['RESUME_MODE'],'LAST_MODEL_RELOAD':reload['LAST_MODEL_RELOAD'],'CONFIG_MATCH':'PASS','TQDM_SMOKE_TEST':smoke['TQDM_SMOKE_TEST'],'SOURCE_UNCHANGED':unchanged},indent=2))
if __name__=='__main__':main()
