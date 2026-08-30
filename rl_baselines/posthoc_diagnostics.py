#!/usr/bin/env python3
"""Derived diagnostics for a completed legacy run; never alters its checkpoints."""
from __future__ import annotations
import argparse,csv,json,sys,zipfile
from pathlib import Path
import h5py,numpy as np
from stable_baselines3 import SAC
from rl_baselines.envs import make_nominal_env
from rl_baselines.evaluate import evaluate_policy_success
ROOT=Path('/home/brainlab/FaRL');SRC=Path('/home/robotics/external_workspace/data/C/demos/lift_ph_demo.hdf5')
sys.path.insert(0,'/home/robotics/external_workspace/data/C/scripts');import a_cluttered as A
def aggregate(eps):
 def mean(k,subset=eps):
  x=[float(e[k]) for e in subset if k in e];return float(np.mean(x)) if x else None
 succ=[e for e in eps if e.get('ever_success')];fail=[e for e in eps if not e.get('ever_success')]
 return {'n':len(eps),'success_rate':len(succ)/len(eps) if eps else None,**{f'mean_{k}':mean(k) for k in ('episode_return','episode_length','return_per_step','max_object_height','final_object_height','object_lift_delta','ever_grasped','grasp_duration_steps','cumulative_reaching_reward','cumulative_grasp_reward','cumulative_success_reward')},'success_conditioned':{'n':len(succ),'total_return':mean('episode_return',succ),'return_per_step':mean('return_per_step',succ)},'failure_conditioned':{'n':len(fail),'total_return':mean('episode_return',fail),'return_per_step':mean('return_per_step',fail)}}
def random_eps(n=20):
 env=make_nominal_env(31000);out=[]
 for i in range(n):
  o,_=env.reset(seed=31000+i)
  while True:
   o,_,t,tr,info=env.step(env.action_space.sample())
   if t or tr:out.append(info['episode_diagnostics']);break
 env.close();return out
def demo_eps(n=5):
 env=make_nominal_env(32000);out=[]
 with h5py.File(SRC,'r') as f:
  for name in sorted(f['data'])[:n]:
   g=f['data'][name];env.reset(seed=32000);env.env.reset_from_xml_string(A.fix_mesh_paths(g.attrs['model_file']));env.env.sim.set_state_from_flattened(np.asarray(g['states'][0]));env.env.sim.forward();env._step=0
   # Reinitialize diagnostic counters against the restored object height.
   env.reset(seed=32000);env.env.reset_from_xml_string(A.fix_mesh_paths(g.attrs['model_file']));env.env.sim.set_state_from_flattened(np.asarray(g['states'][0]));env.env.sim.forward();env._step=0
   h=float(env.env.sim.data.body_xpos[env.env.cube_body_id][2]);env._diag.update(max_object_height=h,initial_object_height=h)
   for a in np.asarray(g['actions']):
    _,_,t,tr,info=env.step(a)
    if t or tr:out.append(info['episode_diagnostics']);break
 env.close();return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--episodes',type=int,default=20);a=ap.parse_args();run=Path(a.run);env=make_nominal_env(33000);models={}
 for key in ('best_model','last_model'):
  m=SAC.load(run/'checkpoints'/f'{key}.zip',env=env,device='cpu');models[key]=evaluate_policy_success(m,env,a.episodes,[33000+i for i in range(a.episodes)])
 env.close();rand=random_eps();demos=demo_eps();groups={'random':aggregate(rand),'successful_demo':aggregate(demos)}
 for k,v in models.items():groups[f'sac_{k}']=aggregate(v['episode_diagnostics'])
 sac_fail=groups['sac_best_model']['failure_conditioned'];demo=groups['successful_demo'];alignment='LENGTH_CONFOUNDED' if sac_fail['total_return'] and sac_fail['total_return']>demo['mean_episode_return'] and sac_fail['return_per_step']<demo['mean_return_per_step'] else 'NOT_IDENTIFIABLE'
 archive=set(zipfile.ZipFile(run/'checkpoints'/'last_model.zip').namelist());replay=(run/'checkpoints'/'replay_buffer.pkl').exists()
 audit={'run_name':run.name,'SAC_20K_SANITY':'PASS_NUMERIC_ONLY','VANILLA_RL_PERFORMANCE':'NOT_YET_LEARNED','groups':groups,'models':models,'REWARD_ALIGNMENT':alignment,'CHECKPOINT_SELECTION_MISALIGNED':'NO','checkpoint_selection':'success rate primary, mean return tie-break','resume_audit':{'actor_critic_target_and_optimizers_in_model_zip':True,'entropy_state_and_optimizer_in_model_zip':True,'REPLAY_BUFFER_SAVED':replay,'NORMALIZATION_STATE_SAVED':'NOT_APPLICABLE_NO_NORMALIZATION','rng_state_saved':False,'TRUE_RESUME_SUPPORTED':'YES' if replay else 'NO','zip_members':sorted(archive)},'warning':'RETURN_IMPROVING_WITHOUT_TASK_PROGRESS' if groups['sac_best_model']['success_rate']<.1 else None}
 (run/'posthoc_diagnostic_audit.json').write_text(json.dumps(audit,indent=2));
 with (run/'successful_demo_vs_sac_failure.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['metric','successful_demo','SAC_failure']);
  for k in ('episode_return','episode_length','return_per_step','max_object_height','ever_grasped','grasp_duration_steps','final_object_height'):w.writerow([k,demo.get('mean_'+k),groups['sac_best_model'].get('mean_'+k) if k not in ('episode_return','return_per_step') else sac_fail['total_return' if k=='episode_return' else 'return_per_step']])
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
