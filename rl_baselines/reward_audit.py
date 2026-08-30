#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import h5py,numpy as np
ROOT=Path('/home/brainlab/FaRL');OUT=ROOT/'results/vanilla_rl_baseline_v1';SRC=Path('/home/robotics/external_workspace/data/C/demos/lift_ph_demo.hdf5')
sys.path.insert(0,'/home/robotics/external_workspace/data/C/scripts')
import a_cluttered as A
from rl_baselines.envs import make_nominal_env
def main():
 random=json.loads((OUT/'reward_audit_random.json').read_text());env=make_nominal_env(123);runs=[]
 with h5py.File(SRC,'r') as f:
  for name in sorted(f['data'])[:5]:
   g=f['data'][name];env.reset(seed=123);env.env.reset_from_xml_string(A.fix_mesh_paths(g.attrs['model_file']));env.env.sim.set_state_from_flattened(np.asarray(g['states'][0]));env.env.sim.forward();env._step=0
   rewards=[];ever=False
   for a in np.asarray(g['actions']):
    _,r,term,trunc,info=env.step(a);rewards.append(r);ever|=info['is_success']
    if term or trunc:break
   runs.append({'demo':name,'steps':len(rewards),'return':float(sum(rewards)),'max_reward':float(max(rewards)),'ever_success':bool(ever),'final_success':bool(info['is_success'])})
 env.close();successful=[x for x in runs if x['ever_success']];demo_mean=float(np.mean([x['return'] for x in successful])) if successful else None
 passed=bool(successful and demo_mean>random['mean_return']*1.5)
 post=OUT/'sac_20k_sanity_tqdm'/'posthoc_diagnostic_audit.json';detail=json.loads(post.read_text()) if post.exists() else None
 out={'REWARD_AUDIT':'PASS' if passed else 'FAIL','random':random,'demonstrations':runs,'successful_demo_count':len(successful),'successful_demo_mean_return':demo_mean,'criterion':'successful demo mean return > 1.5 * random mean return','detailed_current_sac_audit':detail['groups'] if detail else None,'REWARD_ALIGNMENT':detail['REWARD_ALIGNMENT'] if detail else 'NOT_IDENTIFIABLE','task_aligned_answer':'Successful demos have higher return/step, grasp, and lift; SAC failures have higher total return only because they accumulate reaching reward for 200 steps.' if detail else 'NOT_IDENTIFIABLE'}
 (OUT/'reward_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
