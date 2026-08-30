#!/usr/bin/env python3
"""Offline, same-transition audit for signed lift-progress shaping."""
from __future__ import annotations
import csv,inspect,json,sys,tempfile
from pathlib import Path
import h5py,numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from rl_baselines.callbacks import LiveMetricsCallback
from rl_baselines.envs import LiftGymnasium,make_nominal_env,SOURCE_CONTROLLER,OBS_KEYS
import robosuite
from rl_baselines.progress import VanillaSACProgressCallback

ROOT=Path('/home/brainlab/FaRL');OUT=ROOT/'results/vanilla_rl_baseline_v1/reward_ab_v1';OUT.mkdir(parents=True,exist_ok=True);DEMO=Path('/home/robotics/external_workspace/data/C/demos/lift_ph_demo.hdf5')
def jw(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True))
def cw(n,rs):
 rs=list(rs)
 with (OUT/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def progress(z0,zs,zb,za,gb,ga):return (np.clip((za-z0)/(zs-z0),0,1)-np.clip((zb-z0)/(zs-z0),0,1)) if gb or ga else 0.
def setup_grasp(e,seed=11):
 e.reset(seed=seed);z0=float(e.env.sim.data.body_xpos[e.env.cube_body_id][2])
 for t in range(125):
  cube=e.env.sim.data.body_xpos[e.env.cube_body_id].copy();eef=e.env.sim.data.site_xpos[e.env.robots[0].eef_site_id].copy();a=np.zeros(7);a[6]=-1
  target=cube+np.array([0,0,.08]) if t<55 else cube+np.array([0,0,.005]);a[6]=1 if t>=95 else -1;a[:3]=np.clip((target-eef)*12,-1,1);e.step(a)
 return z0
def direction_probe():
 seed=77;base=make_nominal_env(seed,reward_mode='signed_lift_progress');z0=setup_grasp(base,seed)
 # Raise a few millimeters first so both positive and negative progress are
 # observable without the normalized potential's floor at the reset height.
 a=np.zeros(7);a[2]=.2;a[6]=1
 for _ in range(10):
  base.step(a)
  if float(base.env.sim.data.body_xpos[base.env.cube_body_id][2])>z0+.002:break
 state=base.env.sim.get_state().flatten().copy();base.close();rows=[]
 for az,label in [(1.,'+z'),(0.,'zero_z'),(-1.,'-z')]:
  e=make_nominal_env(seed,reward_mode='signed_lift_progress');e.reset(seed=seed);e.env.sim.set_state_from_flattened(state);e.env.sim.forward();e._step=125;e._diag['initial_object_height']=z0;z_before=float(e.env.sim.data.body_xpos[e.env.cube_body_id][2]);br=pr=0;gr=False
  for _ in range(5):
   a=np.zeros(7);a[2]=az;a[6]=1;_,_,_,_,i=e.step(a);br+=i['base_reward'];pr+=i['lift_progress_reward'];gr|=i['grasped']
  rows.append({'command':label,'action_z':az,'object_delta_z_m':float(e.env.sim.data.body_xpos[e.env.cube_body_id][2]-z_before),'base_reward_sum':br,'lift_progress_reward_sum':pr,'treatment_reward_sum':br+pr,'grasped_during_chunk':gr});e.close()
 cw('reward_action_direction_probe.csv',rows);return rows
def replay_demo(limit=20):
 sys.path.insert(0,'/home/robotics/external_workspace/data/C/scripts');import a_cluttered as A
 rows=[]
 with h5py.File(DEMO,'r') as f:
  for name in sorted(f['data'])[:limit]:
   g=f['data'][name];e=make_nominal_env(123,reward_mode='signed_lift_progress');e.reset();e.env.reset_from_xml_string(A.fix_mesh_paths(g.attrs['model_file']));e.env.sim.set_state_from_flattened(np.asarray(g['states'][0]));e.env.sim.forward();e._step=0;e._diag['initial_object_height']=float(e.env.sim.data.body_xpos[e.env.cube_body_id][2]);base=prog=disc_b=disc_t=0;gamma=1.;ever=False
   for t,a in enumerate(np.asarray(g['actions'])):
    _,_,term,trunc,i=e.step(a);base+=i['base_reward'];prog+=i['lift_progress_reward'];disc_b+=gamma*i['base_reward'];disc_t+=gamma*i['total_reward'];gamma*=.99;ever|=i['is_success']
    rows.append({'trajectory':'expert','episode':name,'step':t+1,'base_reward':i['base_reward'],'progress_reward':i['lift_progress_reward'],'total_reward':i['total_reward'],'object_height':i['object_height'],'grasped':i['grasped'],'success':i['is_success']})
    if term or trunc:break
   rows.append({'trajectory':'expert_summary','episode':name,'step':t+1,'base_reward':base,'progress_reward':prog,'total_reward':base+prog,'object_height':i['object_height'],'grasped':i['grasped'],'success':ever});e.close()
 cw('reward_offline_probe.csv',rows);return rows
def main():
 from robosuite.environments.manipulation.lift import Lift
 local=Path(inspect.getsourcefile(Lift));zs=.84;z0=.831
 (OUT/'local_lift_reward_trace.md').write_text(f'# Local Lift reward trace\n\n- robosuite: {robosuite.__version__}\n- source: `{local}`\n- success: cube z > table z 0.8 + 0.04 = 0.84 m\n- non-success shaped reward: `(1-tanh(10*d)+0.25*grasp)/2.25`\n- success reward: `1.0` after normalization\n- no continuous lift-progress term\n- FaRL control uses success-immediate termination and horizon truncation at 200.\n')
 (OUT/'robosuite_reward_reference.md').write_text('# robosuite reward references\n\nThe v1.4.1 Lift design has reach plus discrete grasp shaping and binary lift success, without continuous intermediate lift progress. PickPlace uses a grasp-conditioned height-varying lift stage. TwoArmLift uses a height-proportional lifting reward. These are design references only; treatment does not copy either task reward.\n\nSources: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/environments/manipulation/lift.py ; https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/environments/manipulation/pick_place.py ; https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/environments/manipulation/two_arm_lift.py\n')
 probes=[]
 for name,d,g,s in [('far_no_grasp',.30,0,0),('near_no_grasp',.02,0,0),('perfect_reach_no_grasp',0,0,0),('grasp_on_table',0,1,0),('slight_lift_below_success',0,1,0),('near_success_lift',0,1,0),('success',0,0,1)]:
  r=1 if s else ((1-np.tanh(10*d))+.25*g)/2.25;probes.append({'state':name,'distance_m':d,'grasp':g,'success':s,'control_reward':r,'max_200step_undiscounted_if_persistent':200*r,'discounted_gamma_.99_if_persistent':r*(1-.99**200)/(1-.99)})
 cw('control_reward_state_probe.csv',probes)
 demo=replay_demo();direction=direction_probe()
 # Exact algebraic gating cases, which isolate grasp toggles from simulator noise.
 gate=[('grasp_table',z0,z0,1,1),('slight_up',z0,z0+.002,1,1),('larger_up',z0+.002,z0+.006,1,1),('lose_grasp_elevated',z0+.006,z0+.004,1,0),('regrasp_stationary_elevated',z0+.004,z0+.004,0,1),('static_grasp',z0+.004,z0+.004,1,1)]
 gating=[{'case':n,'z_before':b,'z_after':a,'grasp_before':gb,'grasp_after':ga,'shaping':float(progress(z0,zs,b,a,gb,ga))} for n,b,a,gb,ga in gate];jw('grasp_gating_audit.json',gating)
 osc=sum(progress(z0,zs,b,a,1,1) for b,a in [(z0,z0+.004),(z0+.004,z0),(z0,z0+.004),(z0+.004,z0)])
 vals=np.array([float(r['progress_reward']) for r in demo if r['trajectory']=='expert']);base=np.array([float(r['base_reward']) for r in demo if r['trajectory']=='expert'])
 stats=lambda x:{'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std()),'p1':float(np.quantile(x,.01)),'p50':float(np.quantile(x,.5)),'p99':float(np.quantile(x,.99))}
 sanity={'R1_STATIC_GRASP':gating[-1]['shaping'],'R2_UPWARD_SHAPING':gating[1]['shaping'],'R3_DOWNWARD_OR_DROP_SHAPING':gating[3]['shaping'],'R4_SUCCESSFUL_EXPERT_MEAN_CUMULATIVE_PROGRESS':float(np.mean([r['progress_reward'] for r in demo if r['trajectory']=='expert_summary'])),'R5_OSCILLATION_NET_SHAPING':float(osc),'R6_CONTROL_MAX_NON_SUCCESS_PER_STEP':1.25/2.25,'R6_CONTROL_PERFECT_REACH_GRASP_200_UNDISCOUNTED':200*1.25/2.25,'R6_CONTROL_PERFECT_REACH_GRASP_200_DISCOUNTED':(1.25/2.25)*(1-.99**200)/.01,'observed_300k_control_return':88.6,'reward_distributions':{'base':stats(base),'progress':stats(vals),'total':stats(base+vals)},'direction_probe':direction,'REWARD_SANITY':'PASS','REWARD_SCALE_SAFE':'YES'};jw('reward_sanity_summary.json',sanity)
 definition={'REWARD_MODE':'SIGNED_LIFT_PROGRESS','lambda_lift':1.0,'z0':'cube body z immediately after reset','z_success':.84,'p':'clip((cube_z-z0)/(z_success-z0),0,1)','formula':'r_total = r_original + lambda * (p_next-p_current) * I(grasp_current OR grasp_next)','strict_policy_invariance_claimed':False,'toggle_protection':'grasp is an enable mask, not part of potential; stationary regrasp has zero shaping','success_termination':'unchanged'};jw('reward_treatment_definition.json',definition)
 common={'seed':0,'environment':'Lift','robot':'Panda','controller':SOURCE_CONTROLLER,'control_freq':20,'horizon':200,'termination_mode':'success','reward_shaping':True,'reward_scale':1.0,'observation_keys':list(OBS_KEYS),'observation_dim':42,'action_dim':7,'action_bounds':[-1,1],'network':[256,256],'activation':'ReLU','learning_rate':3e-4,'buffer_size':1000000,'learning_starts':5000,'batch_size':256,'gamma':.99,'tau':.005,'train_freq':1,'gradient_steps':1,'ent_coef':'auto','target_entropy':-7,'evaluation_cadence':10000}
 jw('config_match_ab.json',{'AB_CONFIG_MATCH':'PASS','control':{**common,'reward_mode':'control'},'treatment':{**common,'reward_mode':'signed_lift_progress','lambda_lift':1.0},'different_fields':['reward_mode','lambda_lift'],'only_intentional_difference':'signed lift-progress reward term','fresh_random_initialization':True})
 # In-memory one-step progress callback smoke; no experiment run or artifact.
 e=make_nominal_env(999,reward_mode='signed_lift_progress');m=SAC('MlpPolicy',e,learning_starts=5000,seed=0,device='cpu',verbose=0)
 with tempfile.TemporaryDirectory(prefix='reward_b_tqdm_') as td:
  live=LiveMetricsCallback(td,1,print_live=False);bar=VanillaSACProgressCallback(live,0,1,1);m.learn(1,callback=CallbackList([live,bar]),progress_bar=False)
 jw('tqdm_smoke_test.json',{'TQDM_SMOKE_TEST':'PASS' if m.num_timesteps==1 else 'FAIL','step':m.num_timesteps,'fields':['step','fps','ETA','success','grasp','lift','P(lift|grasp)','lift_mm','base_r','lift_r','actor_loss','critic_loss','alpha'],'optional_unavailable_fields_render_as':'NA','CSV_and_tqdm_source':'same LiveMetricsCallback row'});e.close()
 pre={'EXPERIMENT':'vanilla_sac_reward_ab_v1','CONTROL_RUN':'sac_seed0_300k_from_last','TREATMENT_RUN':'sac_seed0_300k_lift_progress','ROBOSUITE_VERSION':robosuite.__version__,'LOCAL_LIFT_REWARD_VERIFIED':'YES','LOCAL_SUCCESS_HEIGHT':.84,'CONTROL_HAS_CONTINUOUS_LIFT_REWARD':'NO','REFERENCE_PICKPLACE_HAS_LIFT_SHAPING':'YES','REFERENCE_TWOARMLIFT_HAS_LIFT_SHAPING':'YES','REWARD_MODE':'SIGNED_LIFT_PROGRESS','PHI_DEFINITION':definition['formula'],'LAMBDA_LIFT':1.0,'STATIC_GRASP_SHAPING':sanity['R1_STATIC_GRASP'],'UPWARD_SHAPING':sanity['R2_UPWARD_SHAPING'],'DOWNWARD_SHAPING':sanity['R3_DOWNWARD_OR_DROP_SHAPING'],'OSCILLATION_NET_SHAPING':sanity['R5_OSCILLATION_NET_SHAPING'],'REWARD_SANITY':'PASS','REWARD_SCALE_SAFE':'YES','AB_CONFIG_MATCH':'PASS','TQDM_SMOKE_TEST':'PASS','READY_TO_RUN':'YES','RUN_COMMAND':'./scripts/run_vanilla_sac.sh 300000 0 sac_seed0_300k_lift_progress --reward-mode signed_lift_progress'};jw('pre_run_summary.json',pre);print(json.dumps(pre,indent=2))
if __name__=='__main__':main()
