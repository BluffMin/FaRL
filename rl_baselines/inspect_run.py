#!/usr/bin/env python3
"""Print the latest registered run and its newest persisted metrics."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
ROOT=Path('/home/brainlab/FaRL');BASE=ROOT/'results/vanilla_rl_baseline_v1'
def last_csv(p):
 if not p.exists():return None
 rows=list(csv.DictReader(p.open()));return rows[-1] if rows else None
def rows_csv(p):return list(csv.DictReader(p.open())) if p.exists() else []
def num(x,k):
 try:return float(x.get(k,''))
 except:return None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--compact',action='store_true');args=ap.parse_args()
 active=BASE/'active_run.json'
 if not active.exists():print(json.dumps({'status':'NO_REGISTERED_RUN'},indent=2));return
 run=json.loads(active.read_text());out=Path(run['result_dir']);pid=run.get('training_pid');alive=bool(pid and Path(f'/proc/{pid}').exists())
 live=rows_csv(out/'live_metrics.csv');ev=rows_csv(out/'learning_curve.csv');latest=live[-1] if live else None;latest_ev=ev[-1] if ev else None
 successes=[num(x,'success_rate') for x in ev];heights=[num(x,'mean_max_object_height') for x in ev];ent=[num(x,'entropy_coefficient') for x in live];returns=[num(x,'rolling_return') for x in live]
 progress=[num(x,'rolling_object_lift_delta') for x in live];grasp=[num(x,'rolling_grasp_rate') for x in live]
 valid_success=[x for x in successes if x is not None]
 warn=bool(len(returns)>1 and returns[-1] is not None and returns[0] is not None and returns[-1]>returns[0] and valid_success and max(valid_success)<.1 and not any(x and x>0 for x in progress+grasp))
 result={'registry':run,'process_alive':alive,'latest_live_metrics':latest,'latest_evaluation':latest_ev,'best_eval_success':max((x for x in successes if x is not None),default=None),'latest_eval_success':successes[-1] if successes else None,'best_max_object_height':max((x for x in heights if x is not None),default=None),'latest_max_object_height':heights[-1] if heights else None,'grasp_rate_trend':[x for x in grasp if x is not None],'entropy_trend':[x for x in ent if x is not None],'warning':'RETURN_IMPROVING_WITHOUT_TASK_PROGRESS' if warn else None,'summary':json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None,'checkpoints':sorted(p.name for p in (out/'checkpoints').glob('*.zip')) if (out/'checkpoints').exists() else [],'replay_buffer_checkpoint_exists':(out/'checkpoints'/'replay_buffer.pkl').exists(),'TRUE_RESUME_SUPPORTED':bool((out/'checkpoints'/'last_model.zip').exists() and (out/'checkpoints'/'replay_buffer.pkl').exists())}
 if args.compact:
  l=latest or {};e=latest_ev or {}
  print(f"RUN: {run['run_name']}  PROCESS: {'RUNNING' if alive else run.get('status','STOPPED')}  STEP: {l.get('step','NA')}")
  print(f"TRAIN return={l.get('rolling_return','NA')} success={l.get('rolling_success_rate','NA')} grasp={l.get('rolling_grasp_rate','NA')} max_h={l.get('rolling_max_object_height','NA')} ep_len={l.get('rolling_episode_length','NA')}")
  print(f"SAC actor={l.get('actor_loss','NA')} critic={l.get('critic_loss','NA')} alpha={l.get('entropy_coefficient','NA')} saturation={l.get('action_saturation_fraction','NA')}")
  print(f"EVAL n={e.get('episodes','NA')} success={e.get('success_rate','NA')} return={e.get('mean_return','NA')} return/step={e.get('mean_return_per_step','NA')} grasp={e.get('grasp_rate','NA')} max_h={e.get('mean_max_object_height','NA')}")
  if result['warning']:print('WARNING:',result['warning'])
 else:print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
