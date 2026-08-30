#!/usr/bin/env python3
"""Register a user-started training run while preserving interactive output."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/home/brainlab/FaRL');RESULTS=ROOT/'results/vanilla_rl_baseline_v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--run-name',required=True);ap.add_argument('--termination-mode',choices=['success','fixed_horizon'],default='success');ap.add_argument('--reward-mode',choices=['control','signed_lift_progress'],default='control');args=ap.parse_args()
 active=RESULTS/'active_run.json'
 if active.exists():
  old=json.loads(active.read_text());pid=old.get('training_pid')
  if old.get('status')=='RUNNING' and pid and Path(f'/proc/{pid}').exists():raise SystemExit(f"Refusing concurrent run: {old['run_name']} pid={pid}")
 out=RESULTS/args.run_name;out.mkdir(parents=True,exist_ok=True)
 cmd=[str(ROOT/'.venv_rl/bin/python'),'-u','-m','rl_baselines.train_sac','--steps',str(args.steps),'--seed',str(args.seed),'--out',str(out),'--progress','--termination-mode',args.termination_mode,'--reward-mode',args.reward_mode]
 rec={'run_name':args.run_name,'algorithm':'SAC','steps_requested':args.steps,'seed':args.seed,'termination_mode':args.termination_mode,'reward_mode':args.reward_mode,'result_dir':str(out),'command':' '.join(cmd),'started_at':now(),'status':'STARTING','manager_pid':os.getpid()}
 write(RESULTS/'active_run.json',rec);write(out/'run_status.json',rec)
 proc=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy());rec.update(status='RUNNING',training_pid=proc.pid);write(RESULTS/'active_run.json',rec);write(out/'run_status.json',rec)
 try:code=proc.wait();status='COMPLETED' if code==0 else 'FAILED'
 except KeyboardInterrupt:
  proc.terminate();code=proc.wait();status='INTERRUPTED'
 rec.update(status=status,exit_code=code,finished_at=now());write(RESULTS/'active_run.json',rec);write(out/'run_status.json',rec)
 history=RESULTS/'run_history.jsonl'
 with history.open('a') as f:f.write(json.dumps(rec,sort_keys=True)+'\n')
 raise SystemExit(code)
if __name__=='__main__':main()
