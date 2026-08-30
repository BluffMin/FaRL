#!/usr/bin/env python3
"""Register a foreground, user-launched continuation run."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess
from pathlib import Path
ROOT=Path('/home/brainlab/FaRL');RESULTS=ROOT/'results/vanilla_rl_baseline_v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-model',required=True);ap.add_argument('--parent-replay',required=True);ap.add_argument('--parent-step',type=int,required=True);ap.add_argument('--target-total-steps',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--run-name',required=True);ap.add_argument('--termination-mode',choices=['success','fixed_horizon'],default='success');args=ap.parse_args()
 active=RESULTS/'active_run.json'
 if active.exists():
  old=json.loads(active.read_text());pid=old.get('training_pid')
  if old.get('status')=='RUNNING' and pid and Path(f'/proc/{pid}').exists():raise SystemExit(f"Refusing concurrent run: {old['run_name']} pid={pid}")
 out=RESULTS/args.run_name;out.mkdir(parents=True,exist_ok=True)
 cmd=[str(ROOT/'.venv_rl/bin/python'),'-u','-m','rl_baselines.train_sac_resume','--parent-model',str(Path(args.parent_model).resolve()),'--parent-replay',str(Path(args.parent_replay).resolve()),'--parent-step',str(args.parent_step),'--target-total-steps',str(args.target_total_steps),'--seed',str(args.seed),'--out',str(out),'--termination-mode',args.termination_mode]
 parent_run=Path(args.parent_model).resolve().parent.parent.name
 rec={'run_name':args.run_name,'algorithm':'SAC','resume_mode':'TRUE_CONTINUATION','parent_run':parent_run,'parent_step':args.parent_step,'target_total_steps':args.target_total_steps,'seed':args.seed,'termination_mode':args.termination_mode,'result_dir':str(out),'command':' '.join(cmd),'started_at':now(),'status':'STARTING','manager_pid':os.getpid()}
 write(active,rec);write(out/'run_status.json',rec);proc=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy());rec.update(status='RUNNING',training_pid=proc.pid);write(active,rec);write(out/'run_status.json',rec)
 try:code=proc.wait();status='COMPLETED' if code==0 else 'FAILED'
 except KeyboardInterrupt:proc.terminate();code=proc.wait();status='INTERRUPTED'
 rec.update(status=status,exit_code=code,finished_at=now());write(active,rec);write(out/'run_status.json',rec)
 with (RESULTS/'run_history.jsonl').open('a') as f:f.write(json.dumps(rec,sort_keys=True)+'\n')
 raise SystemExit(code)
if __name__=='__main__':main()
