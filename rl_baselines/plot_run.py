#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
def read(p):return list(csv.DictReader(Path(p).open()))
def col(rows,k):return np.asarray([float(r[k]) if r.get(k) not in ('',None) else np.nan for r in rows])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();p=Path(a.run);fig=p/'figures';fig.mkdir(exist_ok=True)
 live=read(p/'live_metrics.csv');ev=read(p/'learning_curve.csv');x=col(live,'step');xe=col(ev,'step')
 plots=[('training_return.png','Rolling training return','rolling_return','Return'),('training_success_rate.png','Rolling training success rate','rolling_success_rate','Success rate')]
 for name,title,key,ylabel in plots:
  f,ax=plt.subplots(figsize=(7,4));ax.plot(x,col(live,key),marker='o',label='training rolling');
  if key=='rolling_success_rate':ax.plot(xe,col(ev,'success_rate'),marker='s',label='deterministic eval (n=20)');ax.set_ylim(-.02,1.02)
  ax.set(title=title,xlabel='Environment steps',ylabel=ylabel);ax.grid(alpha=.25);ax.legend();f.tight_layout();f.savefig(fig/name,dpi=180);plt.close(f)
 f,ax=plt.subplots(figsize=(7,4));ax.plot(x,col(live,'actor_loss'),label='actor loss');ax.plot(x,col(live,'critic_loss'),label='critic loss');ax.set(xlabel='Environment steps',ylabel='Loss',title='SAC losses');ax.grid(alpha=.25);ax.legend();f.tight_layout();f.savefig(fig/'training_losses.png',dpi=180);plt.close(f)
 f,ax=plt.subplots(figsize=(7,4));ax.plot(x,col(live,'entropy_coefficient'),marker='o');ax.set(xlabel='Environment steps',ylabel='Entropy coefficient',title='SAC automatic entropy coefficient');ax.grid(alpha=.25);f.tight_layout();f.savefig(fig/'entropy_coefficient.png',dpi=180);plt.close(f)
 f,ax=plt.subplots(figsize=(7,4));ax.plot(xe,col(ev,'mean_return'),marker='o',label='mean');ax.plot(xe,col(ev,'median_return'),marker='s',label='median');ax.set(xlabel='Environment steps',ylabel='Return',title='Deterministic evaluation return (n=20)');ax.grid(alpha=.25);ax.legend();f.tight_layout();f.savefig(fig/'eval_return.png',dpi=180);plt.close(f)
 def optional(name,rows,xkey,ykey,title,ylabel):
  if not rows or not np.isfinite(col(rows,ykey)).any():return
  f,ax=plt.subplots(figsize=(7,4));ax.plot(col(rows,xkey),col(rows,ykey),marker='o');ax.set(xlabel='Environment steps',ylabel=ylabel,title=title);ax.grid(alpha=.25);f.tight_layout();f.savefig(fig/name,dpi=180);plt.close(f)
 optional('eval_success_rate.png',ev,'step','success_rate','Deterministic evaluation success','Success rate')
 for name,key,title,ylabel in [('episode_length.png','rolling_episode_length','Training episode length','Steps'),('max_object_height.png','rolling_max_object_height','Maximum object height','Meters'),('object_lift_delta.png','rolling_object_lift_delta','Object lift delta','Meters'),('grasp_rate.png','rolling_grasp_rate','Grasp rate','Rate'),('grasp_duration.png','rolling_grasp_duration','Grasp duration','Steps'),('action_norm.png','action_norm','Action norm','L2 norm'),('action_saturation.png','action_saturation_fraction','Action saturation','Fraction'),('return_per_step.png','rolling_return_per_step','Return per step','Reward / step')]:optional(name,live,'step',key,title,ylabel)
 if np.isfinite(col(ev,'success_rate')).any():
  f,ax=plt.subplots(figsize=(6,4));ax.scatter(col(ev,'success_rate'),col(ev,'mean_return'),s=55);ax.set(xlabel='Evaluation success rate',ylabel='Mean return',title='Return versus task success');ax.grid(alpha=.25);f.tight_layout();f.savefig(fig/'return_vs_success.png',dpi=180);plt.close(f)
 if np.isfinite(col(live,'rolling_object_lift_delta')).any() and np.isfinite(col(live,'entropy_coefficient')).any():
  f,ax=plt.subplots(figsize=(6,4));ax.scatter(col(live,'entropy_coefficient'),col(live,'rolling_object_lift_delta'),s=40);ax.set(xlabel='Entropy coefficient',ylabel='Object lift delta',title='Entropy versus task progress');ax.grid(alpha=.25);f.tight_layout();f.savefig(fig/'entropy_vs_task_progress.png',dpi=180);plt.close(f)
 print('\n'.join(str(x) for x in sorted(fig.glob('*.png'))))
if __name__=='__main__':main()
