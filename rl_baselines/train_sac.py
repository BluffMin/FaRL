#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from rl_baselines.callbacks import SuccessEvalCallback,LiveMetricsCallback
from rl_baselines.envs import make_nominal_env
from rl_baselines.evaluate import evaluate_policy_success
ROOT=Path('/home/brainlab/FaRL')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--out',required=True);ap.add_argument('--eval-freq',type=int,default=10000);ap.add_argument('--eval-episodes',type=int,default=20);ap.add_argument('--learning-rate',type=float,default=3e-4);ap.add_argument('--learning-starts',type=int,default=5000);ap.add_argument('--termination-mode',choices=['success','fixed_horizon'],default='success');ap.add_argument('--reward-mode',choices=['control','signed_lift_progress'],default='control');ap.add_argument('--progress',action='store_true',help='show detailed tqdm progress bar');args=ap.parse_args()
 out=Path(args.out);(out/'checkpoints').mkdir(parents=True,exist_ok=True);train=make_nominal_env(args.seed,termination_mode=args.termination_mode,reward_mode=args.reward_mode);ev=make_nominal_env(10000+args.seed,termination_mode=args.termination_mode,reward_mode=args.reward_mode);assert train is not ev and train.env is not ev.env
 model=SAC('MlpPolicy',train,learning_rate=args.learning_rate,buffer_size=1_000_000,learning_starts=args.learning_starts,batch_size=256,tau=.005,gamma=.99,train_freq=1,gradient_steps=1,ent_coef='auto',policy_kwargs={'net_arch':[256,256],'activation_fn':torch.nn.ReLU},seed=args.seed,device='cpu',verbose=1)
 before=torch.cat([p.detach().flatten().cpu() for p in model.policy.parameters()]).clone();live=LiveMetricsCallback(out,1000,print_live=not args.progress);callbacks=[SuccessEvalCallback(ev,out,args.eval_freq,args.eval_episodes,50000,verbose=0 if args.progress else 1),live]
 if args.progress:
  from rl_baselines.progress import VanillaSACProgressCallback
  callbacks.append(VanillaSACProgressCallback(live,0,args.steps,1000))
 cb=CallbackList(callbacks);t=time.time();model.learn(total_timesteps=args.steps,callback=cb,log_interval=10,progress_bar=False);elapsed=time.time()-t
 after=torch.cat([p.detach().flatten().cpu() for p in model.policy.parameters()]);final_n=100 if args.steps>=100000 else 20;final_seeds=[900000+i for i in range(final_n)];final=evaluate_policy_success(model,ev,final_n,final_seeds)
 best=SAC.load(out/'checkpoints'/'best_model.zip',env=ev,device='cpu');best_final=evaluate_policy_success(best,ev,final_n,final_seeds);comparison={'selection_data_note':'best checkpoint was selected only by online 20-episode evaluations; this final set is reporting-only','episodes_per_model':final_n,'best_model':best_final,'last_model':final};(out/'final_model_comparison.json').write_text(json.dumps(comparison,indent=2))
 actor_finite=all(torch.isfinite(p).all() for p in model.actor.parameters());critic_finite=all(torch.isfinite(p).all() for p in model.critic.parameters());ent=float(model.log_ent_coef.detach().exp().cpu()) if model.log_ent_coef is not None else None
 summary={'steps':args.steps,'seed':args.seed,'elapsed_seconds':elapsed,'steps_per_second':args.steps/elapsed,'replay_buffer_size':model.replay_buffer.size(),'replay_buffer_checkpoint':str(out/'checkpoints'/'replay_buffer.pkl'),'parameters_changed':bool(not torch.equal(before,after)),'actor_parameters_finite':bool(actor_finite),'critic_parameters_finite':bool(critic_finite),'entropy_coefficient':ent,'entropy_finite':bool(np.isfinite(ent)),'evaluation':final,'best_model_evaluation':best_final,'train_eval_env_separate':True,'checkpoint':str(out/'checkpoints'/'best_model.zip')}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));train.close();ev.close();print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
