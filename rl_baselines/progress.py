"""Low-overhead, metrics-consistent tqdm for vanilla SAC continuation."""
from __future__ import annotations
import time
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

def show(x,scale=1.):
 if x is None:return 'NA'
 try:return f'{float(x)*scale:.3g}'
 except (TypeError,ValueError):return 'NA'

class VanillaSACProgressCallback(BaseCallback):
 def __init__(self,live_callback,initial_effective_step=100000,target_effective_step=300000,update_interval=1000):
  super().__init__(0);self.live=live_callback;self.initial=initial_effective_step;self.target=target_effective_step;self.interval=update_interval;self.bar=None;self.last=initial_effective_step;self.started=None;self.last_eval_step=None
 def _on_training_start(self):
  self.started=time.monotonic();effective=int(self.model.num_timesteps);self.last=effective
  self.bar=tqdm(total=self.target,initial=effective,desc=f'SAC {self.initial//1000}k->{self.target//1000}k',unit='step',dynamic_ncols=True,mininterval=1.)
 def _on_step(self):
  effective=int(self.model.num_timesteps)
  if effective-self.last<self.interval and effective<self.target:return True
  self.bar.update(max(0,effective-self.last));self.last=effective;row=self.live.rows[-1] if self.live.rows else {}
  elapsed=max(1e-9,time.monotonic()-self.started);fps=(effective-self.initial)/elapsed
  self.bar.set_postfix(ret=show(row.get('rolling_return')),ever=show(row.get('rolling_ever_success_rate')),final=show(row.get('rolling_final_success_rate')),grasp=show(row.get('rolling_grasp_rate')),lift=show(row.get('rolling_lift_rate')),lift_grasp=show(row.get('rolling_p_lift_given_grasp')),lift_mm=show(row.get('rolling_object_lift_delta'),1000),first_s=show(row.get('rolling_first_success_step')),actor=show(row.get('actor_loss')),critic=show(row.get('critic_loss')),alpha=show(row.get('entropy_coefficient')),fps=show(fps))
  curve=self.live.out/'learning_curve.csv'
  if curve.exists():
   import csv
   with curve.open() as f:rows=list(csv.DictReader(f))
   if rows and rows[-1].get('step')!=self.last_eval_step:
    e=rows[-1];self.last_eval_step=e.get('step');tqdm.write(f"[EVAL @ {e.get('step')}] success={e.get('success_rate')} grasp={e.get('grasp_rate')} lift={e.get('lift_rate')} P(lift|grasp)={e.get('p_lift_given_grasp')} median_lift_mm={show(e.get('median_object_lift_delta'),1000)} return={e.get('mean_return')}")
  return True
 def _on_training_end(self):
  if self.bar:self.bar.update(max(0,int(self.model.num_timesteps)-self.last));self.bar.close()
