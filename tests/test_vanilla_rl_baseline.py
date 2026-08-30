import csv,hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
import numpy as np
os.environ.setdefault('PYGLFW_LIBRARY','/workspace/collectenv/lib/libglfw.so.3')
os.environ.setdefault('MUJOCO_PY_MUJOCO_PATH','/workspace/mujoco-2.1.1')
os.environ.setdefault('MUJOCO_GL','osmesa')
ROOT=Path('/home/brainlab/FaRL')

def test_legacy_csv_and_plot_parse():
 from rl_baselines.plot_run import read,col
 p=ROOT/'results/vanilla_rl_baseline_v1/sac_20k_sanity_tqdm'
 rows=read(p/'live_metrics.csv');assert len(rows)==20
 assert np.isfinite(col(rows,'rolling_return')).all()
 assert np.isnan(col(rows,'optional_new_metric')).all()

def test_plot_run_old_and_new_schema(tmp_path):
 old=ROOT/'results/vanilla_rl_baseline_v1/sac_20k_sanity_tqdm'
 subprocess.run([sys.executable,'-m','rl_baselines.plot_run','--run',str(old)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 live={'step':1000,'rolling_return':1,'rolling_success_rate':0,'actor_loss':-1,'critic_loss':.1,'entropy_coefficient':.5,'rolling_episode_length':200,'rolling_max_object_height':.83,'rolling_object_lift_delta':.001,'rolling_grasp_rate':.1,'rolling_grasp_duration':1,'action_norm':1,'action_saturation_fraction':0,'rolling_return_per_step':.005}
 ev={'step':1000,'success_rate':0,'mean_return':1,'median_return':1,'mean_episode_length':200,'mean_return_per_step':.005,'episodes':20,'grasp_rate':0,'mean_max_object_height':.83}
 for name,row in [('live_metrics.csv',live),('learning_curve.csv',ev)]:
  with (tmp_path/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
 subprocess.run([sys.executable,'-m','rl_baselines.plot_run','--run',str(tmp_path)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 assert (tmp_path/'figures/grasp_rate.png').exists() and (tmp_path/'figures/return_per_step.png').exists()

def test_resume_audit_is_honest():
 x=json.loads((ROOT/'results/vanilla_rl_baseline_v1/sac_20k_sanity_tqdm/posthoc_diagnostic_audit.json').read_text())['resume_audit']
 assert x['REPLAY_BUFFER_SAVED'] is False and x['TRUE_RESUME_SUPPORTED']=='NO'
 from rl_baselines.callbacks import SuccessEvalCallback
 assert 'save_replay_buffer' in inspect.getsource(SuccessEvalCallback)

def test_checkpoint_selection_success_first():
 from rl_baselines.callbacks import SuccessEvalCallback
 src=inspect.getsource(SuccessEvalCallback._on_step)
 assert "score=(m['success_rate'],m['lift_rate'],m.get('median_object_lift_delta')" in src

def test_return_per_step_and_action_diagnostics():
 from rl_baselines.envs import make_nominal_env
 e=make_nominal_env(seed=7,horizon=2);o,_=e.reset(seed=7);assert o.shape==(42,) and np.isfinite(o).all()
 a=np.zeros(7,np.float32);o,r,t,tr,info=e.step(a);o,r2,t,tr,info=e.step(a);d=info['episode_diagnostics'];e.close()
 assert tr and np.isclose(d['return_per_step'],d['episode_return']/d['episode_length'])
 assert len(d['action_mean_per_dimension'])==7 and len(d['action_saturation_per_dimension'])==7
 assert isinstance(d['ever_grasped'],bool) and d['max_object_height']>=d['initial_object_height']

def test_original_c_dataset_and_classifier_unchanged():
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert sha('/home/robotics/external_workspace/data/C/demos/lift_ph_demo.hdf5')=='2394eb28fd6fe0a5e733a262689b7bc4141ba1940bd1397652ae6a26ffd105c5'
 assert sha('/home/robotics/external_workspace/train/runs/final_factor_graph_plus_extra/best.pt')=='11ea80908b505e7c15e4e0f43b45bc3611cca115a9fdf4a9fce029628ec47b16'

def test_shell_utilities_syntax():
 subprocess.run(['bash','-n','scripts/run_vanilla_sac.sh','scripts/watch_vanilla_sac.sh','scripts/inspect_vanilla_sac.sh'],cwd=ROOT,check=True)

def test_evaluation_rng_isolation_source():
 from rl_baselines.evaluate import evaluate_policy_success
 src=inspect.getsource(evaluate_policy_success)
 assert 'np.random.get_state()' in src and 'np.random.set_state(rng)' in src
