import numpy as np
from rl_baselines.policy_adapter import VanillaRLPolicy


class _PredictRecorder:
    def __init__(self): self.seen=[]
    def predict(self, obs, deterministic=True):
        self.seen.append(np.array(obs, copy=True))
        return np.asarray(obs[:2], dtype=np.float32), None


def test_action_is_recomputed_from_current_observation():
    policy=VanillaRLPolicy.__new__(VanillaRLPolicy);policy.model=_PredictRecorder()
    obs_t=np.array([1.,2.,3.],dtype=np.float32);obs_next=np.array([4.,5.,6.],dtype=np.float32)
    action_t=policy.act(obs_t,deterministic=True);action_next=policy.act(obs_next,deterministic=True)
    assert np.array_equal(policy.model.seen[0],obs_t)
    assert np.array_equal(policy.model.seen[1],obs_next)
    assert not np.array_equal(action_t,action_next)
