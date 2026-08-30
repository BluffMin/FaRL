# Local Lift reward trace

- robosuite: 1.2.0
- source: `/workspace/collectenv/lib/python3.9/site-packages/robosuite/environments/manipulation/lift.py`
- success: cube z > table z 0.8 + 0.04 = 0.84 m
- non-success shaped reward: `(1-tanh(10*d)+0.25*grasp)/2.25`
- success reward: `1.0` after normalization
- no continuous lift-progress term
- FaRL control uses success-immediate termination and horizon truncation at 200.
