# setting

```bash

git submodule update --init --recursive
.venv\Scripts\python.exe -m pip install -e .

```


```bash
git submodule update --recursive

```

# Test
```bash
.venv\Scripts\python.exe myosuite\utils\examine_env.py --env_name myoLegRoughTerrainWalk-v0
```

# env list
- myoChallengeOslRunRandom-v0
    - https://myosuite.readthedocs.io/en/latest/challenge-doc.html#objective

# My Test
```bash
.venv\Scripts\python.exe my_env_test.py
.venv\Scripts\python.exe my_env_test.py --env_name myoLegRoughTerrainWalk-v0
```

# Prosthesis path
`myosuite\simhive\myo_sim\osl\assets\myolegs_osl_chain.xml`

# Issues

## rendering wierd(solved)
https://github.com/google-deepmind/mujoco/issues/894

https://github.com/google-deepmind/mujoco/issues/639


Solution:
Set the labtop monitor as main monitor.


## rendering issue

```bash
Exception ignored in: <function Renderer.__del__ at 0x00000226550A1B40>
Traceback (most recent call last):
  File "...\myosuite_fork\myosuite\renderer\renderer.py", line 142, in __del__
    self.close()
  File "...\myosuite_fork\myosuite\renderer\mj_renderer.py", line 158, in close
    quit()
  File "C:\Users\...\AppData\Local\Programs\Python\Python310\lib\_sitebuiltins.py", line 26, in __call__
    raise SystemExit(code)
SystemExit: None
```

## ppo log_std problem


### log_std: 0 (not works)
```python
action_distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        
if deterministic:
    actions = mean_actions
else:
    actions = action_distribution.sample()

log_prob = action_distribution.log_prob(actions)
```
-> log_prob became very small value since action space is so large (e.g. -113)
So It will occur underflow later(in the rollback)

### log std: -100(not works)
```python
action_distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        
if deterministic:
    actions = mean_actions
else:
    actions = action_distribution.sample()

log_prob = action_distribution.log_prob(actions)
```
-> `proba_distribution` variance became 0 (underflow)
It occur underflow itself so log_prob is NaN
### log_std: -1(works)
It works!


## FFMPEG cannot find

```
_HAS_FFMPEG, "Cannot find installation of real FFmpeg (which comes with ffprobe).
```

https://www.gyan.dev/ffmpeg/builds/

-> just do
```bash
winget install "FFmpeg (Essentials Build)"
winget install "FFmpeg (Shared)"
```
in CMD

### Deprecated
```
.venv\lib\site-packages\skvideo\io\ffmpeg.py:466: DeprecationWarning: tostring() is deprecated. Use tobytes() instead.
  self._proc.stdin.write(vid.tostring())
```