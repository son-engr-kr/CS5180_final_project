import gymnasium as gym
import myosuite
from stable_baselines3 import PPO
import numpy as np

file_path = 'neumove_models/reference_motions/02-constspeed_reduced_humanoid.npz'
ref_data_npz = np.load(file_path)
keys = ref_data_npz.files
ref_data_dict = {key: ref_data_npz[key] for key in keys}
ref_data_dict = dict(ref_data_npz)
for key in ref_data_dict.keys():
    print(f"key: {key}, type: {type(ref_data_dict[key])}")
# print(f"data keys[{len(ref_data_npz.files)}]: {ref_data_npz.files}")

env = gym.make("myoLegRoughTerrainWalk2DImitationLearning-v0",
               target_reach_range= {
                    'pelvis': ((-.05, -.05, 0), (0.05, 0.05, 0)),
                },
                reference_data = ref_data_dict,
                model_path='neumove_models/gait14dof22musc_cvt3_Right_Toeless_2D.xml',
                out_of_trajectory_threshold=0.45,
                safe_height=0.65)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000, progress_bar=True)
print("done")
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")