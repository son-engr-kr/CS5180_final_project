from myosuite.utils import gym
import numpy as np
from package.config import session_config
import time
if __name__ == '__main__':
    print("===============REFERENCE DATA LOADING================")
    file_path = 'neumove_models/reference_motions/02-constspeed_reduced_humanoid_50Hz_Anchor.npz'
    ref_data_npz = np.load(file_path, allow_pickle=True)
    keys = ref_data_npz.files
    ref_data_dict = {key: ref_data_npz[key] for key in keys}
    ref_data_dict = dict(ref_data_npz)
    print("===============REFERENCE DATA LOADING DONE================")
    env_args_dict = {'model_path':'neumove_models/gait14dof22musc_cvt3_Right_Toeless_2D.xml',
        # env_args_dict = {'model_path':'neumove_models/gait14dof22musc_cvt3_Right_Toeless_3D.xml',
                    # 'reset_type':'init',
                    'target_reach_range': {
                        'pelvis': ((-.05, -.05, 0), (0.05, 0.05, 0)),
                    },
                    'reference_data':ref_data_dict,
                    'reference_data_sample_rate': ref_data_dict["meta_data"].item()["sample_rate"],
                    }
    env_args_dict.update(session_config["env_params"])
    env_name = "myoLegRoughTerrainWalk2DImitationLearning-v0"

    envw = gym.make(env_name) if env_args_dict==None else gym.make(env_name, **(env_args_dict))
    env = envw.unwrapped
    env.mujoco_render_frames = True
    obs, info = env.reset()

    for i in range(10000):
        time.sleep(env.dt)
        # print(env.dt, env.REF_FRAME_SKIP)
        # action, _states = model.predict(obs, deterministic=True)
        # index = env.imitation_step(True,specific_index=i)
        index = env.imitation_step(True,specific_index=None)
