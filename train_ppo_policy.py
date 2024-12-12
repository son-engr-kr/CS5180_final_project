from myosuite.utils import gym
import click
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import stable_baselines3
from package.rl_agent import CustomActorCriticPolicy, CustomLearningCallback

from package.train_handler import TrainSession
from bson import ObjectId



DESC = '''
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots \n
- rollout either learned policies or scripted policies (e.g. see rand_policy class below) \n
USAGE:\n
    $ python examine_env.py --env_name door-v1 \n
    $ python examine_env.py --env_name door-v1 --policy_path myosuite.utils.examine_env.rand_policy \n
    $ python examine_env.py --env_name door-v1 --policy_path my_policy.pickle --mode evaluation --episodes 10 \n
'''
def ppo_train_with_parameters(configs:dict, train_time_step:int, is_rendering_on:bool, find_prev_train_session:bool)->TrainSession:
    seed = 1234
    env_name = "myoLegRoughTerrainWalk2DImitationLearning-v0"
    # env_name = "myoLegStandRandom-v0"

    print("===============REFERENCE DATA LOADING================")
    file_path = 'neumove_models/reference_motions/02-constspeed_reduced_humanoid_50Hz_Anchor.npz'
    ref_data_npz = np.load(file_path, allow_pickle=True)
    # keys = ref_data_npz.files
    # ref_data_dict = {key: ref_data_npz[key] for key in keys}
    ref_data_dict = dict(ref_data_npz)
    print("===============REFERENCE DATA LOADING DONE================")

    print("===================MODIFYING CONFIG===============================")

    # Do nothing

    print("===================MODIFYING CONFIG DONE===============================")


    # env_args = {"model_path":"gait14dof22musc_cvt3_Right_Toeless_2D.xml"}
    # env_args = "{'model_path':'neumove_models/gait14dof22musc_cvt3_Right_Toeless_2D.xml', 'reset_type':'init', 'target_reach_range': {'IFtip': ((.1, -.1, .1), (0.27, .1, .3)),}}"
    env_args_dict = {'model_path':'neumove_models/gait14dof22musc_cvt3_Right_Toeless_2D.xml',
    # env_args_dict = {'model_path':'neumove_models/gait14dof22musc_cvt3_Right_Toeless_3D.xml',
                # 'reset_type':'init',
                'target_reach_range': {
                    'pelvis': ((-.05, -.05, 0), (0.05, 0.05, 0)),
                },
                'reference_data':ref_data_dict,
                'reference_data_sample_rate': ref_data_dict["meta_data"].item()["sample_rate"],
                }
    env_args_dict.update(configs["env_params"])
    np.random.seed(seed)

    
    if is_rendering_on:
        # envw = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
        envw = gym.make(env_name) if env_args_dict==None else gym.make(env_name, **(env_args_dict))
        env = envw.unwrapped
        env.mujoco_render_frames = True
        configs["env_params"]["num_envs"] = 1
        configs["ppo_params"]["n_steps"] = configs["ppo_params"]["batch_size"]
    else:
        env = SubprocVecEnv([lambda: (gym.make(env_name) if env_args_dict==None else gym.make(env_name, **(env_args_dict))).unwrapped for _ in range(configs["env_params"]["num_envs"])])
    # env = envw.unwrapped
    env.seed(seed)

    
    model = stable_baselines3.PPO(
        policy=CustomActorCriticPolicy,
        env=env,
        policy_kwargs=configs["policy_params"],
        verbose=2,
        **configs["ppo_params"],
    )

    configs["policy_info"].update({
        "extractor.policy_net":f"{model.policy.mlp_extractor.policy_net}",
        "extractor.value_net":f"{model.policy.mlp_extractor.value_net}",
        "action_net":f"{model.policy.action_net}",
        "value_net":f"{model.policy.value_net}",
        "ortho_init":f"{model.policy.ortho_init}",
        "share_features_extractor":f"{model.policy.share_features_extractor}"
    })

    train_session = TrainSession(db_name="myosuite_training",
                                 session_config=configs,
                                 find_prev_train_session=find_prev_train_session
                                 )
    
    print(f"{train_session.get_session_config()}")
    last_train_log = train_session.get_last_train_log()
    if last_train_log is not None:
        print(f"load from {last_train_log['model_file_path']}")
        model = model.load(last_train_log["model_file_path"], env=env)
        print(f"model.num_timesteps: {model.num_timesteps}")

    custom_callback = CustomLearningCallback(check_freq=configs["logger_params"]["logging_frequency"],train_session=train_session)
    model.learn(reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1,callback=custom_callback, progress_bar=True)
    env.close()
    print(f"learning done!")

    return train_session
if __name__ == '__main__':
    # main()
    from package.config import session_config

    session_config["logger_params"]["logging_frequency"] = 1e5
    
    session_config["env_params"]["num_envs"] = 1
    session_config["env_params"]["out_of_trajectory_threshold"] = 1.2
    session_config["env_params"]["flag_random_ref_index"] = True
    

    session_config["ppo_params"]["learning_rate"] = 2e-4
    session_config["ppo_params"]["n_steps"] = 1024 * 16 // session_config["env_params"]["num_envs"]
    session_config["ppo_params"]["batch_size"] = 1024*8
    session_config["ppo_params"]["n_epochs"] = 30 # if KL too low -> increase this
    session_config["ppo_params"]["clip_range"] = 0.2
    session_config["ppo_params"]["clip_range_vf"] = 100
    session_config["ppo_params"]["target_kl"] = 0.01 #0.01
    session_config["ppo_params"]["vf_coef"] = 0.5
    session_config["ppo_params"]["ent_coef"] = 0.001
    session_config["ppo_params"]["gae_lambda"] = 0.95


    ppo_train_with_parameters(session_config,
                              train_time_step=10000,
                              is_rendering_on=True,
                              find_prev_train_session=False,
                              )