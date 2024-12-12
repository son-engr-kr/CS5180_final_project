from myosuite.utils import gym
import click
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import stable_baselines3
from package.rl_agent import CustomActorCriticPolicy, CustomLearningCallback
from package.config import session_config
from package.train_handler import TrainSession

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

@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-p', '--policy_path', type=str, help='absolute path of the policy file', default=None)
@click.option('-m', '--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='onscreen')
@click.option('-c', '--camera_name', type=str, default=None, help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='./', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default=None, help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=False, help=('Save the rollout paths'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-rv', '--render_visuals', type=bool, default=False, help=('render the visual keys of the env, if present'))
@click.option('-ea', '--env_args', type=str, default=None, help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))

def main(env_name, policy_path, mode, seed, num_episodes, render, camera_name, output_dir, output_name, save_paths, plot_paths, render_visuals, env_args):

    np.random.seed(seed)
    envw = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env = envw.unwrapped
    env.seed(seed)

    

    env.mujoco_render_frames = True

    model = PPO(
        # "MlpPolicy",
        CustomActorCriticPolicy,
        env,
        learning_rate=3e-6,
        n_steps=128,#intervl between update
        batch_size=64,# batch size
        n_epochs=10,# training iteration
        gamma=0.99,# discount factor
        gae_lambda=0.95,#Generalized Advantage Estimation
        clip_range=0.1,
        clip_range_vf=0.1,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        # tensorboard_log="./ppo_custom_tensorboard/",
        verbose=1,
    )
    model.learn(total_timesteps=1000000)
    print(f"learning done!")
    # model.save("./torch_models_db/ppo1")
    # model = PPO.load("./torch_models_db/ppo1")
def main_with_no_args():
    seed = 123
    env_name = "myoLegRoughTerrainWalk2DImitationLearning-v0"
    # env_name = "myoLegStandRandom-v0"

    print("===============REFERENCE DATA LOADING================")
    file_path = 'neumove_models/reference_motions/02-constspeed_reduced_humanoid_50Hz_Anchor.npz'
    ref_data_npz = np.load(file_path, allow_pickle=True)
    keys = ref_data_npz.files
    ref_data_dict = {key: ref_data_npz[key] for key in keys}
    ref_data_dict = dict(ref_data_npz)
    print("===============REFERENCE DATA LOADING DONE================")

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
    is_rendering_on = False
    
    session_config["env_params"]["out_of_trajectory_threshold"] = 1.2
    session_config["env_params"]["flag_random_ref_index"] = True
    
    env_args_dict.update(session_config["env_params"])

    session_config["logger_params"]["logging_frequency"] = 1e5

    session_config["ppo_params"]["learning_rate"] = 2e-4
    if is_rendering_on:
        session_config["ppo_params"]["n_steps"] = 1024 * 8
    else:
        session_config["ppo_params"]["n_steps"] = 1024 * 32 // session_config["env_params"]["num_envs"]
    session_config["ppo_params"]["batch_size"] = 1024*4
    session_config["ppo_params"]["n_epochs"] = 30
    session_config["ppo_params"]["clip_range"] = 0.2
    session_config["ppo_params"]["clip_range_vf"] = 100
    session_config["ppo_params"]["target_kl"] = None #0.01
    session_config["ppo_params"]["vf_coef"] = 0.5
    session_config["ppo_params"]["ent_coef"] = 0.001
    session_config["ppo_params"]["gae_lambda"] = 0.95


    np.random.seed(seed)
    
    if is_rendering_on:
        # envw = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
        envw = gym.make(env_name) if env_args_dict==None else gym.make(env_name, **(env_args_dict))
        env = envw.unwrapped
        env.mujoco_render_frames = True
    else:
        env = SubprocVecEnv([lambda: (gym.make(env_name) if env_args_dict==None else gym.make(env_name, **(env_args_dict))).unwrapped for _ in range(session_config["env_params"]["num_envs"])])
    # env = envw.unwrapped
    env.seed(seed)
    
    


    
    model = stable_baselines3.PPO(
        policy=CustomActorCriticPolicy,
        env=env,
        policy_kwargs=session_config["policy_params"],
        verbose=2,
        **session_config["ppo_params"],
    )

    session_config["policy_info"].update({
        "extractor.policy_net":f"{model.policy.mlp_extractor.policy_net}",
        "extractor.value_net":f"{model.policy.mlp_extractor.value_net}",
        "action_net":f"{model.policy.action_net}",
        "value_net":f"{model.policy.value_net}",
        "ortho_init":f"{model.policy.ortho_init}",
        "share_features_extractor":f"{model.policy.share_features_extractor}"
    })

    train_session = TrainSession(db_name="myosuite_test",
                                 session_config=session_config,
                                 )
    
    print(f"{train_session.get_session_config()}")
    last_model_log = train_session.get_last_train_log()
    if last_model_log["model_file_path"] is not None:
        print(f"load from {last_model_log["model_file_path"]}")
        model = model.load(last_model_log["model_file_path"], env=env)
        print(f"model.num_timesteps: {model.num_timesteps}")

    custom_callback = CustomLearningCallback(check_freq=session_config["logger_params"]["logging_frequency"],train_session=train_session)
    model.learn(reset_num_timesteps=False, total_timesteps=1e7, log_interval=1,callback=custom_callback, progress_bar=True)
    print(f"learning done!")
    # model.save("./torch_models_db/ppo1")
    # model = PPO.load("./torch_models_db/ppo1")
if __name__ == '__main__':
    # main()
    main_with_no_args()