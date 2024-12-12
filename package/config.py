session_config = {
    # "code_version":"2024-10-23_1",
    "code_update_history":["2024-10-23",
                           "2024-10-25: Changed imitation reward function",
                           "2024-10-25: Imitation index increasing order",
                           "2024-10-26: Initial velocity setting bug fixed",
                           "2024-10-28: DummyVecEnv -> SubprocVecEnv",
                           "2024-10-28: Lumbar extension",
                           "2024-10-29: Changed imitation reward function (-100 -> -5)",
                           "2024-10-29: Maybe Fix environment bugs?(imitation_step increase rate; index increase in reset;)",
                           "2024-10-30: Fixed reference data frame rate bug",
                           "2024-10-30: imitation reward with weight",
                           "2024-10-30: use Parameter when log_std scheduling",
                           "2024-10-31: forward reward(w=0.2)",
                           "2024-11-01: end effector imitation reward",
                           "2024-11-01: increased weight of lumbar_extension imitation reward; limit lumbar_extension range",
                           "2024-11-07: Increased lumbar_extension gainprm-200",
                           "2024-11-07: Removed lumbar penalty(tried head ACC but failed)",
                           "2024-11-19: muscle activation penalty bug fixed",
                           "2024-11-19: remove delta_distance in muscle activation penalty term",
                           "2024-11-19: Bug fix: env config properly applying",
                           "2024-11-30: Reward bug fix: activation penalize square",
                            ]
}

logger_params = {
    "logging_frequency":2e4,# stable baseline logging interval: n_steps * num_envs
}
env_params = {
    "num_envs":8,
    # "max_timestep":4000,
    "safe_height":0.65,
    "out_of_trajectory_threshold":1,
    "flag_random_ref_index":False,
    "frame_skip":20,# 1000 // frame_skip == control_rate
    "target_velocity":1.25,#m/s
    "joint_reward_keys_and_weights":{
        "DEFAULT":1,

        "ankle":1,
        "knee":1,
        "lumbar_extension":1,
    },
    "reward_keys_and_weights":{
        "joint_imitation_reward": 0.7,
        "end_effector_imitation_reward": 0.3,
        "forward_reward": 0.01,
        "muscle_activation_penalize": 0.1,
        "muscle_activation_diff_penalize": 0.1,
        # "head_acceleration_penalize":0.1,
    },
    "custom_max_episode_steps":500,
}
policy_params = {
    # "action_log_std_schedule":[
    #     # (timestep, log_std)
    #     (0,-1.5),
    #     (1e3,-1.8),
    # ]
}
# env = PointMass2D()
ppo_params = {
    # "policy": CustomActorCriticPolicy,
    # "env": env,
    "learning_rate": 3e-4,
    "n_steps": 4096,
    "batch_size": 2048,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    # "verbose": 2
    }

# update on runtime
policy_info = {
}

session_config.update({
    "ppo_params":ppo_params,
    "env_params":env_params,
    "policy_params":policy_params,
    "logger_params":logger_params,
    "policy_info":policy_info,
})