import train_ppo_policy
from itertools import product
from package.mongodb_handler import MongoHandler
from copy import deepcopy

# recursively update the dictionary(See test_bed.ipynb)
def recursive_update(orig_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in orig_dict and isinstance(orig_dict[key], dict):
            recursive_update(orig_dict[key], value)
        else:
            orig_dict[key] = value

if __name__ == '__main__':
    # main()'

    from package.config import session_config

    session_config["logger_params"]["logging_frequency"] = 1e5
    
    session_config["env_params"]["num_envs"] = 64
    session_config["env_params"]["out_of_trajectory_threshold"] = 3.14159
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

    # explanation, key, value list
    explanation_index = 0
    key_index = 1
    value_index = 2
    combinations_candidate_list = [
        ("",
         "target_velocity",[1.25]),
        ("",
         "custom_max_episode_steps",[
            100,200,
        ]),
        ("",
         "out_of_trajectory_threshold",[
            0.6,
        ]),
        ("",
         "joint_reward_keys_and_weights",[
            {"DEFAULT":1, "ankle":1.5, "hip":1, "knee":2, "lumbar_extension":1, "pelvis_tilt":2},
        ]),
        ("rewards-general imitation",
         "reward_keys_and_weights",[
            {"joint_imitation_reward": 0.4, "end_effector_imitation_reward": 0.4, "forward_reward": 0.2},
            {"joint_imitation_reward": 0.8, "end_effector_imitation_reward": 0.1, "forward_reward": 0.1},
        ]),
        ("rewards-activation",
         "reward_keys_and_weights",[
            {"muscle_activation_penalize": 0.3,  "muscle_activation_diff_penalize": 0.3},
            {"muscle_activation_penalize": 0.1,  "muscle_activation_diff_penalize": 0.1},
            {"muscle_activation_penalize": 0.01,  "muscle_activation_diff_penalize": 0.01},
        ]),
    ]
    # combinations_list_old = [kv for kv in combinations_list.items()]

    # print(f"DEBUG:: combinations_list:{combinations_list_old}")

    combinations = list(product(*[c[value_index] for c in combinations_candidate_list]))

    print(f"{len(combinations)=}")
    print(f"{combinations=}")

    mongo_handler = MongoHandler("myosuite_training", "train_archive")
    archive_id = mongo_handler.make_new_train_archive(session_config_combinations=combinations_candidate_list)

    combinated_list = []
    ## Print all combinations
    for combination_group in combinations:
        # new_params = {}
        # print(f"current combination: {combination}")
        combinated = {}
        for c_idx in range(len(combination_group)):
            k = combinations_candidate_list[c_idx][key_index]
            v = combination_group[c_idx]
            # why deep copy? if there is overlap key in combination_group, later one change the prev one.
            recursive_update(combinated, {k:deepcopy(v)})
        combinated_list.append(combinated)
    for idx, combinated in enumerate(combinated_list):
        print(f"{idx}, {combinated=}")

    for combinated in combinated_list:
        session_config["env_params"].update(combinated)
        print(f"DEBUG:: {session_config['env_params']=}")
        train_session = train_ppo_policy.ppo_train_with_parameters(session_config, 7e6, is_rendering_on=False, find_prev_train_session=False)
        mongo_handler.add_session_log_to_archive(archive_id,{
            "session_id":str(train_session.session_id),
            "combination":combinated,
            "entire_configs":train_session.get_session_config()
        })