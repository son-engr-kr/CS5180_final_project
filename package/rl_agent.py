import torch
import torch.nn as nn
import os
import torch.nn.init as init
from torch.distributions import MultivariateNormal
from package.train_handler import TrainSession

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium import spaces
import torch as th
from torch import nn
from typing import Callable, Tuple

# from package.train_handler import TrainSession
from pprint import pprint
import gymnasium as gym
import json
import numpy as np
from datetime import datetime
import torch
torch.autograd.set_detect_anomaly(True)
################################## set device ##################################
# set device to cpu or cuda
# device = torch.device('cpu')
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device: " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device: cpu")

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            nn.Tanh(),
        )
        self._initialize_weights()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.net(observations)
        return x

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

class CustomActorCriticPolicyOld(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        print(f"CustomActorCriticPolicy: kwargs:{kwargs}")
        self.action_log_std_schedule = kwargs.pop('action_log_std_schedule', [(0,0)])
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs,
                                                      features_extractor_class=CustomFeatureExtractor,
                                                      features_extractor_kwargs=dict(features_dim=64))
        # Policy network
        self.action_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.shape[0]),
            nn.Tanh()
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

        # Action distribution
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        # self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
        self.log_std = nn.Parameter(torch.full(action_space.shape, self.action_log_std_schedule[0][1], dtype=torch.float32))
        self.adjust_action_log_std(0)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        mean_actions = self.action_net(features)
        value = self.value_net(features)
        
        action_distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        
        if deterministic:
            actions = mean_actions
        else:
            actions = action_distribution.sample()

        log_prob = action_distribution.log_prob(actions)
        return actions, value, log_prob

    def _predict(self, obs: torch.Tensor, deterministic: bool = False):
        return self.forward(obs, deterministic)[0]

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.extract_features(obs)
        mean_actions = self.action_net(features)
        value = self.value_net(features)
        action_distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = action_distribution.log_prob(actions)
        return log_prob, action_distribution.entropy(), value

    def predict_values(self, obs: torch.Tensor):
        features = self.extract_features(obs)
        return self.value_net(features)
    
    def _initialize_weights(self):
        for layer in list(self.action_net) + list(self.value_net):
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
    def adjust_action_log_std(self, timestep):
        for (t, log_std) in reversed(self.action_log_std_schedule):
            if timestep >= t:
                print(f"Adjust action log std at timestep: {timestep} - {log_std}")
                # self.log_std.data.fill_(log_std)
                self.log_std = nn.Parameter(torch.full(self.log_std.shape, log_std, dtype=torch.float32))
                break

## https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#advanced-example
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, last_layer_dim_pi),
            nn.Tanh(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, last_layer_dim_vf),
            nn.Tanh()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        # kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

class CustomLearningCallback(BaseCallback):
    def __init__(self, *, check_freq: int, train_session:TrainSession, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.train_session = train_session
        # self.ppo

        # self.reward_sum = np.zeros

    def _init_callback(self):
        # if self.save_path is not None:
        #     os.makedirs(self.save_path, exist_ok=True)
        print(f"self.training_env.num_envs:{self.training_env.num_envs}")
        self.rewards_sum = np.zeros(self.training_env.num_envs)
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)
        self.episode_counts = np.zeros(self.training_env.num_envs)
        self.episode_length_counts = np.zeros(self.training_env.num_envs)
        self.current_episode_length_counts = np.zeros(self.training_env.num_envs)
        last_train_log = self.train_session.get_last_train_log()
        if last_train_log is not None:
            self.prev_logging_timestep = last_train_log["num_timesteps"]
        else:
            self.prev_logging_timestep = 0
        # self.model.policy.adjust_action_log_std(self.model.num_timesteps)
        
        # self.num_timesteps = self.train_session.last_num_timestep # initialize after super().__init__()

        # print(f"self.num_timesteps2: {self.num_timesteps}")

        pass

    #called after all envs step done
    def _on_step(self) -> bool:
        # print(f"self.n_calls:{self.n_calls}, self.num_timesteps:{self.num_timesteps}")
        self.current_episode_rewards += self.locals["rewards"]

        for i, done in enumerate(self.locals["dones"]):
            self.current_episode_length_counts[i] += 1
            if done:
                # print(f"done! {self.current_episode_length_counts[i]}")
                self.rewards_sum += self.current_episode_rewards[i]
                self.episode_counts[i] += 1
                self.current_episode_rewards[i] = 0.0
                self.episode_length_counts[i] += self.current_episode_length_counts[i]
                self.current_episode_length_counts[i] = 0


        # if self.n_calls % self.check_freq == 0:
        if self.num_timesteps - self.prev_logging_timestep >= self.check_freq:
            # self.model.policy.adjust_action_log_std(self.model.num_timesteps)
            self.prev_logging_timestep = self.num_timesteps
            # print(f"self.episode_counts:{self.episode_counts}")

            log_data = {
                # 'time': self.logger.name_to_value.get('time/total_timesteps', -1),
                # 'fps': self.logger.name_to_value.get('time/fps', -1),
                # 'iterations': self.logger.name_to_value.get('time/iterations', -1),
                # 'time_elapsed': self.logger.name_to_value.get('time/time_elapsed', -1),
                # 'total_timesteps': self.logger.name_to_value.get('time/total_timesteps', -1),
                'approx_kl': self.logger.name_to_value.get('train/approx_kl', -1),
                'clip_fraction': self.logger.name_to_value.get('train/clip_fraction', -1),
                'clip_range': self.logger.name_to_value.get('train/clip_range', -1),
                'clip_range_vf': self.logger.name_to_value.get('train/clip_range_vf', -1),
                'entropy_loss': self.logger.name_to_value.get('train/entropy_loss', -1),
                'explained_variance': self.logger.name_to_value.get('train/explained_variance', -1),
                'learning_rate': self.logger.name_to_value.get('train/learning_rate', -1),
                'loss': self.logger.name_to_value.get('train/loss', -1),
                'n_updates': self.logger.name_to_value.get('train/n_updates', -1),
                'policy_gradient_loss': self.logger.name_to_value.get('train/policy_gradient_loss', -1),
                'std': self.logger.name_to_value.get('train/std', -1),
                'value_loss': self.logger.name_to_value.get('train/value_loss', -1),

                "num_timesteps": self.model.num_timesteps,
                "average_num_timestep": np.sum(self.episode_length_counts) / np.sum(self.episode_counts),
                # "rewards_avg":logging_reward_avg,
                "average_reward_per_episode":np.sum(self.rewards_sum) / np.sum(self.episode_counts),
                "time": f"{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}",
            }
            for key, value in log_data.items():
                if isinstance(value, np.generic):
                    log_data[key] = value.item()
            # print(f"self.logger:")
            # pprint(log_data)

            self.train_session.make_check_point(model = self.model,
                                                log_dict=log_data)
            self.rewards_sum = np.zeros(self.training_env.num_envs)
            # self.current_episode_rewards = np.zeros(self.training_env.num_envs)
            self.episode_counts = np.zeros(self.training_env.num_envs)
            self.episode_length_counts = np.zeros(self.training_env.num_envs)
            

        return True

