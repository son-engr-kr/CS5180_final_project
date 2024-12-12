""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Cameron Berg (cam.h.berg@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat, intrinsic_euler2quat, quat2euler_intrinsic
import mujoco

class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos',
                        'qvel',
                        # 'tip_pos',
                        'reach_err' # we need this in MujocoEnv: rwd_sparse
                        ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        # "reach": 1.0,
        # "bonus": 4.0,
        # "penalty": 50,
        # "act_reg": 1,
        "joint_imitation_reward": 0.4,
        "end_effector_imitation_reward": 0.4,
        "forward_reward": 0.2,
        "muscle_activation_penalize": -0.01,
        "muscle_activation_diff_penalize": -0.01,

        # "head_acceleration_penalize": 0.2,
    }
    JOINT_RWD_KEYS_AND_WEIGHTS = {}
    REF_FRAME_SKIP = None
    class SimToRefMapping:
        def __init__(self, sim_name:str, ref_name:str, offset:float=0.0, is_flip:bool=False):
            self.sim_name = sim_name
            self.ref_name = ref_name
            self.offset = offset
            self.is_flip = is_flip



    QPOS_SIM_TO_REF = {
        'ankle_angle_l': SimToRefMapping('ankle_angle_l', 'q_ankle_angle_l'),
        'ankle_angle_r': SimToRefMapping('ankle_angle_r', 'q_ankle_angle_r'),
        'hip_flexion_l': SimToRefMapping('hip_flexion_l', 'q_hip_flexion_l'),
        'hip_flexion_r': SimToRefMapping('hip_flexion_r', 'q_hip_flexion_r'),
        'knee_angle_l': SimToRefMapping('knee_angle_l', 'q_knee_angle_l'),
        'knee_angle_r': SimToRefMapping('knee_angle_r', 'q_knee_angle_r'),
        'pelvis_tilt': SimToRefMapping('pelvis_tilt', 'q_pelvis_tilt', is_flip=False),
        'pelvis_tx': SimToRefMapping('pelvis_tx', 'q_pelvis_tx'),
        'pelvis_ty': SimToRefMapping('pelvis_ty', 'q_pelvis_ty', offset=0.925),

        # 'hip_adduction_l': SimToRefMapping('hip_adduction_l', 'q_hip_adduction_l'),
        # 'hip_adduction_r': SimToRefMapping('hip_adduction_r', 'q_hip_adduction_r'),
        # 'hip_rotation_l': SimToRefMapping('hip_rotation_l', 'q_hip_rotation_l'),
        # 'hip_rotation_r': SimToRefMapping('hip_rotation_r', 'q_hip_rotation_r'),
        # 'pelvis_tz': SimToRefMapping('pelvis_tz', 'q_pelvis_tz'),
        'lumbar_extension': SimToRefMapping('lumbar_extension', 'q_lumbar_extension'),

    }
    QVEL_SIM_TO_REF = {
        'ankle_angle_l': SimToRefMapping('ankle_angle_l', 'dq_ankle_angle_l'),
        'ankle_angle_r': SimToRefMapping('ankle_angle_r', 'dq_ankle_angle_r'),
        'hip_flexion_l': SimToRefMapping('hip_flexion_l', 'dq_hip_flexion_l'),
        'hip_flexion_r': SimToRefMapping('hip_flexion_r', 'dq_hip_flexion_r'),
        'knee_angle_l': SimToRefMapping('knee_angle_l', 'dq_knee_angle_l'),
        'knee_angle_r': SimToRefMapping('knee_angle_r', 'dq_knee_angle_r'),
        'pelvis_tilt': SimToRefMapping('pelvis_tilt', 'dq_pelvis_tilt', is_flip=False),
        'pelvis_tx': SimToRefMapping('pelvis_tx', 'dq_pelvis_tx'),
        'pelvis_ty': SimToRefMapping('pelvis_ty', 'dq_pelvis_ty'),

        # 'hip_adduction_l': SimToRefMapping('hip_adduction_l', 'dq_hip_adduction_l'),
        # 'hip_adduction_r': SimToRefMapping('hip_adduction_r', 'dq_hip_adduction_r'),
        # 'hip_rotation_l': SimToRefMapping('hip_rotation_l', 'dq_hip_rotation_l'),
        # 'hip_rotation_r': SimToRefMapping('hip_rotation_r', 'dq_hip_rotation_r'),
        # 'pelvis_tz': SimToRefMapping('pelvis_tz', 'dq_pelvis_tz'),
        'lumbar_extension': SimToRefMapping('lumbar_extension', 'dq_lumbar_extension'),

    }
    ANCHOR_SIM_TO_REF = {
        'ankle_angle_l': SimToRefMapping('ankle_angle_l', 'a_ankle_angle_l'),
        'ankle_angle_r': SimToRefMapping('ankle_angle_r', 'a_ankle_angle_r'),
        'knee_angle_l': SimToRefMapping('knee_angle_l', 'a_knee_angle_l'),
        'knee_angle_r': SimToRefMapping('knee_angle_r', 'a_knee_angle_r'),

    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        # print(f"DEBUG:: kwargs: {kwargs}")
        self._setup(**kwargs)


    def _setup(self,
            target_reach_range:dict,
            joint_random_range:tuple=(0.0,0.0),
            far_th = .35,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        self.joint_random_range = joint_random_range

        

        self.setup_reference_data(data=kwargs.pop("reference_data"))
        self._out_of_trajectory_threshold = kwargs.pop("out_of_trajectory_threshold")
        self._safe_height = kwargs.pop("safe_height")
        self._flag_random_ref_index = kwargs.pop("flag_random_ref_index")
        self._ref_sample_rate = kwargs.pop("reference_data_sample_rate")
        self._target_velocity = kwargs.pop("target_velocity")
        self._step_count_per_episode = 0
        self.CUSTOM_MAX_EPISODE_STEPS = kwargs.pop("custom_max_episode_steps")
        self.REF_FRAME_SKIP = int(kwargs["frame_skip"] * self._ref_sample_rate/1000) #ref sample rate // control rate
        
        self.JOINT_RWD_KEYS_AND_WEIGHTS = kwargs.pop("joint_reward_keys_and_weights")
        ## Parameter check
        for key, weight in self.JOINT_RWD_KEYS_AND_WEIGHTS.items():
            if key == "DEFAULT":
                continue
            for qv in self.QPOS_SIM_TO_REF.values():
                if key in qv.sim_name:
                    break
            else:  # If the inner loop didn't break, the key wasn't found
                raise KeyError(f"Key '{key}' not found in any qv.sim_name")

        self.DEFAULT_RWD_KEYS_AND_WEIGHTS.update(kwargs.pop("reward_keys_and_weights"))

        print("===============================PARAMETERS=============================")
        print(f"self._ref_sample_rate:{self._ref_sample_rate}")
        print(f"REF_FRAME_SKIP:{self.REF_FRAME_SKIP}")# should be 1
        print(f"self.JOINT_RWD_KEYS_AND_WEIGHTS:{self.JOINT_RWD_KEYS_AND_WEIGHTS}")
        print(f"self.DEFAULT_RWD_KEYS_AND_WEIGHTS:{self.DEFAULT_RWD_KEYS_AND_WEIGHTS}")
        print("===============================PARAMETERS=============================")

        self.prev_muscle_activations_for_reward = None

        #phys: 1000hz
        # control 50hz : 50 * 20 = 1000hz
        # ref 50hz: 500hz 10skip: 20 * 500 / 1000

        
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]

        # find geometries with ID == 1 which indicates the skins
        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        # Change the alpha value to make it transparent
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0

        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

        #bipass the bug(?)
        for _ in range(30):
            super().step(np.zeros(self.sim.model.nu))
        

    def get_obs_dict(self, sim):
        # TODO observation - tx exclude
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time]) # they use time separately like t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)

        qpos = []
        for mapping in self.QPOS_SIM_TO_REF.values():
            if mapping.sim_name == 'pelvis_tx':
                pass
            else:
                qpos.append(self.sim.data.joint(mapping.sim_name).qpos[0].copy())
        qvel = []
        for mapping in self.QVEL_SIM_TO_REF.values():
            qvel.append(self.sim.data.joint(mapping.sim_name).qvel[0].copy())
        obs_dict['qpos'] = np.array(qpos) # 7 + 1 elements
        obs_dict['qvel'] = np.array(qvel) # 7 + 2 elements
        if sim.model.na>0:
            # BaseV0 Add the key like this: obs_keys.append("act")
            obs_dict['act'] = sim.data.act[:].copy() # 22 elements
            # obs_dict['act'] = np.array([])

        # # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])
        # for isite in range(len(self.tip_sids)):
        #     obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
        #     obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        # obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        obs_dict['reach_err'] = np.array([])
        # print(f"DEBUG:: obs_dict: {obs_dict}")
        return obs_dict

    def _get_qpos_diff(self):
        # print(f"DEBUG:: ref index in reward: {self._imitation_index}")

        def get_qpos_diff(mapping:ReachEnvV0.SimToRefMapping):
            sign = 1.0 if not mapping.is_flip else -1.0
            diff = self.sim.data.joint(mapping.sim_name).qpos[0].copy() - (sign*self._reference_data[mapping.ref_name][self._imitation_index] + mapping.offset)
            return diff
        diff_array = []
        weight_array = []
        for qv in self.QPOS_SIM_TO_REF.values():
            if qv.sim_name == 'pelvis_ty' or qv.sim_name == 'pelvis_tx':
                pass
            else:
                diff = get_qpos_diff(qv)
                diff_array.append(diff)
                for key, weight in self.JOINT_RWD_KEYS_AND_WEIGHTS.items():
                    if key in qv.sim_name:
                        weight_array.append(weight)
                        # print(f"DEBUG::{key},{qv.sim_name}")
                        break
                else:#if for not break
                    # print(f"DEBUG::{'DEFAULT'},{qv.sim_name}")
                    weight_array.append(self.JOINT_RWD_KEYS_AND_WEIGHTS["DEFAULT"])
        return np.array(diff_array), np.array(weight_array)
    def _get_end_effector_diff(self):
        body_pos = self.sim.data.body('pelvis').xpos.copy()
        diff_array = []
        for mapping in self.ANCHOR_SIM_TO_REF.values():
            sim_anchor = self.sim.data.joint(mapping.sim_name).xanchor.copy() - body_pos
            ref_anchor = self._reference_data[mapping.ref_name][self._imitation_index]
            diff = np.linalg.norm(sim_anchor - ref_anchor)
            diff_array.append(diff)
        return diff_array
    def get_reward_dict(self, obs_dict):
        # print(f"DEBUG:: sim.data.time: {self.sim.data.time}, dt: {self.dt}")
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        # vel_dist = np.linalg.norm(obs_dict['qvel'], axis=-1)
        # act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        far_th = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf
        # # near_th = len(self.tip_sids)*.0125
        near_th = len(self.tip_sids)*.050



        qpos_diff_array, weight_array = self._get_qpos_diff()
        anchor_diff_array = self._get_end_effector_diff()

        # print(f"DEBUG:: dt: {self.dt}")
        # print(f"DEBUG:: diff_array:{diff_array}\nweight_array:{weight_array}\nresult:{weight_array * np.exp(-8 * np.square(diff_array))}")
        # print(f"DEBUG:: np.exp(-5 * np.square(diff_array)):{np.exp(-8 * np.square(diff_array))}")
        qpos_reward = self.dt * np.mean(weight_array * np.exp(-8 * np.square(qpos_diff_array))) #TODO: make as a training hyper-parameter?
        anchor_reward = self.dt * np.mean(np.exp(-5 * np.square(anchor_diff_array))) #TODO: make as a training hyper-parameter?
        # print(f"DEBUG::np.exp(-5 * np.square(anchor_diff_array)): {np.exp(-5 * np.square(anchor_diff_array))}")
        # print(f"DEBUG:: imitation_reward:{imitation_reward}")

        forward_reward = self.dt * np.exp(-5 * np.square(self.sim.data.joint("pelvis_tx").qvel[0].copy() - self._target_velocity))

        # print(f"DEBUG:: {self.sim.data.actuator('lumbar_extension_motor').ctrl[0].copy()=}")
        # print(f"DEBUG:: {self.sim.data.act[:].copy()=}, {self.sim.data.actuator('lumbar_extension_motor').ctrl[0].copy()=}")
        muscle_activations = np.concatenate((self.sim.data.act[:].copy(), 
                                              np.array([self.sim.data.actuator('lumbar_extension_motor').ctrl[0].copy()]).reshape(1,)))
        # muscle_activations = np.concatenate(self.sim.data.act[:].copy(), np.array([self.sim.data.actuator('lumbar_extension_motor').ctrl[0].copy()]))
        # print(f"DEBUG:: {len(muscle_activations)=}, {len(self.sim.data.ctrl)=}")
        # print(f"DEBUG:: {muscle_activations=}")

        # delta_distance = self.dt * self.sim.data.joint("pelvis_tx").qvel[0].copy()
        # print(f"DEBUG:: {muscle_activations=}, {delta_distance=}")
        # what if delta_distance is zero?
        # muscle_activation_penalize = self.dt * np.sum(muscle_activations ** 2) / max(delta_distance, 0.001 * self.dt)

        # The weight sould be negative. Not muscle_activation_penalize itself.
        # print(f"DEBUG:: {muscle_activations=}") # muscle: 0, 1, motor: -1, 1
        muscle_activation_penalize = self.dt * np.mean(np.exp(-4 * np.square(muscle_activations)))
        # case a) if delta_distance == 0 => muscle_activation_penalize -> -infinite
        # case b) if delta_distance < 0 => muscle_activation_penalize -> positive (which sould not)

        # The weight sould be negative. Not muscle_activation_penalize itself.
        if self.prev_muscle_activations_for_reward is not None:
            muscle_activation_diff_penalize = self.dt * np.mean(np.exp(-4 * np.square(self.prev_muscle_activations_for_reward - muscle_activations)))
            # print(f"DEBUG:: {muscle_activation_diff_penalize=}")
        else:
            muscle_activation_diff_penalize = 0
        self.prev_muscle_activations_for_reward = muscle_activations

        # head_acceleration_penalize = - self.dt * np.abs(self.sim.data.actuator("lumbar_extension_motor").ctrl[0].copy())
        # mujoco.mj_rnePostConstraint(self.sim.model, self.sim.data)
        # print(f"DEBUG:: self.sim.data.body('head').cacc.copy(): {self.sim.data.body('head').cacc.copy()}, norm: {np.linalg.norm(self.sim.data.body('head').cacc.copy())}")
        # head_acceleration_penalize = - self.dt * np.linalg.norm(self.sim.data.body("head").cacc.copy())

        rwd_dict = collections.OrderedDict((
            # Imitation Reward
            ('joint_imitation_reward', qpos_reward),
            ('end_effector_imitation_reward', anchor_reward),
            ('forward_reward', forward_reward),
            ('muscle_activation_penalize', muscle_activation_penalize),
            ('muscle_activation_diff_penalize', muscle_activation_diff_penalize),
            # ('head_acceleration_penalize',head_acceleration_penalize),
            # Optional Keys
            # ('reach',   10.-1.*reach_dist -10.*vel_dist),
            # ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            # ('act_reg', -100.*act_mag),
            # ('penalty', -1.*(reach_dist>far_th)),
            # # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            # ('done',    reach_dist > far_th),
            ('done',    self._get_done()),
        ))
        # print(f"reach_dist:{reach_dist}, far_th:{far_th}")
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    def _get_done(self):
        # if self._imitation_index is None:
        #     self._imitation_index = 0
        #     self._follow_reference_motion()

        # imitation_reward_th = np.exp(-0.5 * np.sum(np.square(np.array(0.3))))
        # if imitation_reward < imitation_reward_th:
        #     print(f"DEBUG:: RESET!! imitation_reward:{imitation_reward}<{imitation_reward_th}")
        #     return True
        pelvis_height = self.sim.data.joint('pelvis_ty').qpos[0].copy()
        if pelvis_height < self._safe_height:
            # print(f"DEBUG:: RESET!! pelvis_height:{pelvis_height}")
            return True
        return False
    def DEBUG_print_diff(self, prefix:str=""):
        print(f"DEBUG:: {prefix}")
        def change_qpos(sim_name:str, ref_name:str, offset:float=0.0,flip:bool=False):
            sign = 1.0 if not flip else -1.0
            qpos_idx = self.sim.model.joint(sim_name).qposadr
            print(f"DEBUG:: {sim_name} vs {ref_name} {self.sim.data.qpos[qpos_idx]} vs {sign*self._reference_data[ref_name][self._imitation_index] + offset}, {self.sim.data.qpos[qpos_idx] - (sign*self._reference_data[ref_name][self._imitation_index] + offset)}")
        for qv in self.QPOS_SIM_TO_REF.values():
            change_qpos(qv.sim_name, qv.ref_name, qv.offset, qv.is_flip)
    def _follow_reference_motion(self, object_qpos, object_qvel, x_follow:bool):
        for q_sim_to_ref in self.QPOS_SIM_TO_REF.values():
            
            sign = 1.0 if not q_sim_to_ref.is_flip else -1.0
            qpos_idx = self.sim.model.joint(q_sim_to_ref.sim_name).qposadr
            object_qpos[qpos_idx] = sign*self._reference_data[q_sim_to_ref.ref_name][self._imitation_index] + q_sim_to_ref.offset
            # self.sim.data.qpos[qpos_idx] = sign*self._reference_data[qv.ref_name][self._imitation_index] + qv.offset
            if not x_follow and q_sim_to_ref.sim_name == 'pelvis_tx':
                object_qpos[qpos_idx] = 0
        for q_sim_to_ref in self.QVEL_SIM_TO_REF.values():
            sign = 1.0 if not q_sim_to_ref.is_flip else -1.0
            qvel_idx = self.sim.model.joint(q_sim_to_ref.sim_name).qposadr
            object_qvel[qvel_idx] = sign*self._reference_data[q_sim_to_ref.ref_name][self._imitation_index] + q_sim_to_ref.offset
            # self.sim.data.qvel[qvel_idx] = 5 * sign*self._reference_data[qv.ref_name][self._imitation_index] + qv.offset
    def imitation_step(self, x_follow:bool, specific_index:int = None):
        if specific_index is None:
            self._imitation_index += self.REF_FRAME_SKIP
            if self._imitation_index >= self._reference_data_length:
                self._imitation_index = 0
        else:
            self._imitation_index = specific_index
        self._follow_reference_motion(self.sim.data.qpos, self.sim.data.qvel, x_follow)
        # should call this but I don't know why
        # next_obs, reward, terminated, truncated, info = super().step(np.zeros(self.sim.model.nu))
        # return (next_obs, reward, False, False, info)
        self.forward()
        return self._imitation_index
        # pass
    # overrides(BaseV0)
    def step(self, a, **kwargs):
        self._imitation_index += self.REF_FRAME_SKIP
        if self._imitation_index < self._reference_data_length:
            is_out_of_index = False
        else:
            is_out_of_index = True
            self._imitation_index = self._reference_data_length - 1
        # print(f"DEBUG:: ref index in step:  {self._imitation_index}")
        
        next_obs, reward, terminated, truncated, info = super().step(a, **kwargs)
        # self.DEBUG_print_diff()
        if is_out_of_index:
            is_out_of_trajectory = False
            reward = 0
            truncated = True
        else:
            diff_array, weight_array = self._get_qpos_diff()
            is_out_of_trajectory = np.any(np.abs(diff_array) >self._out_of_trajectory_threshold)
        self._step_count_per_episode += 1
        is_over_time_limit = self._step_count_per_episode >= self.CUSTOM_MAX_EPISODE_STEPS
        
        return (next_obs, reward, terminated, truncated or is_out_of_trajectory or is_over_time_limit, info)
        
    
    def setup_reference_data(self, data:dict):
        self._reference_data = data
        self._imitation_index = 0
        self._follow_reference_motion(self.sim.data.qpos, self.sim.data.qvel, False)
        self._reference_data_length = 0
        for key in self._reference_data.keys():
            if key != "meta_data":
                self._reference_data_length = len(self._reference_data[key])
                break



    # generate a valid target
    def generate_targets(self):
        for site, span in self.target_reach_range.items():
            sid = self.sim.model.site_name2id(site)
            sid_target = self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid_target] = self.sim.data.site_xpos[sid].copy() + self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()
    
    def get_pelvis_x(self):
        return self.sim.data.joint("pelvis_tx").qpos[0].copy()
    def reset(self, **kwargs):
        rng = np.random.default_rng()# TODO: refactoring random to use seed
        self._step_count_per_episode = 0
        if self._flag_random_ref_index:
            self._imitation_index = rng.integers(0, int(len(self._reference_data['q_pelvis_tx']) * 0.8))
        else:
            self._imitation_index = 0
        # print(f"DEBUG:: ref index in reset:  {self._imitation_index}")
        # generate random targets
        # new_qpos = self.generate_qpos()# TODO: should set qvel too.
        # self.sim.data.qpos = new_qpos
        self._follow_reference_motion(self.sim.data.qpos, self.sim.data.qvel, False)
        
        #TODO: Do we need or not????? reset does not do the step things. right? 
        # self._imitation_index += self.REF_SIM_FRAME_RATE
        self.sim.forward()
        self.generate_targets()

        # sync targets to sim_obsd
        self.robot.sync_sims(self.sim, self.sim_obsd)


        # generate resets
        obs = super().reset(reset_qpos= self.sim.data.qpos, reset_qvel=self.sim.data.qvel, **kwargs)
        return obs
