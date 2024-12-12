from package.mongodb_handler import MongoHandler
from bson import ObjectId
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pprint
import time
import os
import stable_baselines3
import pprint
from package.train_handler import TrainSession
from myosuite.utils import gym
import cv2
import mujoco
from datetime import datetime
import json
from copy import deepcopy

import skvideo.io
class GaitPropertyAnalyzer:
    def __init__(self):
        self.last_right_heelstrike_time = 0
        self.max_gait_duration = -1
        self.min_gait_duration = -1

        ## TODO: stride, ground clearance
class GaitCycleData:
    def __init__(self):
        self.gait_duration = 0
        self.sum_of_activation_squre = 0

        self.pelvis_x_series = []


        ## TODO: refactor later
        self.sim_datas = None
        self.hip_flexion_r = []
        self.hip_flexion_l = []
        self.knee_angle_r = []
        self.knee_angle_l = []
        self.ankle_angle_r = []
        self.ankle_angle_l = []
        self.r_foot_sensor = []
        self.l_foot_sensor = []
        self.r_toes_sensor = []
        self.l_toes_sensor = []
        self.pelvis_vx = []

def plot_result(object_id1:ObjectId,
                *,
                result_dir:str,
                show_train_log_plot:bool,
                flag_watch_trained_sim:bool,
                flag_need_video_rendering:bool,
                object1_idx:int=-1,
                )->GaitPropertyAnalyzer:
    db_name = "myosuite_training"
    mongo_handler = MongoHandler(db_name, "train_sessions")
    gait_property = GaitPropertyAnalyzer()

    session = mongo_handler.get_session_by_id_without_train_log(object_id1)
    print(f"session1: {pprint.pformat(session)}")
    train_session = TrainSession(db_name=db_name,
                                session_id=object_id1,
                                )
    base_data_file_path = 'neumove_models/reference_motions/02-constspeed_reduced_humanoid_segmented.npz'
    base_data_npz = np.load(base_data_file_path, allow_pickle=True)
    # keys = base_data_npz.files
    # ref_data_dict = {key: base_data_npz[key] for key in keys}
    base_data_dict = dict(base_data_npz)
                                

    fig, axs = plt.subplots(6,1, figsize=(15, 10))
    def plotting(mongo_handler:MongoHandler, object_id:ObjectId, ax:matplotlib.axes.Axes,
                *,
                x_key:str,
                y_key:str,
                window_size:int=1,
                # x_tick_interval:float,
                # y_tick_interval:float,
                title:str,
                color:tuple=None,
                linestyle:str=None,
                marker:str=None,
                y_log_scale:bool=False,
                y_mapper:callable=None):
        train_info_list = mongo_handler.get_train_log_list(object_id)
        if train_info_list is None or len(train_info_list) == 0:
            print("No train log data")
            return
        # print(train_info_list[0])
        ### logic under here ###

        std_values = np.array([item["std"] for item in train_info_list])
        start_index = np.where(std_values > -1)[0][0]
        # print(f"indices where std_values > -1: {start_index}")

        x_values = np.array([item[x_key] for item in train_info_list])[start_index:]
        y_values = np.array([item[y_key] for item in train_info_list])[start_index:]
        if y_mapper is not None:
            y_values = y_mapper(y_values)

        if y_log_scale:
            ax.set_yscale("log")

        y_ma = np.convolve(y_values, np.ones(window_size)/window_size, mode='valid')
        y_std = np.array([np.std(y_values[i:i+window_size]) for i in range(len(y_values) - window_size + 1)])
        ax.plot(x_values[window_size-1:], y_ma, color=color, marker=marker, linestyle=linestyle, label=f"{object_id}", linewidth=2)
        ax.fill_between(x_values[window_size-1:], y_ma - y_std, y_ma + y_std, alpha=0.2, color='m')

        # Add a legend
        ax.legend()
        ax.set_title(title)
        ax.grid(True)

        # Set x-axis limits
        #TODO: wierd when 6e7 ( xlim set to 1.5e7 )
        current_x_min, current_x_max = ax.get_xlim()
        x_range = x_values[-1] - x_values[0]
        x_tick_interval = 10 ** np.floor(np.log10(x_range / 5))
        while x_range / x_tick_interval > 10:
            x_tick_interval *= 5
        new_x_max = np.ceil((x_values[-1]) / x_tick_interval) * x_tick_interval
        
        if new_x_max > current_x_max:
            ax.set_xlim(0, new_x_max)

        current_y_min, current_y_max = ax.get_ylim()
        y_range = np.max(y_values) - np.min(y_values)
        # print(f"y_range: {y_range}") #TODO: why there is a error without this print?
        y_tick_interval = 10 ** np.floor(np.log10(y_range / 5))
        while y_range / y_tick_interval > 5:
            y_tick_interval *= 5
        new_y_min = np.floor((np.min(y_values) - y_tick_interval) / y_tick_interval) * y_tick_interval
        new_y_max = np.ceil((np.max(y_values) + y_tick_interval) / y_tick_interval) * y_tick_interval
        

        if np.isnan(new_y_min) or np.isinf(new_y_min) or new_y_min > current_y_min:
            new_y_min = current_y_min  # or set to a default value
        if np.isnan(new_y_max) or np.isinf(new_y_max) or new_y_max < current_y_max:
            new_y_max = current_y_max
        ax.set_ylim(new_y_min, new_y_max)

        x_ticks = np.arange(0, new_x_max + x_tick_interval, x_tick_interval)
        x_ticks = [int(tick) if tick.is_integer() else tick for tick in x_ticks]
        ax.set_xticks(x_ticks)

        if not y_log_scale:
            y_ticks_0 = np.arange(new_y_min, new_y_max + y_tick_interval, y_tick_interval)
            y_ticks_0 = [int(tick) if tick.is_integer() else tick for tick in y_ticks_0]
            ax.set_yticks(y_ticks_0)
        

    def plotting_all(mongo_handler, objectId, axs,
                    *,
                    color:tuple,
                    line_style:str
                    ):
        plotting(mongo_handler, objectId, axs[0],
                x_key="num_timesteps",
                y_key="average_reward_per_episode",
                title="reward",
                # x_tick_interval=2e5,
                # y_tick_interval=10,
                color=color,
                marker=None,
                y_log_scale=True,
                linestyle=line_style)
        plotting(mongo_handler, objectId, axs[1],
                x_key="num_timesteps",
                y_key="approx_kl",
                title="KL",
                # x_tick_interval=2e5,
                # y_tick_interval=10,
                color=color,
                marker=None,
                linestyle=line_style)
        plotting(mongo_handler, objectId, axs[2],
                x_key="num_timesteps",
                y_key="value_loss",
                title="value loss",
                # x_tick_interval=2e5,
                # y_tick_interval=10,
                color=color,
                marker=None,
                linestyle=line_style)
        plotting(mongo_handler, objectId, axs[3],
                x_key="num_timesteps",
                y_key="policy_gradient_loss",
                title="policy grad loss",
                # x_tick_interval=2e5,
                # y_tick_interval=10,
                color=color,
                marker=None,
                linestyle=line_style)
        plotting(mongo_handler, objectId, axs[4],
                x_key="num_timesteps",
                y_key="loss",
                title="loss",
                # x_tick_interval=2e5,
                # y_tick_interval=10,
                color=color,
                marker=None,
                linestyle=line_style)
        
        plotting(mongo_handler, objectId, axs[5],
                x_key="num_timesteps",
                y_key="std",
                title="log std",
                # x_tick_interval=2e5,
                # y_tick_interval=10,
                color=color,
                marker=None,
                linestyle=line_style,
                y_mapper=np.log,
                )
        

    # If we need multiple column, We can pass modified axs.
    plotting_all(mongo_handler, object_id1,axs,
                color = (0,0,0,1),
                line_style="-")

    fig.tight_layout()

    fig.savefig(os.path.join(result_dir, f"{object_id1}_{train_session.get_last_train_log()['num_timesteps']}.png"))
    fig.savefig(os.path.join(result_dir, f"{object_id1}_{train_session.get_last_train_log()['num_timesteps']}.svg"), format="svg")
    if show_train_log_plot:
        plt.show()

    ############## Save session config ###################
    session_config_file_path = os.path.join(result_dir, f"{object_id1}_session_config.txt")
    
    with open(session_config_file_path, 'w') as f:
        json.dump(session["session_config"], f, indent=4)



    ################# replay trained policy in sim ########################


    np.random.seed(0)
    # session = mongo_handler.get_session_by_id(objectId1)

    #TODO: remove data loading in analyze
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
    
    ## TODO: Do we need this?
    if "flag_random_ref_index" not in session["session_config"]["env_params"]:
        env_args_dict["flag_random_ref_index"] = True
    env_args_dict.update(session["session_config"]["env_params"])
    
    
    ##========================= Modify params for replay ======================
    modifed_env_params_for_replay = {}
    modifed_env_params_for_replay["custom_max_episode_steps"] = 1000000000 # make infinite episode
    modifed_env_params_for_replay["out_of_trajectory_threshold"] = 1000000
    # print(f"session_info: {pprint.pformat(session)}")
    for key in modifed_env_params_for_replay:
        if key not in env_args_dict:
            assert False, f"Error: The key '{key}' is not found in env_args_dict."
    env_args_dict.update(modifed_env_params_for_replay)
    print(f"env params modified for replay: {modifed_env_params_for_replay}")
    ##!========================= Modify params for replay ======================
    
    env_name = "myoLegRoughTerrainWalk2DImitationLearning-v0"

    envw = gym.make(env_name) if env_args_dict==None else gym.make(env_name, **(env_args_dict))
    env = envw.unwrapped
    env.seed(0)

    _ = env.unwrapped.sim.renderer.render_offscreen(camera_id=1,
                                                    width=1920,
                                                    height=1080)
    print(f"DEBUG:: {getattr(env.sim.renderer, 'render_offscreen', None)=}")
    print(f"DEBUG:: {env.sim.renderer.render_offscreen.__doc__=}")
    env.sim.renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
    
    train_log = train_session.get_specific_index_train_log(object1_idx)
    if train_log is not None:
        print(f"load from {train_log['model_file_path']}")
        model = stable_baselines3.PPO.load(train_log["model_file_path"], env=env)
        print(f"model.num_timesteps: {model.num_timesteps}")

        # flag_watch_trained_sim = input("Do you want to view the trained simulation? If not, it will be saved. (y/n)")
        if flag_watch_trained_sim:
            env.mujoco_render_frames = True
            # evaluation
            obs, info = env.reset()
            for time_step in range(10000):
                time.sleep(env.dt)
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, truncated, info = env.step(action)
                # env.render()
                if done or truncated:
                    obs, info = env.reset()
        else:
            
            video_path = os.path.join(result_dir,f"{object_id1}_t{model.num_timesteps}.mp4")
            
            env.mujoco_render_frames = False
            

            # max_length = 3 * 50#3sec * 50Hz TODO: freq from sim
            sim_datas = {
                "hip_flexion_r": [],
                "hip_flexion_l": [],
                "knee_angle_r": [],
                "knee_angle_l": [],
                "ankle_angle_r": [],
                "ankle_angle_l": [],
                "r_foot_sensor": [],
                "l_foot_sensor": [],
                "r_toes_sensor": [],
                "l_toes_sensor": [],

                "pelvis_vx": [],
            }
            
            gait_cycle_datas:list[GaitCycleData] = [GaitCycleData()]

            fig_l, axs_l = plt.subplots(4, 2, figsize=(10, 8))
            fig_r, axs_r = plt.subplots(4, 2, figsize=(10, 8))
            # evaluation
            frames = []
            obs, info = env.reset()
            # frame = env.sim.renderer.render_offscreen(camera_id=1,
            #                                                         width=1920,
            #                                                         height=1080)
            # print(f"DEBUG:: frame: {frame}")
            # frames.append(frame)
            DEFAULT_GAIT_STEP_DURATION = 1.2 * 50
            TARGET_VELOCITY = session["session_config"]["env_params"]["target_velocity"]
            flag_left_foot_contacted_atleast_once = False
            foot_sensor_threshold = 10
            gait_cycle_index = 0
            heel_strike_r_x = []
            toe_off_r_x = []
            for time_step in range(600):
                # time.sleep(env.dt)
                action, _states = model.predict(obs, deterministic=True)


                obs, rewards, done, truncated, info = env.step(action)

                frame = env.sim.renderer.render_offscreen(camera_id=1,
                                                                    width=1920,
                                                                    height=1080)
                data_length = len(sim_datas["knee_angle_r"])

                

                def draw_joint_angle(axs, row:int, data_r, data_l, std_r, std_l ,heel_strike_r_x,toe_off_r_x,plot_name:str, ylim:tuple[float,float],
                                     ref_data_l=None, ref_data_r=None):
                    ## R plot
                    if ref_data_r is not None:
                        axs[row,1].plot(ref_data_r, color='#777777')

                    axs[row,1].plot(data_r, color='#000000')#, label=f'Right {plot_name} Angle')
                    if std_r is not None:
                        axs[row, 1].fill_between(range(len(data_r)), 
                                [r - s for r, s in zip(data_r, std_r)], 
                                [r + s for r, s in zip(data_r, std_r)], 
                                color='#555555', alpha=0.5)
                    if len(heel_strike_r_x) > 0:
                        axs[row,1].scatter(heel_strike_r_x, [data_r[idx] for idx in heel_strike_r_x],
                                        marker="v",s=30, c="#ff0000", alpha=1)#, label=f'Heel Strike')#, cmap='viridis')
                    if len(toe_off_r_x) > 0:
                        axs[row,1].scatter(toe_off_r_x, [data_r[idx] for idx in toe_off_r_x],
                                        marker="^", s=30, c="#0000ff", alpha=1)#, label=f'Toe Off')#, cmap='viridis')
                    axs[row,1].set_ylim(*ylim)
                    axs[row,1].set_title(f'Right {plot_name} Angle')
                    axs[row,1].set_xlabel('Time Steps')
                    axs[row,1].set_ylabel('Angle')
                    # axs[row,1].legend()

                    ## L plot
                    if ref_data_l is not None:
                        axs[row,0].plot(ref_data_l, color='#777777')
                    axs[row,0].plot(data_l, color='#000000')#, label=f'Left {plot_name} Angle',)
                    if std_l is not None:
                        axs[row, 0].fill_between(range(len(data_l)), 
                                [r - s for r, s in zip(data_l, std_l)], 
                                [r + s for r, s in zip(data_l, std_l)], 
                                color='#555555', alpha=0.5)
                    axs[row,0].set_ylim(*ylim)
                    axs[row,0].set_title(f'Left {plot_name} Angle')
                    axs[row,0].set_xlabel('Time Steps')
                    axs[row,0].set_ylabel('Angle')
                    # axs[row,0].legend()
                def draw_figure():
                    draw_joint_angle(axs_l, 0, np.rad2deg(sim_datas[f"{'hip_flexion'}_r"]), np.rad2deg(sim_datas[f"{'hip_flexion'}_l"]),None, None, heel_strike_r_x, toe_off_r_x, "Hip", (-50,50))
                    draw_joint_angle(axs_l, 1, np.rad2deg(sim_datas[f"{'knee_angle'}_r"]), np.rad2deg(sim_datas[f"{'knee_angle'}_l"]), None, None, heel_strike_r_x, toe_off_r_x,  "Knee", (-70,10))
                    draw_joint_angle(axs_l, 2, np.rad2deg(sim_datas[f"{'ankle_angle'}_r"]), np.rad2deg(sim_datas[f"{'ankle_angle'}_l"]), None, None, heel_strike_r_x, toe_off_r_x,  "Ankle", (-30,30))

                    ## Draw Pelvis x Velocity

                    # axs_l[3,0].cla()
                    axs_l[3,0].plot(sim_datas[f"pelvis_vx"], color='#000000')#, label=f'Pelvis Velocity')
                    axs_l[3,0].axhline(y=TARGET_VELOCITY, color='#555555', linestyle='--')#, label='Target Velocity')
                    axs_l[3,0].set_xlabel('Time Steps')
                    axs_l[3,0].set_ylabel('Velocity [m/s]')
                    axs_l[3,0].set_ylim(0,4)
                    # axs_l[3,0].legend()


                    ###########################################################

                    for axs_row in axs_l:
                        for ax in axs_row:
                            if data_length < DEFAULT_GAIT_STEP_DURATION:
                                ax.set_xlim(0,DEFAULT_GAIT_STEP_DURATION)
                    for axs_row in axs_r:
                        for ax in axs_row:
                            if data_length < DEFAULT_GAIT_STEP_DURATION:
                                ax.set_xlim(0,DEFAULT_GAIT_STEP_DURATION)
                    
                    ## graph to numpy array
                    # ax.axis('off')  # hide axis
                    # plt.tight_layout()
                    # plt.draw()

                    ## fig L draw
                    fig_l.tight_layout()
                    fig_l.canvas.draw()
                    # plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    plot_img_l = np.array(fig_l.canvas.buffer_rgba(), dtype=np.uint8)
                    plot_img_l = plot_img_l.reshape(fig_l.canvas.get_width_height()[::-1] + (4,))
                    # plt.close(fig)

                    plot_img_rgb_l = cv2.cvtColor(plot_img_l, cv2.COLOR_RGBA2RGB)

                    target_height = frame.shape[0] // 2
                    aspect_ratio = plot_img_rgb_l.shape[1] / plot_img_rgb_l.shape[0]
                    target_width = int(target_height * aspect_ratio)

                    plot_img_rgb_l = cv2.resize(plot_img_rgb_l, (target_width, target_height))
                    frame[0:plot_img_rgb_l.shape[0], 0:plot_img_rgb_l.shape[1]] = plot_img_rgb_l

                    ## fig R draw
                    fig_r.tight_layout()
                    fig_r.canvas.draw()
                    # plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    plot_img_r = np.array(fig_r.canvas.buffer_rgba(), dtype=np.uint8)
                    plot_img_r = plot_img_r.reshape(fig_r.canvas.get_width_height()[::-1] + (4,))
                    # plt.close(fig)

                    plot_img_rgb_r= cv2.cvtColor(plot_img_r, cv2.COLOR_RGBA2RGB)

                    target_height = frame.shape[0] // 2
                    aspect_ratio = plot_img_rgb_r.shape[1] / plot_img_rgb_r.shape[0]
                    target_width = int(target_height * aspect_ratio)

                    plot_img_rgb_r = cv2.resize(plot_img_rgb_r, (target_width, target_height))
                    frame[0:plot_img_rgb_r.shape[0]:, frame.shape[1] - plot_img_rgb_r.shape[1]:] = plot_img_rgb_r

                    ## drawing end
                
                r_foot_sensor_value = env.sim.data.sensor('r_foot').data.copy()
                l_foot_sensor_value = env.sim.data.sensor('l_foot').data.copy()
                r_toes_sensor_value = env.sim.data.sensor('r_toes').data.copy()
                l_toes_sensor_value = env.sim.data.sensor('l_toes').data.copy()
                
                window_size = 2
                if data_length >= window_size:
                    ##detect heel strike and foot contact in realtime
                    

                    r_foot_sensor_sum_value =\
                            np.average(np.concatenate([[r_foot_sensor_value],sim_datas["r_foot_sensor"][-window_size+1:]]))\
                                + np.average(np.concatenate([[r_toes_sensor_value], sim_datas["r_toes_sensor"][-window_size+1:]]))

                    prev_r_foot_sensor_sum_value = np.average(sim_datas["r_foot_sensor"][-window_size:]) + np.average(sim_datas["r_toes_sensor"][-window_size:])

                    # foot prev, current
                    flag_right_foot_prev_contact = prev_r_foot_sensor_sum_value > foot_sensor_threshold
                    flag_right_foot_currently_contact = r_foot_sensor_sum_value > foot_sensor_threshold

                    r_foot_heel_strike =  (not flag_right_foot_prev_contact) and flag_right_foot_currently_contact
                    r_foot_toe_off = flag_right_foot_prev_contact and (not flag_right_foot_currently_contact)
                    if r_foot_heel_strike:
                        draw_figure()
                        for axs_row_l, axs_row_r in zip(axs_l, axs_r):
                            axs_row = np.concatenate((axs_row_l, axs_row_r))
                            for ax in axs_row:
                                ax.set_xlim(0,data_length-1)
                                ax.set_xticks(np.linspace(0, data_length - 1, num=5))
                                ax.set_xticklabels([f"{int(x / (data_length - 1) * 100)}%" for x in np.linspace(0, data_length - 1, num=5)])
                                ax.set_xlabel('Gait Cycle')

                        fig_l.savefig(os.path.join(result_dir,f"{object_id1}_t{model.num_timesteps}_gait{gait_cycle_index}_L.png"))
                        fig_l.savefig(os.path.join(result_dir,f"{object_id1}_t{model.num_timesteps}_gait{gait_cycle_index}_L.svg"), format="svg")
                        fig_r.savefig(os.path.join(result_dir,f"{object_id1}_t{model.num_timesteps}_gait{gait_cycle_index}_R.png"))
                        fig_r.savefig(os.path.join(result_dir,f"{object_id1}_t{model.num_timesteps}_gait{gait_cycle_index}_R.svg"), format="svg")
                        gait_cycle_index += 1
                        gait_cycle_datas[-1].sim_datas = deepcopy(sim_datas)
                        for key in sim_datas:
                            sim_datas[key].clear()

                        gait_cycle_datas.append(GaitCycleData())
                
                ### plot preprocessing
                for axs_row_l, axs_row_r in zip(axs_l, axs_r):
                    axs_row = np.concatenate((axs_row_l, axs_row_r))
                    for ax in axs_row:
                        ax.cla() #clear
                        # ax.cla() #set to default axis

                        ax.xaxis.set_major_locator(mticker.AutoLocator())
                        ax.xaxis.set_minor_locator(mticker.NullLocator())  # If you had removed minor ticks

                        ax.yaxis.set_major_locator(mticker.AutoLocator())
                        ax.yaxis.set_minor_locator(mticker.NullLocator())

                        # Reset the tick formatters to default
                        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

                        # Reset axis labels if they were cleared
                        ax.set_xlabel('')  # Or set to desired label
                        ax.set_ylabel('')  # Or set to desired label



                sim_datas["hip_flexion_r"].append(env.sim.data.joint('hip_flexion_r').qpos[0])
                sim_datas["hip_flexion_l"].append(env.sim.data.joint('hip_flexion_l').qpos[0])
                sim_datas["knee_angle_r"].append(env.sim.data.joint('knee_angle_r').qpos[0])
                sim_datas["knee_angle_l"].append(env.sim.data.joint('knee_angle_l').qpos[0])
                sim_datas["ankle_angle_r"].append(env.sim.data.joint('ankle_angle_r').qpos[0])
                sim_datas["ankle_angle_l"].append(env.sim.data.joint('ankle_angle_l').qpos[0])
                sim_datas["r_foot_sensor"].append(r_foot_sensor_value)
                sim_datas["l_foot_sensor"].append(l_foot_sensor_value)
                sim_datas["r_toes_sensor"].append(r_toes_sensor_value)
                sim_datas["l_toes_sensor"].append(l_toes_sensor_value)

                # print(f"DEBUG:: {env.sim.data.act[:].copy()=}")
                # print(f"DEBUG:: {gait_cycle_datas[-1].sum_of_activation_squre=}")
                # print(f'DEBUG:: {env.sim.data.joint("pelvis_tx").qpos[0].copy()=}')
                gait_cycle_datas[-1].sum_of_activation_squre += np.sum(np.square(env.sim.data.act[:].copy())) * env.dt
                gait_cycle_datas[-1].gait_duration += env.dt
                gait_cycle_datas[-1].pelvis_x_series.append(env.sim.data.joint("pelvis_tx").qpos[0].copy())


                sim_datas["pelvis_vx"].append(env.sim.data.joint("pelvis_tx").qvel[0].copy())

                data_length = len(sim_datas['r_foot_sensor']) # calc length again incase it cleared before
                

                # Scatter heel strike and toe off
                heel_strike_r_x.clear()
                toe_off_r_x.clear()
                for temp_time_step in range(data_length):
                    if temp_time_step > window_size:#window_size + 1 for prev
                        temp_prev_r_foot_contact = np.average(sim_datas["r_foot_sensor"][temp_time_step-window_size-1:temp_time_step-1]) > foot_sensor_threshold
                        temp_prev_r_toes_contact = np.average(sim_datas["r_toes_sensor"][temp_time_step-window_size-1:temp_time_step-1]) > foot_sensor_threshold
                        temp_prev_r_foot_contact_or = temp_prev_r_foot_contact or temp_prev_r_toes_contact
                        temp_current_r_foot_contact = np.average(sim_datas["r_foot_sensor"][temp_time_step-window_size:temp_time_step]) > foot_sensor_threshold
                        temp_current_r_toes_contact = np.average(sim_datas["r_toes_sensor"][temp_time_step-window_size:temp_time_step]) > foot_sensor_threshold
                        temp_current_r_foot_contact_or = temp_current_r_foot_contact or temp_current_r_toes_contact

                        # heel strike R
                        if not temp_prev_r_foot_contact_or and temp_current_r_foot_contact_or:
                            heel_strike_r_x.append(temp_time_step)
                        # toe off R
                        if temp_prev_r_foot_contact_or and not temp_current_r_foot_contact_or:
                            toe_off_r_x.append(temp_time_step)
                ###########################################################
                
                
                
                
                if flag_need_video_rendering:
                    draw_figure()
                    

                # plot_img_rgb = cv2.resize(plot_img_rgb, (frame.shape[1] // 4, frame.shape[0] // 4))
                # frame[0:plot_img_rgb.shape[0], 0:plot_img_rgb.shape[1]] = plot_img_rgb
                if time_step == 0:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(result_dir,f"{object_id1}_t{model.num_timesteps}_0.png"), frame_bgr)

                frames.append(frame)

                # if len(sim_datas["knee_angle_r"]) >= max_length:
                #     print(sim_datas["r_foot_sensor"])
                #     for key in sim_datas:
                #         sim_datas[key].clear()

                # env.render()
                if done or truncated:
                    obs, info = env.reset()
                    for key in sim_datas:
                        sim_datas[key].clear()

            ### FINAL ANALYZE
            # activation_squres = np.array([gait_cycle_data.sum_of_activation_squre for gait_cycle_data in gait_cycle_datas])
            # gait_durations = np.array([gait_cycle_data.gait_duration for gait_cycle_data in gait_cycle_datas])
            activation_squre_sums = []
            gait_durations = []
            activation_squre_per_distance = []
            distances = []
            kinematics = {
                "hip_flexion_r": [],
                "hip_flexion_l": [],
                "knee_angle_r": [],
                "knee_angle_l": [],
                "ankle_angle_r": [],
                "ankle_angle_l": [],
            }
            for idx, gait_cycle_data in enumerate(gait_cycle_datas):
                if idx > 0 and idx < len(gait_cycle_datas) - 1 and gait_cycle_data.gait_duration > 0.3 and gait_cycle_data.sim_datas is not None:
                    activation_squre_sums.append(gait_cycle_data.sum_of_activation_squre)
                    gait_durations.append(gait_cycle_data.gait_duration)
                    distance = gait_cycle_data.pelvis_x_series[-1] - gait_cycle_data.pelvis_x_series[0]
                    distances.append(distance)
                    activation_squre_per_distance.append(gait_cycle_data.sum_of_activation_squre/distance)
                    for k in kinematics.keys():
                        kinematics[k].append(gait_cycle_data.sim_datas[k])

            mean = np.mean(activation_squre_per_distance)
            std = np.std(activation_squre_per_distance)

            
            fig, ax = plt.subplots()

            # Plot the ratios
            ax.bar(range(len(activation_squre_per_distance)), activation_squre_per_distance, label='Activation/Distance')

            ax.axhline(mean, color='#000000', linestyle='-', label='Mean')
            ax.axhline(mean + std, color='#777777', linestyle='--', label='+ 1 SD')
            ax.axhline(mean - std, color='#777777', linestyle='--', label='- 1 SD')

            # Add labels and title
            ax.set_xlabel('Gait Cycle Index')
            ax.set_ylabel('Activation Square Sum / Distance')
            ax.set_title('Activation Square Sum to Distance Ratio for Each Gait Cycle')

            ax.set_ylim(0, 5)

            # Add a legend
            ax.legend()

            # Display the plot
            fig.savefig(os.path.join(result_dir, f"activation_square_{object_id1}_{train_session.get_last_train_log()['num_timesteps']}.png"))
            fig.savefig(os.path.join(result_dir, f"activation_square_{object_id1}_{train_session.get_last_train_log()['num_timesteps']}.svg"), format="svg")

            ############################################################
            fig, axs = plt.subplots(3, 2, figsize=(10, 8))
            interpolated_data_dict = {}
            for key in kinematics.keys():
                interpolated_data = []
                common_x = np.linspace(0, 100, 101)
                for series in kinematics[key]:
                    original_x = np.linspace(0, 100, len(series))
                    interpolated_series = np.interp(common_x, original_x, series)
                    interpolated_data.append(interpolated_series)

                # Compute the average across all interpolated data
                average_data = np.mean(interpolated_data, axis=0)
                std_data = np.std(interpolated_data, axis=0)

                interpolated_data_dict[key] = {"average_data":average_data,
                                               "std_data":std_data,
                                               }
                
                
            draw_joint_angle(axs, 0, 
                             np.rad2deg(interpolated_data_dict[f"{'hip_flexion'}_r"]["average_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'hip_flexion'}_l"]["average_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'hip_flexion'}_r"]["std_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'hip_flexion'}_l"]["std_data"]),
                             [],[],
                             "Hip", (-50,50),
                             ref_data_r= np.rad2deg(base_data_dict["q_hip_flexion_r"]),
                             ref_data_l= np.rad2deg(base_data_dict["q_hip_flexion_l"]),
                             )
            draw_joint_angle(axs, 1, 
                             np.rad2deg(interpolated_data_dict[f"{'knee_angle'}_r"]["average_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'knee_angle'}_l"]["average_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'knee_angle'}_r"]["std_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'knee_angle'}_l"]["std_data"]),
                             [],[],
                             "Knee", (-70,10),
                             ref_data_r= np.rad2deg(base_data_dict["q_knee_angle_r"]),
                             ref_data_l= np.rad2deg(base_data_dict["q_knee_angle_l"]),
                             )
            draw_joint_angle(axs, 2, 
                             np.rad2deg(interpolated_data_dict[f"{'ankle_angle'}_r"]["average_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'ankle_angle'}_l"]["average_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'ankle_angle'}_r"]["std_data"]),
                             np.rad2deg(interpolated_data_dict[f"{'ankle_angle'}_l"]["std_data"]),
                             [],[],
                             "Ankle", (-30,30),
                             ref_data_r= np.rad2deg(base_data_dict["q_ankle_angle_r"]),
                             ref_data_l= np.rad2deg(base_data_dict["q_ankle_angle_l"]),
                             )

            for axs_row in axs:
                for ax in axs_row:
                    ax.set_xlim(0,100)
                    ax.set_xticks(np.linspace(0, 100, num=5))
                    ax.set_xticklabels([f"{int(x)}%" for x in np.linspace(0, 100, num=5)])
                    ax.set_xlabel('Gait Cycle')
            fig.tight_layout()

            fig.savefig(os.path.join(result_dir,f"average_kinematics_{object_id1}_t{model.num_timesteps}.png"))
            fig.savefig(os.path.join(result_dir,f"average_kinematics_{object_id1}_t{model.num_timesteps}.svg"), format="svg")

            ### SAVE VIDEO
            
            skvideo.io.vwrite(video_path, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})

    env.close()
    print("Analyze done!")

    return gait_property


if __name__ == "__main__":
    object_id1_str = input("object id 1: ")
    if object_id1_str == "":
        object_id1 = None
    else:
        object_id1 = ObjectId(object_id1_str)

    object1_idx_str = input("object 1 idx(if no input, it will be last): ")
    if object1_idx_str == "":
        object1_idx = -1
    else:
        object1_idx = int(object1_idx_str)

    flag_watch_trained_sim = input("Do you want to view the trained simulation? If not, it will be saved. (y/n)")
    if flag_watch_trained_sim == "y":
        flag_watch_trained_sim = True
    else:
        flag_watch_trained_sim = False

    flag_need_video_rendering = input("Do you want to render the video (y/n)")
    if flag_need_video_rendering == "y":
        flag_need_video_rendering = True
    else:
        flag_need_video_rendering = False

    now = datetime.now()
    formatted_timestamp = now.strftime("%Y_%m_%d__%H_%M_%S")
    result_dir = os.path.join("train_results", f"session_{formatted_timestamp}_{object_id1}")
    os.makedirs(result_dir, exist_ok=True)
    plot_result(object_id1,
                result_dir=result_dir,
                show_train_log_plot=True,
                object1_idx=object1_idx,
                flag_watch_trained_sim=flag_watch_trained_sim,
                flag_need_video_rendering=flag_need_video_rendering
                )