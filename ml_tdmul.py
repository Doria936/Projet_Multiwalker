from __future__ import annotations
import supersuit as ss
from pettingzoo.sisl import multiwalker_v9
from stable_baselines3 import PPO, TD3
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import os
import time
import glob
import numpy as np
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import pyautogui

dirpath = "TD3/"
log_dir = "./logs/"

nb_walkers = 3

def capture_window(title, output_filename="output.avi", fps=60, record_seconds=30):
    # 查找特定标题的窗口
    window = gw.getWindowsWithTitle(title)[0]
    # 确保窗口是活动的和最前面的（根据你的系统设置，这可能需要调整）
    window.activate()
    window.moveTo(0, 0)

    # 获取窗口的位置和大小
    x, y, width, height = window.left, window.top, window.width, window.height

    # 定义编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # 计算录制的帧数
    num_frames = fps * record_seconds

    for _ in range(num_frames):
        # 对指定区域进行屏幕捕捉
        img = pyautogui.screenshot(region=(x, y, width, height))
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色从BGR到RGB
        out.write(frame)  # 写入帧到文件

    # 完成后释放资源
    out.release()
    cv2.destroyAllWindows()


def get_save_path(env, steps, time):
    return dirpath + str(nb_walkers) + "/" + time +"/" + f"{steps}steps"


def create_models_for_agents(env):
    # Initialisation des modèles pour chaque agent
    models = []

    for _ in range(nb_walkers):
        noise = NormalActionNoise(mean=0.1 * np.zeros(4), sigma=0.2 * np.ones(4))

        model = TD3(
            "MlpPolicy", 
            env, 
            verbose=0, 
            learning_starts=100_000,
            learning_rate=1e-3,
            buffer_size=1e6,
            action_noise = noise
            # tensorboard_log=log_dir
        )
        models.append(model)

    return models

def load(latest_policy):
    return TD3.load(latest_policy)


def train_sisl_supersuit_multiple_models(env_fn, steps:int = 10_000, **kwargs):

    env = env_fn.parallel_env(**kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, 8, base_class="stable_baselines3")

    models = create_models_for_agents(env)

    # Pour chaque agent, on entraîne son propre modèle

    for agent_idx in range(len(models)):
        model = models[agent_idx]
        #env.seed(np.random.randint(0, 2**31 - 1))
        env.reset()

        model.learn(total_timesteps=steps, progress_bar=True, log_interval=100000)

    ti = time.strftime('%Y%m%d-%H%M%S')
    
    for i, model in enumerate(models):
        model.save(get_save_path(env,steps, i, ti))

    env.close()


def eval_multiple_models(env_fn, models, num_games: int = 100, render_mode: str = None, **env_kwargs):
    env = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    rewards = {agent: 0 for agent in range(nb_walkers)}
    
    for ct in range(num_games):
        obs = env.reset()  # Reset the environment and get the initial observations
        done = [False] * nb_walkers
        trunc = [False] * nb_walkers

        obs = obs[0]

        while not all(done) or not all(trunc):
            actions = []  # Initialize a list to hold actions for all agents
            
            for agent_idx in range(nb_walkers):
                
                if done[agent_idx]:  # If the agent is done, set its action to None
                    actions.append(None)
                    # agents_obs.insert(agent_idx, None)
                else:
                    action, _ = models[agent_idx].predict(obs[agent_idx], deterministic=True)  # Get the action from the model
                    actions.append(action)  # Append the action for this agent
            
            # Take a step with all the agents at once
            result = env.step(actions)  # Pass the actions for all agents simultaneously
            
            # Depending on the structure of the result, unpack accordingly
            # Example: if result is a tuple with 4 elements (obs, rewards, dones, infos)
            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, done, trunc, info = result
            else:
                print("Unexpected step result format:", len(result))
                break  

            for i in range(nb_walkers):
               rewards[i] += reward[i]  # Accumulate the reward for each agent 

        print(f"Game {ct + 1}: rewards={rewards}")

    avg_reward = sum(rewards.values()) / len(rewards)
    print(f"Average reward: {avg_reward}")
    return avg_reward


def get_models():
    try:
        latest_policy = max(glob.glob(dirpath + str(nb_walkers) + '/' + f"*/"))
    except ValueError:
        print("Policy not found.")
        exit(0)

    models = []

    for p in glob.glob(latest_policy + "*.zip"):
        m = load(p)
        models.append(m)
    
    return models
    

def get_one_model():
    try:
        # print(os.listdir(dirpath))
        latest_policy = max(glob.glob(dirpath + str(nb_walkers) + '/' + f"*/"))
        # latest_policy = max(glob.glob(dirpath + str(2) + '/' + f"*/"))
        # latest_policy = os.listdir(dirpath)[1]
        print(f'{latest_policy=}')
    except ValueError:
        print("Policy not found.")
        exit(0)

    models = []
    # m = load(latest_policy)
    # models.append(m)

    for p in glob.glob(latest_policy + "*.zip"):
        print(f'{p=}')
        m = load(p)
        models.append(m)
    
    return models.pop()


def get_save_path_one_walker(env, steps, time):
    return dirpath + str(1) + "/" + time +"/" + f"0walker_{steps}steps"

def create_one_model(env):
    # Initialisation des modèles pour chaque agent
    
    # noise = NormalActionNoise(mean=0.1 * np.zeros(4), sigma=0.2 * np.ones(4))

    noise = OrnsteinUhlenbeckActionNoise(mean= 0 * np.ones(4), sigma=0.2 * np.ones(4))
    
    model = TD3(
        "MlpPolicy", 
        env, 
        verbose=0, 
        learning_starts=100_000,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        action_noise = noise,
        batch_size=512,
        tensorboard_log=log_dir
    )
    return model


def train_sisl_supersuit_one_models(env_fn, steps:int = 10_000, **kwargs):

    env = env_fn.parallel_env(**kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, 8, base_class="stable_baselines3")

    model = create_one_model(env)

    # Pour chaque agent, on entraîne son propre modèle
     
    #env.seed(np.random.randint(0, 2**31 - 1))
    env.reset()

    # tensorboard_callback = TensorboardCallback()
    eval_callback = EvalCallback(env, best_model_save_path=dirpath+str(nb_walkers)+'/', log_path=log_dir, eval_freq=10000, deterministic=True, render=True)
    model.learn(total_timesteps=steps, progress_bar=True, callback=[eval_callback])

    # model.learn(total_timesteps=steps, progress_bar=True)

    ti = time.strftime('%Y%m%d-%H%M%S')
    
    # model.save(get_save_path_one_walker(env,steps, ti))
    model.save(get_save_path(env,steps, ti))

    env.close()

def eval_one_model(env_fn, model, num_games: int = 100, render_mode: str = None, **env_kwargs):
    env = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    rewards_all = np.zeros((num_games,nb_walkers))
    frames = []
    for ct in range(num_games):
        rewards = {agent: 0 for agent in range(nb_walkers)}
        obs = env.reset()  # Reset the environment and get the initial observations
        done = [False] * nb_walkers
        trunc = [False] * nb_walkers

        obs = obs[0]

        while not all(done) or not all(trunc):
            actions = []  # Initialize a list to hold actions for all agents
            
            for agent_idx in range(nb_walkers):
                
                if done[agent_idx]:  # If the agent is done, set its action to None
                    actions.append(None)
                    # agents_obs.insert(agent_idx, None)
                else:
                    action, _ = model.predict(obs[agent_idx], deterministic=True)  # Get the action from the model
                    actions.append(action)  # Append the action for this agent
            
            # Take a step with all the agents at once
            result = env.step(actions)  # Pass the actions for all agents simultaneously
            
            # Depending on the structure of the result, unpack accordingly
            # Example: if result is a tuple with 4 elements (obs, rewards, dones, infos)
            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, done, trunc, info = result
            else:
                print("Unexpected step result format:", len(result))
                break  

            for i in range(nb_walkers):
               rewards[i] += reward[i]  # Accumulate the reward for each agent 

        for i in range(3):
            rewards_all[ct][i] = rewards[i]
        print(f"Game {ct + 1}: rewards={rewards}")
        print(f'{rewards_all=}')

    avg_reward = sum(rewards.values()) / len(rewards)
    avg_rewards_all_games = np.mean(rewards_all, axis=1)
    # print(f"Average reward: {avg_reward}")
    print(f"All average rewards: {avg_rewards_all_games}")
    return avg_reward

if __name__ == "__main__":
    # env_fn = multiwalker_v9
    # env_kwargs = {'n_walkers':nb_walkers,'shared_reward':False, "fall_reward": -10, "terminate_reward" : -20, "terminate_on_fall" : False, "remove_on_fall" : False}

    # # Entraînement
    # start = time.time()
    # train_sisl_supersuit_multiple_models(env_fn, steps=1_500_000, **env_kwargs)
    # end = time.time()
    # print(f'Training time: {end-start}')
   
    # # Évaluation
    
    # models = get_models()

    # eval_multiple_models(env_fn, models, num_games=5, render_mode="human", **env_kwargs)

    env_fn = multiwalker_v9
    env_kwargs = {'n_walkers':nb_walkers, 'shared_reward':False, "fall_reward": -1, "terminate_reward" : -10, 
                  "terminate_on_fall" : False, "remove_on_fall" : False, "forward_reward" : 2,
                  "max_cycles":1100}
    # meilleur config pr l'instant

    # Entraînement
    start = time.time()
    # train_sisl_supersuit_one_models(env_fn, steps=3_000_000, **env_kwargs)
    end = time.time()
    print(f'Training time: {end-start}')
   
    # Évaluation
    # nb_walkers_eval = 2
    model = get_one_model()
    # env_kwargs_eval = {'n_walkers':nb_walkers_eval,'shared_reward':False, "fall_reward": -5, "terminate_reward" : -1, 
    #               "terminate_on_fall" : False, "remove_on_fall" : False, "forward_reward" : 0.3,
    #               "max_cycles":500}
    eval_one_model(env_fn, model, num_games=5, render_mode="human", **env_kwargs)
    
    # 使用窗口标题调用函数
    # capture_window("Multiwalker",output_filename=str(time.time)+'.avi')