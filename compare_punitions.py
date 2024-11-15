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
import matplotlib.pyplot as plt

dirpath = "TD3\\"
nb_walkers = 3
tensordir = "log/"



def load(latest_policy):
    return TD3.load(latest_policy)


def eval_one_model(env_fn, model, num_games: int = 100, render_mode: str = None, **env_kwargs):
    env = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)


    rewards = {agent: 0 for agent in range(nb_walkers)}
    reward_plot = []

    for ct in range(num_games):
        obs = env.reset()  # Reset the environment and get the initial observations
        done = [False] * nb_walkers
        trunc = [False] * nb_walkers

        obs = obs[0]
        tmp = 0

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
               tmp += reward[i]
            
        reward_plot.append(tmp / 3)

        
    avg_reward = sum(rewards.values()) / len(rewards)
    print(f"Average reward: {avg_reward}")
    return reward_plot



def get_four_models():
    m100 = "TD3\\1\\20241115-150036/"
    m20 = "TD3\\1\\20241115-154102/"
    m10 = "TD3\\1\\20241115-163110/"
    m1 = "TD3\\1\\20241115-171115/"
    
    lis = {-100 : m100, -20 : m20, -10 : m10, -1 : m1}
    models = {}

    for jey, path in lis.items():
        for p in glob.glob(path + "*.zip"):
            m = load(p)
            models[jey] = m
    
    return models


if __name__ == "__main__":
    env_fn = multiwalker_v9
    env_kwargs = {'n_walkers':nb_walkers,'shared_reward':False, "fall_reward": -10, "terminate_reward" : -20, "terminate_on_fall" : False, "remove_on_fall" : False}

   
    models = get_four_models()
    rewards = {}
    for key, model in models.items():

        blip = eval_one_model(env_fn, model, num_games=3, render_mode=None, **env_kwargs)
        rewards[key] = blip
    

    print(rewards)

    for key, val in rewards.items():
        plt.plot(val, marker = 'o',  label = key)
    

    plt.xlabel("N° de partie")
    plt.ylabel("Récompense obtenue")

    plt.title("Comparaison des récompenses obtenues par 4 modèles entrainés en faisant varier la sévérité des punitions")

    plt.legend()
    plt.show()

        