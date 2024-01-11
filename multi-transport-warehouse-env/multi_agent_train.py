import time

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from  stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import numpy as np
from grid_multi_agents import MultiAgentWarehouseEnv
import pygame
import sys

def learning_rate_f(progress_remaining):
    """
        Learning rate function that starts at 0.001 and linearly decays to 0.
        """
    return 0.001 * (1 + np.cos(np.pi * (1-progress_remaining))) / 2


def cos_annealing_lr(progress_remaining):

    return 0.001 * (1+ np.cos(np.pi * progress_remaining)) /2


if __name__ == '__main__':
    block_pos = np.array([
        [0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ])
    env = MultiAgentWarehouseEnv(
        10, 10,
        [[5, 3], [6, 6],[9,4],[0,4]],  # 目标位置
        block_pos,  # 障碍物位置
        [[1, 1], [2, 2],[7,1],[4,7]]  # 智能体初始位置
    )
    env = make_vec_env(lambda: env, n_envs=1)
    env.reset()
    env.render()
    #time.sleep(100)

    model_PPO = PPO("MlpPolicy", env, verbose=1, seed=42, learning_rate=cos_annealing_lr, clip_range=0.2,
    tensorboard_log="./PPO_multi_agent_warehouse_tensorboard/ppo_2_agents_lr_cos_cr0.2", policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-4)))
    eval_callback = EvalCallback(env, best_model_save_path='./logs', log_path='./logs', eval_freq=500, deterministic=True, render=False)

    #model_DDPG = DDPG("MlpPolicy", env, verbose=1, seed=42, learning_rate=1e-4, tensorboard_log="./DDPG_multi_agent_warehouse_tensorboard/", policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-4)))

    model_PPO.learn(
        total_timesteps=300000,
        log_interval=1,
        tb_log_name="ppo_multi_agent_warehouse"
    )
    #model_DDPG.save("./model_saved/ddpg_multi_agent_warehouse.pth")
    model_PPO.save("./model_saved/ppo_multi_agent_warehouse.pth")
    print("Training finished!")
    pygame.quit()
    sys.exit()