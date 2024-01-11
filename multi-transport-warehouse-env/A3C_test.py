from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


import gymnasium as gym
import numpy as np
import warehouse_multi_transporter
import warehouse_transporter_env

def learning_rate_f(progress_remaining):
    """
        Learning rate function that starts at 0.001 and linearly decays to 0.
        """
    return 0.001 * (1 + np.cos(np.pi * (1-progress_remaining))) / 2


def cos_annealing_lr(progress_remaining):

    return 0.001 * (1+ np.cos(np.pi * progress_remaining)) /2

if __name__ == '__main__':
    env1 = warehouse_multi_transporter.warehouse_multi_agent(6, 6, 1, [(0, 5)], (4, 2), 4,
                                                             ((1, 2), (3, 3), (2, 2), (3, 2), (0, 2)))

    target_pos = np.array((4, 5))
    print(target_pos.shape)
    agent_pos = np.array((6, 9))
    block_pos = np.array([
        [0, -1, -1, -1, 0, 0, -1, 0, 0, 0],
        [0, 0, -1, 0, 0, -1, 0, 0, -1, 0],
        [0, 0, -1, 0, 0, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, -1, 0, 0, -1, 0, 0],
        [0, -1, 0, 0, 0, 0, -1, 0, -1, 0],
        [0, -1, 0, 0, 0, 0, -1, 0, -1, 0],
        [-1, 0, -1, 0, 0, 0, 0, 0, 0, 0]
    ])

    env2 = warehouse_transporter_env.warehouse_env(10, 10, target_pos, block_pos, agent_pos)
    model_A2C = A2C("MlpPolicy", env2, seed=42, learning_rate=learning_rate_f, verbose=1,
                tensorboard_log="./a2c_warehouse_tensorboard/",
                policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-4)))

    model_PPO = PPO("MlpPolicy", env2, seed=42, learning_rate=1e-3, verbose=1, tensorboard_log="./PPO_warehouse_tensorboard/", policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-4)))
    eval_callback = EvalCallback(env2, best_model_save_path='./logs', log_path='./logs', eval_freq=500,
                                 deterministic=True, render=False)

    # model_A2C.learn(total_timesteps=30000)
    # model_A2C.save("A2C_model_10x10.pth")
    model_PPO.learn(total_timesteps=50000)
    model_PPO.save("PPO_model_10x10.pth")