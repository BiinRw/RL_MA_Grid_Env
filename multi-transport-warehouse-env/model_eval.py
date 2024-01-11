from stable_baselines3 import A2C
import warehouse_transporter_env
import numpy as np

target_pos = np.array((4, 5))
print(target_pos.shape)
agent_pos = np.array((5, 9))
block_pos = np.array([
    [0, -1, -1, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, -1, 0, 0, -1, 0, 0, -1, 0],
    [0, 0, -1, 0, 0, -1, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, -1, 0, 0, -1, 0, 0],
    [0, -1, 0, 0, 0, 0, -1, 0, -1, 0],
    [0, -1, 0, 0, 0, 0, -1, 0, -1, 0],
    [-1, 0, -1, 0, 0, 0, 0, 0, 0, 0]
])

eval_env = warehouse_transporter_env.warehouse_env(10, 10, target_pos, block_pos, agent_pos)
loaded_model = A2C.load("model_test_10000")

obs = eval_env.reset()
for i in range(1000):
    action, _states = loaded_model.predict(obs, deterministic= True)
    print("action:" ,action)
    obs, reward, done, info = eval_env.step(action)
    #eval_env.render()
    if done:
        obs = eval_env.reset()