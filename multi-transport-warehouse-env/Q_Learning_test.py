import random
import warehouse_multi_transporter
import numpy as np
import gymnasium as gym
import plot_utils

def q_learning(env:gym.Env, episodes, alpha=0.3, gamma=0.99, epsilon=0.4):
    q_table = np.zeros((env.n_agents, env.width, env.height, env.n_actions))

    print(q_table.shape)
    rewards = []
    test_reward = 0

    chart_x = []
    chart_y = []

    for episode in range(episodes):
        state = env.reset()
        done = False


        episode_reward = 0

        while not done:
            actions = [None] * len(env.action_space)
            for i in range(env.n_agents):
                x, y = state[i]
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()[i]
                else:
                    action = np.argmax(q_table[i, x, y])
                actions[i] = action
            for i in range(env.n_agents):
                next_state, reward, done_i, _ = env.step(actions)
                #图形化观测
                #if episode>900:
                env.render()
                next_x, next_y = next_state[i]
                value = alpha*(reward[i] + gamma * np.max(q_table[i, next_x, next_y]) - q_table[i, x, y, action])
                value = value + q_table[i, x, y, action]
                q_table[i, x, y, action] = value
                state = next_state
                episode_reward += reward[i]
            done = all(done_i)
        rewards.append(episode_reward)
        test_reward = episode_reward + test_reward
        if (episode+1)%100==0:
            chart_x.append(episode+1)
            chart_y.append(test_reward/100)
            print("episode:", episode, "avg_reward:", test_reward/100)
            test_reward = 0
    print((chart_x,chart_y))
    plot_utils.line_chart(chart_x, chart_y,"avg_rewards", "episodes", "avg_reward")
    return q_table, rewards

#env = warehouse_multi_transporter.warehouse_multi_agent(5,5,2,[(0,4),(2,4)],[(0,1),(1,3)],4,((1,2),(3,1),(1,1),(2,2),(3,2),(0,2)))
env = warehouse_multi_transporter.warehouse_multi_agent(6,6,3,[(0,5),(2,5),(1,5)],[(4,2),(1,3),(1,0)],4,((1,2),(3,3),(2,2),(3,2),(0,2)))

num_episodes = 1000
q_table, rewards = q_learning(env,num_episodes)