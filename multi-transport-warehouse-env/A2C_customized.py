import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import warehouse_multi_transporter
import sys
import time
import warehouse_transporter_env

class Actor2Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Actor2Critic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128,  1)
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

def compute_returns(rewards:list, discount_factor:float):
    R = 0;
    returns = []
    for r in rewards[::-1]:
        R = r + discount_factor * R
        returns.insert(0, R)
    return returns

def train(num_agents, env:gym.Env, num_episodes, discount_factor = 0.99, learning_rate = 0.001):
    num_inputs = 100
    num_outputs = 4
    print("num_inputs:",num_inputs)
    print("num_outputs:", num_outputs)
    model = Actor2Critic(num_inputs, num_outputs)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    num_steps = 200

    for episode in range(num_episodes):
        state = env.reset()

        log_probs = []
        values = []
        rewards = []

        for step in range(num_steps):
            state = torch.FloatTensor(state)
            state_flat = state.flatten()
            #print("state_flat:",state_flat)
            policy, value = model(state_flat)
            #print("policy size:", policy.shape)
            #print("policy:", policy)
            #print("value shape", value.shape)
            action = torch.distributions.Categorical(policy).sample()
            print("action:", int(action.item()))
            next_state, reward, done, _ = env.step(int(action.item()))
            if episode >=9:
                env.render()
                time.sleep(0.05)
            log_prob = torch.distributions.Categorical(policy).log_prob(action)
            #print("log_prob:",  log_prob)
            log_probs.append(log_prob.view(1))
            values.append(value)
            rewards.append(reward)
            if done==True:
                print("-"*20+"done"+"-"*20)
                break

            state = next_state
        returns = compute_returns(rewards, discount_factor)
        log_probs = torch.cat(log_probs)
        returns = torch.tensor(returns)
        #print("returns.shape:",returns.shape)
        returns = returns.detach()
        values = torch.cat(values)

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        optimizer.step()
    torch.save(model.state_dict(), './model_saved/A2C_test.pth')
if __name__=="__main__":
    env1 = warehouse_multi_transporter.warehouse_multi_agent(10,10,2,[(0,9),(1,9)],[(1,1),(2,3)],4,((1,2),(3,3),(2,2),(3,2),(0,2),(1,6),(1,5),(7,8),(6,2),(6,4)))
    target_pos = np.array((4,4))
    print(target_pos.shape)
    agent_pos = np.array((0,9))
    block_pos = np.array([
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, -1, -1, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    env2 = warehouse_transporter_env.warehouse_env(10,10,target_pos,block_pos,agent_pos)
    train(2,env2,100)
    sys.exit()