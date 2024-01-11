import time

import numpy as np
import gym
from gym import spaces
import pygame
import sys


class MultiAgentWarehouseEnv(gym.Env):
    def __init__(self, grid_width, grid_height, target_positions, block_positions, agent_positions):
        super(MultiAgentWarehouseEnv, self).__init__()

        self.num_agents = len(agent_positions)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.target_positions = np.array(target_positions)
        self.block_positions = np.array(block_positions)
        self.agent_positions = np.array(agent_positions, dtype=int)
        self.current_positions = self.agent_positions
        self.all_done = [False] * self.num_agents

        # 动作空间和观测空间
        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)
        self.observation_space = spaces.Box(low=0, high=max(grid_width, grid_height), shape=(self.num_agents, 2), dtype=int)

        # Pygame初始化
        self.cell_size = 50
        self.window_size = (grid_width * self.cell_size, grid_height * self.cell_size)
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Multi Agent Warehouse Environment")
    def reset(self):
        # 重置智能体位置
        self.current_positions = self.agent_positions
        self.all_done = [False] * self.num_agents
        self.render()
        return self.current_positions

    def step(self, actions):
        rewards = [0] * self.num_agents
        done = False
        next_positions = np.copy(self.current_positions)
        self.render()
        time.sleep(0.03)

        # 更新每个智能体的位置
        for i, action in enumerate(actions):
            if self.all_done[i]:
                continue
            if action == 0:  # 上
                next_positions[i][1] = max(next_positions[i][1] - 1, 0)
            elif action == 1:  # 下
                next_positions[i][1] = min(next_positions[i][1] + 1, self.grid_height - 1)
            elif action == 2:  # 左
                next_positions[i][0] = max(next_positions[i][0] - 1, 0)
            elif action == 3:  # 右
                next_positions[i][0] = min(next_positions[i][0] + 1, self.grid_width - 1)

        for i in range(self.num_agents):
            if self.is_moveable(next_positions[i]):
                self.current_positions[i] = next_positions[i]

        # 检查碰撞和到达目标
        for i in range(self.num_agents):
            if self.all_done[i]:
                continue
            elif self.is_collision(i, next_positions) or not self.is_moveable(next_positions[i]):
                rewards[i] = -1  # 碰撞惩罚
            elif np.array_equal(next_positions[i], self.target_positions[i]):
                rewards[i] = 100
                self.all_done[i] = True
                # 如果任何智能体到达目标，环境重置
            else:
                rewards[i] = -1  # 未到达目标的小惩罚

        done = all(self.all_done)
        reward = sum(rewards)
        return self.current_positions, reward, done, {}

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))

        # 绘制网格
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                if self.block_positions[x, y]:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)

        # 绘制所有智能体和它们的目标位置
        for i in range(self.num_agents):
            agent_pos = self.current_positions[i]
            target_pos = self.target_positions[i]

            agent_rect = pygame.Rect(agent_pos[0] * self.cell_size, agent_pos[1] * self.cell_size, self.cell_size,
                                     self.cell_size)
            pygame.draw.circle(self.screen, (0, 0, 255), agent_rect.center, int(self.cell_size / 2))

            target_rect = pygame.Rect(target_pos[0] * self.cell_size, target_pos[1] * self.cell_size, self.cell_size,
                                      self.cell_size)
            pygame.draw.rect(self.screen, (0, 255, 0), target_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def is_collision(self, agent_index, next_positions):
        for i in range(self.num_agents):
            if i != agent_index and np.array_equal(next_positions[i], next_positions[agent_index]):
                return True
        return False

    def is_block(self, position):
        return self.block_positions[position[0], position[1]]

    def is_moveable(self, position):
        if self.is_block(position):
            return False
        else:
            for i in range(self.num_agents):
                if self.all_done[i] and np.array_equal(position, self.target_positions[i]):
                    return False
            return True

    def all_complete(self, next_positions):
        for i in range(self.num_agents):
            if not np.array_equal(next_positions[i], self.target_positions[i]):
                return False
        return True



