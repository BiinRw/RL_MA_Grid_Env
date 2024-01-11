import sys

import gym
import numpy as np
import torch
from gym import spaces
import pygame
import time

class warehouse_env(gym.Env):

    #智能体环境输入：网格大小、目标位置、仓库中初始货物位置.
    #在
    def __init__(self, grid_width:int, grid_height:int,target_pos:np.ndarray, block_pos:np.ndarray, agent_pos:np.ndarray):
        #判断输入的类型是否合规：
        if target_pos.shape != (2,) or target_pos.dtype != np.int:
            raise TypeError("TypeError raised in initialing target_pos")
        if block_pos.shape !=(grid_width,grid_height):
            raise TypeError("TypeError raised in initialing block_pos")
        if agent_pos.shape != (2,) or agent_pos.dtype != np.int:
            raise TypeError("TypeError raised in initialing agent_pos")

        ##基础属性
        self.init_pos = agent_pos
        self.width = grid_width
        self.height = grid_height
        self.target_pos = target_pos
        self.block_pos = block_pos
        self.agent_pos = agent_pos
        self.grid_env = self._init_gird_env(self.block_pos, self.agent_pos, self.target_pos)

        ##动作空间定义，智能体四种动作分别对应前后左右的移动
        self.action_space = spaces.Discrete(4)

        ##观测空间，智能体的整个运行的网格环境，使用Box定义
        self.observation_space = gym.spaces.Box(low=0,high=self.width-1, shape=(self.width,self.height), dtype=np.int)

        ##智能体位置
        self.cur_pos = None
        self.exit_pos = None
        self.current_pos = self.agent_pos

        ##图形化参数
        self.cell_size = 50
        self.window_size =(800, 800)
        ##初始化pygame图形化环境
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Warehouse Multi Transporter Environment")

    #根据初始状态，使用一个网格二维数组来标识当前智能体的运行环境
    def _init_gird_env(self, block_pos, agent_pos, target_pos)-> np.ndarray:
        warehouse_gird = np.zeros((self.width,self.height))
        warehouse_gird = warehouse_gird + self.block_pos
        x,y = self.agent_pos[0], self.agent_pos[1]
        if warehouse_gird[x][y] == -1:
            raise ValueError("agent_pos error")
        else:
            warehouse_gird[x][y] = 1
        target_x, target_y = self.target_pos[0], self.target_pos[1]
        if warehouse_gird[target_x][target_y] !=0:
            raise ValueError("target_pos error")
        else:
            warehouse_gird[target_x][target_y] = 2
        # for row in warehouse_gird:
        #     print(' '.join('{:5.0f}'.format(item) for item in row))
        return warehouse_gird

    #重置观测空间
    def reset(self):
        #init_pos = self.init_pos
        #init_pos = self._init_pos()
        init_pos = np.array((6, 9))
        self.current_pos = init_pos
        self.grid_env = self._init_gird_env(self.block_pos, init_pos, self.target_pos)
        grid_state = self.grid_env
        self.render()
        #time.sleep(1)
        return grid_state

    def step(self, action:int):
        done = False
        flag = self._make_action(action)
        self.render()
        #print("action:",action,"current_pos: ",self.current_pos,"target_pos: ", self.target_pos)
        if flag == True:
            if np.array_equal(self.current_pos, self.target_pos):
                reward = 100
                done = True
                print("-"*20+"done"+"-"*20)
            else:
                reward = -1
        else:
            reward = -1
        return self.grid_env, reward, done, {}




    def _init_pos(self)->list:
        random_row = np.random.randint(self.height)
        print(random_row)
        current_loc = (random_row, self.width - 1)
        print(current_loc)
        self.agent_pos = current_loc
        self.current_pos = current_loc
        return self.agent_pos


    def render(self) ->  None:
        # clear environment
        self.screen.fill((255,255,255))
        # 绘制网络
        for i in range(self.width):
            for j in range(self.height):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        #得到智能体当前位置
        agent_rect = pygame.Rect(self.current_pos[1] * self.cell_size,
                                self.current_pos[0] * self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_rect.center, int(self.cell_size / 2))
        #绘制目标货物的位置
        target_rect = pygame.Rect(self.target_pos[1] * self.cell_size,
                                    self.target_pos[0] * self.cell_size,
                                    self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), target_rect)
        #绘制仓库中货物的位置
        for i in range(self.block_pos.shape[0]):
            for j in range(self.block_pos.shape[1]):
                if self.block_pos[i][j] == -1:
                    block_rect = pygame.Rect(j * self.cell_size,
                                     i * self.cell_size,
                                     self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (0, 0, 0), block_rect)
        #文本信息，字体的定义
        font = pygame.font.Font(None, 36)
        value = 23
        text = font.render(f"value: {value}", True, (200,200,100))
        #self.screen.blit(text,(j * self.cell_size // 2 - text.get_width() // 2,  i * self.cell_size // 2 - text.get_height() // 2))
        pygame.display.flip()

        pygame.time.wait(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


    def _make_action(self, agent_action:int):
        #print("agent_action:", agent_action)
        x, y = self.current_pos

        if agent_action == 0: #上
            new_y = max(y-1,0)
            new_x = x
        elif agent_action ==1: #下
            new_y = min(y+1,self.height-1)
            new_x = x
        elif agent_action ==2:
            new_y = y #左
            new_x = max(x-1, 0)
        elif agent_action ==3: #右
            new_y = y
            new_x = min(x+1, self.width-1)
        new_loc = (new_x,new_y)

        if new_loc == (x,y):
            return False
        elif self.grid_env[new_loc[0]][new_loc[1]] == 0 or self.grid_env[new_loc[0]][new_loc[1]] == 2:
            self.grid_env[new_loc[0]][new_loc[1]] = 1
            self.grid_env[x][y] = 0
            self.current_pos = np.array(new_loc)
            return True
        elif self.grid_env[new_loc[0]][new_loc[1]] == -1:
            return False

