import sys

import gym
import numpy as np
from gym import spaces
import pygame


class warehouse_multi_agent(gym.Env):

    #智能体环境输入：网格大小、智能体数量、每个智能体的初始位置、每个智能体当前时间步的目标位置、仓库中初始货物位置
    def __init__(self, grid_width:int, grid_height:int, n_agents:int, start_pos:list, target_pos:list, n_actions:int, block_pos:tuple):

        ##基础属性
        self.width = grid_width
        self.height = grid_height
        self.n_agents = n_agents
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.n_actions = n_actions
        self.block_pos = block_pos

        act_space = [self.n_actions] * self.n_agents
        print("act_space:", act_space)
        print("type(act_space):",type(act_space))

        ##动作空间
        self.action_space = spaces.MultiDiscrete(act_space)
        print("self.action_space.nvec:", self.action_space.nvec[0])
        #self.action_space = gym.spaces.Tuple([spaces.Discrete(n_actions) for _ in range(n_agents)])
        ##观测空间
        # self.observation_space = gym.spaces.Tuple(
        #     (gym.spaces.Discrete(self.width),
        #     gym.spaces.Discrete(self.height))
        # )
        self.observation_space = gym.spaces.MultiDiscrete(
            ([self.width, self.height] * n_agents
             )
        )
        print(self.observation_space)
        print(type(self.observation_space))
        ##智能体位置
        self.cur_pos = None

        self.exit_pos = None

        self.current_pos = self.start_pos

        ##图形化参数
        self.cell_size = 50
        self.window_size =(800, 800)
        ##初始化pygame图形化环境
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Warehouse Multi Transporter Environment")

    def step(
            self,
            action
    ):
        reward = []
        done = []
        for i in range(self.n_agents):
            if not self.is_task_completed(i):
                flag = self._make_action(i, action[i])
                if flag == True and self.is_task_completed(i):
                    reward.append(100.0)
                elif flag == False:
                    reward.append(-1.0)
                else:
                    reward.append(-0.1)
            else:
                reward.append(0)
            done.append(self.is_task_completed(i))
        #array = np.array(self.current_pos, dtype=object).flatten()
        return self.current_pos, reward, done, {}


    def reset(self):
        self.current_pos = self._init_pos()
        print(self.current_pos)
        array = np.array(self.current_pos,dtype=object).flatten()
        array_transpose = np.transpose(array)
        print(array_transpose)
        return self.current_pos

    def _init_pos(self)->list:
        for i in range(self.n_agents):
            random_row = np.random.randint(self.height)
            print(random_row)
            current_loc = (random_row, self.width - 1)
            print(current_loc)
            self.start_pos[i]=current_loc
        return self.start_pos

    def render(self) ->  None:
        # clear environment
        self.screen.fill((255,255,255))

        # 绘制网络
        for i in range(self.width):
            for j in range(self.height):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        #得到智能体当前位置
        for i in range(self.n_agents):
            agent_rect = pygame.Rect(self.current_pos[i][1] * self.cell_size,
                                     self.current_pos[i][0] * self.cell_size,
                                     self.cell_size, self.cell_size)
            pygame.draw.circle(self.screen, (0, 0, 255), agent_rect.center, int(self.cell_size / 2))

        #绘制目标货物的位置
        for i in range(len(self.target_pos)):
            target_rect = pygame.Rect(self.target_pos[i][1] * self.cell_size,
                                      self.target_pos[i][0] * self.cell_size,
                                      self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), target_rect)

        #绘制仓库中货物的位置
        for i in range(len(self.block_pos)):
            block_rect = pygame.Rect(self.block_pos[i][1] * self.cell_size,
                                     self.block_pos[i][0] * self.cell_size,
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

    def _make_action(self, agent_label, agent_action):
        #print("agent_action:", agent_action)
        x, y = self.current_pos[agent_label]

        if agent_action ==0: #上
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
        elif new_loc not in self.block_pos and self.action_validate(new_loc, self.block_pos, self.current_pos, agent_label):
            self.current_pos[agent_label] = new_loc
            return True
        else:
            return False

    def action_validate(self, new_pos:tuple, block_pos:tuple, agent_current_pos:tuple,agent_label:int):
        #判断是否与船块位置冲突
        for coord in block_pos:
            if new_pos==coord:
                return False
        #判断新位置是否与其他智能体位置冲突
        for i in range(len(agent_current_pos)):
            if i != agent_label:
                if new_pos==agent_current_pos[i]:
                    return False
        return True

    def is_task_completed(self, agent_label):
        if self.current_pos[agent_label]==self.target_pos[agent_label]:
            return True
        else:
            return False

if __name__ == '__main__':
    env = warehouse_multi_agent(10,10,2,[(0,9),(1,9)],[(1,1),(2,3)],4,((1,2),(3,3),(2,2),(3,2),(0,2),(1,6),(1,5),(7,8),(6,2),(6,4)))
    env.reset()
    env.render()
    env.step([0,0])
    env.render()
    env.step([0,0])