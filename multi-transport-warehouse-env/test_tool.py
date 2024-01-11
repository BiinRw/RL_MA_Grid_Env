import warehouse_multi_transporter

env = warehouse_multi_transporter.warehouse_multi_agent(6,6,3,[(0,5),(2,5),(3,5)],[(1,1),(4,2),(5,2)],4,((1,2)))
print(env.action_space.sample())
print(env.observation_space.sample())
obs = env.reset()
print(obs)