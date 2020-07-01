from gym.envs.registration import register
 
register(id='ConnectFour-v0', 
    entry_point='gym_connectfour.envs:ConnectFourEnv', 
)
