from pettingzoo.classic import leduc_holdem_v4
import pprint as pp
import numpy as np


env = leduc_holdem_v4.env(render_mode="ansi")
env.reset(seed=np.random.randint(1_000))

print(env.last())



#Is this code for two agents playing against eachother, or am I coding my actions which are being sent to the environment??
#You're playing with yourself :-)



for agent in env.agent_iter(): 
    print(agent)
    #what does this function do right here -> I think it gives list of agents in env so that you can iterate over them, 
    # and I guess it does so in a way that continues until game is over, couldn't find documentation for it, fuck it we ball
    #why is info, I see why, not supported


    observation, reward, termination, truncation, info = env.last()
    print(f'obs: {observation}\n,reward: {reward},termination: {termination},truncation: {truncation},info {info}')

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        # this is where you would insert your policy
        # TODO: Insert policy

        action = env.action_space(agent).sample(mask)

        # if(agent == 'player_0'):
        #     action = 2
        #     env.step(action)
        #     print(env.last())
        #     continue
        if(observation['action_mask'][3] == 1):
            action = 3
        print(f'Imma do {action}')

    env.step(action)


# print(env.rewards)
print('render')
env.render()
env.close()
