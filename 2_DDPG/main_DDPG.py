"""
Traditional Controller Design
Designer: Lin Cheng  2018.01.22

"""

########################### Package  Input  #################################

import matplotlib.pyplot as plt
import numpy as np
import SmallStateControl
from DDPG_Morvan import ddpg

############################ Object and Method  ####################################

env = SmallStateControl.SSCPENV()

s_dim = env.state_dim
print("环境状态空间维度为", s_dim)
print('-----------------------------\t')
a_dim = env.action_dim
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')
a_bound = env.abound
print("环境动作空间的上界为", a_bound)
print('-----------------------------\t')

reload_flag = False
ddpg = ddpg(a_dim, s_dim, a_bound, reload_flag)

###############################  Training  ####################################


max_Episodes = 300000
Learning_Start = False
var = 10  # control exploration
step_me = np.zeros([max_Episodes])
reward_me = np.zeros([max_Episodes])



for i in range(max_Episodes):
    state_now = env.reset_random()
    ep_reward = 0
    j = 0
    state_now_sequence = np.empty((0, 2))
    action_sequence = np.empty((0, 1))
    reward_sequence = np.empty((0, 1))
    state_next_sequence = np.empty((0, 2))
    done_sequence = np.empty((0, 1))
    while True:
        action = ddpg.choose_action(state_now)
        action = np.clip(np.random.normal(action, var), 0, 20)  # add randomness to action selection for exploration
        state_next, reward, done, info = env.step(action[0])

        state_now_sequence = np.vstack((state_now_sequence, state_now))
        action_sequence = np.vstack((action_sequence, action))
        reward_sequence = np.vstack((reward_sequence, reward))
        state_next_sequence = np.vstack((state_next_sequence, state_next))
        done_sequence = np.vstack((done_sequence, done))



        if Learning_Start:
            ddpg.learn()
            var *= .99999  # decay the action randomness
            RENDER = True
        else:
            if ddpg.pointer > ddpg.MEMORY_CAPACITY:
                Learning_Start = True

        state_new = state_next
        ep_reward += reward
        j += 1

        if done:
            print('Episode:', i, ' ep_reward: %i' % int(ep_reward), 'step', j,  'Explore: %.2f' % var, )
            # if ep_reward > -300:
            break
    TD_n = 20
    for kk in range(len(state_now_sequence[:, 0])):
        if kk + TD_n - 1 < len(state_now_sequence[:, 0]) - 1:
            state_now = state_now_sequence[kk, :]
            action = action_sequence[kk, 0]
            reward = np.sum(reward_sequence[kk: kk + TD_n, 0])
            state_next = state_next_sequence[kk + TD_n - 1, :]
            done = False
        else:
            state_now = state_now_sequence[kk, :]
            action = action_sequence[kk, 0]
            reward = np.sum(reward_sequence[kk:, 0])
            state_next = state_next_sequence[-1, :]
            done = True
        ddpg.store_transition(state_now, action, reward, state_next, np.array([done * 1.0]))

    reward_me[i] = ep_reward
    if var < 1:
        break

ddpg.net_save()

plt.figure(1)
plt.plot(reward_me)
plt.savefig("reward_me.png")


plt.show()
















#
