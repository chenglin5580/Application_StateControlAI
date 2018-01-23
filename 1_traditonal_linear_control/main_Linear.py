"""
Traditional Controller Design
Designer: Lin Cheng  2018.01.22

"""

########################### Package  Input  #################################

import matplotlib.pyplot as plt
import numpy as np
import SmallStateControl

############################ Object and Method  ####################################

env = SmallStateControl.SSCPENV()

s_dim = env.state_dim
print("环境状态空间维度为", s_dim)
print('-----------------------------\t')
a_dim = env.action_dim
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')
# a_bound = env.abound
# print("环境动作空间的上界为", a_bound)
# print('-----------------------------\t')


###############################  Control  ####################################

state_now = env.reset()

i_index = 0
state_track = []
action_track = []
time_track = []
action_ori_track = []
reward_track = []
while True:

    omega = 10
    state_next, reward, done, info = env.step(omega)

    state_track.append(state_now.copy())
    action_track.append(info['action'])
    time_track.append(info['time'])
    action_ori_track.append(info['u_ori'])
    reward_track.append(info['reward'])


    state_now = state_next

    if done:
        break

plt.figure(1)
plt.plot(time_track, [x[0] for x in state_track])
plt.grid()

#
plt.figure(2)
plt.plot(time_track, action_track)
plt.plot(time_track, action_ori_track)
plt.grid()

plt.figure(3)
plt.plot(time_track, reward_track)
plt.grid()
plt.show()
#
#
