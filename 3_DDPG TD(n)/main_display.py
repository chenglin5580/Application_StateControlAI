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
# print("环境动作空间的上界为", a_bound)
# print('-----------------------------\t')

reload_flag = True
ddpg = ddpg(a_dim, s_dim, a_bound, reload_flag)

###############################  Training  ####################################


state_now = env.reset()

i_index = 0



state_track = []
action_track = []
time_track = []
action_ori_track = []
reward_track = []
omega_track = []
penalty_track = []
reward_me = 0
while True:

    omega = ddpg.choose_action(state_now)
    state_next, reward, done, info = env.step(omega[0])

    state_track.append(state_now.copy())
    action_track.append(info['action'])
    time_track.append(info['time'])
    action_ori_track.append(info['u_ori'])
    reward_track.append(info['reward'])
    omega_track.append(float(omega))
    penalty_track.append(info['penalty'])


    state_now = state_next
    reward_me += reward

    if done:
        break

print(reward_me)
plt.figure(1)
plt.plot(time_track, [x[0] for x in state_track])
plt.grid()
plt.title('x')

#
plt.figure(2)
plt.plot(time_track, action_track)
plt.plot(time_track, action_ori_track)
plt.title('action')
plt.grid()

plt.figure(3)
plt.plot(time_track, reward_track)
plt.grid()
plt.title('reward')

plt.figure(4)
plt.plot(time_track, omega_track)
plt.grid()
plt.title('omega')

plt.figure(5)
plt.plot(time_track, penalty_track)
plt.grid()
plt.title('penalty')


plt.show()
#





#
