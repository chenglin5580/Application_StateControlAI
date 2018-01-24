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
    state_now = env.reset()
    ep_reward = 0
    j = 0

    while True:
        action = ddpg.choose_action(state_now)
        action = np.clip(np.random.normal(action, var), 0, 20)  # add randomness to action selection for exploration
        state_next, reward, done, info = env.step(action[0])


        ddpg.store_transition(state_now, action, reward, state_next, np.array([done * 1.0]))

        if Learning_Start:
            ddpg.learn()
            var *= .99995  # decay the action randomness
            RENDER = True
        else:
            if ddpg.pointer > ddpg.MEMORY_CAPACITY:
                Learning_Start = True

        state_new = state_next
        ep_reward += reward
        j += 1

        if done:
            print('Episode:', i, ' ep_reward: %.4f' % ep_reward, 'step', j,  'Explore: %.2f' % var, )
            # if ep_reward > -300:
            break

    reward_me[i] = ep_reward
    if var < 0.1:
        break


ddpg.net_save()

# plt.figure(1)
# plt.plot(reward_me)
# plt.savefig("reward_me.png")
#
#
# plt.show()





###############################  test  ####################################

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
