"""
Dueling DQN & Natural DQN comparison

Lin Cheng 2018.01.15

"""

## package input
import gym
import numpy as np
# from DDQN_tensorflow_newVersion import Dueling_DQN_method
from DDQN_tensorflow import Dueling_DQN_method
import matplotlib.pyplot as plt
import SmallStateControl


# 导入environment
env = SmallStateControl.SSCPENV()

state_dim = env.state_dim + 1
print("环境状态空间维度为", state_dim)
print('-----------------------------\t')
action_sequence = np.linspace(1, 20, 10)
action_dim = len(action_sequence)
print("环境动作空间维度为", action_dim)
print('-----------------------------\t')


Dueling_DQN = Dueling_DQN_method(action_dim, state_dim, reload_flag=True)


###############################  Training  ####################################


x_now = env.reset()
state_now = np.hstack((x_now, np.array([3])))
action_old = int(state_now[-1])
state_now = np.reshape(state_now, [1, state_dim])

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

    action = Dueling_DQN.chose_action(state_now, train=False)
    omega = action_sequence[action]
    x_next, reward, done, info = env.step(omega)
    state_next = np.hstack((x_next, np.array(action)))
    state_next = np.reshape(state_next, [1, state_dim])
    if action != action_old:
        reward -= 10 / 500


    state_track.append(state_now[0].copy())
    action_track.append(info['action'])
    time_track.append(info['time'])
    action_ori_track.append(info['u_ori'])
    reward_track.append(info['reward'])
    omega_track.append(float(omega))
    penalty_track.append(info['penalty'])

    state_now = state_next
    action_old = action
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
plt.savefig('omega_Less.png')
plt.title('omega')

plt.figure(5)
plt.plot(time_track, penalty_track)
plt.grid()
plt.title('penalty')


plt.show()














