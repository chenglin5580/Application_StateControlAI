"""
Traditional Controller Design
Designer: Lin Cheng  2018.01.22
Simplified by MrFive
"""

########################### Package  Input  #################################

import matplotlib.pyplot as plt
import numpy as np
from DDPG import ddpg

import SmallStateControl

############################ Object and Method  ####################################

if __name__ == '__main__':
    env = SmallStateControl.SSCPENV()

    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.abound
    ddpg = ddpg(a_dim, s_dim, a_bound, e_greedy_end=0.1, e_liner_times=10000)

    state_now = env.reset()
    state_track = []
    action_track = []
    time_track = []
    action_ori_track = []
    reward_track = []
    omega_track = []
    max_Episodes = 150
    ep_reward = 0

    for episode in range(max_Episodes):
        state_now = env.reset()
        ep_reward = 0
        step = 0
        while True:
            action = ddpg.choose_action(state_now)
            state_next, reward, done, info = env.step(action)

            if episode == max_Episodes - 1:
                ddpg.train = False
                state_track.append(state_now.copy())
                action_track.append(info['action'])
                time_track.append(info['time'])
                action_ori_track.append(info['u_ori'])
                reward_track.append(info['reward'])
                omega_track.append(action)

            ddpg.store_transition(state_now, action, reward, state_next, np.array([done * 1.0]))
            if step % 5 == 0:
                ddpg.learn()

            state_new = state_next
            ep_reward += reward
            step += 1
            if done:
                print('Episode:', episode, ' ep_reward: %.4f' % ep_reward, 'step', step, 'Explore: %.2f' % ddpg.epsilon)
                break

    print('game over')
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

    plt.show()
