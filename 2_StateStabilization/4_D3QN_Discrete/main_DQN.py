import matplotlib.pyplot as plt
from D3QN import DQN

from SmallStateControlDis import SSCPENV

if __name__ == "__main__":
    # maze game
    env = SSCPENV()
    RL = DQN(n_actions=env.n_action,
             n_features=2,
             learning_rate=0.01,
             gamma=0.9,
             e_greedy_end=0.1,
             memory_size=3000,
             e_liner_times=10000,
             batch_size=256,
             output_graph=False,
             double=True,
             dueling=True,
             train=True
             )
    # train part
    if RL.train:
        step = 0
        ep_reward = 0
        episodes = 300
        for episode in range(episodes):
            ep_reward = 0
            observation = env.reset()  # initial observation
            while True:
                action = RL.choose_action(observation)               # RL choose action based on observation
                observation_, reward, done, info = env.step(action)  # RL get next observation and reward
                ep_reward += reward
                RL.store_transition(observation, action, reward, observation_)  # store memory

                if RL.memory_counter > RL.memory_size:
                    RL.learn()

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    break
                step += 1
            print('Episode:', episode + 1, '/', episodes,' ep_reward: %.4f' % ep_reward, 'epsilon: %.3f' % RL.epsilon)
        # save net
        RL.net_save()
        # end of game
        print('train over')

    # display part
    state_track = []
    action_track = []
    time_track = []
    action_ori_track = []
    omega_track = []
    ep_reward = 0
    observation = env.reset()
    while True:
        # RL choose action based on observation
        action = RL.choose_action(observation)
        # RL take action and get next observation and reward
        observation_, reward, done, info = env.step(action)

        # store track
        state_track.append(observation.copy())
        action_track.append(info['action'])
        time_track.append(info['time'])
        action_ori_track.append(info['u_ori'])
        omega_track.append(info['omega'])

        # swap observation
        ep_reward += reward
        observation = observation_

        # break while loop when end of this episode
        if done:
            print('ep_reward: %.4f' % ep_reward)
            break

    # plot
    plt.figure(1)
    plt.plot(time_track, [x[0] for x in state_track])
    plt.grid()
    plt.title('x')

    plt.figure(2)
    plt.plot(time_track, action_track)
    plt.plot(time_track, action_ori_track)
    plt.title('action')
    plt.grid()

    plt.figure(3)
    plt.plot(time_track, omega_track)
    plt.title('omega')
    plt.grid()

    plt.show()
