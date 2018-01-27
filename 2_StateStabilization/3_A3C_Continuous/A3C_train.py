

########################### Package  Import  #################################
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
# from SmallStateControl import SSCPENV as Object_AI  # 程林， 状态跟踪
from StateStabilizationProblem import SSCPENV as Object_AI # 程林， 状态镇定
tf.set_random_seed(2)

########################### User Setting  #################################

Training_flag = True                                # True 训练  False Displau
OUTPUT_GRAPH = False                                # True 输出tensorboard  False 不输出
LOG_DIR = './log'                                   # tensorboard 输出地址
N_WORKERS = multiprocessing.cpu_count()             # cpu数目
MAX_EP_STEP = 600                                   # 片段长度
MAX_GLOBAL_EP = 2000                                # 总迭代片段数目
GLOBAL_NET_SCOPE = 'Global_Net'                     # Global Actor-Critic 标志
UPDATE_GLOBAL_ITER = 50                             # A3C pull/Push 间隔
GAMMA = 0.9                                         #
ENTROPY_BETA = 0.01                                 # exploration
LR_A = 0.0001                                       # learning rate for actor
LR_C = 0.001                                        # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
modelpath = sys.path[0] + '/my_net/data.chkp'       # 权重存放地址


env = Object_AI()                                   # environment
N_S = env.state_dim                                 # state维度
N_A = env.action_dim                                # action维度
A_BOUND = env.abound - np.mean(env.abound)          # action范围

########################### Method  #################################

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                with tf.name_scope('wrap_a_out'):
                    mu = mu * A_BOUND[1]  #归一化反映射，防止方差为零

            with tf.name_scope('choose_a'):  # use local params to choose action
                self.A = tf.clip_by_value(mu, A_BOUND[0], A_BOUND[1])  # 根据actor给出的分布，选取动作


        else:   # worker, local net, calculate losses
            with tf.variable_scope(scope):
                # 网络引入
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')   #状态
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A') #动作
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  #目标价值

                # 网络构建
                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)  #mu 均值 sigma 均方差

                # 价值网络优化
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4  #归一化反映射，防止方差为零


                normal_dist = tf.distributions.Normal(mu, sigma)  #tf自带的正态分布函数

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)  #概率的log值
                    exp_v = log_prob * tf.stop_gradient(td)  #stop_gradient停止梯度传递的意思
                    entropy = normal_dist.entropy()  # encourage exploration，香农熵，评价分布的不确定性，
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)   #actor的优化目标是价值函数最大

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])  # 根据actor给出的分布，选取动作
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)   #计算梯度
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)   #计算梯度

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):  #把全局的pull到本地
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):  #根据本地的梯度，优化global的参数
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):  # 网络定义
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 30, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # 函数：执行push动作
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # 函数：执行pull动作
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # 函数：选择动作action
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]



class Worker(object):
    def __init__(self, name, globalAC):
        self.env = Object_AI()
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP  #GLOBAL_EP 是Global的片段个数
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []  #类似于memory，存储运行轨迹
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):  #MAX_EP_STEP每个片段的最大个数
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)+np.mean(env.abound)  #选取动作
                s_, r, done, info = self.env.step(a)

                # done = True if ep_t == MAX_EP_STEP - 1 else False  #算法运行结束条件

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:         # 每个片段结束，输出一下结果
                    # if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                    GLOBAL_RUNNING_R.append(ep_r)
                    # else:
                    #     GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %.4f" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":

    SESS = tf.Session()
    ###############################  Train  ####################################

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')   #actor优化器定义
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')   #critic优化器定义
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # 定义global ， 不过只需要它的参数空间
        workers = []
        # Create worker
        for i in range(N_WORKERS):  # N_WORKERS 为cpu个数
            i_name = 'W_%i' % i   # worker name，形如W_1
            workers.append(Worker(i_name, GLOBAL_AC))  #添加名字为W_i的worker

    actor_saver = tf.train.Saver()


    if Training_flag:
        COORD = tf.train.Coordinator()
        SESS.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            if os.path.exists(LOG_DIR):
                shutil.rmtree(LOG_DIR)
            tf.summary.FileWriter(LOG_DIR, SESS.graph)

        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)

        plt.figure(6)
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.xlabel('step')
        plt.ylabel('Total moving reward')

        actor_saver.save(SESS, modelpath)
    else:
        actor_saver.restore(SESS, modelpath)


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

        omega = GLOBAL_AC.choose_action(state_now) + np.mean(env.abound)
        # omega = workers[0].AC.choose_action_S(state_now)

        state_next, reward, done, info = env.step(omega)

        state_track.append(state_now.copy())
        action_track.append(info['action'])
        time_track.append(info['time'])
        action_ori_track.append(info['u_ori'])
        reward_track.append(info['reward'])
        omega_track.append(float(omega))
        # penalty_track.append(info['penalty'])


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

    # plt.figure(5)
    # plt.plot(time_track, penalty_track)
    # plt.grid()
    # plt.title('penalty')


    plt.show()