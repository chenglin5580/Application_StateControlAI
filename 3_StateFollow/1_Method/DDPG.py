'''
Add API by MrFive
DDPG Method
这里面的DDPG是双层网络
'''
import tensorflow as tf
import numpy as np
import sys

tf.set_random_seed(2)


###############################  DDPG  ####################################

class ddpg(object):
    def __init__(
            self,
            a_dim,  # 动作的维度
            s_dim,  # 状态的维度
            a_bound,  # 动作的上下限
            e_greedy_end=0.1,  # 最后的探索值 e_greedy*幅值
            e_liner_times=1000,  # 探索值经历多少次学习变成e_end
            epilon_init=1,  # 表示1倍的幅值作为初始值
            LR_A=0.0001,  # Actor的学习率
            LR_C=0.0002,  # Critic的学习率
            GAMMA=0.9,  # reward discount
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=10000,  # 记忆池容量
            BATCH_SIZE=256,  # 批次数量
            units=64,
            reload_flag=False,  # 是否读取
            train=True  # 训练的时候有探索
    ):

        # DDPG网络参数
        self.method = 'MovFan'
        self.LR_A = LR_A  # learning rate for actor
        self.LR_C = LR_C  # learning rate for critic
        self.GAMMA = GAMMA  # reward discount
        self.TAU = TAU  # soft replacement
        self.MEMORY_CAPACITY = MEMORY_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.units = units

        self.epsilon_init = epilon_init  # 初始的探索值
        self.epsilon = self.epsilon_init
        self.epsilon_end = e_greedy_end
        self.e_liner_times = e_liner_times
        self.train = train

        self.pointer = 0
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.iteration = 0
        self.modelpath = sys.path[0] + '/data.chkp'

        # DDPG构建
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)  # 存储s,a,r,s_,done
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
            self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
            self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [
            [tf.assign(ta, (1 - self.TAU) * ta + self.TAU * ea), tf.assign(tc, (1 - self.TAU) * tc + self.TAU * ec)]
            for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        tf.summary.scalar('td_error', td_error)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

        self.actor_saver = tf.train.Saver()
        if reload_flag:
            self.actor_saver.restore(self.sess, self.modelpath)
        else:
            self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        action = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        return np.clip(np.random.normal(action, abs(self.a_bound[0] - self.a_bound[1]) / 2 * self.epsilon),
                       self.a_bound[0], self.a_bound[1])  # add randomness to action selection for exploration

    def learn(self):
        if self.pointer < self.MEMORY_CAPACITY:
            pass
        else:
            self.sess.run(self.soft_replace)
            indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
            bs_ = bt[:, self.s_dim + self.a_dim + 1: 2 * self.s_dim + self.a_dim + 1]
            done_ = bt[:, -1:]

            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.done: done_})
            self.iteration += 1
            if self.train == True:
                self.epsilon = max(self.epsilon - (self.epsilon_init - self.epsilon_end) / self.e_liner_times,
                                   self.epsilon_end)
            else:
                self.epsilon = 0

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = self.units
            net = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, abs(self.a_bound[0] - self.a_bound[1]) / 2, name='scaled_a') + np.mean(self.a_bound)

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = self.units
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            q = tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
            return q

    def net_save(self):
        self.actor_saver.save(self.sess, self.modelpath)
