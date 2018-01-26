
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(5)
import sys


class Dueling_DQN_method:

    def __init__(self, action_dim, state_dim, reload_flag=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.memory_counter = 0
        self.memory_size = 1000
        self.memory = np.empty([self.memory_size, 2*self.state_dim+1+1+1])
        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decrease = 0.0001
        self.epsilon_min = 0.1
        self.learning_rate = 0.01
        self.learn_step_counter = 0
        self.replace_target_limit = 100
        self.modelpath = sys.path[0] + '/my_net/data.chkp'

        self.build_model()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.actor_saver = tf.train.Saver()
        if reload_flag:
            self.actor_saver.restore(self.sess, self.modelpath)
        else:
            self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.global_variables_initializer())


    def build_model(self):

        # eval network
        self.s = tf.placeholder(tf.float32, [None, self.state_dim], name='s')  # state input
        self.q_target = tf.placeholder(tf.float32, [None, self.action_dim], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_network'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 100, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_dim, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2_value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                self.value_eval = tf.matmul(l1, w2) + b2

            with tf.variable_scope('l2_advantage'):
                w3 = tf.get_variable('w3', [n_l1, self.action_dim], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.action_dim], initializer=w_initializer, collections=c_names)
                self.advantage_eval = tf.matmul(l1, w3) + b3

            with tf.variable_scope('l3_q_eval'):
                self.q_eval = self.value_eval + self.advantage_eval - tf.reduce_mean(self.advantage_eval, axis=1,
                                                                                     keep_dims=True)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.q_target))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # build target network
        self.s_ = tf.placeholder(tf.float32, [None, self.state_dim], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_dim, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2_value_stream'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                self.value_target_next = tf.matmul(l1, w2) + b2

            with tf.variable_scope('l2_advantage_stream'):
                w3 = tf.get_variable('w3', [n_l1, self.action_dim], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.action_dim], initializer=b_initializer, collections=c_names)
                self.advantage_target_next = tf.matmul(l1, w3) + b3

            with tf.variable_scope('l3_q_next'):
                self.q_next = self.value_target_next + self.advantage_target_next - tf.reduce_mean(self.advantage_target_next,
                                                                                              axis=1, keep_dims=True)

    def memory_store(self, state_now, action, reward, state_next, done):

        action = np.reshape(action, [1, 1])
        reward = np.reshape(reward, [1, 1])
        done = np.reshape(done, [1, 1])
        transition = np.hstack((state_now, action, reward, state_next, done))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def chose_action(self, state, train=True):

        if train and (np.random.uniform() < self.epsilon):
            # action = self.action_space.sample()
            action = np.random.randint(0, self.action_dim)
        else:
            q_eval = self.sess.run(self.q_eval, feed_dict={self.s: state})
            # q_eval = self.model_eval.predict(state)
            action = np.argmax(q_eval)
        return action



    def Learn(self):

        if self.learn_step_counter % self.replace_target_limit == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            # print('memory is not full')

        batch_memory = self.memory[sample_index, :]

        batch_state = batch_memory[:, :self.state_dim]
        batch_action = batch_memory[:, self.state_dim].astype(int)
        batch_reward = batch_memory[:, self.state_dim+1]
        batch_state_next = batch_memory[:, -self.state_dim-1:-1]
        batch_done = batch_memory[:, -1]

        q_target = self.sess.run(self.q_eval, feed_dict={self.s: batch_state})
        q_next1 =  self.sess.run(self.q_eval, feed_dict={self.s: batch_state_next})
        q_next2 = self.sess.run(self.q_next, feed_dict={self.s_: batch_state_next})
        batch_action_withMaxQ =  np.argmax(q_next1, axis=1)
        batch_index11 = np.arange(self.batch_size, dtype=np.int32)
        q_next_Max = q_next2[batch_index11, batch_action_withMaxQ]
        # q_target[batch_index11, batch_action] = batch_reward + (1-batch_done)*self.gamma * q_next_Max
        q_target[batch_index11, batch_action] = batch_reward + self.gamma * q_next_Max

        # train
        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: batch_state,
                                                self.q_target: q_target,
                                                })
        self.learn_step_counter += 1

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrease

    def model_save(self):
        # model save
        self.actor_saver.save(self.sess, self.modelpath)





































