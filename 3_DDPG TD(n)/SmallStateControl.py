import numpy as np


class SSCPENV(object):
    def __init__(self, x_dim=2, action_dim=1, init_x=None):
        self.x_dim = x_dim
        self.action_dim = action_dim
        self.abound = np.array([10, 10])
        self.init_x = init_x
        self.state_dim = self.x_dim
        self.t = 0
        self.delta_t = 0.01
        self.xd = 5
        self.xd_dot = 0
        self.xd_dot2 = 0
        self.total_time = 5
        self.x = self.reset()
        self.u_bound = 30 * np.array([-1, 1])

    def reset(self):
        if self.init_x:
            return self.init_x
        else:
            self.t = 0
            self.x = np.zeros(self.x_dim)
            return self.x

    def reset_random(self):
        if self.init_x:
            return self.init_x
        else:
            self.t = 0
            self.x = np.zeros(self.x_dim)
            self.x = np.clip(np.random.normal(self.x, 2), -2, 2)
            return self.x

    def render(self):
        pass

    def step(self, omega):

        # 控制律
        x = self.x[0]
        x_dot = self.x[1]
        delta_x = x - self.xd
        delta_x_dot = x_dot - self.xd_dot
        u = - 2 * x - 3 - omega ** 2 * delta_x - 2 * omega * delta_x_dot + self.xd_dot2
        u_origin = u

        # Penalty Calculation
        u_norm = abs((u - np.mean(self.u_bound)) / abs(self.u_bound[0] - self.u_bound[1]) * 2)
        Penalty_bound = 0.75
        if u_norm < Penalty_bound:
            Satu_Penalty = 0
        else:
            if u_norm > 8:
                u_norm = 8
            Satu_Penalty = - 1000 * (np.exp(0.1 * (u_norm - Penalty_bound)) - 1)


        # 限幅
        if u > np.max(self.u_bound):
            u = np.max(self.u_bound)
        elif u < np.min(self.u_bound):
            u = np.min(self.u_bound)


        # 微分方程
        A = np.array([[0, 1], [2, 0]])
        B = np.array([0, 1])
        B_con = np.array([0, 3])
        x_dot = np.dot(A, self.x) + np.dot(B, u) + B_con
        self.x += self.delta_t * x_dot
        self.t = self.t + self.delta_t

        # Reward Calculation
        reward = (omega + Satu_Penalty / 3) / 500

        info = {}
        info['action'] = u
        info['time'] = self.t
        info['u_ori'] = u_origin
        info['reward'] = reward
        info['penalty'] = Satu_Penalty
        if self.t > self.total_time:
            done = True
            if abs(delta_x) > 1:
                reward += - 20
        else:
            done = False

        # Return
        return self.x, reward, done, info
