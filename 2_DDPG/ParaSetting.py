import numpy as np
class Para_setting(object):
    def __init__(self):
        self.var_ini = 10
        self.var_end = 2
        self.var_decend  = 0.994  #0.994 需要267步伐
        step_decend = np.log2(self.var_end/self.var_ini) / np.log2(self.var_decend)
        self.max_Episodes = 300
        if step_decend > self.max_Episodes:
            self.max_Episodes = int(step_decend) + 100





