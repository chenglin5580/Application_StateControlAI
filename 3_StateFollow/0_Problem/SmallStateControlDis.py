import numpy as np
from SmallStateControl import SSCPENV as SSP


# 继承连续

class SSCPENV(SSP):
    def __init__(self, x_dim=2, action_dim=1, init_x=None):
        super().__init__(x_dim, action_dim, init_x)
        self.abound = np.linspace(1, 12, 10)
        self.n_action = len(self.abound)

    def step(self, omega):
        if self.action_dim == 1:
            if type(omega) == np.ndarray:
                omega = omega[0]
        omega = self.abound[int(omega)]
        return super().step(omega)
