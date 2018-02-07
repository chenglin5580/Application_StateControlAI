import numpy as np
from SS_Pro_Con import SSCPENV as SSP


# 继承连续

class SSCPENV(SSP):
    def __init__(self, x_dim=2, action_dim=1):
        super().__init__(x_dim, action_dim)
        self.abound = np.linspace(1, 12, 10)
        self.action_dim = len(self.abound)

    def step(self, omega):
        if self.action_dim == 1:
            if type(omega) == np.ndarray:
                omega = omega[0]
        omega = self.abound[int(omega)]
        return super().step(omega)
