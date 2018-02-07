

import A3C

# from SmallStateControl import SSCPENV as Object_AI
from SS_Pro_Con import SSCPENV as Object_AI

env = Object_AI()
para = A3C.Para(env,
                MAX_GLOBAL_EP=2000,
                UPDATE_GLOBAL_ITER=20,
                GAMMA=0.9,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.0001, )
RL = A3C.A3C(para)
RL.run()
# RL.display()
