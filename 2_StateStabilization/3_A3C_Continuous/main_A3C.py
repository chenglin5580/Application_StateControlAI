

############################ Package import  ####################################

import A3C
from StateStabilizationProblem import SSCPENV as Object_AI # 程林， 状态镇定


############################ Object and Method  ####################################

env = Object_AI()

RL = A3C.A3C(env, A3C_flag=True)
# RL.run()
