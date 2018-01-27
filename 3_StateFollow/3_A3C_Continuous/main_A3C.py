

# from SmallStateControl import SSCPENV as Object_AI
from StateStabilizationProblem import SSCPENV as Object_AI
import A3C
env = Object_AI()
para = A3C.Para(env,
                MAX_GLOBAL_EP=2000,
                UPDATE_GLOBAL_ITER=50,
                GAMMA=0.9,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.001,)
RL = A3C.A3C(para)
RL.run()
