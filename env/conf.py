import numpy as np

# HYPER-PARAMETERS

# True  => single pendulum
# False => double pendulum
if True:  

    ## REPLAY BUFFER
    MINIBATCH_SIZE          = 64
    REPLAY_START_SIZE       = 128
    REPLAY_MEMORY_SIZE      = 50_000
    REPLAY_SAMPLE_STEP      = 1

    ## TRAINING
    GAMMA                   = 0.95  
    N_EPISODES              = 10_000
    LEARNING_RATE           = 0.001
    MAX_EPISODE_LENGTH      = 150                
    NETWORK_RESET           = 32

    ## EPSILON-GREEDY
    INITIAL_EXPLORATION     = 1.   
    EXPLORATION_DECAY       = 1000
    FINAL_EXPLORATION       = 0.001

    ## MODEL
    N_JOINTS                = 1
    NU                      = 11
    UMAX                    = 5.0
    DT                      = 0.1
    NDT                     = 10
    NOISE                   = 0.
    WITHSinCos              = False

else:
    ## REPLAY BUFFER
    MINIBATCH_SIZE          = 128
    REPLAY_START_SIZE       = 256
    REPLAY_MEMORY_SIZE      = 100_000
    REPLAY_SAMPLE_STEP      = 1

    ## TRAINING
    GAMMA                   = 0.95  
    N_EPISODES              = 10_000
    LEARNING_RATE           = 0.001
    MAX_EPISODE_LENGTH      = 200                
    NETWORK_RESET           = 5*MAX_EPISODE_LENGTH

    ## EPSILON-GREEDY
    INITIAL_EXPLORATION     = 1.   
    EXPLORATION_DECAY       = 1000
    FINAL_EXPLORATION       = 0.001

    ## MODEL
    N_JOINTS                = 2
    NU                      = 21
    UMAX                    = 5.0
    DT                      = 0.1
    NDT                     = 10
    NOISE                   = 0.
    WITHSinCos              = True
    

## VISUAL
GEPETTO_GUI_VISUAL      = 500
if N_JOINTS == 1:
    X_0                     = np.array([[np.pi], [0.]])
elif N_JOINTS == 2:
    X_0                     = np.array([[np.pi, 0.], [0., 0.]])
else:
    print("undefined behaviour")

