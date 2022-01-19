# HYPER-PARAMETERS

## REPLAY BUFFER
MINIBATCH_SIZE          = 64
REPLAY_START_SIZE       = 5_000
REPLAY_MEMORY_SIZE      = 100_000
REPLAY_SAMPLE_STEP      = 4

## TRAINING
GAMMA                   = 0.95  
N_EPISODES              = 10_000
LEARNING_RATE           = 0.001
MAX_EPISODE_LENGTH      = 150
NETWORK_RESET           = 5*MAX_EPISODE_LENGTH

## EPSILON-GREEDY
INITIAL_EXPLORATION     = 1.   
EXPLORATION_DECAY       = 0.001
FINAL_EXPLORATION       = 0.001

## VISUAL
GEPETTO_GUI_VISUAL = 500

## MODEL
N_JOINTS                = 2
NU                      = 21
UMAX                    = 5.0
DT                      = 0.1
NDT                     = 1
NOISE                   = 0.
