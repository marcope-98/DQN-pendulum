# HYPER-PARAMETERS

## REPLAY BUFFER
MINIBATCH_SIZE          = 64
REPLAY_START_SIZE       = 5_000#1_000              #    50 000
REPLAY_MEMORY_SIZE      = 100_000#50_000             # 1 000 000
REPLAY_SAMPLE_STEP      = 4

## TRAINING
GAMMA                   = 0.95                # 0.99  
N_EPISODES              = 10_000#2_500              # 5 000 also works
LEARNING_RATE           = 0.001#0.001              # 0.00025 
MAX_EPISODE_LENGTH      = 150                # 150 also works
NETWORK_RESET           = 5*MAX_EPISODE_LENGTH

## EPSILON-GREEDY
INITIAL_EXPLORATION     = 1.   
EXPLORATION_DECAY       = 0.001
FINAL_EXPLORATION       = 0.001              # 0.1
FINAL_EXPLORATION_FRAME = 1_000_000      # for linear epsilon greedy decay (not used)

## VISUAL
GEPETTO_GUI_VISUAL = 500

## MODEL
N_JOINTS                = 2
NU                      = 21
UMAX                    = 5.0
DT                      = 0.1
NDT                     = 1
NOISE                   = 0.
