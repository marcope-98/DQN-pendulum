# HYPER-PARAMETERS

## Replay buffer
MINIBATCH_SIZE      = 32
REPLAY_START_SIZE   = 1_000              #    50 000
REPLAY_MEMORY_SIZE  = 50_000             # 1 000 000
REPLAY_SAMPLE_STEP  = 4

## TRAINING
GAMMA               = 0.9                # 0.99  
N_EPISODES          = 2_500              # 5 000 also works
NETWORK_RESET       = 100
LEARNING_RATE       = 0.001              # 0.00025 
MAX_EPISODE_LENGTH  = 120                # 150 also works

## EPSILON-GREEDY
INITIAL_EXPLORATION = 1.   
EXPLORATION_DECAY   = 0.001
FINAL_EXPLORATION   = 0.001              # 0.1
FINAL_EXPLORATION_FRAME = 1_000_000      # for linear epsilon greedy decay (not used)

## VISUAL
GEPETTO_GUI_VISUAL = 500