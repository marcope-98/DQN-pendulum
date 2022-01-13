LEARNING_RATE       = 0.001
REPLAY_MEMORY_SIZE  = 1_000_000
N_EPISODES          = 2500           # number of training episodes
MAX_EPISODE_LENGTH  = 100
INITIAL_EXPLORATION = 1     # initialize the exploration probability to 1
EXPLORATION_DECAY   = 0.001 # exploration decay for exponential decreasing

#EXPLORATION_DECAY   = 0.0005 # exploration decay for exponential decreasing
FINAL_EXPLORATION   = 0.001 # minimum of exploration proba
MINIBATCH_SIZE      = 50
GAMMA               = 0.9 
NETWORK_RESET       = 25