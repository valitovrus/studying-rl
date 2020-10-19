# %%
import random
import numpy as np
import gym

# %%
frozenLake = gym.make("FrozenLake-v0")

# %%
actionSize = frozenLake.action_space.n
stateSize = frozenLake.observation_space.n
qtable = np.zeros((stateSize, actionSize))

# %%
epochs = 50*1000
testEpochs = 100 
maxSteps = 99

learningRate = 0.7 
gamma = 0.618 # why? 

epsilon = 1.0 
startEpsilon = 1.0
endEpsilon = 0.01 
decayRate = 0.01 
