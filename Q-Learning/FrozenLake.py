# %%
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
# %%
class ExpTradeOff:
    def __init__(self, startEpsilon, endEpsilon, decayRate):
        self.epsilon = startEpsilon
        self.startEpsilon = startEpsilon
        self.endEpsilon = endEpsilon
        self.decayRate = decayRate

    def NextEpoch(self, epoch):
        self.epsilon = self.endEpsilon + (self.startEpsilon - self.endEpsilon) * np.exp(-self.decayRate * epoch)
    def ShouldExplore(self):
        threshold = random.uniform(0,1)
        return threshold < self.epsilon

# %%
frozenLake = gym.make("FrozenLake-v0")

# %%
epochs = 50*1000
maxSteps = 99

learningRate = 0.8
gamma = 0.95

# %%
actionSize = frozenLake.action_space.n
stateSize = frozenLake.observation_space.n
qtable = np.zeros((stateSize, actionSize))
expTradeOff = ExpTradeOff(1.0, 0.01, 0.005)

# statistics
rewards = []
explorationRate = []
stepsPerEpoch = []
for epoch in range(epochs):
    state = frozenLake.reset()
    step = 0
    done = False
    total_rewards = 0
    exploredAtEpoch = 0
    stepsAtEpoch = 0
    for step in range(maxSteps):
        if expTradeOff.ShouldExplore():
            action = frozenLake.action_space.sample()
            exploredAtEpoch+=1
        else:
            action = np.argmax(qtable[state,:])
        
        newState, reward, done, info = frozenLake.step(action)
        qtable[state,action] = qtable[state,action] + learningRate * (reward + gamma * np.max(qtable[newState,:])-qtable[state,action])
        total_rewards += reward
        state = newState
        stepsAtEpoch += 1
        if done:
            break

    expTradeOff.NextEpoch(epoch)
    rewards.append(total_rewards)
    explorationRate.append(exploredAtEpoch)
    stepsPerEpoch.append(stepsAtEpoch)
# %%
print(qtable[:2,:])
print ("Score over time: " +  str(sum(rewards)/epochs))
plt.plot(rewards)
#%%
print ("Exploration/exploitation trade-off")
plt.plot(explorationRate)

# %% 
print ("Steps per epoch")
plt.plot(stepsPerEpoch)
# %%
totalReward = 0
for episode in range(1000):
    state = frozenLake.reset()
    step = 0
    done = False
    frozenLake.render()

    for step in range(maxSteps):
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = frozenLake.step(action)
        if done:
            frozenLake.render()
            totalReward += reward # reward is 1 for the target cell, otherwise 0
            print("Steps taken: ",step)
            break
        # frozenLake.render()
        state = new_state
print("\nTotalReward:", totalReward)
print("\nover")

# %%
frozenLake.close()
# %%
