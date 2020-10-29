# %%
import random
import numpy as np
import gym
import matplotlib.pyplot as plt

# %% 
class QStatistics:
    
    def __init__(self):
        self.rewards = []
        self.explorationRate = []
        self.stepsPerEpoch = []
        self.exploredAtEpoch = 0
        self.stepsAtEpoch = 0
        self.rewardsAtEpoch = 0

    def NextEpoch(self):
        self.rewards.append(self.rewardsAtEpoch)
        self.explorationRate.append(self.exploredAtEpoch)
        self.stepsPerEpoch.append(self.stepsAtEpoch)

        self.exploredAtEpoch = 0
        self.stepsAtEpoch = 0
        self.rewardsAtEpoch = 0
    
    def StepTaken(self, reward):
        self.rewardsAtEpoch+=reward
        self.stepsAtEpoch+=1

    def Explored(self):
        self.exploredAtEpoch+=1


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
stat = QStatistics()

for epoch in range(epochs):
    state = frozenLake.reset()
    step = 0
    done = False
    for step in range(maxSteps):
        if expTradeOff.ShouldExplore():
            action = frozenLake.action_space.sample()
            stat.Explored()
        else:
            action = np.argmax(qtable[state,:])
        
        newState, reward, done, info = frozenLake.step(action)
        qtable[state,action] = qtable[state,action] + learningRate * (reward + gamma * np.max(qtable[newState,:])-qtable[state,action])
        stat.StepTaken(reward)
        state = newState

        if done:
            break

    expTradeOff.NextEpoch(epoch)
    stat.NextEpoch()
# %%
print(qtable[:2,:])
print ("Score over time: " +  str(sum(stat.rewards)/epochs))
plt.plot(stat.rewards)
#%%
print ("Exploration/exploitation trade-off")
plt.plot(stat.explorationRate)

# %% 
print ("Steps per epoch")
plt.plot(stat.stepsPerEpoch)
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
