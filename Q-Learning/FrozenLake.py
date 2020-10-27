# %%
import random
import numpy as np
import gym

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
rewards = []

# %%
for epoch in range(epochs):
    state = frozenLake.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(maxSteps):
        if expTradeOff.ShouldExplore():
            action = frozenLake.action_space.sample()
        else:
            action = np.argmax(qtable[state,:])
        
        newState, reward, done, info = frozenLake.step(action)
        qtable[state,action] = qtable[state,action] + learningRate * (reward + gamma * np.max(qtable[newState,:]-qtable[state,action]))
        total_rewards += reward
        state = newState
        if done:
            break

    expTradeOff.NextEpoch(epoch)
    rewards.append(total_rewards)
# %%
print(qtable)
print ("Score over time: " +  str(sum(rewards)/epochs))
# %%
state = frozenLake.reset()
step = 0
done = False
frozenLake.render()

for step in range(maxSteps):
    action = np.argmax(qtable[state,:])
    new_state, reward, done, info = frozenLake.step(action)
    if done:
        frozenLake.render()
        print("Steps taken: ",step)
        break
    # frozenLake.render()
    state = new_state

print("\nover")

# %%
frozenLake.close()