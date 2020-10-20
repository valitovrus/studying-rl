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
gamma = 0.618  # why?
# %%
class ExpTradeOff:
    epsilon = 1.0
    startEpsilon = 1.0
    endEpsilon = 0.01
    decayRate = 0.01

    def NextEpoch(self, epoch):
        self.epsilon = self.endEpsilon + (self.startEpsilon - self.endEpsilon) * np.exp(-self.decayRate * epoch)
    def ShouldExplore(self):
        threshold = random.uniform(0,1)
        return threshold < self.epsilon


# %%
expTradeOff = ExpTradeOff()
for epoch in range(epochs):
    state = frozenLake.reset()
    step = 0
    done = False
    
    for step in range(maxSteps):
        if expTradeOff.ShouldExplore:
            action = frozenLake.action_space.sample()
        else:
            action = np.argmax(qtable[state,:])
        
        newState, reward, done, info = frozenLake.step(action)
        qtable[state,action] += learningRate * (reward + gamma * np.max(qtable[newState,:]-qtable[state,action]))
        state = newState
        if done:
            break

    expTradeOff.NextEpoch(epoch)

# %%
print(qtable)
# %%
state = frozenLake.reset()
step = 0
done = False
frozenLake.render()

for step in range(5):
    action = np.argmax(qtable[state,:])
    new_state, reward, done, info = frozenLake.step(action)
    if done:
        frozenLake.render()
        print("Steps taken: ",step)
        break
    frozenLake.render()
    state = new_state

print("\nover")

# %%
env.close()