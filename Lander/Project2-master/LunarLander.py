import random

import gym
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *
from theano import tensor as T
import time

np.random.seed(20)
random.seed(200)

Memory_Max = 100000
Batch_size = 32


def moving_average(arr, window):
    val = np.cumsum(arr, dtype=float)
    val[window:] = val[window:] - val[:-window]
    return val[window - 1:] / window


def huber_loss(y_true, y_pred):
    max_delta = 1.0
    error = y_true - y_pred
    abs_error = np.abs(error)
    loss1 = 0.5 * T.square(error)
    loss2 = max_delta * abs_error - 0.5 * T.square(max_delta)
    loss = T.switch(T.le(abs_error, max_delta), loss1, loss2)
    return T.mean(loss)


class ReplayMemory:
    def __init__(self, memoryCapacity):
        self.memory_Capacity = memoryCapacity
        self.memory = []

    def push_Memory(self, transitions):
        if len(self.memory) >= self.memory_Capacity:
            #self.memory.append(transitions)
            del self.memory[0] ## removing first entry
        self.memory.append(transitions)

    def sampleMemory(self):
        return random.sample(self.memory, Batch_size)

    def getMemory(self):
        return self.memory


class QModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(Batch_size, activation='relu', input_shape=(numStates,)))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(Batch_size, activation='relu'))
        self.model.add(Dense(Batch_size, activation='relu'))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(numActions, activation='linear'))
        optimizer = RMSprop(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss=huber_loss)

    def model_Train(self, train_x, train_y):
        self.model.fit(train_x, train_y, verbose=0)

    def model_Predict(self, test_x):
        return self.model.predict(test_x)

    def get_trained_model(self):
        return self.model


class Actor:
    def __init__(self):
        self.steps = 0
        self.decayRate = 70

        self.gamma = 0.99
        self.eps_Start = 1.0
        self.eps_End = 0.1

    def take_action(self, state):
        state = state.reshape((state.shape[0], 1))
        threshold = self.eps_End + (self.eps_Start - self.eps_End) * np.exp(-1. * self.steps / self.decayRate)

        if np.random.rand(1) < threshold:
            action = np.random.randint(0, numActions)
        else:

            ##debug this
            action = np.argmax(model.model_Predict(state.T))
        self.steps += 1
        return action

    def observe_replayMemory(self, *args):
        replay_memory.push_Memory(args)

    def optimize(self):
        if len(replay_memory.memory) < Batch_size:
            return
        else:
            random_batch = replay_memory.sampleMemory()
            final_next_state_batch = np.zeros(numStates)
            curr_state_batch = [s[2] for s in random_batch]
            next_state_batch = [final_next_state_batch if s[3] is None else s[3] for s in random_batch]
            np_curr = np.asarray(curr_state_batch)
            np_next = np.asarray(next_state_batch)

            curr_action_values = model.model_Predict(np_curr)
            next_action_values = model.model_Predict(np_next)
            train_X = np.zeros((len(random_batch), numStates))
            train_Y = np.zeros((len(random_batch), numActions))
            for i in range(len(random_batch)):
                curr_state = random_batch[i][2]
                next_state = random_batch[i][3]
                action = random_batch[i][0]
                reward = random_batch[i][1]
                target_Value = curr_action_values[i]
                ### terminal state
                if next_state is None:
                    target_Value[action] = reward
                else:
                    # blah = np.max(next_action_values[i])
                    target_Value[action] = reward + self.gamma * (np.max(next_action_values[i]))
                train_X[i] = curr_state
                train_Y[i] = target_Value

        model.model_Train(train_X, train_Y)


class Environment:
    def __init__(self, env):
        self.env = env

    def run(self):
        cumm_reward = []
        for i in range(1500):
            tic = time.clock()
            state = self.env.reset()
            done = False
            Reward = 0
            while not done:
                self.env.render()
                action = actor.take_action(state)
                next_state, reward, done, info = self.env.step(action)
                ##debug
                if done:
                    next_state = None

                transitions = (action, reward, state, next_state)
                replay_memory.push_Memory(transitions)
                actor.optimize()
                Reward += reward
                state = next_state
            toc = time.clock()
            print("reward: ", Reward)
            cumm_reward.append(Reward)
            latest_arr = cumm_reward[-100:]
            exp_mean = np.mean(latest_arr)
            if i!=0 and i%100==0:
                np.save("LunarLanderPlain-reward"+str(i),cumm_reward)
            print ("Episode time:" ,toc-tic)
            print("Mean100:", exp_mean)
            if exp_mean >=201:
                return cumm_reward



lunar_lander = gym.make('LunarLander-v2')
numStates = lunar_lander.observation_space.shape[0]
numActions = lunar_lander.action_space.n
actor = Actor()
env = Environment(lunar_lander)
model = QModel()
replay_memory = ReplayMemory(Memory_Max)
print(replay_memory.getMemory())
try:
    tic = time.clock()
    totalRewards =  env.run()
    np.save("TotalRewards-Plain",totalRewards)
    print("Total runTime:", time.clock()-tic)
## after termination, save the model
finally:
    pass
    trainedModel = model.get_trained_model()
    trainedModel.save("Plain.h5")
