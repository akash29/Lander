import random

import gym
from keras.layers import *
from keras.models import Sequential, load_model
from keras.optimizers import *
from theano import tensor as T
import time
import sys

np.random.seed(20)
random.seed(200)

Memory_Max = 200000
Batch_size = 32
Update_Frequency = 50


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
            del self.memory[0]  ## removing first entry
        self.memory.append(transitions)

    def sampleMemory(self):
        num = min(Batch_size, len(self.memory))
        return random.sample(self.memory, num)



class QModel:

    def create_model(self):
        model = Sequential()
        model.add(Dense(Batch_size, activation='relu', input_shape=(numStates,)))
        # self.model.add(BatchNormalization())
        model.add(Dense(Batch_size, activation='relu'))
        model.add(Dense(Batch_size,activation='relu'))
        # self.model.add(BatchNormalization())
        model.add(Dense(numActions, activation='linear'))
        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss=huber_loss)
        return model

    def model_Train(self, model, train_x, train_y):
        model.fit(train_x, train_y, verbose=0)

    def model_Predict(self, model, test_x):
        return model.predict(test_x)

    def update_weights(self, model1, model2):
        weights = model1.get_weights()
        model2.set_weights(weights)


class Actor:
    def __init__(self):
        self.steps = 0
        self.decayRate = 70

        self.gamma = 0.99
        self.eps_Start = 1.0
        self.eps_End = 0.1

    def take_action(self, state,threshold):
        state = state.reshape((state.shape[0],1))

        if np.random.rand(1) < threshold:
            action = np.random.randint(0, numActions)
        else:
            action = np.argmax(qmodel.model_Predict(model_1, state.T))
        return action

    def take_random_actions(self):
        return np.random.randint(0,numActions)

    def observe_replayMemory(self, transitions):
        replay_memory.push_Memory(transitions)

        if self.steps % Update_Frequency == 0:
            qmodel.update_weights(model_1,model_2)

    def decay(self):
        threshold = self.eps_End + (self.eps_Start - self.eps_End) * np.exp(-1. * self.steps / self.decayRate)
        self.steps+=1
        return threshold

    def optimize(self):

        random_batch = replay_memory.sampleMemory()
        final_next_state_batch = np.zeros(numStates)
        curr_state_batch = [s[2] for s in random_batch]
        next_state_batch = [final_next_state_batch if s[3] is None else s[3] for s in random_batch]
        np_curr = np.asarray(curr_state_batch)
        np_next = np.asarray(next_state_batch)

        curr_action_values = qmodel.model_Predict(model_1, np_curr)
        next_action_values = qmodel.model_Predict(model_2, np_next)
        train_X = np.zeros((len(random_batch), numStates))
        train_Y = np.zeros((len(random_batch), numActions))
        for i in range(len(random_batch)):
            curr_state = random_batch[i][2]
            next_state = random_batch[i][3]
            action = random_batch[i][0]
            reward = random_batch[i][1]
            target_Value = curr_action_values[i]
            if next_state is None:
                target_Value[action] = reward
            else:
                target_Value[action] = reward + self.gamma * next_action_values[i][np.argmax(curr_action_values[i])]
            train_X[i] = curr_state
            train_Y[i] = target_Value

        qmodel.model_Train(model_1, train_X, train_Y)


class Runner:
    def __init__(self, env):
        self.env = env

    def run(self):
        cumm_reward = []
        for i in range(5000):
            tic = time.clock()
            state = self.env.reset()
            done = False
            Reward = 0
            epsilon = actor.decay()
            while not done:
                self.env.render()
                action = actor.take_action(state,epsilon)
                next_state, reward, done, info = self.env.step(action)
                ##debug
                if done:
                    next_state = None

                transitions = (action, reward, state, next_state)
                actor.observe_replayMemory(transitions)
                actor.optimize()
                Reward += reward
                state = next_state



            toc = time.clock()
            print("reward: ", Reward)
            cumm_reward.append(Reward)
            latest_arr = cumm_reward[-100:]
            exp_mean = np.mean(latest_arr)
            if i!=0 and i%100==0:
                np.save("TotalRewards-DDQN"+str(i), cumm_reward)
            print("Episode time:", toc - tic)
            print("Mean100:", exp_mean)
            if exp_mean >= 200:
                return cumm_reward

    def testRun(self):
        cumm_reward = []
        for i in range(100):
            state = self.env.reset()
            done = False
            Reward = 0
            while not done:
                self.env.render()
                state = state.reshape((state.shape[0], 1))
                action = np.argmax(qmodel.model_Predict(model_1, state.T))
                next_state, reward, done, info = self.env.step(action)
                ##debug
                if done:
                    next_state = None

                Reward += reward
                state = next_state

            print("reward: ", Reward)
            cumm_reward.append(Reward)
        np.save("TestRun-Rewards",cumm_reward)


if __name__ == "__main__":
    runmode = sys.argv[1]
    lunar_lander = gym.make('LunarLander-v2')
    lunar_lander.seed(20)
    numStates = lunar_lander.observation_space.shape[0]
    numActions = lunar_lander.action_space.n
    actor = Actor()
    env = Runner(lunar_lander)
    qmodel = QModel()
    model_1 = qmodel.create_model()
    model_2 = qmodel.create_model()
    replay_memory = ReplayMemory(Memory_Max)
    if runmode == "Train":
        try:
            tic = time.clock()
            totalRewards = env.run()
            np.save("TotalRewards-DDQN", totalRewards)
            print("Total runTime:", time.clock() - tic)
        ## after termination, save the model
        finally:
            model_1.save("lunarLander-Trained-DDQN.h5")
    else:
        del model_1
        del model_2
        model_1 = load_model("lunarLander-Trained-DDQN.h5",custom_objects={'huber_loss':huber_loss})
        env.testRun()

