import numpy as np
import matplotlib.pyplot as plt


def moving_average(arr, window):
    val = np.cumsum(arr, dtype=float)
    val[window:] = val[window:] - val[:-window]
    return val[window - 1:] / window


def plotRewards_compare(arr1,label1,arr2,label2,title,save=False):
    x = np.arange(0,len(arr1))
    y = arr1
    plt.plot(x,y,label=label1)
    x2 = np.arange(0,len(arr2))
    y2= arr2
    plt.plot(x2,y2,label=label2)

    plt.axhline(y=200.0,color='r',linestyle='--')
    plt.xlabel("Number of episodes")
    plt.ylabel("Rewards")
    plt.title("Average Reward")
    plt.legend(loc='upper right')
    if save:
        fig = plt.gcf()
        fig.savefig(title)
    plt.show()


def plotRewards(arr,title,save=False):
    x = np.arange(0,len(arr))
    y = arr
    plt.plot(x,y)
    plt.axhline(y=200.0,color='r',linestyle='--')
    plt.xlabel("Number of episodes")
    plt.ylabel("Rewards")
    plt.title("Average Reward")
    if save:
        fig = plt.gcf()
        fig.savefig(title)
    plt.show()

if __name__ == "__main__":
    temp = np.load("TotalRewards-DDQN.npy")
    avg = moving_average(temp,100)
    plotRewards(avg,"TrainRewards-DDQN",True)