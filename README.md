# Lander
Required dependencies:

1)Install Python3. Version used in this project is 3.5.2

2)Install numpy

3)clone and install gym environment

4)Install Keras and Theano. Theano will be the backend used for this project. Hence, set the keras backend to use theano.

5)Install matplotlib

The project was tested on PyCharm IDE. It should work with any IDE. Using an IDE to run the code is most recommneded.

LunarLander-DDQN.py is the main code. It accepts an input argument to set - "Train" or "Test" mode. runConfig.png provides a example of the run configuration in pyCharm. Other IDEs would use a similar setting. On terminal keras.load function throws an exception. Hence, discouraged.
Following is an example of the configuration in IDE - C:\Users\prath\Anaconda3\python.exe -s D:/CS7642/projectTest/LunarLander-DDQN.py Test

2)The above file could be directly run on "Test" mode as the trained model lunarLander-Trained-DDQN.h5 is saved for re-use.

3)TotalRewards-DDQN.npy and TestRun-Rewards.npy are the saved rewards for train and test run respectively.

4)Plot_Utils.py is a utils file to generate any rewards plot.

Following result from test:
![Alt Text](https://j.gifs.com/59MoOB.gif)
