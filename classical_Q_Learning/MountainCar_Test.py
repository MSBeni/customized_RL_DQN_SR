import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the gym mountain car environment
env = gym.make('MountainCar-v0')


# creating a numpy array holding the bins for the observations of the car (position and velocity)
def create_bins(number_of_bins_per_observation):
    """
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
    """
    pos_bins = np.linspace(-1.2, 0.6, number_of_bins_per_observation)
    vel_bins = np.linspace(-0.07, 0.07, number_of_bins_per_observation)
    bins = np.array([pos_bins, vel_bins])
    return bins


