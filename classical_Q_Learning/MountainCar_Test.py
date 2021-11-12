import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the gym mountain car environment
env = gym.make('MountainCar-v0')

# Set hyper parameters
EPOCHS = 30000
BURN_IN = 100
epsilon = 1

EPSILON_END = 10000
EPSILON_REDUCE = 0.0001

ALPHA = 0.8
GAMMA = 0.9


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


all_bins = create_bins(25)


# Creating a function to discretize an observation
def discretize_observation(observations, bins):
    binned_observations = list()
    for i in range(len(observations)):
        binned_observations.append(np.digitize(observations[i], bins[i], right=False))
    # Using tuple for later indexing
    return tuple(binned_observations)


# CREATE THE Q TABLE
q_table = np.zeros([25, 25, env.action_space.n])


def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    """
    Defining a function to select an action based on epsilon greedy -- Exploration and Exploitation
    :param epsilon:
    :param q_table:
    :param discrete_state:
    :return: action
    """
    rand_num = np.random.random()
    if rand_num > epsilon:
        action = np.argmax(q_table[discrete_state])
    else:
        action = env.action_space.sample()
    return action


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    """
    Bellman Equation Calculation of the next q value
    :param old_q_value:
    :param reward:
    :param next_optimal_q_value:
    :return:
    """
    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


log_interval = 100  # How often do we update the plot? (Just for performance reasons)
# Here we set up the routine for the live plotting of the achieved points ######
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()

max_position_log = []  # to store all achieved points
mean_positions_log = []  # to store a running mean of the last 30 results
epochs = []  # store the epoch for plotting


def reduce_epsilon(epsilon, epoch):
    if BURN_IN <= epoch <= EPSILON_END:
        epsilon -= EPSILON_REDUCE

    return epsilon


###################### TRAINING TASKS ##########################
for epoch in range(EPOCHS):
    # TODO: Get initial observation and discretize them. Set done to False
    #########################################
    observstions = env.reset()
    discretized_observations = discretize_observation(observstions, all_bins)
    done = False

    # These lines are for plotting.
    max_position = -np.inf
    epochs.append(epoch)

    # As long as current run is alive (i.e not done) perform the following steps:
    while not done:  # Perform current run as long as done is False (as long as there is still time to reach the top)

        # Select action according to epsilon-greedy strategy
        action = epsilon_greedy_action_selection(epsilon, q_table, discretized_observations)

        # Perform selected action and get next state. Do not forget to discretize it
        new_state, reward, done, info = env.step(action)
        position, velocity = new_state
        new_discretized_obs = discretize_observation(new_state, all_bins)

        # Get old Q-value from Q-Table and get next optimal Q-Value
        old_Q = q_table[discretized_observations + (action,)]
        next_opt_Q = np.max(q_table[new_discretized_obs])
        # Compute next Q-Value and insert it into the table
        next_Q = compute_next_q_value(old_Q, reward, next_opt_Q)
        q_table[discretized_observations + (action,)] = next_Q
        # Update the old state with the new one
        discretized_observations = new_discretized_obs

        ##  Only for plotting the results - store the highest point the car is able to reach
        if position > max_position:
            max_position = position

    # Reduce epsilon
    epsilon = reduce_epsilon(epsilon, epoch)
    ##############################################################################

    max_position_log.append(max_position)  # log the highest position the car was able to reach
    running_mean = round(np.mean(max_position_log[-30:]), 2)  # Compute running mean of position over the last 30 epochs
    mean_positions_log.append(running_mean)  # and log it

    ################ Plot the points and running mean ##################
    if epoch % log_interval == 0:
        ax.clear()
        ax.scatter(epochs, max_position_log)
        ax.plot(epochs, max_position_log)
        ax.plot(epochs, mean_positions_log, label=f"Running Mean: {running_mean}")
        plt.legend()
        fig.canvas.draw()

env.close()
plt.show()


# Thanks to the www.pieriandata.com provided data