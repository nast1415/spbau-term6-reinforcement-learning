import gym
import numpy as np


# Make mountain car domain using OpenAI gym
env = gym.make('MountainCar-v0')

# Get info about boundaries of position and speed from environment
left_position, min_velocity = env.observation_space.low
right_position, max_velocity = env.observation_space.high

# Number of tilings and tiles size set to 9 and 9 (but it is also good result for (9, 9) and (11, 11))
# other values are worse
number_of_tilings = 10
tiles_size = 10

# Set offset value for position and velocity (as delta between max and min values divided by the number of tilings)
position_shift = (right_position - left_position) / number_of_tilings
velocity_shift = (max_velocity - min_velocity) / number_of_tilings

# Create tilings for position and velocity with given offset
position_tiling = np.zeros((number_of_tilings, tiles_size + 1))
velocity_tiling = np.zeros((number_of_tilings, tiles_size + 1))

for i in range(number_of_tilings):
    position_tiling[i] = np.linspace(left_position - position_shift * (1 - i / (number_of_tilings - 1)),
                                     right_position + position_shift * i / (number_of_tilings - 1), tiles_size + 1)
    velocity_tiling[i] = np.linspace(min_velocity - velocity_shift * (1 - i / (number_of_tilings - 1)),
                                     max_velocity + velocity_shift * i / (number_of_tilings - 1), tiles_size + 1)
# print(position_tiling)
# print(velocity_tiling)

# Next two functions are supporting for SARSA with tile coding algorithm


# Function to get tiles by state (position, velocity)
def get_tiles(position, velocity):
    return np.argmin(position_tiling < position, axis=1) * tiles_size + np.argmin(velocity_tiling < velocity, axis=1) \
           + np.arange(number_of_tilings) * tiles_size ** 2

number_of_actions = env.action_space.n
q = np.zeros((number_of_tilings * tiles_size ** 2, number_of_actions))


# Function to get next action by state (position, velocity)
def get_action(position, velocity):
    return np.argmax(q[get_tiles(position, velocity)].sum(axis=0))

# Set attributes for SARSA algorithm
alpha = 0.6
gamma = 0.99

number_of_episodes = 500
number_of_steps = env.spec.timestep_limit

print("Please wait, SARSA algorithm is working...")

# SARSA with tile coding algorithm (repeated 'number_of_episodes' times)
for episode in range(number_of_episodes):
    state = env.reset()

    for step in range(number_of_steps):
        a = get_action(*state)
        t = get_tiles(*state)

        next_state, reward, is_terminal, smth1 = env.step(a)
        q[t, a] += alpha / number_of_tilings * (reward + gamma * np.max(q[get_tiles(*next_state)].sum(axis=0)) -
                                                q[t, a].sum(axis=0))
        state = next_state

        if is_terminal:
            break

next_state = env.reset()
sum_reward = 0

# Play mountain_car and find summary reward
while True:
    env.render()
    next_state, reward, terminal, smth2 = env.step(get_action(*next_state))
    sum_reward += reward
    if terminal:
        break

print("Summary reward: ", sum_reward)