import numpy as np
import matplotlib.pyplot as plt
import random


# Updated lion and cow grid from task2 (now lion needs to take the cow at the right up corner and bring it to the start)
class LionAndCowGrid:
    def __init__(self, size_of_grid):
        self.size_of_grid = size_of_grid

        self.num_of_states = 2 * size_of_grid ** 2

        self.actions = [(0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1), (0, 0, 0)]
        self.num_of_actions = len(self.actions)

        self.possible_actions = {}

        for i in range(size_of_grid ** 2):
            s = (i, 0)
            self.possible_actions[s] = []
            s = (i, 1)
            self.possible_actions[s] = []

        for i in range(size_of_grid ** 2):
            s = (i, 0)
            self.possible_actions[s].append(self.get_possible_actions(s))
            s = (i, 1)
            self.possible_actions[s].append(self.get_possible_actions(s))

        self.num_of_actions = len(self.actions)

    # Function that return reward if we are in state s and do action a
    def get_reward(self, s, a):
        if (s[0] == 1 and s[1] == 1 and a == (-1, 0, 0)) or \
                (s[0] == self.size_of_grid and s[1] == 1 and a == (0, -1, 0)):
            return 100
        else:
            return 0

    # Function that convert int number of a state to it's coordinates on the grid
    def get_coordinates_by_state(self, s):
        return s[0] % self.size_of_grid, s[0] // self.size_of_grid, s[1]

    # Function that convert two coordinates of the state to it's int number (from 0 to size_of_grid ** 2 - 1)
    def get_state_by_coordinates(self, s_coord):
        x = s_coord[0]
        y = s_coord[1]
        z = s_coord[2]
        return y * self.size_of_grid + x, z

    def get_new_state(self, s, a):
        return self.get_state_by_coordinates((self.get_coordinates_by_state(s)[0] + a[0],
                                              self.get_coordinates_by_state(s)[1] + a[1],
                                              self.get_coordinates_by_state(s)[2] + a[2]))

    # Function that return for state actions available from this state
    def get_possible_actions(self, s):
        # For 3 corner cells only two actions are possible and for cell with cow only 1 action is possible
        if s[0] == 0 and s[1] == 0:
            return [(1, 0, 0), (0, 1, 0)]
        if s[0] == 0 and s[1] == 1:
            return [(0, 0, 0)]
        if s[0] == self.size_of_grid - 1:
            return [(-1, 0, 0), (0, 1, 0)]
        if s[0] == self.size_of_grid * (self.size_of_grid - 1):
            return [(1, 0, 0), (0, -1, 0)]
        if s[0] == self.size_of_grid ** 2 - 1 and s[1] == 0:
            return [(0, 0, 1)]
        if s[0] == self.size_of_grid ** 2 - 1 and s[1] == 1:
            return [(0, -1, 0), (-1, 0, 0)]

        # For edge cells three actions are possible
        if s[0] % self.size_of_grid == 0:
            return [(1, 0, 0), (0, 1, 0), (0, -1, 0)]
        if (s[0] + 1) % self.size_of_grid == 0:
            return [(0, -1, 0), (0, 1, 0), (-1, 0, 0)]
        if s[0] < self.size_of_grid:
            return [(1, 0, 0), (0, 1, 0), (-1, 0, 0)]
        if s[0] > self.size_of_grid * (self.size_of_grid - 1):
            return [(1, 0, 0), (0, -1, 0), (-1, 0, 0)]

        # For other cells 4 actions are possible
        return [(0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]


# Supporting function for SARSA and Q-learning algorithm
def get_action_according_to_eps_greedy_q(q_func, grid, s, eps):
    q_max = 0
    maxes = []

    actions = grid.possible_actions[s][0]
    actions_len = len(actions)
    for j in range(actions_len):
        a = actions[j]
        if q_func[(s, a)] == q_max:
            maxes.append(a)
        if q_func[(s, a)] > q_max:
            q_max = q_func[(s, a)]
            maxes = [a]

    id = random.randrange(len(maxes))
    next_move = maxes[id]
    p = random.random()

    if p < eps:
        return actions[random.randrange(actions_len)]
    else:
        return next_move


# Q-learning algorithm implementation
def q_learning_algo(grid, num_of_episodes, alpha, discount, eps):
    # Initialization section
    q_func = {}
    iterations = []
    size_of_grid = grid.size_of_grid

    for i in range(size_of_grid ** 2):
        ss = [(i, 0), (i,  1)]

        for s in ss:
            actions = grid.possible_actions[s][0]
            actions_len = len(actions)
            for j in range(actions_len):
                a = actions[j]
                q_func[(s, a)] = 0

    # Main section of the algorithm
    for i in range(num_of_episodes):
        s = (0, 0)
        cur_iter = 0
        while not s == (0, 1):
            cur_iter += 1
            a = get_action_according_to_eps_greedy_q(q_func, grid, s, eps)
            r = grid.get_reward(s, a)
            s1 = grid.get_new_state(s, a)
            a1 = get_action_according_to_eps_greedy_q(q_func, grid, s1, eps)
            q_func[(s, a)] = q_func[(s, a)] + alpha * (r + discount * q_func[(s1, a1)] - q_func[(s, a)])
            s = s1

        iterations.append(cur_iter)

        if i % 100 == 0:
            print(i)

    return q_func, iterations


# SARSA algorithm implementation
def sarsa_algo(grid, num_of_episodes, alpha, discount, eps):
    # Initialization section
    q_func = {}
    iterations = []
    size_of_grid = grid.size_of_grid

    for i in range(size_of_grid ** 2):
        ss = [(i, 0), (i,  1)]

        for s in ss:
            actions = grid.possible_actions[s][0]
            actions_len = len(actions)
            for j in range(actions_len):
                a = actions[j]
                q_func[(s, a)] = 0

    # Main section of the algorithm
    for i in range(num_of_episodes):
        s = (0, 0)
        a = get_action_according_to_eps_greedy_q(q_func, grid, s, eps)

        cur_iter = 0
        while not s == (0, 1):
            cur_iter += 1
            r = grid.get_reward(s, a)
            s1 = grid.get_new_state(s, a)
            a1 = get_action_according_to_eps_greedy_q(q_func, grid, s1, eps)
            q_func[(s, a)] = q_func[(s, a)] + alpha * (r + discount * q_func[(s1, a1)] - q_func[(s, a)])
            s = s1
            a = a1

        iterations.append(cur_iter)

        if i % 10000 == 0:
            print(i)

    return q_func, iterations


# Function for drawing plots
def draw_plot(name, y):
    plt.plot(y, color="blue")
    plt.title("Learning curve for " + name)
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode")
    plt.savefig("learning_curve_" + name + ".png")
    plt.show()


def main():
    lion_and_cow_grid = LionAndCowGrid(10)
    eps = 10e-5
    discount = 0.8

    q_func, iterations = sarsa_algo(lion_and_cow_grid, 75, 1, discount, eps)
    print(iterations)
    draw_plot("SARSA", iterations)

    print("------- Task 4. Updated Lion And Cow Grid -------")
    print()
    print("------- SARSA algorithm implementation -------")

    best_action_1 = {}
    best_action_2 = {}

    for i in range(9):
        best_action_1[i] = get_action_according_to_eps_greedy_q(q_func, lion_and_cow_grid, (i, 0), eps)
        best_action_2[i] = get_action_according_to_eps_greedy_q(q_func, lion_and_cow_grid, (i, 1), eps)

    print("Best actions for lion to find a cow: ", best_action_1)
    print("Best actions for lion to take cow back to the left corner: ", best_action_2)


    print()
    print("------- Q-learning algorithm implementation -------")

    q_func, iterations = q_learning_algo(lion_and_cow_grid, 75, 1, discount, eps)
    print(iterations)
    draw_plot("Q learning", iterations)

    best_action_1 = {}
    best_action_2 = {}

    for i in range(9):
        best_action_1[i] = get_action_according_to_eps_greedy_q(q_func, lion_and_cow_grid, (i, 0), eps)
        best_action_2[i] = get_action_according_to_eps_greedy_q(q_func, lion_and_cow_grid, (i, 1), eps)

    print("Best actions for lion to find a cow: ", best_action_1)
    print("Best actions for lion to take cow back to the left corner: ", best_action_2)



if __name__ == "__main__":
    main()