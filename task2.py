import numpy as np


class LionAndCowGrid:
    def __init__(self, size_of_grid):
        self.size_of_grid = size_of_grid

        self.num_of_states = size_of_grid ** 2

        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        self.num_of_actions = len(self.actions)

        self.possible_actions = []

        for i in range(size_of_grid ** 2):
            self.possible_actions.append([])

        for i in range(size_of_grid ** 2):
            self.possible_actions[i].append(self.get_possible_actions(i))

        self.num_of_actions = len(self.actions)

    # Function that return reward if we are in state s and do action a
    def get_reward(self, s, a):
        if ((s == self.size_of_grid ** 2 - 2) and (a == (1, 0))) \
                or ((s == self.size_of_grid * (self.size_of_grid - 1) - 1) and (a == (0, 1))):
            return 100
        else:
            return 0

    # Function that convert int number of a state to it's coordinates on the grid
    def get_coordinates_by_state(self, s):
        return s % self.size_of_grid, s // self.size_of_grid

    # Function that convert two coordinates of the state to it's int number (from 0 to size_of_grid ** 2 - 1)
    def get_state_by_coordinates(self, s_coord):
        x = s_coord[0]
        y = s_coord[1]
        return y * self.size_of_grid + x

    def get_new_state(self, s, a):
        return self.get_state_by_coordinates((self.get_coordinates_by_state(s)[0] + a[0],
                                              self.get_coordinates_by_state(s)[1] + a[1]))

    # Function that return for state actions available from this state
    def get_possible_actions(self, s):
        # For 3 corner cells only two actions are possible and for cell with cow only 1 action is possible
        if s == 0:
            return [(1, 0), (0, 1)]
        if s == self.size_of_grid - 1:
            return [(-1, 0), (0, 1)]
        if s == self.size_of_grid * (self.size_of_grid - 1):
            return [(1, 0), (0, -1)]
        if s == self.size_of_grid ** 2 - 1:
            return [(0, 0)]

        # For edge cells three actions are possible
        if s % self.size_of_grid == 0:
            return [(1, 0), (0, 1), (0, -1)]
        if (s + 1) % self.size_of_grid == 0:
            return [(0, -1), (0, 1), (-1, 0)]
        if s < self.size_of_grid:
            return [(1, 0), (0, 1), (-1, 0)]
        if s > self.size_of_grid * (self.size_of_grid - 1):
            return [(1, 0), (0, -1), (-1, 0)]

        # For other cells 4 actions are possible
        return [(0, -1), (1, 0), (0, 1), (-1, 0)]


def policy_iteration_algo(grid, discount, eps):
    size_of_grid = grid.size_of_grid

    # Policy is a numpy array, where indexes are number of states and values are actions for each state
    # At first we initialize policy and value function with nulls
    policy = []
    for i in range(size_of_grid ** 2):
        policy.append((0, 0))

    v_func = np.zeros(size_of_grid ** 2)

    # Then we initialize policy with possible random actions
    for s in range(size_of_grid ** 2):
        actions = grid.possible_actions[s][0]
        actions_len = len(actions)
        id = np.random.randint(0, actions_len, size=1)[0]
        policy[s] = actions[id]

    # print("Null value function: ", v_func)
    # print("Random policy: ", policy)
    # print("----------------------")

    number_of_iterations = 0

    while True:
        number_of_iterations += 1

        # Policy evaluation
        while True:
            delta = 0
            for s in range(size_of_grid ** 2):
                v = v_func[s]
                s1 = grid.get_new_state(s, policy[s])
                v_func[s] = grid.get_reward(s, policy[s]) + discount * v_func[s1]
                delta = max(delta, np.abs(v - v_func[s]))
            if delta < eps:
                break

        # Policy improvement
        is_policy_stable = True
        for s in range(size_of_grid ** 2):
            old_action = policy[s]
            possible_actions = grid.possible_actions[s][0]
            policy_val = 0
            for a in possible_actions:
                s1 = grid.get_new_state(s, a)
                if policy_val <= grid.get_reward(s, a) + discount * v_func[s1]:
                    policy[s] = a
                    policy_val = grid.get_reward(s, a) + discount * v_func[s1]
            if old_action != policy[s]:
                is_policy_stable = False
        if is_policy_stable:
            return v_func, policy, number_of_iterations


def value_iteration_algo(grid, discount, eps):
    size_of_grid = grid.size_of_grid

    # Initialize value function with nulls
    v_func = np.zeros(size_of_grid ** 2)

    # At first we initialize policy and value function with nulls
    policy = []
    for i in range(size_of_grid ** 2):
        policy.append((0, 0))

    # Then we initialize policy with possible random actions
    for s in range(size_of_grid ** 2):
        actions = grid.possible_actions[s][0]
        actions_len = len(actions)
        id = np.random.randint(0, actions_len, size=1)[0]
        policy[s] = actions[id]

    number_of_iterations = 0

    while True:
        delta = 0
        number_of_iterations += 1

        for s in range(size_of_grid ** 2):
            v = v_func[s]

            possible_actions = grid.possible_actions[s][0]
            v_func[s] = 0

            for a in possible_actions:
                s1 = grid.get_new_state(s, a)
                if v_func[s] <= grid.get_reward(s, a) + discount * v_func[s1]:
                    v_func[s] = grid.get_reward(s, a) + discount * v_func[s1]
            delta = max(delta, np.abs(v - v_func[s]))

        if delta < eps:
            break
    print("Value function after value_iteration algorithm: ", v_func)

    for s in range(size_of_grid ** 2):
        possible_actions = grid.possible_actions[s][0]
        policy_val = 0
        for a in possible_actions:
            s1 = grid.get_new_state(s, a)
            if policy_val <= grid.get_reward(s, a) + discount * v_func[s1]:
                policy[s] = a
                policy_val = grid.get_reward(s, a) + discount * v_func[s1]

    return policy, number_of_iterations


def main():
    lion_and_cow_grid = LionAndCowGrid(10)
    eps1 = 10e-5
    eps2 = 10e-2
    eps3 = 10
    eps4 = 100
    eps_array = [eps1, eps2, eps3, eps4]
    discount = 0.8

    print("------- Task 2. Policy and value iteration algorithms for Lion And Cow Grid -------")

    for eps in eps_array:
        print("---------------------------------")
        print("For treshhold: ", eps)
        print("---------------------------------")
        print("----- Policy iteration algo -----")
        v_func, policy, number_of_iterations = policy_iteration_algo(lion_and_cow_grid, discount, eps)
        print("Value function after policy_iteration algorithm: ", v_func)
        print("Optimal policy after policy_iteration algorithm: ", policy)
        print(str(number_of_iterations), "iterations are needed to find optimal policy")
        print("---------------------------------")

        print("----- Value iteration algo -----")
        policy_value, number_of_iterations = value_iteration_algo(lion_and_cow_grid, discount, eps)
        print("Optimal policy after value_iteration algorithm: ", policy_value)
        print(str(number_of_iterations), "iterations are needed to find optimal policy")


if __name__ == "__main__":
    main()