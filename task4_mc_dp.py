import numpy as np
from new_grid_modification import GridMod23


def get_policy_result_dp(grid):
    s = grid.grid_desk[0, 0, tuple(sorted(grid.set_of_cows))]
    r = 0
    while not s.is_terminal:
        action = s.policy
        x, y = s.make_step(action)
        print(x, y)

        if (x, y) in grid.set_of_cows:
            new_set_of_cows = tuple(sorted(set(s.set_of_cows).difference([(x, y)])))
            s = grid.grid_desk[x, y, new_set_of_cows]
        else:
            s = grid.grid_desk[x, y, s.set_of_cows]

        r += grid.get_reward(s)

    return r


def get_policy_result_mc(grid):
    s = grid.grid_desk[0, 0, tuple(sorted(grid.set_of_cows))]
    r = 0
    while not s.is_terminal:
        x, y = s.make_step(s.possible_actions[np.argmax(s.policy)])

        if (x, y) in grid.set_of_cows:
            new_set_of_cows = tuple(sorted(set(s.set_of_cows).difference([(x, y)])))
            s = grid.grid_desk[x, y, new_set_of_cows]
        else:
            s = grid.grid_desk[x, y, s.set_of_cows]

        r += grid.get_reward(s)

    return r


# DP policy iteration algorithm implementation
def value_iteration_algo(grid, alpha, eps):
    # Initialization section
    it = 0

    # Main section
    while True:
        it += 1
        if it % 100 == 0:
            print('Iteration', it)
        delta = 0

        for position in grid.grid_desk:
            s = grid.grid_desk[position]
            v = s.value

            for action in s.possible_actions:
                x, y = s.make_step(action)

                if (x, y) in grid.set_of_cows:
                    new_set_of_cows = tuple(sorted(set(s.set_of_cows).difference([(x, y)])))
                    s_new = grid.grid_desk[x, y, new_set_of_cows]
                else:
                    s_new = grid.grid_desk[x, y, s.set_of_cows]

                # Get reward after making a step
                r = grid.get_reward(s_new)
                if s == s_new:
                    r = 0

                v_cur = r + alpha * s_new.value
                if v_cur > s.value:
                    s.value = v_cur
                    s.policy = action

                delta = max(delta, abs(v - s.value))

        if delta < eps:
            break

    return it


# Implementation of MC algorithm
def on_policy_first_visit_mc_control_algo(grid, eps, forever_const):

    for i in range(forever_const):
        s = grid.grid_desk[0, 0, tuple(sorted(grid.set_of_cows))]
        state_action_array = []
        # Generate an episode using policy
        while not s.is_terminal:
            action = s.get_action()
            state_action_array.append((s, action))

            x, y = s.make_step(s.possible_actions[action])

            if (x, y) in grid.set_of_cows:
                new_set_of_cows = tuple(sorted(set(s.set_of_cows).difference([(x, y)])))
                s = grid.grid_desk[x, y, new_set_of_cows]
            else:
                s = grid.grid_desk[x, y, s.set_of_cows]

        # For each (s, a) pair appearing at the episode...
        for pos, (s, action) in enumerate(state_action_array[:-1]):
            if (s, action) not in state_action_array[:i]:
                s.returns[action].append(100 - (len(state_action_array) - pos + 1) * 2)
            if len(s.returns[action]) > 0:
                s.q_func[action] = np.array(s.returns[action]).mean()
            else:
                s.q_func[action] = 0

        s, action = state_action_array[-1]
        s.returns[action].append(100)
        s.q_func[action] = np.array(s.returns[action]).mean()

        # For each state in S...
        for s, _ in state_action_array:
            best_action = np.argmax(s.q_func)
            s.policy = np.zeros_like(s.q_func) + eps / s.number_of_possible_actions
            s.policy[best_action] = 1 - eps + eps / s.number_of_possible_actions


def main():
    # Set parameters of the grid and create grid
    size_of_grid = 3
    cows_array_1 = [(2, 2)]
    cows_array_2 = [(2, 2), (0, 2)]
    cows_array_4 = [(2, 2), (0, 2), (2, 0), (1, 1)]

    grid1 = GridMod23(size_of_grid, cows_array_1, 0)
    grid2 = GridMod23(size_of_grid, cows_array_2, 0)
    grid4 = GridMod23(size_of_grid, cows_array_4, 0)

    alpha = 0.8
    eps = 0.02

    # For one cow on grid
    # Summary reward is 86 because optimal way to the cow and back in the 3x3 grid with current parameters takes 7 steps
    # so summary reward for the optimal way is -2 * 7 + 100 = 86
    it = value_iteration_algo(grid1, alpha, eps)
    reward = get_policy_result_dp(grid1)
    print("Grid 3x3 with one cow")
    print("Value iteration algorithm make " + str(it) + " iterations")
    print("Summary reward: " + str(reward))
    print("---------------------------------------------------")

    # For two cows on grid
    # Summary reward is also 86, because optimal way here is equal to the optimal way on grid with one cow
    it = value_iteration_algo(grid2, alpha, eps)
    reward = get_policy_result_dp(grid2)
    print("Grid 3x3 with two cows")
    print("Value iteration algorithm make " + str(it) + " iterations")
    print("Summary reward: " + str(reward))
    print("---------------------------------------------------")

    # For four cows on grid
    # Summary reward is 82 because optimal way to the cow and back in the 3x3 grid with current parameters takes 9 steps
    # so summary reward for the optimal way is -2 * 9 + 100 = 82
    it = value_iteration_algo(grid4, alpha, eps)
    reward = get_policy_result_dp(grid4)
    print("Grid 3x3 with four cows")
    print("Value iteration algorithm make " + str(it) + " iterations")
    print("Summary reward: " + str(reward))
    print("---------------------------------------------------")


    #MC algo
    grid1 = GridMod23(size_of_grid, cows_array_1, 1)
    grid4 = GridMod23(size_of_grid, cows_array_4, 1)

    on_policy_first_visit_mc_control_algo(grid1, 0.1, 50)
    reward = get_policy_result_mc(grid1)
    print("Grid 3x3 with one cow")
    print("MC algorithm")
    print("Summary reward: " + str(reward))
    print("---------------------------------------------------")

    on_policy_first_visit_mc_control_algo(grid4, 0.1, 50)
    reward = get_policy_result_mc(grid4)
    print("Grid 3x3 with four cows")
    print("MC algorithm")
    print("Summary reward: " + str(reward))
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()