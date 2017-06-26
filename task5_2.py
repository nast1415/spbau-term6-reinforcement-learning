import numpy as np
from queue import PriorityQueue
from new_grid_modification import GridMod234


#np.random.seed(0)


def get_policy_result_dyna(grid):
    s = grid.grid_desk[0, 0, tuple(sorted(grid.set_of_cows))]
    r = 0
    while not s.is_terminal:
        max_val = None
        max_arg = 0
        for i in s.q_func:
            if max_val is None:
                max_val = s.q_func[i]
                max_arg = i
            elif max_val < s.q_func[i]:
                max_val = s.q_func[i]
                max_arg = i

        x, y = s.make_step(max_arg)
        print(x, y)

        if (x, y) in grid.set_of_cows:
            new_set_of_cows = tuple(sorted(set(s.set_of_cows).difference([(x, y)])))
            s = grid.grid_desk[x, y, new_set_of_cows]
        else:
            s = grid.grid_desk[x, y, s.set_of_cows]

        r += grid.get_reward(s)

    return r


# DYNA-Q algorithm implementation
def dyna_q_algo(grid, alpha, gamma, forever_const):
    observed_states_array = []
    all_states = list(grid.grid_desk.keys())
    n = 50

    for i in range(forever_const):
        # Select current nonterminal state
        while True:
            state_id = np.random.choice(range(len(all_states)))
            s = grid.grid_desk[all_states[state_id]]
            if not s.is_terminal:
                break

        # Add current state to an array of observed states if it is not in this array
        if not s in observed_states_array:
            observed_states_array.append(s)

        # Choose action according to eps-greedy strategy
        action = s.get_action()
        x, y = s.make_step(action)

        # Add action to the taken actions array if it is not in this array for this state
        if not action in s.taken_actions:
            s.taken_actions.append(action)

        # Check if the cow in the current cell, than we should take it
        if (x, y) in grid.set_of_cows:
            new_set_of_cows = tuple(sorted(set(s.set_of_cows).difference([(x, y)])))
            s_new = grid.grid_desk[x, y, new_set_of_cows]
        else:
            s_new = grid.grid_desk[x, y, s.set_of_cows]

        # Get reward after making a step
        r = grid.get_reward(s_new)
        if s == s_new:
            r = 0

        # Find maximal value of q_function in state s_new
        max_q_value = None
        for j in s_new.q_func:
            if max_q_value is None:
                max_q_value = s_new.q_func[j]
            else:
                max_q_value = max(max_q_value, s_new.q_func[j])

        # Recalculate q function and model
        s.q_func[action] += alpha * (r + gamma * max_q_value - s.q_func[action])
        s.model[action] = ((s_new.x, s_new.y, s_new.set_of_cows), r)

        for j in range(n):
            # Random previously observed state
            state_id = np.random.choice(np.arange(len(observed_states_array)))
            s = observed_states_array[state_id]

            # Random previously taken action in s
            action_id = np.random.choice(np.arange(len(s.taken_actions)))
            action = s.taken_actions[action_id]

            # Get info from model
            (x, y, set_of_cows), r = s.model[action]
            s_new = grid.grid_desk[x, y, set_of_cows]

            # Find maximal value of q_function in state s_new
            max_q_value = None
            for j in s_new.q_func:
                if max_q_value is None:
                    max_q_value = s_new.q_func[j]
                else:
                    max_q_value = max(max_q_value, s_new.q_func[j])

            # Recalculate q function
            s.q_func[action] += alpha * (r + gamma * max_q_value - s.q_func[action])


# DYNA-Q+ with prioritized sweeping algorithm implementation
def dyna_q_plus_algo(grid, alpha, gamma, theta, forever_const):
    # Initialization section
    all_states = list(grid.grid_desk.keys())
    pqueue = PriorityQueue()
    n = 50

    # Repeat forever
    for i in range(forever_const):
        # Select current nonterminal state
        while True:
            state_id = np.random.choice(range(len(all_states)))
            s = grid.grid_desk[all_states[state_id]]
            if not s.is_terminal:
                break

        # Choose action according to policy
        max_val = None
        action = 0
        for j in s.q_func:
            if max_val is None:
                max_val = s.q_func[j]
                action = j
            elif max_val < s.q_func[j]:
                max_val = s.q_func[j]
                action = j

        x, y = s.make_step(action)

        # Check if the cow in the current cell, than we should take it
        if (x, y) in grid.set_of_cows:
            new_set_of_cows = tuple(sorted(set(s.set_of_cows).difference([(x, y)])))
            s_new = grid.grid_desk[x, y, new_set_of_cows]
        else:
            s_new = grid.grid_desk[x, y, s.set_of_cows]

        # Get reward after making a step
        r = grid.get_reward(s_new)
        if s == s_new:
            r = 0

        # Recalculate model function
        s.model[action] = ((s_new.x, s_new.y, s_new.set_of_cows), r)

        # Find maximal value of q_function in state s_new
        max_q_value = None
        for j in s_new.q_func:
            if max_q_value is None:
                max_q_value = s_new.q_func[j]
            else:
                max_q_value = max(max_q_value, s_new.q_func[j])

        # Set p value and add state and action to the PQueue if p > theta
        p = abs(r + gamma * max_q_value - s.q_func[action])
        if p > theta:
            pqueue.put((-p, (s, action)))

        for _ in range(n):
            # Repeat while PQueue is not empty
            if pqueue.empty():
                break

            p_val, (s, action) = pqueue.get()
            (x, y, set_of_cows), r = s.model[action]
            s_new = grid.grid_desk[x, y, set_of_cows]

            # Find maximal value of q_function in state s_new
            max_q_value = None
            for j in s_new.q_func:
                if max_q_value is None:
                    max_q_value = s_new.q_func[j]
                else:
                    max_q_value = max(max_q_value, s_new.q_func[j])

            # Recalculate q function for state s
            s.q_func[action] += alpha * (r + gamma * max_q_value - s.q_func[action])

            # Build set of states and actions predicted to lead to s
            pred_set = []
            for cell in grid.grid_desk:
                model = grid.grid_desk[cell].model
                for action in model:
                    (x, y, set_of_cows), r = model[action]
                    if s.x == x and s.y == y and s.set_of_cows == set_of_cows:
                        pred_set.append((grid.grid_desk[cell], action, r))

            for s_new, action_new, r_new in pred_set:
                # Find maximal value of q_function in state s_new
                max_q_value = None
                for k in s.q_func:
                    if max_q_value is None:
                        max_q_value = s.q_func[k]
                    else:
                        max_q_value = max(max_q_value, s.q_func[k])

                p = abs(r_new + gamma * max_q_value - s_new.q_func[action_new])
                if p > theta:
                    pqueue.put((-p, (s_new, action_new)))


def main():
    # Set parameters of the grid and create grid
    size_of_grid = 10
    cows_array_2 = [(9, 9)]

    grid2_stochastic = GridMod234(size_of_grid, cows_array_2, 3)

    alpha = 0.8
    gamma = 1
    theta = 0.02

    dyna_q_algo(grid2_stochastic, alpha, gamma, 3000)
    reward = get_policy_result_dyna(grid2_stochastic)
    print("Grid 10x10 with two cows")
    print("DYNA-Q algorithm")
    print("Summary reward: " + str(reward))
    print("---------------------------------------------------")

    dyna_q_plus_algo(grid2_stochastic, alpha, gamma, theta, 2000)
    print("start testing")
    reward = get_policy_result_dyna(grid2_stochastic)
    print("Grid 10x10 with two cows")
    print("DYNA-Q+ with prioritized sweeping algorithm")
    print("Summary reward: " + str(reward))
    print("---------------------------------------------------")

if __name__ == "__main__":
    main()