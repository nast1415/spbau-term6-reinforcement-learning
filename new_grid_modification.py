import numpy as np
from itertools import combinations


np.random.seed(0)

'''
    This is file with all lion and cow grid modifications for exercises 4 and 5:
    * Modification 1 (GridCellMod1 and GridMod1 classes) is for SARSA and Q learning algorithms from ex 4
    * Modification 2 (GridCellMod2 and GridMod234 classes) is for DP value iteration algorithm from ex 4
    * Modification 3 (GridCellMod3 and GridMod234 classes) is for on_policy_first_visit_mc_control algorithm from ex 4
    * Modification 4 (GridCellMod4 and GridMod234 classes) is for DYNA-Q and DYNA-Q+ algorithms from ex 5
    * Stochastic modification (GridCellStochastic and GridMod234 classes) is for second part of ex 5

'''


class GridCellMod1:
    def __init__(self, x, y, is_cow_in_cell, grid_size, cows_number):
        self.x = x
        self.y = y
        self.is_cow_in_cell = is_cow_in_cell
        self.grid_size = grid_size
        self.cows_number = cows_number

        self.is_terminal = False

        self.ACTIONS = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, 1), 'DOWN': (0, -1)}

        self.possible_actions = []

        if x < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['RIGHT'])
        if x > 0:
            self.possible_actions.append(self.ACTIONS['LEFT'])
        if y < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['UP'])
        if y > 0:
            self.possible_actions.append(self.ACTIONS['DOWN'])

        self.number_of_possible_actions = len(self.possible_actions)
        self.q_func = np.zeros(self.number_of_possible_actions)

    # Function to make a step according to eps-greedy strategy
    def make_step(self):
        x, y = self.x, self.y
        eps = 0.1

        if len(self.possible_actions) > 1:
            p_array = eps / (self.number_of_possible_actions - 1) * np.ones(self.number_of_possible_actions)
            p_array[np.argmax(self.q_func)] = 1 - eps
        else:
            p_array = np.ones(1)

        action_id = np.random.choice(range(len(self.possible_actions)), p=p_array)
        action = self.possible_actions[action_id]

        x += action[0]
        y += action[1]

        return x, y, action_id

    def set_terminal(self):
        self.is_terminal = True
        self.possible_actions = [(0, 0)]
        self.q_func = [0]


class GridCellMod2:
    def __init__(self, x, y, set_of_cows, grid_size):
        self.x = x
        self.y = y
        self.set_of_cows = set_of_cows
        self.grid_size = grid_size
        self.value = 0

        self.is_terminal = False

        self.ACTIONS = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, 1), 'DOWN': (0, -1)}

        self.possible_actions = []

        if x < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['RIGHT'])
        if x > 0:
            self.possible_actions.append(self.ACTIONS['LEFT'])
        if y < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['UP'])
        if y > 0:
            self.possible_actions.append(self.ACTIONS['DOWN'])

        self.number_of_possible_actions = len(self.possible_actions)
        self.policy = self.possible_actions[0]

    # Function to make a step according to the given action
    def make_step(self, action):
        x, y = self.x, self.y
        x += action[0]
        y += action[1]
        return x, y

    def set_terminal(self):
        self.is_terminal = True
        self.possible_actions = [(0, 0)]
        self.policy = [(0, 0)]
        self.value = 0


class GridCellMod3:
    def __init__(self, x, y, set_of_cows, grid_size):
        self.x = x
        self.y = y
        self.set_of_cows = set_of_cows
        self.grid_size = grid_size

        self.is_terminal = False

        self.ACTIONS = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, 1), 'DOWN': (0, -1)}

        self.possible_actions = []

        if x < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['RIGHT'])
        if x > 0:
            self.possible_actions.append(self.ACTIONS['LEFT'])
        if y < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['UP'])
        if y > 0:
            self.possible_actions.append(self.ACTIONS['DOWN'])

        self.number_of_possible_actions = len(self.possible_actions)
        self.policy = np.ones(self.number_of_possible_actions) / self.number_of_possible_actions
        self.q_func = np.zeros(self.number_of_possible_actions)
        self.returns = [[] for _ in range(self.number_of_possible_actions)]

    # Function to make a step according to the given action
    def make_step(self, action):
        x, y = self.x, self.y
        x += action[0]
        y += action[1]
        return x, y

    def get_action(self):
        id = np.random.choice(np.arange(self.number_of_possible_actions), p=self.policy)
        return id

    def set_terminal(self):
        self.is_terminal = True
        self.possible_actions = [(0, 0)]
        self.policy = [1]
        self.q_func = [0]


class GridCellMod4:
    def __init__(self, x, y, set_of_cows, grid_size):
        self.x = x
        self.y = y
        self.set_of_cows = set_of_cows
        self.q_func = {}
        self.grid_size = grid_size

        self.is_terminal = False
        # Add model dictionary (for each state and action returns next state and reward)
        self.model = {}

        self.ACTIONS = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, 1), 'DOWN': (0, -1)}

        self.possible_actions = []
        self.taken_actions = []

        if x < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['RIGHT'])
            self.model[self.ACTIONS['RIGHT']] = ((x + 1, y, self.set_of_cows), -2)
        if x > 0:
            self.possible_actions.append(self.ACTIONS['LEFT'])
            self.model[self.ACTIONS['LEFT']] = ((x - 1, y, self.set_of_cows), -2)
        if y < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['UP'])
            self.model[self.ACTIONS['UP']] = ((x, y + 1, self.set_of_cows), -2)
        if y > 0:
            self.possible_actions.append(self.ACTIONS['DOWN'])
            self.model[self.ACTIONS['DOWN']] = ((x, y - 1, self.set_of_cows), -2)

        self.number_of_possible_actions = len(self.possible_actions)
        for action in self.possible_actions:
            self.q_func[action] = 0

    # Function to make a step according to the given action
    def make_step(self, action):
        x, y = self.x, self.y
        x += action[0]
        y += action[1]
        return x, y

    def get_action(self):
        eps = 0.1

        max_val = None
        max_arg = 0
        for i in self.q_func:
            if max_val is None:
                max_val = self.q_func[i]
                max_arg = i
            elif max_val < self.q_func[i]:
                max_val = self.q_func[i]
                max_arg = i

        p = np.ones(self.number_of_possible_actions) * eps / (self.number_of_possible_actions - 1)
        q_items = list(self.q_func.items())

        for i, (k, v) in enumerate(q_items):
            if k == max_arg:
                p[i] = 1 - eps

        return q_items[np.random.choice(np.arange(self.number_of_possible_actions), p=p)][0]

    def set_terminal(self):
        self.is_terminal = True
        self.possible_actions = [(0, 0)]
        self.q_func = {(0, 0): 0}
        self.model = {(0, 0): ((self.x, self.y, self.set_of_cows), 0)}

    def __lt__(self, other):
        return False


class GridCellStochastic:
    def __init__(self, x, y, set_of_cows, grid_size):
        self.x = x
        self.y = y
        self.set_of_cows = set_of_cows
        self.q_func = {}
        self.grid_size = grid_size

        self.is_terminal = False
        # Add model dictionary (for each state and action returns next state and reward)
        self.model = {}

        self.ACTIONS = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, 1), 'DOWN': (0, -1)}

        self.possible_actions = []
        self.taken_actions = []

        if x < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['RIGHT'])
            self.model[self.ACTIONS['RIGHT']] = ((x + 1, y, self.set_of_cows), -2)
        if x > 0:
            self.possible_actions.append(self.ACTIONS['LEFT'])
            self.model[self.ACTIONS['LEFT']] = ((x - 1, y, self.set_of_cows), -2)
        if y < grid_size - 1:
            self.possible_actions.append(self.ACTIONS['UP'])
            self.model[self.ACTIONS['UP']] = ((x, y + 1, self.set_of_cows), -2)
        if y > 0:
            self.possible_actions.append(self.ACTIONS['DOWN'])
            self.model[self.ACTIONS['DOWN']] = ((x, y - 1, self.set_of_cows), -2)

        self.number_of_possible_actions = len(self.possible_actions)
        for action in self.possible_actions:
            self.q_func[action] = 0

    # Function to make a step according to the given action (with 70% chance of success)
    def make_step(self, action):
        x, y = self.x, self.y

        is_success = np.random.rand() < 0.7
        if is_success or (action == (1, 0) and x == 0) or (action == (0, 1) and y == 0) or (
                action == (-1, 0) and x == self.grid_size - 1) or (action == (0, -1) and y == self.grid_size - 1):
            x += action[0]
            y += action[1]
        else:
            x -= action[0]
            y -= action[1]

        return x, y

    def get_action(self):
        eps = 0.1

        max_val = None
        max_arg = 0
        for i in self.q_func:
            if max_val is None:
                max_val = self.q_func[i]
                max_arg = i
            elif max_val < self.q_func[i]:
                max_val = self.q_func[i]
                max_arg = i

        p = np.ones(self.number_of_possible_actions) * eps / (self.number_of_possible_actions - 1)
        q_items = list(self.q_func.items())

        for i, (k, v) in enumerate(q_items):
            if k == max_arg:
                p[i] = 1 - eps

        return q_items[np.random.choice(np.arange(self.number_of_possible_actions), p=p)][0]

    def set_terminal(self):
        self.is_terminal = True
        self.possible_actions = [(0, 0)]
        self.q_func = {(0, 0): 0}
        self.model = {(0, 0): ((self.x, self.y, self.set_of_cows), 0)}

    def __lt__(self, other):
        return False


class GridMod1:
    def __init__(self, size_of_grid, cows_array):
        self.size_of_grid = size_of_grid
        self.cows_array = cows_array
        self.cows_number = len(cows_array)

        self.grid_desk = {}
        for i in range(size_of_grid):
            for j in range(size_of_grid):
                self.grid_desk[i, j, 0] = GridCellMod1(i, j, False, self.size_of_grid, 0)

        for cow_position in cows_array:
            self.grid_desk[cow_position[0], cow_position[1], 0] = GridCellMod1(cow_position[0], cow_position[1], True,
                                                                               self.size_of_grid, 0)

        self.grid_desk[0, 0, self.cows_number] = \
            GridCellMod1(0, 0, False, self.size_of_grid, self.cows_number)
        self.grid_desk[0, 0, self.cows_number].set_terminal()

    def get_reward(self, cell):
        x = cell.x
        y = cell.y
        cur_cows_number = cell.cows_number
        if (x == 0) and (y == 0) and (cur_cows_number == self.cows_number):
            return 100 * cur_cows_number
        else:
            return -2


class GridMod234:
    def __init__(self, size_of_grid, set_of_cows, algo_id):
        self.size_of_grid = size_of_grid
        self.set_of_cows = set_of_cows
        self.cows_number = len(self.set_of_cows)

        # Generate all subsets of set_of_cows
        all_cows_subsets = []
        for i in range(self.cows_number + 1):
            cur_combinations = []
            for comb in combinations(self.set_of_cows, i):
                cur_combinations += [list(comb)]
            all_cows_subsets += sorted(cur_combinations)

        self.all_cows_subsets = all_cows_subsets

        # Now third parameter of grid desk is set of cows which lion hasn't taken yet
        # (at the beginning it is set of all cows on grid, at the end it is empty set)
        self.grid_desk = {}
        for i in range(size_of_grid):
            for j in range(size_of_grid):
                for k in self.all_cows_subsets:
                    if algo_id == 0:
                        self.grid_desk[i, j, tuple(sorted(k))] = GridCellMod2(i, j, tuple(sorted(k)), self.size_of_grid)
                    elif algo_id == 1:
                        self.grid_desk[i, j, tuple(sorted(k))] = GridCellMod3(i, j, tuple(sorted(k)), self.size_of_grid)
                    elif algo_id == 2:
                        self.grid_desk[i, j, tuple(sorted(k))] = GridCellMod4(i, j, tuple(sorted(k)), self.size_of_grid)
                    else:
                        self.grid_desk[i, j, tuple(sorted(k))] = \
                            GridCellStochastic(i, j, tuple(sorted(k)), self.size_of_grid)
        # Lion need to return at the (0, 0) position without any cows left on the grid
        self.grid_desk[0, 0, ()].set_terminal()

    def get_reward(self, cell):
        x = cell.x
        y = cell.y
        set_of_cows = cell.set_of_cows
        if (x == 0) and (y == 0) and (len(set_of_cows) == 0):
            return 100
        else:
            return -2


def main():
    size_of_grid = 3
    cows_array = [(2, 2)]

    grid = GridMod1(size_of_grid, cows_array)
    print(grid.grid_desk[2, 2, 0].is_cow_in_cell)

    cell = GridCellMod1(2, 1, False, 3, 2)
    i, j, ind = cell.make_step()
    print(i, j)


if __name__ == "__main__":
    main()
