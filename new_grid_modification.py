import numpy as np


class GridCell:
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
        self.value = np.zeros(self.number_of_possible_actions)

    def __str__(self):
        print("(", self.x, ',', self.y, ")")

    # Function to make a step according to eps-greedy strategy
    def make_step(self):
        x, y = self.x, self.y
        eps = 0.1

        if len(self.possible_actions) > 1:
            p_array = eps / (self.number_of_possible_actions - 1) * np.ones(self.number_of_possible_actions)
            p_array[np.argmax(self.value)] = 1 - eps
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
        self.value = [0]


class Grid:
    def __init__(self, size_of_grid, cows_array):
        self.size_of_grid = size_of_grid
        self.cows_array = cows_array
        self.cows_number = len(cows_array)

        self.grid_desk = {}
        for i in range(size_of_grid):
            for j in range(size_of_grid):
                self.grid_desk[i, j, 0] = GridCell(i, j, False, self.size_of_grid, 0)

        for cow_position in cows_array:
            self.grid_desk[cow_position[0], cow_position[1], 0] = GridCell(cow_position[0], cow_position[1], True,
                                                           self.size_of_grid, 0)

        self.grid_desk[0, 0, self.cows_number] = \
            GridCell(0, 0, False, self.size_of_grid, self.cows_number)
        self.grid_desk[0, 0, self.cows_number].set_terminal()

    def get_reward(self, cell):
        x = cell.x
        y = cell.y
        cur_cows_number = cell.cows_number
        if (x == 0) and (y == 0) and (cur_cows_number == self.cows_number):
            return 100 * cur_cows_number
        else:
            return -2



def main():
    size_of_grid = 3
    cows_array = [(2, 2)]

    grid = Grid(size_of_grid, cows_array)
    print(grid.grid_desk[2, 2, 0].is_cow_in_cell)

    cell = GridCell(2, 1, False, 3, 2)
    i, j, ind = cell.make_step()
    print(i, j)


if __name__ == "__main__":
    main()







