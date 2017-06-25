import matplotlib.pyplot as plt
from new_grid_modification import GridMod1


# SARSA algorithm implementation
def sarsa_algorithm(grid, number_of_episodes, alpha, gamma):
    # Initialization section
    reward_array = []

    # Main section
    for i in range(number_of_episodes):
        if i % 100 == 0:
            print('Episode', i)
        # Start at the left bottom corner of the grid desk
        s = grid.grid_desk[0, 0, 0]

        x, y, action_id = s.make_step()

        reward_during_episode = 0

        while True:
            cows_number = s.cows_number

            # After making a step, we can get into the situation when we're first time at this cell with this number of cows.
            # In this situation we must check if it is necessary to change 'is_cow_in_cell' flag.
            # It is necessary if we took the cow from this cell some steps ago.
            # If it is true, for some x, y, cow1 this flag was set to 'False' (after taking cow we set this flag
            # to 'False' for the grid cell with same coordinates and cows_number = current_cows_number + 1)
            if (x, y, cows_number) not in grid.grid_desk.keys():
                is_cow_flag = True
                for x1, y1, cow1 in grid.grid_desk:
                    if x1 == x and y1 == y:
                        is_cow_flag &= grid.grid_desk[x, y, cow1].is_cow_in_cell
                grid.grid_desk[x, y, cows_number] = GridCellMod1(x, y, is_cow_flag, grid.size_of_grid, cows_number)

            # If after the step we're in the cell with cow, we will take this cow and set 'is_cow_in_cell' flag to False
            # for the grid cell with this coordinates and number_of_cows = current_number_of_cows + 1
            # (to prevent the situation when we take this cow another time just after we take it)
            if grid.grid_desk[x, y, cows_number].is_cow_in_cell:
                if (x, y, cows_number + 1) not in grid.grid_desk.keys():
                    grid.grid_desk[x, y, cows_number + 1] = GridCellMod1(x, y, False, grid.size_of_grid, cows_number + 1)
                s_new = grid.grid_desk[x, y, cows_number + 1]
            else:
                s_new = grid.grid_desk[x, y, cows_number]

            # Get reward after making a step
            r = grid.get_reward(s_new)

            # Re-calculate q value
            x_next, y_next, action_id_next = s_new.make_step()
            s.q_func[action_id] += alpha * (r + gamma * s_new.q_func[action_id_next] - s.q_func[action_id])

            # Add current reward (for plot)
            reward_during_episode += r

            # At the terminal cell we add reward and number of iterations during the episode to the corresponding arrays
            # and end the episode
            if s_new.is_terminal:
                reward_array.append(reward_during_episode)
                break

            # If cell is not terminal - we update values and repeat
            s = s_new
            x, y, action_id = x_next, y_next, action_id_next

    return reward_array


# Q learning algorithm implementation
def q_learning_algorithm(grid, number_of_episodes, alpha, gamma):
    # Initialization section
    reward_array = []

    # Main section
    for i in range(number_of_episodes):
        if i % 100 == 0:
            print('Episode', i)
        # Start at the left bottom corner of the grid desk
        s = grid.grid_desk[0, 0, 0]

        x, y, action_id = s.make_step()

        reward_during_episode = 0

        while True:
            cows_number = s.cows_number

            # After making a step, we can get into the situation when we're first time at this cell with this number of cows.
            # In this situation we must check if it is necessary to change 'is_cow_in_cell' flag.
            # It is necessary if we took the cow from this cell some steps ago.
            # If it is true, for some x, y, cow1 this flag was set to 'False' (after taking cow we set this flag
            # to 'False' for the grid cell with same coordinates and cows_number = current_cows_number + 1)
            if (x, y, cows_number) not in grid.grid_desk.keys():
                is_cow_flag = True
                for x1, y1, cow1 in grid.grid_desk:
                    if x1 == x and y1 == y:
                        is_cow_flag &= grid.grid_desk[x, y, cow1].is_cow_in_cell
                grid.grid_desk[x, y, cows_number] = GridCellMod1(x, y, is_cow_flag, grid.size_of_grid, cows_number)

            # If after the step we're in the cell with cow, we will take this cow and set 'is_cow_in_cell' flag to False
            # for the grid cell with this coordinates and number_of_cows = current_number_of_cows + 1
            # (to prevent the situation when we take this cow another time just after we take it)
            if grid.grid_desk[x, y, cows_number].is_cow_in_cell:
                if (x, y, cows_number + 1) not in grid.grid_desk.keys():
                    grid.grid_desk[x, y, cows_number + 1] = GridCellMod1(x, y, False, grid.size_of_grid, cows_number + 1)
                s_new = grid.grid_desk[x, y, cows_number + 1]
            else:
                s_new = grid.grid_desk[x, y, cows_number]

            # Get reward after making a step
            r = grid.get_reward(s_new)

            s.q_func[action_id] += alpha * (r + gamma * max(s_new.q_func) - s.q_func[action_id])

            # Add current reward (for plot)
            reward_during_episode += r

            # At the terminal cell we add reward and number of iterations during the episode to the corresponding arrays
            # and end the episode
            if s_new.is_terminal:
                reward_array.append(reward_during_episode)
                break

            # If cell is not terminal - we update values and repeat
            s = s_new
            x, y, action_id = s.make_step()
    return reward_array


def main():
    # Set parameters of the grid and create grid
    size_of_grid = 3
    cows_array_1 = [(2, 2)]
    cows_array_2 = [(2, 2), (0, 2)]
    cows_array_3 = [(2, 2), (0, 2), (2, 0)]

    grid1 = GridMod1(size_of_grid, cows_array_1)
    grid2 = GridMod1(size_of_grid, cows_array_2)
    grid3 = GridMod1(size_of_grid, cows_array_3)

    # Set parameters for the q learning and SARSA algorithms
    alpha = 1
    gamma = 0.8
    number_of_episodes = 100

    # Results for 1 cow
    reward_sarsa = sarsa_algorithm(grid1, number_of_episodes, alpha, gamma)
    reward_q = q_learning_algorithm(grid1, number_of_episodes, alpha, gamma)

    lbl_sarsa, = plt.plot(range(number_of_episodes), reward_sarsa, label="sarsa")
    lbl_q, = plt.plot(range(number_of_episodes), reward_q, label="q learnng")
    plt.legend(handles=[lbl_sarsa, lbl_q])
    plt.title("SARSA and Q learning on 3x3 grid with 1 cow")
    plt.xlabel('number of episodes')
    plt.ylabel('reward during episode')
    plt.savefig("q_sarsa_reward_plot_1cow")
    plt.show()

    # Results for 2 cows
    reward_sarsa = sarsa_algorithm(grid2, number_of_episodes, alpha, gamma)
    reward_q = q_learning_algorithm(grid2, number_of_episodes, alpha, gamma)

    lbl_sarsa, = plt.plot(range(number_of_episodes), reward_sarsa, label="sarsa")
    lbl_q, = plt.plot(range(number_of_episodes), reward_q, label="q learnng")
    plt.legend(handles=[lbl_sarsa, lbl_q])
    plt.title("SARSA and Q learning on 3x3 grid with 2 cows")
    plt.xlabel('number of episodes')
    plt.ylabel('reward during episode')
    plt.savefig("q_sarsa_reward_plot_2cows")
    plt.show()

    # Results for 3 cows
    reward_sarsa = sarsa_algorithm(grid3, number_of_episodes, alpha, gamma)
    reward_q = q_learning_algorithm(grid3, number_of_episodes, alpha, gamma)

    lbl_sarsa, = plt.plot(range(number_of_episodes), reward_sarsa, label="sarsa")
    lbl_q, = plt.plot(range(number_of_episodes), reward_q, label="q learnng")
    plt.legend(handles=[lbl_sarsa, lbl_q])
    plt.title("SARSA and Q learning on 3x3 grid with 3 cows")
    plt.xlabel('number of episodes')
    plt.ylabel('reward during episode')
    plt.savefig("q_sarsa_reward_plot_3cows")
    plt.show()


if __name__ == "__main__":
    main()