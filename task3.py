import matplotlib.pyplot as plt
import random


# Class for the black jack game
class BlackJack:
    RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
    VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 10, 'Q': 10,
              'K': 10}

    def __init__(self):
        self.dealer = []
        self.player = []
        self.is_game_run = False
        self.game_result = 0

    def begin_game(self):
        self.is_game_run = True
        self.give_card_to_player(self.dealer)
        self.player = [4, 5]

        if self.get_score(self.player) == 21:
            self.game_result = 1
            self.is_game_run = False
        if self.get_score(self.dealer) == 21:
            self.game_result = 2
            self.is_game_run = False
        #self.print_current_state()

        return self.is_game_run, self.game_result

    # Add one card to the player's or dealer's array (number of the card in the RANK list)
    def give_card_to_player(self, player):
        player.append(random.randrange(0, 13))

    # Return current score of the player according to his cards
    def get_score(self, player):
        score = 0
        for card in player:
            score += self.VALUES[self.RANKS[card]]
        return score

    # Function for hit move: player get another card and there are three situations: he win, he lopse or game continue
    def hit(self):
        self.give_card_to_player(self.player)
        #self.print_current_state()
        if self.get_score(self.player) == 21:
            self.game_result = 1
            self.is_game_run = False
        elif self.get_score(self.player) > 21:
            self.game_result = 2
            self.is_game_run = False

    # Function for stick move: dealer get cards until his score is smaller than player's score.
    # Then there are three situations: dealer get 21 and player loose, dealer get more that 21 and player win or they
    # have the same score, so it is a tie
    def stick(self):
        while self.get_score(self.dealer) < self.get_score(self.player):
            self.give_card_to_player(self.dealer)
        #self.print_current_state()

        if self.get_score(self.dealer) > 21:
            self.game_result = 1
            self.is_game_run = False
        elif self.get_score(self.dealer) == self.get_score(self.player):
            self.game_result = 3
            self.is_game_run = False
        else:
            self.game_result = 2
            self.is_game_run = False

    # Function that print dealer's and player's cards
    def print_current_state(self):
        print("Dealer cards: ", self.dealer)
        print("Player cards: ", self.player)

# Special functions for the on-policy first-visit MC control algorithm


#  This function return one of two actions: 'hit' or 'stick' according to the given policy and state
# (state is a pair of dealer's and player's scores)
def get_action_by_policy(policy, dealer_score, player_score):
    p1 = policy[(player_score, dealer_score, "hit")]
    p2 = policy[(player_score, dealer_score, "stick")]

    if p1 >= p2:
        return "hit"
    else:
        return "stick"


# This function return average result after 'num_of_episodes' episodes of the game according to given policy
def get_average_policy_result(policy, num_of_episodes):
    win_score = 0
    for i in range(num_of_episodes):
        blackjack = BlackJack()
        blackjack.begin_game()

        while blackjack.is_game_run:
            while blackjack.get_score(blackjack.player) < 11:
                blackjack.hit()

            if not blackjack.is_game_run:
                break

            next_move = get_action_by_policy(policy, blackjack.get_score(blackjack.dealer),
                                             blackjack.get_score(blackjack.player))
            if next_move == "hit":
                blackjack.hit()
            else:
                blackjack.stick()
        if blackjack.game_result == 1:
            win_score += 1
        elif blackjack.game_result == 2:
            win_score -= 1

    return win_score / num_of_episodes


# In next algorithm we have states as (player_score, dealer_score) pairs and two actions: 'hit' or 'stick'
# Function for on-policy mc algorithm
def on_policy_first_visit_mc_control_algo(eps, forever_const):
    # Initialization section
    q_func = {}
    returns = {}
    policy = {}

    # Array with forever_const / 1000 elements with average win_results for current policy (to draw a plot)
    results_for_plot = []
    coordinates_for_plot = []

    # Probability of the actions we choose randomly
    hit_probability = 1 / 2

    for i in range(11, 21, 1):
        for j in range(1, 21, 1):
            q_func[(i, j, "hit")] = 0
            q_func[(i, j, "stick")] = 0
            returns[(i, j, "hit")] = []
            returns[(i, j, "stick")] = []
            policy[(i, j, "hit")] = hit_probability
            policy[(i, j, "stick")] = 1 - hit_probability

    # Main section of the algorithm
    for i in range(forever_const):
        state_action_set = set()
        state_set = set()

        blackjack = BlackJack()
        blackjack.begin_game()

        while blackjack.is_game_run:
            while blackjack.get_score(blackjack.player) < 11:
                blackjack.hit()

            if not blackjack.is_game_run:
                break

            dealer_score = blackjack.get_score(blackjack.dealer)
            player_score = blackjack.get_score(blackjack.player)

            next_move = get_action_by_policy(policy, blackjack.get_score(blackjack.dealer), blackjack.get_score(blackjack.player))

            state_action_set.add((player_score, dealer_score, next_move))
            state_set.add((player_score, dealer_score))

            if next_move == "hit":
                blackjack.hit()
            else:
                blackjack.stick()

            for sa_pair in state_action_set:
                if blackjack.game_result == 1:
                    returns[sa_pair].append(1)
                elif blackjack.game_result == 2:
                    returns[sa_pair].append(-1)
                else:
                    returns[sa_pair].append(0)
                q_func[sa_pair] = sum(returns[sa_pair]) * 1.0 / len(returns[sa_pair])

            for s in state_set:
                if q_func[s[0], s[1], "hit"] > q_func[s[0], s[1], "stick"]:
                    right_action = "hit"
                    wrong_action = "stick"
                else:
                    right_action = "stick"
                    wrong_action = "hit"

                policy[(s[0], s[1], right_action)] = 1 - eps + eps / 2
                policy[(s[0], s[1], wrong_action)] = eps / 2

        if i % 1000 == 0:
            # print(i)
            results_for_plot.append(get_average_policy_result(policy, 100))
            coordinates_for_plot.append(i)

    return policy, results_for_plot, coordinates_for_plot


def draw_plot(results_for_plot, coordinates_for_plot):
    plt.plot(coordinates_for_plot, results_for_plot)
    plt.show()


def main():
    policy, results_for_plot, coordinates_for_plot = on_policy_first_visit_mc_control_algo(0.1, 100000)
    for i in range(11, 21):
        for j in range(1, 11, 1):
            for k in ["hit", "stick"]:
                print(i, j, k, ":", policy[(i, j, k)])

    draw_plot(results_for_plot, coordinates_for_plot)


if __name__ == "__main__":
    main()
