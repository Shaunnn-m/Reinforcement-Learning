import numpy as np
import random
import sys
from FourRooms import FourRooms

def initialize_q_table(states, actions):
    return np.zeros((states, actions))

def choose_action(state, q_table, epsilon):
    return random.choice([FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]) if random.uniform(0, 1) < epsilon \
        else np.argmax(q_table[state])

def update_q_table(q_table, state, action, reward, new_state, alpha, gamma):
    best_next_action = np.argmax(q_table[new_state])
    q_table[state][action] += alpha * (reward + gamma * q_table[new_state][best_next_action] - q_table[state][action])

def get_state_index(x, y):
    return (x - 1) * 11 + (y - 1)  # Adjusted for 1-based index

def stochastic_action(action, stochastic=False):
    if stochastic and random.random() < 0.2:  # 20% chance to deviate
        return random.choice([a for a in range(4) if a != action])
    return action

def calculate_reward(grid_cell, collected_packages):
    if grid_cell == FourRooms.RED and FourRooms.RED not in collected_packages:
        return 0  # Reward for collecting the first package (Red)
    elif grid_cell == FourRooms.GREEN and FourRooms.RED in collected_packages and FourRooms.GREEN not in collected_packages:
        return 0  # Reward for collecting the second package (Green)
    elif grid_cell == FourRooms.BLUE and FourRooms.GREEN in collected_packages and FourRooms.BLUE not in collected_packages:
        return 100  # Reward for collecting the third package (Blue)
    elif grid_cell > 0:
        return -10  # Penalty for collecting any package out of order
    return -1  # Small penalty for any movement that doesn't result in correct package collection

def main(stochastic=False):
    epsilon = 0.1  # Exploration rate
    alpha = 0.5    # Learning rate
    gamma = 0.9    # Discount factor
    episodes = 20000  # Number of training episodes
    max_steps_per_episode = 1000  # Maximum steps in each episode

    fourRoomsObj = FourRooms(scenario='rgb', stochastic=stochastic)  # Pass the stochastic parameter
    q_table = initialize_q_table(121, 4)  # 121 states (11x11 grid), 4 actions

    for episode in range(episodes):
        fourRoomsObj.newEpoch()
        state = get_state_index(*fourRoomsObj.getPosition())
        terminal = False
        steps = 0
        collected_packages = []

        while not terminal and steps < max_steps_per_episode:
            action = choose_action(state, q_table, epsilon)
            action = stochastic_action(action, stochastic)  # Apply stochastic effect based on the flag
            grid_cell, new_pos, packages_remaining, terminal = fourRoomsObj.takeAction(action)
            new_state = get_state_index(*new_pos)

            if (grid_cell == FourRooms.GREEN and FourRooms.RED not in collected_packages) or \
                    (grid_cell == FourRooms.BLUE and (FourRooms.RED not in collected_packages or FourRooms.GREEN not in collected_packages)):
                terminal = True
                reward = -100  # Terminate with a heavy penalty for picking the wrong package
                state = get_state_index(*fourRoomsObj.getPosition())  # Reset position
                collected_packages = []  # Reset collected packages
                q_table = initialize_q_table(121, 4)  # Reset Q-table
            else:
                reward = calculate_reward(grid_cell, collected_packages)
                if grid_cell > 0 and grid_cell <= 3:
                    collected_packages.append(grid_cell)

            update_q_table(q_table, state, action, reward, new_state, alpha, gamma)
            state = new_state
            steps += 1

    fourRoomsObj.showPath(-1,"Scenario_3.png")

if __name__ == "__main__":
    # Check if a stochastic flag is provided in command-line arguments
    stochastic = '-stochastic' in sys.argv
    main(stochastic)
