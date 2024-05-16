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
    return (x - 1) * 11 + (y - 1)

def stochastic_action(action, stochastic=False):
    if stochastic and random.random() < 0.2:  # 20% chance to deviate
        return random.choice([a for a in range(4) if a != action])
    return action

def main(stochastic=False):
    epsilon = 0.1  # Exploration rate
    alpha = 0.5    # Learning rate
    gamma = 0.9    # Discount factor
    episodes = 500  # Number of training episodes

    fourRoomsObj = FourRooms(scenario='simple', stochastic=stochastic)
    q_table = initialize_q_table(121, 4)  # 121 states (11x11 grid), 4 actions

    for episode in range(episodes):
        fourRoomsObj.newEpoch()
        state = get_state_index(*fourRoomsObj.getPosition())
        terminal = False

        while not terminal:
            action = choose_action(state, q_table, epsilon)
            action = stochastic_action(action, stochastic)  # Apply stochastic effect based on the flag
            _, new_pos, packages_remaining, terminal = fourRoomsObj.takeAction(action)
            reward = 10 if packages_remaining == 0 else -1  # Reward for collecting a package, penalty for moving
            new_state = get_state_index(*new_pos)
            update_q_table(q_table, state, action, reward, new_state, alpha, gamma)
            state = new_state

        # Decrease epsilon or adjust alpha if necessary to improve learning over time

    # Displaying the final learned path
    fourRoomsObj.showPath(-1,"Scenario_1.png")

if __name__ == "__main__":
    # Check if a stochastic flag is provided in command-line arguments
    stochastic = '-stochastic' in sys.argv
    main(stochastic)
