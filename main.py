import os
import sys

import pygame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from environment import Environment
from constants import STATE_SIZE, ACTION_SPACE
from agent import DQNAgent

def main(mode="human"):
    if mode == "human":
        env = Environment(train=False)
        state = env.reset()
        done = False
        while not done:
            action = "none"  # Default action if no key is pressed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        action = 0  # Move drone right
                    elif event.key == pygame.K_LEFT:
                        action = 1  # Move drone left
                    elif event.key == pygame.K_DOWN:
                        action = 2  # Drop grenade
            # Perform the action in the environment
            state, reward, done, _ = env.step(action)
            print(state)
            # Render the environment if renderMode is enabled
            env.render()
            print(f"Reward: {reward}\n")
    elif mode == "train":
        env = Environment()
        episodes = 8000
        agent = DQNAgent(STATE_SIZE, len(ACTION_SPACE))

        save_dir = "brains"
        os.makedirs(save_dir, exist_ok=True)
        plot_data = []

        for episode in range(0, episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                env.render()
                pygame.event.get()  # Important: Keep this to prevent freezing

                agent.remember(state, next_state, action, reward, done)  # Store experience in replay buffer
                agent.train()  # Train the agent (using a minibatch from the replay buffer)

                state = next_state
                total_reward += reward

            if (episode + 1) % 10 == 0:
                agent.decay_epsilon()

            plot_data.append((episode, total_reward))
            print(f"Episode {episode+1}/{episodes} Total Reward: {total_reward} Epsilon: {agent.epsilon:.2f} Steps: {env.steps}\n")

        df = pd.DataFrame(plot_data)
        df.to_csv("plot_data.csv", index=False, header=False)

        # Save the trained model (optional, but highly recommended)
        torch.save(agent.q_network.state_dict(), os.path.join(save_dir, "trained_model.pth"))
    elif mode == "test":
        model_path = os.path.join("brains", "trained_model.pth")
        env = Environment(train=False)
        episodes = 50
        agent = DQNAgent(STATE_SIZE, len(ACTION_SPACE))
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.target_q_network.load_state_dict(torch.load(model_path))

        agent.q_network.eval()  # Set the Q-network to eval mode
        agent.target_q_network.eval() # Set the target Q-network to eval mode as well


        for episode in range(0, episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.act(state, greedy=True)
                next_state, reward, done, _ = env.step(action)
                env.render()
                pygame.event.get()
                state = next_state
                total_reward += reward
            print(f"Episode {episode+1}/{episodes} Total Reward: {total_reward} Steps: {env.steps}\n")

    elif mode == "plot":
        # Read data from the CSV file
        data = np.genfromtxt('plot_data.csv', delimiter=',')
        # Extract x and y values
        x = data[:, 0]
        y = data[:, 1]
        # Ensure the length of y is divisible by 1000
        num_entries = len(y)
        trimmed_length = (num_entries // 1000) * 1000  # Trim to the nearest multiple of 1000
        y_trimmed = y[:trimmed_length]  # Trim the y array
        x_trimmed = x[:trimmed_length]  # Trim the x array

        # Reshape y_trimmed into groups of 1000
        y_reshaped = y_trimmed.reshape(-1, 1000)  # Reshape into (num_groups, 1000)
        x_reshaped = x_trimmed.reshape(-1, 1000)  # Reshape into (num_groups, 1000)

        # Compute the mean for each group of 1000 episodes
        y_averaged = np.mean(y_reshaped, axis=1)  # Shape: (num_groups,)
        x_averaged = np.mean(x_reshaped, axis=1)  # Shape: (num_groups,)

        # Plot the averaged rewards
        plt.figure(figsize=(10, 6))
        plt.plot(x_averaged, y_averaged, marker='o', linestyle='-', color='b', label='Average Reward per 1000 Episodes')
        plt.xlabel('Episode Number')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per 1000 Episodes')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("No argument provided.")