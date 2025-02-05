import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Define a simpler architecture
        self.fc1 = nn.Linear(state_size, 128)  # Input to first hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, 32)          # Third hidden layer
        self.fc4 = nn.Linear(32, action_size) # Output layer (Q-values for each action)
    
    def forward(self, state):
        # Apply activation functions after each layer
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)  # Output Q-values for each action
    
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Initialize the Q-network and optimizer
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def act(self, state, greedy=False):
        if not greedy and random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)  # Keep the unsqueeze here
        self.q_network.eval() # Set to eval mode for inference
        with torch.no_grad(): # Disable gradients for inference
            q_values = self.q_network(state)
        self.q_network.train() # Set back to train mode
        return torch.argmax(q_values).item()

    def train(self, state, next_state, action, reward, done):  # Batched inputs
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.float)

        current_q_values = self.q_network(state)
        current_q_value = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():  # No gradient computation for target
            next_q_values = self.q_network(next_state)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done) * self.gamma * max_next_q_value

        loss = F.mse_loss(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
