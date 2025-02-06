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

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.position = 0  # Keep track of where to overwrite

    def push(self, state, next_state, action, reward, done):
        experience = (state, next_state, action, reward, done)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience  # Overwrite oldest experience
        self.position = (self.position + 1) % self.buffer_size # Circular buffer

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None # Not enough samples in the buffer yet
        indices = random.sample(range(len(self.buffer)), batch_size)
        experiences = [self.buffer[idx] for idx in indices]
        states, next_states, actions, rewards, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        return states, next_states, actions, rewards, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, tau=0.001, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        # Initialize the Q-network and optimizer
        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())  
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.memory = ReplayBuffer(buffer_size)  # Create an instance of ReplayBuffer
        self.batch_size = batch_size

    def act(self, state, greedy=False):
        if not greedy and random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)  # Keep the unsqueeze here
        self.q_network.eval() # Set to eval mode for inference
        with torch.no_grad(): # Disable gradients for inference
            q_values = self.q_network(state)
        self.q_network.train() # Set back to train mode
        return torch.argmax(q_values).item()

    def train(self):  # Batched inputs
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
        states, next_states, actions, rewards, dones = batch # Get the batch from the memory

        current_q_values = self.q_network(states)
        current_q_value = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones) * self.gamma * max_next_q_value

        loss = F.mse_loss(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network, self.target_q_network, self.tau)

    def remember(self, state, next_state, action, reward, done):
        self.memory.push(state, next_state, action, reward, done)  # Use the ReplayBuffer's push method

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
