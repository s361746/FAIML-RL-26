import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class Policy(torch.nn.Module):
    
    """
    Network constructor.
    Initializes the architecture of the Actor and the Critic.
    """
    def __init__(self, state_space, action_space, mode="actor_critic"):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.mode = mode
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Actor network
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        # Critic network
        if self.mode == "actor_critic":
            self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
            self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
            self.fc3_critic_mean = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    """
    Initializes the weights of all linear modules in the network using 
    the Xavier normal distribution and sets the biases to zero.
    Helps prevent vanishing/exploding gradient problems at the start of training.
    """
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    """
    Executes the forward pass of the neural network.
    """
    def forward(self, x):

        # Actor
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        # Critic
        if self.mode == "actor_critic":
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))
            critic_value = self.fc3_critic_mean(x_critic)
        
        if self.mode == "reinforce":
            return normal_dist, None
        else:
            return normal_dist, critic_value


class Agent(object):

    """
    Initializes the RL Agent.
    """
    def __init__(self, policy, device='cpu', baseline=None):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        
        # Configure baseline for REINFORCE
        if self.policy.mode == "reinforce":
            self.baseline = baseline if baseline is not None else 0
            
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    """
    Processes collected data, calculates losses, 
    and updates the policy network parameters.
    """
    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        if self.policy.mode == "reinforce":     
            # TASK 2:
            # Compute discounted returns
            returns = self._discount_rewards(rewards, self.gamma)
            
            # Compute policy gradient loss function given actions and returns
            advantages = returns - self.baseline
            policy_loss = -(action_log_probs * advantages).mean()
            
            # Compute gradients and step the optimizer
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
        else:
            # TASK 3:
            # Compute bootstrapped discounted return estimates
            normal_dist, values = self.policy(states)
            values = values.squeeze(-1)
            _, next_values = self.policy(next_states)
            next_values = next_values.squeeze(-1).detach()
            
            # N-step backward algorithm to compute correct targets, if the state is terminal, the bootstrap value is 0 
            targets = torch.zeros_like(rewards)
            running_return = 0.0 if done[-1] else next_values[-1]
            
            for t in reversed(range(len(rewards))):
                if done[t]:
                    running_return = 0.0
                running_return = rewards[t] + self.gamma * running_return
                targets[t] = running_return
                
            advantages = targets - values.detach() 
            
            # Compute actor loss and critic loss
            critic_loss = F.mse_loss(values, targets)
            actor_loss = -(action_log_probs * advantages).mean()

            # Compute the mean entropy of the distribution
            entropy = normal_dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy
            
            # Compute gradients and step the optimizer
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients to avoid exploding gradients in continuous control tasks
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

        return    
    
    """
    Calculates the cumulative discounted returns for each timestep 
    of an episode
    """
    @staticmethod
    def _discount_rewards(r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r    

    """
    Determines the appropriate action vector given the current environment state.
    """
    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None
        else:   # Sample from the distribution
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob
    
    """
    Saves a single step transition into the agent's short-term memory buffer.
    """
    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)