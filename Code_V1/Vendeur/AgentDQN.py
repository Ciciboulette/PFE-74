import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from DQN_V2 import DeepQNetwork
from Replay_memory import ReplayMemory

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_dim, action_dim, replay_memory_dim=1e5, batch_dim=64, 
        gamma=0.99, learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity_dim = int(replay_memory_dim)
        self.batch_dim = batch_dim
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)

        self.network = DeepQNetwork(state_dim, action_dim, seed).to(device)
        self.target_network = DeepQNetwork(state_dim, action_dim, seed).to(device)
        #optim.Adam = a stochastic optimization
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        

        # Replay memory
        self.memory = ReplayMemory(self.capacity_dim)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    ########################################################
    # STEP() method
    #
    def step(self, state, action, next_state, reward):
        # Save experience in replay memory
        self.memory.push(state, action, next_state, reward)
        
        # Learn every UPDATE_RATE time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.can_provide_sample(self.batch_dim):
                experiences = self.memory.sample(self.batch_dim)
                self.learn(experiences, self.gamma)


	########################################################
    # ACT() method
    #
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


	########################################################
    # LEARN() method
    # Update value parameters using given batch of experience tuples.
    def learn(self, experiences, gamma, DQN=True):
        
        """
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, s', r) tuples 
            gamma (float): discount factor
        """

        states, actions, next_states, rewards = experiences

        # Get Q values from current observations (s, a) using model nextwork
        Qsa = self.network(states).gather(1, actions)

        #Regular (Vanilla) DQN
        #************************
            # Get max Q values for (s',a') from target model
        Qsa_prime_target_values = self.target_network(next_states).detach()
        Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)        

        
        # Compute Q targets for current states 
        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))
        
        # Compute loss (error)
        loss = F.mse_loss(Qsa, Qsa_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network, self.tau)


    ########################################################
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    #update la target network
    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
