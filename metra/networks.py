import torch
import torch.nn as nn
from torch.distributions import Normal

class Representation(nn.Module):
    """The representation function phi(s) from the paper (traj_encoder)."""
    def __init__(self, state_dim, skill_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, skill_dim),
        )

    def forward(self, state):
        return self.net(state)

class Actor(nn.Module):
    """The skill-conditioned policy pi(a|s, z)."""
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, state_dim, action_dim, skill_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=1)
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, skill):
        mean, log_std = self.forward(state, skill)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    """The SAC critic (Q-function)."""
    def __init__(self, state_dim, action_dim, skill_dim, hidden_dim):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action, skill):
        sa = torch.cat([state, action, skill], 1)
        q1 = self.net1(sa)
        q2 = self.net2(sa)
        return q1, q2