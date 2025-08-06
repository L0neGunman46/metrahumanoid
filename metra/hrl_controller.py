# metra/hrl_controller.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HierarchicalController:
    """High-level controller for downstream tasks (HRL approach)."""
    
    def __init__(self, state_dim, skill_dim, task_dim, config):
        self.config = config
        self.device = config["device"]
        self.skill_dim = skill_dim
        self.skill_length = config.get("skill_length", 50)  # K steps per skill
        
        # High-level policy network
        self.high_level_policy = nn.Sequential(
            nn.Linear(state_dim + task_dim, config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], skill_dim),
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.high_level_policy.parameters(), lr=config["lr"])
        
    def select_skill(self, state, task_info):
        """Select skill z for current state and task."""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        task_tensor = torch.FloatTensor(task_info.reshape(1, -1)).to(self.device)
        
        input_tensor = torch.cat([state_tensor, task_tensor], dim=1)
        
        with torch.no_grad():
            skill_logits = self.high_level_policy(input_tensor)
            if self.config.get("discrete_skills", False):
                # Discrete skills
                skill_probs = torch.softmax(skill_logits, dim=1)
                skill_idx = torch.multinomial(skill_probs, 1).item()
                z = np.zeros(self.skill_dim)
                z[skill_idx] = 1.0
                z = z - np.mean(z)  # Zero-center
            else:
                # Continuous skills
                z = skill_logits.cpu().numpy().flatten()
                z = z / (np.linalg.norm(z) + 1e-8)  # Normalize
                
        return z
    
    def update(self, state, task_info, skill, reward, next_state, done):
        """Update high-level policy using rewards."""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        task_tensor = torch.FloatTensor(task_info.reshape(1, -1)).to(self.device)
        skill_tensor = torch.FloatTensor(skill.reshape(1, -1)).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        input_tensor = torch.cat([state_tensor, task_tensor], dim=1)
        skill_logits = self.high_level_policy(input_tensor)
        
        if self.config.get("discrete_skills", False):
            # Cross-entropy loss for discrete skills
            skill_idx = np.argmax(skill)
            loss = -torch.log_softmax(skill_logits, dim=1)[0, skill_idx] * reward_tensor
        else:
            # MSE loss for continuous skills
            loss = torch.mean((skill_logits - skill_tensor) ** 2) * (-reward_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()