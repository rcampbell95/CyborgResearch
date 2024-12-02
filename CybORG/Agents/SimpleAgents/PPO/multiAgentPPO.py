
import numpy as np
import torch
import torch.nn as nn
from CybORG.Agents.SimpleAgents.PPO.ppo import PPONetwork
from torch.distributions import Categorical

class MultiAgentPPO:
    """Multi-Agent PPO implementation for drone swarm dfense. 
    Manages multiple PPO networks, one for each drone agent, coordinating their training
    and decision making to protect the swarm against cyber attacks.

    Args: 
        num_agents (int): Number of drone agents.
        input_dim (int): Dimension of state space
        output_dim (int): Number of possibel actions
    
    Attributes:
        device (torch.device): Device for computation 
        networks (dict): Maps agents Ids to their repective PPO newtowrks
        optimizers (dict): Maps agents Ids to their Adam Optimizers
        clip_epsilon (float): PPO clipping parameters
        gamma (float): Discount factor for future rewards
        gae_lambda (flaot): Lambda for Gae 
    
    
    """
    def __init__(self, num_agents=18, input_dim=None, output_dim=7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        
        # Create network for each blue agent
        self.networks = {
            f'blue_agent_{i}': PPONetwork(num_drones=num_agents).to(self.device)
            for i in range(num_agents)
        }
        
        # Initialize optimizers with learning rate 3e-4
        self.optimizers = {
            agent_id: torch.optim.Adam(network.parameters(), lr=3e-4)
            for agent_id, network in self.networks.items()
        }
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def get_actions(self, observations):
        """
        Get actions for all agents based on their observations.

        Args:
            observations (dict): Maps agents Ids to thier observations

        Returns:
            tuple: (action, action_probs, values)
                - action (dict): Selected Actions for each agent
                - action_pros (dict): Probability of selected actions
                - values(dict): Estimated state values
        """
        actions = {}
        action_probs = {}
        values = {}
        
        for agent_id, obs in observations.items():
            if agent_id in self.networks:
                state = torch.FloatTensor(obs).to(self.device)
                with torch.no_grad():
                    probs, value = self.networks[agent_id](state)
                    dist = Categorical(probs)
                    action = dist.sample()
                    actions[agent_id] = action.item()
                    action_probs[agent_id] = probs[action].item()
                    values[agent_id] = value.item()
        
        return actions, action_probs, values

    def update(self, experiences):
        """Update policy and value networks using PPO algorithm.
            
            Args:
                experiences (dict): Contains trajectories for each agent with keys:
                    - states: List of observations
                    - actions: List of actions taken
                    - probs: Action probabilities
                    - rewards: Received rewards
                    - dones: Episode termination flags
                    - values: Estimated state values
        """

        for agent_id, exp in experiences.items():
            if agent_id not in self.networks:
                continue
                
            states = torch.FloatTensor(np.array(exp['states'])).to(self.device)
            actions = torch.LongTensor(exp['actions']).to(self.device)
            old_probs = torch.FloatTensor(exp['probs']).to(self.device)
            rewards = torch.FloatTensor(exp['rewards']).to(self.device)
            dones = torch.FloatTensor(exp['dones']).to(self.device)
            values = torch.FloatTensor(exp['values']).to(self.device)
            
            # Compute advantages using GAE
            advantages = []
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(10):  # Multiple epochs
                current_probs, current_values = self.networks[agent_id](states)
                dist = Categorical(current_probs)
                
                ratio = torch.exp(dist.log_prob(actions) - torch.log(old_probs))
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * ((current_values - returns) ** 2).mean()
                total_loss = actor_loss + critic_loss
                
                self.optimizers[agent_id].zero_grad()
                total_loss.backward()
                self.optimizers[agent_id].step()