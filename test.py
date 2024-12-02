import numpy as np
import torch
import torch.nn as nn
from CybORG.Agents.SimpleAgents.PPO.multiAgentPPO import MultiAgentPPO
from CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator
from CybORG import CybORG
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper

def main():
    # Initialize environment
    sg = DroneSwarmScenarioGenerator()
    cyborg = CybORG(sg, 'sim')
    env = PettingZooParallelWrapper(cyborg)
    
    # Initialize PPO agent
    ppo = MultiAgentPPO()
    
    # Training loop
    episodes = 100
    max_steps = 500
    
    for episode in range(episodes):
        observations = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in observations.keys()}
        
        experiences = {agent_id: {
            'states': [], 'actions': [], 'probs': [], 'rewards': [], 
            'dones': [], 'values': []
        } for agent_id in observations.keys()}
        
        for step in range(max_steps):
            actions, probs, values = ppo.get_actions(observations)
            next_observations, rewards, dones, _ = env.step(actions)
            
            # Store experiences
            for agent_id in observations.keys():
                experiences[agent_id]['states'].append(observations[agent_id])
                experiences[agent_id]['actions'].append(actions[agent_id])
                experiences[agent_id]['probs'].append(probs[agent_id])
                experiences[agent_id]['rewards'].append(rewards[agent_id])
                experiences[agent_id]['dones'].append(dones[agent_id])
                experiences[agent_id]['values'].append(values[agent_id])
                episode_rewards[agent_id] += rewards[agent_id]
            
            observations = next_observations
            if all(dones.values()):
                break
        
        # Update policy
        ppo.update(experiences)
        
        # Log progress
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
        print(f"Episode {episode + 1}, Average Reward: {avg_reward}")

if __name__ == "__main__":
    main()