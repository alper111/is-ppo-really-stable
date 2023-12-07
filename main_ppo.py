import sys

import numpy as np
import torch
import gym

import models
import utils

env = gym.make(sys.argv[1])
state_dim = torch.tensor(env.observation_space.shape).prod()
action_dim = env.action_space.shape[0]
dv = "cpu"
agent = models.PPOAgent(state_dim=state_dim, hidden_dim=64, action_dim=action_dim,
                        dist="gaussian", num_layers=2, device=dv, lr_p=0.0003,
                        lr_v=0.001, K=80, batch_size=-1, eps=0.2, c_clip=1.0, c_v=0.5,
                        c_ent=0.01, bn=False)
it = 0
update_iter = 4000
max_timesteps = 1000
show_freq = 10
gamma = 0.99
lmbda = 0.97

for epi in range(5000):
    epi_rew = []
    while agent.memory.size < update_iter:
        states, actions, logprobs, rewards = [], [], [], []
        obs, _ = env.reset()
        obs = torch.tensor(obs.reshape(-1) / 255.0, dtype=torch.float)
        for t in range(max_timesteps):
            # forward policy
            with torch.no_grad():
                m = agent.dist(obs)
                action = m.sample()
                logprob, _ = agent.logprob(obs, action)
            states.append(obs.cpu())
            obs, rew, done, _, _ = env.step(action.tolist())
            obs = torch.tensor(obs.reshape(-1) / 255.0, dtype=torch.float)
            actions.append(action.cpu())
            logprobs.append(logprob.cpu())
            rewards.append(rew)
            if done:
                break

        # add the last state for advantage estimation
        states.append(obs.cpu())
        if not done:
            with torch.no_grad():
                rewards.append(agent.value(obs).item())
        else:
            rewards.append(0)

        states = torch.stack(states)
        actions = torch.stack(actions)
        logprobs = torch.stack(logprobs)
        rewards = torch.tensor(rewards, dtype=torch.float)
        # gae
        with torch.no_grad():
            values = agent.value(states).reshape(-1)
            if done:
                values[-1] = 0
            advantages = rewards[:-1] + gamma * values[1:] - values[:-1]
        discounted_adv = utils.discount(advantages, gamma * lmbda).cpu()
        cumrew = rewards[:-1].sum().item()
        epi_rew.append(cumrew)
        rewards = utils.discount(rewards, gamma=gamma)[:-1].cpu()

        for i in range(states.shape[0]-1):
            agent.record(states[i].clone(), actions[i].clone(), logprobs[i].clone(),
                         rewards[i].clone(), discounted_adv[i].clone())

    agent.policy.train()
    agent.value.train()
    loss = agent.update()
    agent.policy.eval()
    agent.value.eval()
    agent.reset_memory()
    print(f"Epi: {epi+1}, reward: {np.mean(epi_rew):.3f}, loss: {loss:.3f}")

    if (epi+1) % show_freq == 0:
        env_vis = gym.make(sys.argv[1], render_mode="human")
        obs, _ = env_vis.reset()
        obs = torch.tensor(obs.reshape(-1) / 255.0, dtype=torch.float)
        for t in range(max_timesteps):
            # forward policy
            with torch.no_grad():
                m = agent.dist(obs)
                action = m.sample()
            obs, _, done, _, _ = env_vis.step(action.tolist())
            obs = torch.tensor(obs.reshape(-1) / 255.0, dtype=torch.float)
            env.render()
            if done:
                break
        env_vis.close()

