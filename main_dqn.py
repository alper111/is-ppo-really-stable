import time

import torch
import gym
import crafter

import blocks
import models


env = gym.make("CrafterReward-v1")
state_dim = torch.tensor(env.observation_space.shape).prod()
action_dim = env.action_space.n
network = blocks.MLP([state_dim]+[128]*2+[action_dim])
dv = "cuda"
agent = models.DQNAgent(network, dv, 0.0001, 128, gamma=0.99, decay=0.99, buffer_length=100000)
it = 0
update_freq = 1

while agent.iter < 1e6:
    obs = env.reset()
    obs = torch.tensor(obs.reshape(-1) / 255.0, dtype=torch.float, device=dv)
    done = False
    r = 0.0
    loss = 0.0
    t = 0
    render = False
    while not done:
        if render:
            env.render()
            time.sleep(0.01)
        action = agent.action(obs)
        obs_new, reward, done, _ = env.step(action.tolist())
        r += reward
        t += 1

        obs_new = torch.tensor(obs_new.reshape(-1) / 255.0, dtype=torch.float, device=dv)
        reward = torch.tensor(reward, dtype=torch.float, device=dv)
        d = torch.tensor(done, dtype=torch.float, device=dv)

        agent.record(obs, action, reward, obs_new, d)
        it += 1
        if it % update_freq == 0:
            loss += agent.update()
            it = 0
        obs = obs_new
    loss /= t
    print(f"Reward: {r:.3f}, loss: {loss:.3f}, t: {t}, eps: {agent.epsilon:.3f}, iter: {agent.iter:,}")
