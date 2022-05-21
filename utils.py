import torch


def discount(rewards, gamma):
    """
    Discount the reward trajectory.
    Parameters
    ----------
    rewards : list of float
        Reward trajectory.
    gamma : float
        Discount factor.
    Returns
    -------
    discounted_rewards : list of float
        Discounted reward trajectory.
    """
    R = 0.0
    discounted_rewards = []
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, device=rewards.device)
    return discounted_rewards
