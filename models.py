"""Several reinforcement learning algorithms."""
import os
import copy
import pickle

import torch

import blocks


class PPOAgent:
    """Proximal policy optimization method."""

    def __init__(self, state_dim, hidden_dim, action_dim, dist, num_layers, device, lr_p, lr_v, K,
                 batch_size, eps, c_clip, c_v, c_ent, bn):
        """
        Initialize a PPO agent.

        Parameters
        ----------
        state_dim : int
            State dimension (input dimension).
        hidden_dim : int
            Hidden dimension of MLP layers.
        action_dim : int
            Action dimension.
        dist : {"gaussian", "categorical"}
            Distribution used to represent the policy output.
        num_layers : int
            Number of layers in MLP.
        device : torch.device
            Device of MLP parameters and other parameters.
        lr_p : float
            Learning rate for policy network.
        lr_v : float
            Learning rate for value network.
        K : int
            Number of optimization steps for each update iteration. This is
            same for both policy network and value network.
        batch_size : int
            Batch size. If -1, full batch is used.
        eps : float
            Clipping parameter of PPO.
        c_clip : float
            Policy loss coefficient.
        c_v : float
            Value loss coefficient.
        c_ent : float
            Entropy loss coefficient.
        bn : bool
            Whether to use batch norm or not.
        """
        self.device = device
        self.K = K
        self.batch_size = batch_size
        self.dst = dist
        self.memory = Memory(keys=["state", "action", "reward", "logprob", "adv"], buffer_length=-1)
        self.eps = eps
        self.c_clip = c_clip
        self.c_v = c_v
        self.c_ent = c_ent
        if self.dst == "gaussian" and action_dim > 1:
            self.multivariate = True
        else:
            self.multivariate = False

        policy_layer = [state_dim] + [hidden_dim] * num_layers + [action_dim]
        value_layer = [state_dim] + [hidden_dim] * num_layers + [1]
        self.policy = blocks.MLP(layer_info=policy_layer, batch_norm=bn)
        self.value = blocks.MLP(layer_info=value_layer, batch_norm=bn)

        log_std = -0.5 * torch.ones(action_dim, dtype=torch.float)
        self.log_std = torch.nn.ParameterList([torch.nn.Parameter(log_std)])

        self.optimizer = torch.optim.Adam(
            params=[
                {"params": self.policy.parameters(), "lr": lr_p},
                {"params": self.log_std.parameters(), "lr": lr_p},
                {"params": self.value.parameters(), "lr": lr_v}],
            amsgrad=True)
        self.criterion = torch.nn.MSELoss()

    def dist(self, x):
        """
        Policy output as a distribution for a given state.

        Parameters
        ----------
        x : torch.tensor
            State.

        Returns
        -------
        m : torch.distribution
            Policy output as a distribution.
        """
        out = self.policy(x)
        if self.dst == "categorical":
            m = torch.distributions.multinomial.Categorical(logits=out)
        else:
            std = blocks.expln(self.log_std[0])
            m = torch.distributions.normal.Normal(out, std)
        return m

    def logprob(self, s, a):
        """
        Log probability of an action for a given state with the current policy.

        Parameters
        ----------
        s : torch.tensor
            State.
        a : torch.tensor
            Action.

        Returns
        -------
        logprob : torch.tensor
            Log probability.
        entropy : torch.tensor
            Entropy.
        """
        m = self.dist(s)
        logprob = m.log_prob(a)
        entropy = m.entropy()
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        return logprob, entropy

    def action(self, x, std=True):
        """
        Sample action from the current policy.

        Parameters
        ----------
        x : torch.tensor
            State.
        std : bool, optional.
            If False, then the mean action will be sampled.

        Returns
        -------
        action : torch.tensor
            Sampled action.
        logprob : torch.tensor
            Log probability of the sampled action with the current policy and
            state.
        """
        with torch.no_grad():
            m = self.dist(x)
            if std:
                action = m.sample()
            else:
                action = self.policy(x)
            logprob = m.log_prob(action)
            if self.multivariate:
                logprob = logprob.sum(dim=-1)
        return action, logprob

    def record(self, state, action, logprob, reward, adv):
        """
        Append an experience to the memory.

        Parameters
        ----------
        state : torch.tensor
            State.
        action : torch.tensor
            Action.
        logprob : torch.tensor
            Log probability.
        reward : torch.tensor
            Reward. This is expected to be discounted.
        adv : torch.tensor
            Advantage. This is expected to be GAE advantage.

        Returns
        -------
        None
        """
        dic = {"state": state, "action": action, "reward": reward, "logprob": logprob, "adv": adv}
        self.memory.append(dic)

    def reset_memory(self):
        """
        Clear all experiences from the memory.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.memory.clear()

    def loss(self, dic):
        """
        Calculate PPO loss.

        Parameters
        ----------
        dic : dict
            This should contain state, action, logprob, reward, and adv.

        Returns
        -------
        loss : torch.tensor
            PPO loss value.
        """
        state = dic["state"].to(self.device)
        action = dic["action"].to(self.device)
        logprob = dic["logprob"].to(self.device)
        reward = dic["reward"].to(self.device)
        adv = dic["adv"].to(self.device)

        # policy loss
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        new_logp, entropy = self.logprob(state, action)
        ratio = torch.exp(new_logp - logprob)
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * adv
        policy_loss = - self.c_clip * torch.min(surr1, surr2)
        # value loss
        v_bar = self.value(state).reshape(-1)
        value_loss = self.c_v * self.criterion(v_bar, reward)
        # entropy loss
        entropy_loss = - self.c_ent * entropy
        # total loss
        loss = (policy_loss + value_loss + entropy_loss).mean()
        return loss

    def update(self):
        """
        Perform a PPO update.

        Parameters
        ----------
        None

        Returns
        -------
        avg_loss : torch.tensor
            Total loss for this update.
        """

        self.policy.to(self.device)
        self.value.to(self.device)
        self.log_std.to(self.device)

        avg_loss = 0.0
        for _ in range(self.K):
            if self.batch_size == -1:
                res = self.memory.get_all()
            else:
                res = self.memory.sample_n(self.batch_size)

            loss = self.loss(res)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        avg_loss /= self.K
        self.policy.cpu()
        self.value.cpu()
        self.log_std.cpu()

        return avg_loss

    def save(self, path, ext=None):
        """
        Save policy and value networks.

        Parameters
        ----------
        path : str
            Save folder. If the path does not exist, a new folder created
            recursively.
        ext : str, optional
            Extension of the save name. This is useful for saving multiple
            models at several epochs.

        Returns
        -------
        None
        """
        if not os.path.exists(path):
            os.makedirs(path)
        pname = "policy"
        vname = "value"
        stdname = "logstd"
        if ext:
            pname = pname + ext + ".ckpt"
            vname = vname + ext + ".ckpt"
            stdname = stdname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        vname = os.path.join(path, vname)
        stdname = os.path.join(path, stdname)
        torch.save(self.policy.eval().state_dict(), pname)
        torch.save(self.value.eval().state_dict(), vname)
        torch.save(self.log_std.state_dict(), stdname)

    def load(self, path, ext=None):
        """
        Load policy and value networks.

        Parameters
        ----------
        path : str
            Path of the model folder.
        ext : str
            Name extension.

        Returns
        -------
        None
        """
        pname = "policy"
        vname = "value"
        stdname = "logstd"
        if ext:
            pname = pname + ext + ".ckpt"
            vname = vname + ext + ".ckpt"
            stdname = stdname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        vname = os.path.join(path, vname)
        stdname = os.path.join(path, stdname)
        self.policy.load_state_dict(torch.load(pname))
        self.value.load_state_dict(torch.load(vname))
        self.log_std.load_state_dict(torch.load(stdname))


class DQNAgent:
    """Deep Q-learning agent."""

    def __init__(self, network, device, lr, batch_size, gamma, epsilon=1.0, decay=0.999,
                 decay_iter=200, target_update_iter=100, buffer_length=1000000):
        self.device = device
        self.batch_size = batch_size
        self.memory = Memory(keys=["s", "a", "r", "sn", "d"], buffer_length=buffer_length)
        self.network = network.to(device)
        self.target_net = copy.deepcopy(network)
        self.optimizer = torch.optim.Adam(lr=lr, params=self.network.parameters())
        self.epsilon = epsilon
        self.eps_decay_rate = decay
        self.eps_decay_iter = decay_iter
        self.target_update_iter = target_update_iter
        self.gamma = gamma
        self.iter = 0
        self.criterion = torch.nn.MSELoss()

    def action(self, x, epsilon=True):
        with torch.no_grad():
            out = self.network(x.to(self.device))
        if epsilon:
            if torch.rand(1) < self.epsilon:
                a = torch.randint(0, out.shape[-1], (), device=self.device)
            else:
                a = out.argmax()
        else:
            a = out.argmax()
        return a

    def record(self, state, action, reward, state_n, done):
        dic = {"s": state, "a": action, "r": reward, "sn": state_n, "d": done}
        self.memory.append(dic)

    def reset_memory(self):
        self.memory.clear()

    def loss(self, dic):
        state = dic["s"].to(self.device)
        action = dic["a"].to(self.device)
        reward = dic["r"].to(self.device)
        state_n = dic["sn"].to(self.device)
        done = dic["d"].to(self.device)
        batch_size = state.shape[0]

        with torch.no_grad():
            q_next, _ = self.target_net(state_n).max(dim=-1)
        q_next = q_next.reshape(-1)
        target = reward + (1-done)*self.gamma*q_next
        q_now = self.network(state)[torch.arange(batch_size), action]
        loss = self.criterion(q_now, target)
        return loss

    def update_target_net(self):
        self.target_net = copy.deepcopy(self.network)

    def update(self):
        if self.batch_size*10 <= self.memory.size:
            sample = self.memory.sample_n(self.batch_size)
            loss = self.loss(sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter += 1
            if self.iter % self.eps_decay_iter == 0:
                self.epsilon = max(self.epsilon * self.eps_decay_rate, 0.1)

            if self.iter % self.target_update_iter == 0:
                self.update_target_net()
            loss = loss.item()
        else:
            loss = 0.0
        return loss

    def save(self, path, ext="", save_memory=False):
        if not os.path.exists(path):
            os.makedirs(path)
        name = os.path.join(path, "network" + ext + ".pt")
        optim_name = os.path.join(path, "optim" + ext + ".pt")
        torch.save(self.network.eval().cpu().state_dict(), name)
        torch.save(self.optimizer.state_dict(), optim_name)
        self.network.to(self.device)

        file = open(os.path.join(path, "epsilon" + ext + ".pkl"), "wb")
        pickle.dump({"epsilon": self.epsilon}, file)
        if save_memory:
            self.memory.save(path, ext)

    def load(self, path, ext="", load_memory=False):
        name = os.path.join(path, "network" + ext + ".pt")
        # optim_name = os.path.join(path, "optim" + ext + ".pt")
        self.network.load_state_dict(torch.load(name))
        # self.optimizer.load_state_dict(torch.load(optim_name))

        # file = open(os.path.join(path, "epsilon" + ext + ".pkl"), "rb")
        # self.epsilon = pickle.load(file)["epsilon"]
        # if load_memory:
        #     self.memory.load(path, ext)


class PGAgent:
    """Vanilla policy gradient agent."""

    def __init__(self, state_dim, hidden_dim, action_dim, dist, num_layers, device, lr, batch_size):
        """
        Initialize a PG agent.

        Parameters
        ----------
        state_dim : int
            State dimension (input dimension).
        hidden_dim : int
            Hidden dimension of MLP layers.
        action_dim : int
            Action dimension.
        dist : {"gaussian", "categorical"}
            Distribution used to represent the policy output.
        num_layers : int
            Number of layers in MLP.
        device : torch.device
            Device of MLP parameters and other parameters.
        lr : float
            Learning rate.
        batch_size : int
            Batch size. If -1, full batch is used.
        """
        self.device = device
        self.batch_size = batch_size
        self.dst = dist
        self.memory = Memory(keys=["logprob", "reward"], buffer_length=-1)
        if self.dst == "gaussian" and action_dim > 1:
            self.multivariate = True
        else:
            self.multivariate = False

        policy_layer = [state_dim] + [hidden_dim] * num_layers + [action_dim]

        self.policy = blocks.MLP(layer_info=policy_layer)
        self.policy.to(device)

        log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
        self.log_std = torch.nn.Parameter(log_std)

        self.optimizer = torch.optim.Adam(
            lr=lr,
            params=[
                {"params": self.policy.parameters()},
                {"params": self.log_std}
            ], amsgrad=True)

    def dist(self, x):
        out = self.policy(x)
        if self.dst == "categorical":
            m = torch.distributions.multinomial.Categorical(logits=out)
        else:
            dim = out.shape[-1]
            mu = torch.tanh(out[..., :dim//2])
            logstd = out[..., dim//2:]
            std = 0.2 + torch.nn.functional.softplus(logstd)
            m = torch.distributions.normal.Normal(mu, std)
        return m

    def logprob(self, s, a):
        m = self.dist(s)
        logprob = m.log_prob(a)
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
        return logprob

    def action(self, x, std=True):
        m = self.dist(x)
        if std:
            action = m.sample()
        else:
            with torch.no_grad():
                out = self.policy(x)
                action = torch.tanh(out[..., :out.shape[-1]//2])
        logprob = m.log_prob(action)
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
        return action, logprob

    def record(self, logprob, reward):
        dic = {"logprob": logprob, "reward": reward}
        self.memory.append(dic)

    def reset_memory(self):
        self.memory.clear()

    def loss(self, dic):
        logprob = dic["logprob"]
        reward = dic["reward"]
        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        loss = -reward * logprob
        return loss.mean()

    def update(self):
        avg_loss = 0.0
        if self.batch_size == -1:
            res = self.memory.get_all()
        else:
            res = self.memory.sample_n(self.batch_size)

        loss = self.loss(res)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        avg_loss += loss.item()
        return avg_loss

    def save(self, path, ext=None):
        if not os.path.exists(path):
            os.makedirs(path)
        pname = "policy"
        if ext:
            pname = pname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        torch.save(self.policy.eval().cpu().state_dict(), pname)
        self.policy.train().to(self.device)

    def load(self, path, ext=None):
        pname = "policy"
        if ext:
            pname = pname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        self.policy.load_state_dict(torch.load(pname))


class Memory:
    def __init__(self, keys, buffer_length=-1):
        self.buffer = {}
        self.keys = keys
        for key in keys:
            self.buffer[key] = []
        self.buffer_length = buffer_length
        self.size = 0

    def clear(self):
        for key in self.keys:
            del self.buffer[key][:]
        self.size = 0

    def append(self, dic):
        if self.buffer_length != -1 and self.size == self.buffer_length:
            for key in self.keys:
                self.buffer[key] = self.buffer[key][1:]
            self.size -= 1
        for key in self.keys:
            self.buffer[key].append(dic[key])
        self.size += 1

    def peek_n(self, n, from_start=False):
        if from_start:
            idx = list(range(self.size-n, self.size))
        else:
            idx = list(range(n))
        return self.get_by_idx(idx)

    def sample_n(self, n):
        r = torch.randperm(self.size)
        idx = r[:n]
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        res = {}
        for key in self.keys:
            res[key] = torch.stack([self.buffer[key][i] for i in idx])
        return res

    def get_all(self):
        idx = list(range(self.size))
        return self.get_by_idx(idx)

    def save(self, path, ext=""):
        metadata = {
            "keys": self.keys,
            "buffer_length": self.buffer_length
        }
        file = open(os.path.join(path, "buffer_info"+ext+".pkl"), "wb")
        pickle.dump(metadata, file)
        file.close()

        torch.save(self.get_all(), os.path.join(path, "buffer"+ext+".pt"))

    def load(self, path, ext=""):
        self.clear()
        metadata = pickle.load(open(os.path.join(path, "buffer_info"+ext+".pkl"), "rb"))
        self.buffer_length = metadata["buffer_length"]
        self.keys = metadata["keys"]
        for key in self.keys:
            self.buffer[key] = []

        memory = torch.load(os.path.join(path, "buffer"+ext+".pt"))
        memsize = memory[self.keys[0]].shape[0]
        print(memsize)
        for i in range(memsize):
            self.append({key: memory[key][i] for key in memory})
