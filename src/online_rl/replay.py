"""
Time-major replay buffer for vectorised online RL (torch, on the training device).

Stores, per env and step, the observation before acting, the action taken and an episode id. A sample is a
transition (s_t, a_t, s_{t+1}, a_{t+1}) inside one episode plus a future state sf = s_{t+1+k} of the NEXT state,
k ~ Geometric(1 - gamma) renormalized to the steps left in that episode (as dre_iclr's `renormalize_geom`, and
the MetaWorld dataset's future frames); k = 0 when s_{t+1} is the last stored step of its episode. Transitions
that cross an episode boundary (time-limit or termination, whose terminal observation auto-reset overwrites) are
never sampled.
"""

import math

import torch


class ReplayBuffer:
    def __init__(self, capacity, num_envs, obs_dim, action_dim, device, max_episode_length):
        self.cap, self.num_envs, self.device = capacity, num_envs, device
        self.window = min(max_episode_length, capacity)
        self.obs = torch.zeros(capacity, num_envs, obs_dim, device=device)
        self.act = torch.zeros(capacity, num_envs, action_dim, device=device)
        self.ep = torch.full((capacity, num_envs), -1, dtype=torch.long, device=device)
        self.head = 0  # number of steps written so far (logical time)

    def add(self, obs, action, ep_id):
        i = self.head % self.cap
        self.obs[i], self.act[i], self.ep[i] = obs, action, ep_id
        self.head += 1

    @property
    def size(self):
        return min(self.head, self.cap) * self.num_envs

    def _valid_start(self, b):
        """Logical times t (and envs) such that s_t and s_{t+1} are stored and in the same episode."""
        lo = max(0, self.head - self.cap)
        t = torch.randint(lo, self.head - 1, (b,), device=self.device)
        e = torch.randint(self.num_envs, (b,), device=self.device)
        for _ in range(100):
            bad = self.ep[t % self.cap, e] != self.ep[(t + 1) % self.cap, e]
            if not bad.any():
                break
            n = int(bad.sum())
            t[bad] = torch.randint(lo, self.head - 1, (n,), device=self.device)
            e[bad] = torch.randint(self.num_envs, (n,), device=self.device)
        return t, e, ~bad

    def sample(self, b, gamma):
        t, e, ok = self._valid_start(b)
        t, e = t[ok], e[ok]
        # Steps stored after s_{t+1} in the same episode, h in [0, window].
        steps = t[:, None] + 2 + torch.arange(self.window, device=self.device)[None]
        same = (steps <= self.head - 1) & (self.ep[steps % self.cap, e[:, None]] == self.ep[t % self.cap, e][:, None])
        h = same.long().cumprod(1).sum(1)
        # k ~ Geometric(1 - gamma) conditioned on k <= h: k = ceil(log(1 - u (1 - gamma^h)) / log gamma).
        log_g = math.log(max(gamma, 1e-10))
        cdf = (1 - torch.exp(torch.clamp(h * log_g, min=-700))).clamp(min=1e-10)
        u = torch.rand(len(t), device=self.device)
        k = torch.ceil(torch.log((1 - u * cdf).clamp(min=1e-10)) / log_g).long()
        k = torch.where(h > 0, torch.minimum(k.clamp(min=1), h.clamp(min=1)), torch.zeros_like(k))
        i0, i1, i_f = t % self.cap, (t + 1) % self.cap, (t + 1 + k) % self.cap
        return {"obs": self.obs[i0, e], "act": self.act[i0, e], "next_obs": self.obs[i1, e],
                "next_act": self.act[i1, e], "future_obs": self.obs[i_f, e]}

    def sample_obs(self, b):
        """Uniform stored observations (for exploration goals)."""
        lo = max(0, self.head - self.cap)
        t = torch.randint(lo, self.head, (b,), device=self.device) % self.cap
        return self.obs[t, torch.randint(self.num_envs, (b,), device=self.device)]
