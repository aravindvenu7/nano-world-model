"""
Online GTSE-r / GTSE-b agent on the object-token bilinear model (models/object_bilinear.py).

Loss: the conditional GTSE TD loss of dre_iclr (scalar score, `full=False`), as in
experiments/train_experiment_gtse_bilinear.py, on masked goal-token sets: every sample draws a subset of goal
objects, and the loss only counts that subset, so one network learns the occupancy of any subset of cubes
(BuilderBench tasks and exploration goals often use only some cubes).
  loss = (1 - gamma) * loss_s_ + gamma * loss_f
  loss_s_: score matching at s_{t+1} given (s_t, a_t);  loss_f: TD at sf, online (s_t, a_t) vs target (s_{t+1}, a_{t+1}).

Critic: Q(s, a, g) = sum_k dt * score(s, a, g, t_k), t_k on [0, t1_val) with nfe steps (dre_iclr `_compute_critic`,
conditional part). Because the readout is bilinear, score(t) = <G_t(g), M_t(s, a)> with G_t a goal-only tensor, so
Q = sum_k dt <G_k, M_k>; with phi_time=False (GTSE-r) M does not depend on t and Q = <sum_k dt G_k, M(s, a)>.

Actions: gradient ascent on Q in action space from `num_candidates` uniform starts, then the argmax (the `grad`
action mode of dre/ogbench/toy_bgrw_online), epsilon-greedy with uniform actions.
"""

import copy

import torch

from models.object_bilinear import ObjectBilinearTimeScore, scalar_score
from utils.gtse_targets import epsilon_target


class GTSEAgent:
    def __init__(self, cfg, goal_space, state_dim, action_dim, device):
        m, g = cfg.model, cfg.gtse
        phi_time = m.phi_time if m.phi_time is not None else cfg.agent == "gtse_b"
        if cfg.agent not in ("gtse_r", "gtse_b"):
            raise ValueError(f"agent must be 'gtse_r' or 'gtse_b', got {cfg.agent}")
        self.model = ObjectBilinearTimeScore(state_dim, goal_space.dim, action_dim, hidden_size=m.hidden_size, depth=m.depth,
                                             num_heads=m.num_heads, mlp_ratio=m.mlp_ratio, readout_heads=m.readout_heads,
                                             time_embedding=m.time_embedding, time_dim=m.time_dim,
                                             phi_time=phi_time, psi_time=m.psi_time).to(device)
        self.target = copy.deepcopy(self.model).requires_grad_(False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=g.lr)
        self.gs, self.device, self.action_dim = goal_space, device, action_dim
        self.gamma, self.tau, self.noise, self.huber = g.discount, g.target_tau, g.goal_noise_std, g.huber
        self.temp_coeff, self.t1, self.nfe, self.grad_clip = g.temp_coeff, g.t1_val, g.nfe, g.grad_clip
        self.plan = cfg.planning

    # ── training ─────────────────────────────────────────────────────────────

    def _masks(self, b, loss_mask_cubes):
        mask = self.gs.random_mask(b, loss_mask_cubes)
        return mask, self.gs.feat_mask[None] * mask[..., None].float()

    def _score(self, model, state, action, x_t, t, mask):
        return scalar_score(model(state, action, x_t, t, mask, self.gs.types), mask, self.gs.feat_mask)

    def update(self, batch, env, loss_mask_cubes=None):
        st, st1 = env.state_tokens(batch["obs"]), env.state_tokens(batch["next_obs"])
        x_next, sf = self.gs.from_obs(batch["next_obs"]), self.gs.from_obs(batch["future_obs"])
        b = st.shape[0]
        mask, fm = self._masks(b, loss_mask_cubes)
        t = torch.rand(b, device=self.device) * (1.0 - 1e-5)  # t ~ U[0, 1), shared by both branches
        tt = t.view(b, 1, 1)

        # Immediate branch: score at s_{t+1} given (s_t, a_t).
        s_ = x_next + self.noise * torch.randn_like(x_next)
        x0 = torch.randn_like(s_)
        xt = (x0 * torch.sqrt(1 - tt ** 2) + tt * s_) * fm
        lam, tgt = epsilon_target(x0, s_, t, self.temp_coeff, mask=fm)
        loss_s = ((tgt - lam * self._score(self.model, st, batch["act"], xt, t, mask)) ** 2).mean()

        # TD branch at sf: online (s_t, a_t) vs target (s_{t+1}, a_{t+1}).
        sf = sf + self.noise * torch.randn_like(sf)
        x0f = torch.randn_like(sf)
        xtf = (x0f * torch.sqrt(1 - tt ** 2) + tt * sf) * fm
        lam_f, _ = epsilon_target(x0f, sf, t, self.temp_coeff, mask=fm)
        with torch.no_grad():
            tgt_f = self._score(self.target, st1, batch["next_act"], xtf, t, mask)
        residual = lam_f * (tgt_f - self._score(self.model, st, batch["act"], xtf, t, mask))
        if self.huber:
            delta = 1.0 / torch.sqrt(torch.clamp(1.0 - t, 1e-5, 1.0))
            per = torch.where(residual.abs() <= delta, 0.5 * residual ** 2, delta * (residual.abs() - 0.5 * delta))
        else:
            per = residual ** 2
        loss_f = per.mean()
        loss = (1 - self.gamma) * loss_s + self.gamma * loss_f

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip if self.grad_clip else float("inf"))
        self.opt.step()
        with torch.no_grad():  # Polyak target update
            for p, tp in zip(self.model.parameters(), self.target.parameters()):
                tp.mul_(1 - self.tau).add_(p, alpha=self.tau)
        return {"loss": loss.item(), "loss_s_": loss_s.item(), "loss_f": loss_f.item(), "grad_norm": gnorm.item()}

    # ── critic and action selection ──────────────────────────────────────────

    def _time_grid(self):
        dt0 = self.t1 / self.nfe
        return torch.arange(self.nfe, device=self.device) * dt0, dt0

    @torch.no_grad()
    def goal_tensor(self, goal, mask):
        """G_k(g) per time t_k, (E, K, H, hd, hd), such that score(s, a, g, t_k) = <G_k, M_k(s, a)>."""
        m = self.model
        tk, _ = self._time_grid()
        e, k, ng = goal.shape[0], len(tk), goal.shape[1]
        q = m.goal_queries(goal.repeat_interleave(k, 0), tk.repeat(e), mask.repeat_interleave(k, 0), self.gs.types)
        h = m.readout_heads
        hd = q.shape[-1] // h
        q = q.view(e, k, ng, h, hd) * mask[:, None, :, None, None]
        u = (self.gs.feat_mask @ m.readout_out.weight).view(ng, h, hd)  # W_out^T f_i: summing the valid output dims
        return torch.einsum("eknha,nhv->ekhav", q, u)

    def q_values(self, G, obs_tokens, actions):
        """G (E, K, H, hd, hd); obs_tokens (E, Ns, D_s); actions (E, C, A) -> Q (E, C).

        <G, M> with M = sum_j k_j v_j^T is computed as sum_j k_j^T G v_j, so the hd x hd memory is never built
        per candidate: memory per candidate is (tokens x hidden) instead of hd^2 per head."""
        m = self.model
        e, c = actions.shape[:2]
        tk, dt0 = self._time_grid()
        st = obs_tokens.repeat_interleave(c, 0)
        if not m.phi_time:  # M does not depend on t: Q = <sum_k dt G_k, M(s, a)>
            key, val = m.context_kv(st, actions.reshape(e * c, -1), torch.zeros(e * c, device=self.device))
            key, val = key.view(e, c, *key.shape[1:]), val.view(e, c, *val.shape[1:])  # (E, C, M, H, hd)
            kg = torch.einsum("ecmha,ehab->ecmhb", key, G.sum(1) * dt0)
            return (kg * val).sum((2, 3, 4))
        k = len(tk)
        key, val = m.context_kv(st.repeat_interleave(k, 0), actions.reshape(e * c, -1).repeat_interleave(k, 0), tk.repeat(e * c))
        key, val = key.view(e, c, k, *key.shape[1:]), val.view(e, c, k, *val.shape[1:])  # (E, C, K, M, H, hd)
        kg = torch.einsum("eckmha,ekhab->eckmhb", key, G * dt0)
        return (kg * val).sum((2, 3, 4, 5))

    def act(self, obs_tokens, goal, mask, epsilon):
        """Gradient ascent on Q from uniform starts, argmax, then epsilon-greedy. Returns (E, A) in [-1, 1]."""
        p = self.plan
        e = obs_tokens.shape[0]
        G = self.goal_tensor(goal, mask)
        a = (torch.rand(e, p.num_candidates, self.action_dim, device=self.device) * 2 - 1).requires_grad_(True)
        for _ in range(p.plan_steps):
            grad, = torch.autograd.grad(self.q_values(G, obs_tokens, a).sum(), a)
            a = (a + p.plan_lr * grad).clamp(-1, 1).detach().requires_grad_(True)
        with torch.no_grad():
            q = self.q_values(G, obs_tokens, a)
            best = a[torch.arange(e, device=self.device), q.argmax(1)].detach()
            explore = torch.rand(e, device=self.device) < epsilon
            best[explore] = torch.rand(int(explore.sum()), self.action_dim, device=self.device) * 2 - 1
        return best

    def state_dict(self):
        return {"model": self.model.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict()}
