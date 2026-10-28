# Online RL on BuilderBench (GTSE-r / GTSE-b)

Online RL from scratch on [BuilderBench](https://github.com/rajghugare19/builderbench) with the two bilinear GTSE critics, GTSE-r and GTSE-b, ported from `dre_iclr/ogbench/impls/agents/`. The agent sees the environment's state vector as a set of object tokens; there are no images and no VAE. It has no actor network: actions come from gradient ascent on the critic.

Entry point: `src/main_builderbench.py`, configured by `src/configs/rl_builderbench.yaml`. This is independent of the MetaWorld pipeline (`main_occupancy.py`, `main_gtse*.py`).

## Setup

On top of the `nanowm` environment (`environment.yml` lists these too):

```bash
pip install "mujoco>=3.3.6" "mujoco-mjx>=3.3.6" "jax>=0.6.2" optax flax etils ml_collections
```

On Python 3.10 this resolves to JAX 0.6.2 and MuJoCo 3.14 and changes no existing package version. JAX runs on the CPU (`JAX_PLATFORMS=cpu`, set by the entry point): the physics is MuJoCo's C++ rollout either way, and torch keeps the GPU.

The BuilderBench environment code is vendored in `src/builderbench/` (MIT license, upstream commit `de9130b`), including the generated task files in `src/builderbench/tasks/`. To regenerate them: `cd src/builderbench && python create_task_data.py`.

## Quick start

```bash
# Single task: train and evaluate on one task
python src/main_builderbench.py env_id=cube-2-task1 agent=gtse_r goal_space=pos

# Self-supervised play: explore with self-chosen goals, evaluate on every task with that many cubes
python src/main_builderbench.py env_id=cube-3-play agent=gtse_b goal_space=full exploration.goal_source=mega
```

Outputs go to `$RESULTS_DIR/<timestamp>-builderbench-<env_id>-<agent>/`:

| File | Contents |
| --- | --- |
| `metrics.jsonl` | one JSON line per log or evaluation |
| `tb/` | TensorBoard scalars (`logging.tensorboard=true`) |
| `checkpoint.pt` | model, target model, optimizer, config, env steps |

The entry point defaults `RESULTS_DIR` and the cache directories to `/zfsauton/scratch/avenugo2`; exported values take precedence.

## Protocols

| `env_id` | Protocol | Goals while training | Evaluation |
| --- | --- | --- | --- |
| `cube-N-taskK` | single task | the task's target, from the env | the same task, success rate |
| `cube-N-play` | self-supervised play | self-chosen exploration goals | every task with N cubes, success rate per task |

In play, each env reset carries a random task target; the agent ignores it while training and only evaluation uses it. Exploration goals are resampled whenever an episode ends:

- `exploration.goal_source=buffer`: a random stored cube configuration.
- `exploration.goal_source=mega`: the lowest-density configuration among `mega_candidates` stored ones, under a Gaussian KDE of `mega_kde_size` stored configurations (as BuilderBench's `play_ppo_mega`).

A random subset of cubes counts in each exploration goal (each cube kept with probability 1/2, at least one).

Evaluation runs one episode per eval env on the env's task targets with no exploration noise. A success counts if the BuilderBench `success` metric is positive at any step before the episode ends, the same rule as BuilderBench's evaluator.

## Agents

| `agent` | Context tower φ | Goal tower ψ | Source |
| --- | --- | --- | --- |
| `gtse_r` | (s, a), no time input | (x_t, t) | `gtse_r.py`, README command (`psi_time=True`) |
| `gtse_b` | (s, a, t) | (x_t, t) | `gtse_legendre.py` with `full=False` (`ContrastiveTimeScoreFn`) |

Override either with `model.phi_time` / `model.psi_time`.

## Model

`src/models/object_bilinear.py`, `ObjectBilinearTimeScore`. It uses the same two-tower bilinear design as the MetaWorld GTSE-b/r model (`models/nanowm_bilinear.py`), built from NanoWM's `TransformerBlock` (adaLN-Zero, SwiGLU) and `TimestepEmbedder`.

**Tokens.** For N cubes, the observation (`12 + 13N` numbers) is split into:

| Token | Features | Dims |
| --- | --- | --- |
| gripper | position, orientation, linear velocity, finger joints | 12, padded to 13 |
| cube (one per cube) | position, orientation, linear velocity, angular velocity | 13 |
| action (context tower only) | the 5-dim action | 5 |

Positions are standardized with the workspace bounds; velocities are divided by `env.vel_scale` and angular velocities by `env.angvel_scale`.

**Towers.**

- Context tower φ: the gripper, cube and action tokens, `model.depth` blocks, time from adaLN when φ sees `t`.
- Goal tower ψ: the noised goal tokens, `model.depth` blocks with a key mask so any subset of goal objects can be the goal.
- Readout: `score_i = W_out · M(s, a)ᵀ · Q_i` for each goal token, where `M(s, a) = Kᵀ V` comes from the context tokens. This is attention without the softmax, with nothing nonlinear after it, so the score is bilinear. `W_out` starts at zero, so the score starts at 0 everywhere.

There are no position embeddings, so the model is unchanged by reordering the cubes. This matches BuilderBench's permutation-invariant success check, where any cube can fill any target.

## Goal space

`goal_space` sets what the GTSE density is over:

- `pos`: one token per cube, its standardized position (3 dims).
- `full`: the state tokens themselves, gripper and cubes (13 dims).

A goal is a set of tokens plus a mask saying which count. Task and exploration goals only fix cube positions, so in the `full` space they set the target position, an identity orientation and zero velocities for each cube, and leave the gripper out.

## Training

**Loss.** The conditional GTSE TD loss of `dre_iclr` with the scalar score target (`full=False`, `utils/gtse_targets.py`), as in the MetaWorld GTSE-b/r trainers:

- `loss_s_`: score matching at `s_{t+1}` given `(s_t, a_t)`.
- `loss_f`: TD at a future state `sf`, regressing the online score at `(s_t, a_t)` onto the target network's score at `(s_{t+1}, a_{t+1})`.
- `loss = (1 − γ) · loss_s_ + γ · loss_f`, Huber on the TD branch by default.

Each sample draws a random subset of goal objects and the loss counts only that subset, so one network learns the occupancy of any subset of cubes. In single-task runs the subset has exactly as many cubes as the task; in play it is random. In the `full` space the gripper token is kept with probability 1/2.

**Replay** (`src/online_rl/replay.py`). A time-major ring buffer of `buffer_capacity` steps per env. A sample is `(s_t, a_t, s_{t+1}, a_{t+1})` inside one episode, plus `sf = s_{t+1+k}`, a geometric future of the next state: `k ~ Geometric(1 − γ)` renormalized to the steps left in that episode (`k = 0` at the episode's last stored step). Transitions that cross an episode boundary are never sampled.

**Loop** (`src/online_rl/train.py`).

1. Collect `rollout_length` steps from each of `num_envs` envs: uniform random actions for the first `warmup_env_steps`, then the planner below.
2. Take `grad_steps_per_iter` updates of batch `batch_size`.
3. Every `logging.eval_every` iterations, evaluate; every `logging.save_every`, save a checkpoint.

## Action selection

The critic integrates the scalar score over time, as `dre_iclr`'s `_compute_critic` (conditional part):

```latex
Q(s, a, g) = \sum_{k=0}^{nfe-1} \Delta t \, \mathrm{score}(s, a, g, t_k), \quad t_k = k \, \Delta t, \quad \Delta t = t_1 / nfe
```

with `t1 = gtse.t1_val` and `nfe = gtse.nfe`. Because the readout is bilinear, each term is an inner product of a goal-only tensor `G_k(g)` and the context memory `M_k(s, a)`. The goal part is computed once per env step. For GTSE-r, `M` does not depend on `t`, so each candidate action needs one context-tower pass whatever `nfe` is; for GTSE-b it needs `nfe` passes.

Each step, for each env:

1. Draw `planning.num_candidates` uniform actions in [−1, 1].
2. Take `planning.plan_steps` gradient-ascent steps on Q with step size `planning.plan_lr`, clipping to [−1, 1].
3. Keep the candidate with the highest Q.
4. With probability `planning.epsilon`, replace it with a uniform random action (training only).

## Critic snapshots

Every `diagnostics.every_env_steps` env steps (5000 by default; 0 turns it off), the run writes two figures to `<run dir>/diagnostics/`, modeled on `dre/ogbench/toy_bgrw_online/diagnostics.py`. The first snapshot is taken right after warmup, before any update, as the untrained baseline.

The state has 12 + 13N dimensions, so each figure is a 2D slice. A `diagnostics.grid` × `diagnostics.grid` grid covers the workspace (x, y), and everything else is held at an anchor state. The anchor is a start configuration of the eval env, pinned at the first snapshot together with its task goal, so snapshots compare across training.

| Figure | Panels |
| --- | --- |
| `step_<N>_cube.png` | cube visitation from the replay buffer; state axis (cube `diagnostics.cube` moved to each (x, y) at height `diagnostics.table_z`, goal fixed, critic at the planner's action); goal axis (state and action at the anchor, goal = a cube at (x, y)), as `cdr` and as `log M = cdr + log N(g)` |
| `step_<N>_gripper.png` | gripper visitation; state axis with the gripper moved to each (x, y) at its anchor height |

Each row ends with the mean critic over grid points the buffer has and has not visited. Their difference, unvisited minus visited, is the optimism statistic from the toy runs. It is logged with the goal-axis peak as `diag/*` in `metrics.jsonl` and TensorBoard. The critic plotted is the one the agent acts on, `Q = Σ Δt · score(t_k)`.

## Configuration

`src/configs/rl_builderbench.yaml`; override any key on the command line. The defaults are untuned.

| Key | Default | Meaning |
| --- | --- | --- |
| `env_id` | `cube-1-task1` | `cube-N-taskK` or `cube-N-play` |
| `agent` | `gtse_r` | `gtse_r` or `gtse_b` |
| `goal_space` | `pos` | `pos` or `full` |
| `num_envs` / `num_eval_envs` | 256 / 64 | parallel envs for training / evaluation |
| `num_threads` | 8 | MuJoCo rollout threads |
| `total_env_steps` | 10,000,000 | training length, summed over envs |
| `warmup_env_steps` | 50,000 | random actions before the first update |
| `rollout_length` | 16 | steps per env between update phases |
| `grad_steps_per_iter` | 16 | updates per iteration |
| `batch_size` | 1024 | transitions per update |
| `buffer_capacity` | 4000 | stored steps per env |
| `model.hidden_size` / `depth` / `num_heads` | 256 / 4 / 4 | per tower |
| `model.time_embedding` / `time_dim` | `fourier` / 256 | Fourier of −log(1 − t), or `linear`: sinusoidal of (1 − t)·999 |
| `gtse.discount` | 0.99 | TD discount and future-goal parameter |
| `gtse.target_tau` | 0.005 | Polyak rate |
| `gtse.huber` | true | Huber loss on the TD branch |
| `gtse.lr` | 3e-4 | Adam learning rate |
| `gtse.t1_val` / `gtse.nfe` | 0.925 / 30 | critic integration range and points |
| `planning.num_candidates` / `plan_steps` / `plan_lr` | 16 / 5 / 0.1 | gradient-ascent planner |
| `planning.epsilon` | 0.1 | exploration noise while training |
| `exploration.goal_source` | `buffer` | play only: `buffer` or `mega` |
| `env.vel_scale` / `angvel_scale` | 1.0 / 10.0 | velocity scaling in the tokens |
| `diagnostics.every_env_steps` | 5000 | env steps between critic snapshots; 0 = off |
| `diagnostics.grid` / `cube` / `table_z` | 40 / 0 / 0.02 | snapshot grid size, swept cube, cube height |

## Code layout

```
src/
├── main_builderbench.py          (entry point)
├── configs/rl_builderbench.yaml  (config)
├── builderbench/                 (vendored BuilderBench environments, wrappers, task data)
├── models/object_bilinear.py     (object-token bilinear time-score model)
├── utils/gtse_targets.py         (GTSE score target, shared with the MetaWorld trainers)
└── online_rl/
    ├── builderbench_env.py       (JAX env adapter, tokenization, goal spaces)
    ├── replay.py                 (replay buffer, future-state sampling)
    ├── gtse_agent.py             (loss, critic, gradient-ascent action selection)
    ├── diagnostics.py            (critic snapshots)
    └── train.py                  (training loop, exploration goals, evaluation)
```

## Changes to shared code

The MetaWorld paths are unchanged. These edits are pass-throughs whose defaults keep the old behavior:

- `Attention` and `TransformerBlock` (`models/nanowm_occupancy.py`) accept an optional `attn_mask`.
- The bilinear readout in `models/nanowm_bilinear.py` is split into `bilinear_memory` and `bilinear_apply`.
- `epsilon_target` moved to `utils/gtse_targets.py` and gained an optional `mask`.

With identical weights and inputs, the outputs match the pre-change code exactly (maximum difference 0.0):

- the occupancy model, against the committed version, with all five action-injection modes;
- the GTSE-b/r bilinear model;
- `epsilon_target`.

## Limitations

- All hyperparameters are first guesses; none has been tuned on BuilderBench.
- Transitions into a terminal state (a cube leaving the workspace) are not trained on: auto-reset overwrites the terminal observation, and those transitions cross an episode boundary.
- There is no marginal network; the critic is the conditional term only, as in the MetaWorld GTSE ports.
- Not ported from `dre_iclr`: the actor, Beta time sampling, `tmean` / `tstd` input normalization (fixed workspace scaling is used instead), and Legendre quadrature.
- With `permutation_invariant_reward=false` BuilderBench matches cubes to targets by index, but the model cannot tell cubes apart, so it only fits the default permutation-invariant setting.
