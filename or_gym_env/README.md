# OR-Gym Simple Supply Chain Environment for MIRACL

This directory contains an OR-Gym/Gymnasium-style replacement for the original Messiah-based simple supply-chain environment used in the MIRACL experiments. The environment is implemented in `simple_state.py` and is designed to work with the accompanying notebook `metah_s2_5000ts_orgym.ipynb`.

The goal is to provide a runnable, shareable simple supply-chain benchmark for meta-training, fine-tuning, and evaluation without requiring the proprietary Messiah simulator.

## Files

```text
simple_state.py                  # OR-Gym-style simple SC environment
metah_s2_5000ts_orgym.ipynb      # Notebook for MIRACL meta-training + SB3 fine-tuning
README.md                        # This file
```

Optional:

```text
data_input_v2.xlsx               # Optional workbook with SC costs, emissions, inventory, and demand
```

If `data_input_v2.xlsx` is unavailable, `simple_state.py` falls back to embedded simple-network values extracted from the original workbook.

## Environment summary

`simple_state.py` defines a simple multi-echelon supply-chain environment with:

- **7 nodes**: supplier, factories, retailers, and markets
- **10 total edges**, of which **8 are controllable action edges**
- **2 products**: raw and finished goods
- **3 objectives**: profit, negative GHG emissions, and negative service-level inequality
- **100 periods** by default
- OR-Gym/Gymnasium-compatible `reset()` and `step()` API

The raw vector reward follows the convention:

```python
[profit, -emission, -service_inequality]
```

This means emissions and service inequality are represented as negative rewards because the RL agent maximises all objectives. Values closer to zero are better for the second and third objectives.

Each `step()` returns both:

```python
info["mo_reward"]      # normalised vector reward used for scalarised training
info["mo_reward_raw"]  # raw vector reward for analysis/reporting
```

## Main classes

```python
SimpleSupplyChainEnv      # stochastic training environment
FTSimpleSupplyChainEnv    # fixed-demand/fixed-cost fine-tuning environment
SimpleState               # alias for SimpleSupplyChainEnv
TestSimpleState           # alias for FTSimpleSupplyChainEnv
make_env                  # factory for RLlib/SB3/Gymnasium usage
```

Use `SimpleState` / `fixed_demand=False` for meta-training and `TestSimpleState` / `fixed_demand=True` for fine-tuning or deterministic evaluation.

## Installation

Create an environment with the main dependencies:

```bash
pip install numpy pandas openpyxl gymnasium stable-baselines3 matplotlib pygmo
```

For MIRACL meta-training with RLlib, install the Ray/RLlib version used by your project, for example:

```bash
pip install "ray[rllib]==2.3.1"
```

The notebook also expects the MIRACL codebase containing:

```text
algo_ray/rllib/algorithms/maml/maml_psa.py
```

Set the base directory through the notebook or environment variable:

```bash
export MIRACL_BASE_DIR=/home/rifnyrachman7/_metamorl
export ORGYM_ENV_DIR=/path/to/folder/containing/simple_state.py
```

## Quick smoke test

Run from the directory containing `simple_state.py`:

```bash
python simple_state.py
```

Expected output includes the action and observation space shapes, plus one sample reward:

```text
simple_state.py: action_space= ... observation_space= ...
step reward= ... mo_reward= ... terminated= ...
```

You can also test inside Python:

```python
from simple_state import make_env

env = make_env({"fixed_demand": True, "num_periods": 100, "seed": 0})
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(env.action_space)
print(env.observation_space)
print(reward)
print(info["mo_reward"])
print(info["mo_reward_raw"])
```

## Using the environment with Stable-Baselines3

For SB3 fine-tuning, use the fixed-demand environment. It is usually safer to enable normalised actions, include demand in the observation, and start debugging with non-extreme scalarisation weights.

```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from simple_state import make_env

NUM_PERIODS = 100
weights = np.array([0.6, 0.2, 0.2], dtype=np.float64)

def make_sb3_env(weights, seed=None):
    env = make_env({
        "fixed_demand": True,
        "num_periods": NUM_PERIODS,
        "weights": weights,
        "seed": seed,
        "normalize_actions": True,
        "max_order_quantity": 200.0,
        "include_demand_in_obs": True,
    })
    return Monitor(env)

env = make_sb3_env(weights, seed=0)
model = PPO(
    MlpPolicy,
    env,
    learning_rate=1e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    seed=0,
)
model.learn(total_timesteps=5_000)
```

Important: when `fixed_demand=True`, do **not** also pass `randomise_costs=False` or `randomise_demand=False` to `make_env`. The fixed-demand class already sets these internally.

## Using the environment with RLlib

The notebook registers two RLlib environments:

```python
TRAIN_ENV_ID = "miracl_simple_train_orgym"
FT_ENV_ID = "miracl_simple_ft_orgym"
```

The training environment uses stochastic demand/cost settings:

```python
from gymnasium.wrappers import TimeLimit
from ray.tune.registry import register_env
from simple_state import make_env

NUM_PERIODS = 100

def make_train_env(env_config=None):
    cfg = dict(env_config or {})
    cfg.setdefault("fixed_demand", False)
    cfg.setdefault("num_periods", NUM_PERIODS)
    env = make_env(cfg)
    return TimeLimit(env, max_episode_steps=cfg["num_periods"])

register_env("miracl_simple_train_orgym", make_train_env)
```

Use this registered environment in the MIRACL/MAML configuration.

## Notebook workflow

The notebook `metah_s2_5000ts_orgym.ipynb` is organised as follows:

1. Set paths and runtime settings.
2. Import MIRACL, RLlib, SB3, and `simple_state.py`.
3. Run a smoke test for the OR-Gym-style environment.
4. Register RLlib environments.
5. Optionally run MIRACL/MAML meta-training.
6. Optionally run SB3 fine-tuning across 21 scalarisation weights.
7. Save and reload CSV results.
8. Compute hypervolume, sparsity, and EUM curves.
9. Plot performance curves.

The notebook uses safety flags to avoid accidentally launching long training runs:

```python
RUN_META_TRAINING = False
RUN_FINE_TUNING = False
```

Set them to `True` only when you want to run those stages.

For a quick debugging run, use a small budget and balanced weights:

```python
RUN_FINE_TUNING = True
USE_RLLIB_WARM_START = False

TOTAL_STEPS = 5_000
RECORD_EVERY = 1_000
RUNS = 1
EVAL_EPISODES = 3

ACTIVE_WEIGHT_GRID = np.array([
    [0.8, 0.1, 0.1],
    [0.6, 0.2, 0.2],
    [0.4, 0.3, 0.3],
], dtype=np.float64)
```

Then make sure the fine-tuning loop iterates over `ACTIVE_WEIGHT_GRID` rather than the full `WEIGHT_GRID_21`.

## Scalarisation weights

The full 21-weight grid used in the notebook is:

```python
WEIGHT_GRID_21 = np.array([
    [0.0, 0.0, 1.0], [0.0, 0.2, 0.8], [0.0, 0.4, 0.6], [0.0, 0.6, 0.4], [0.0, 0.8, 0.2],
    [0.0, 1.0, 0.0], [0.2, 0.0, 0.8], [0.2, 0.2, 0.6], [0.2, 0.4, 0.4], [0.2, 0.6, 0.2],
    [0.2, 0.8, 0.0], [0.4, 0.0, 0.6], [0.4, 0.2, 0.4], [0.4, 0.4, 0.2], [0.4, 0.6, 0.0],
    [0.6, 0.0, 0.4], [0.6, 0.2, 0.2], [0.6, 0.4, 0.0], [0.8, 0.0, 0.2], [0.8, 0.2, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)
```

For debugging, avoid starting with extreme weights such as `[0, 0, 1]`, because the policy ignores profit and emissions under that scalarisation. Use balanced weights first to verify learning behaviour.

## Evaluation

A useful evaluator should track both normalised and raw returns:

```python
def evaluate_model(model, weights, n_eval_episodes=3, seed=123):
    scalar_returns = []
    raw_returns = []
    norm_returns = []

    for ep in range(n_eval_episodes):
        env = make_sb3_env(weights, seed=seed + ep)
        obs, info = env.reset(seed=seed + ep)
        done = False
        scalar_total = 0.0
        raw_total = np.zeros(3, dtype=np.float64)
        norm_total = np.zeros(3, dtype=np.float64)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            scalar_total += float(reward)
            raw_total += np.asarray(info["mo_reward_raw"], dtype=np.float64)
            norm_total += np.asarray(info["mo_reward"], dtype=np.float64)

        scalar_returns.append(scalar_total)
        raw_returns.append(raw_total)
        norm_returns.append(norm_total)

    return {
        "scalar_mean": float(np.mean(scalar_returns)),
        "raw_mean": np.mean(raw_returns, axis=0),
        "norm_mean": np.mean(norm_returns, axis=0),
    }
```

Use `raw_mean` for reporting business quantities and `norm_mean` / `scalar_mean` to diagnose whether PPO is improving the reward it actually trains on.

## Common issues

### `NameError: WEIGHT_GRID is not defined`

The notebook uses `WEIGHT_GRID_21`, not `WEIGHT_GRID`. For debugging, define a separate grid:

```python
ACTIVE_WEIGHT_GRID = WEIGHT_GRID_21[:3].copy()
```

and change the fine-tuning loop to:

```python
for weight_id, weights in enumerate(ACTIVE_WEIGHT_GRID):
    ...
```

### `TypeError: build_config() got multiple values for keyword argument 'randomise_costs'`

This happens when passing `randomise_costs` or `randomise_demand` while also using `fixed_demand=True`. Remove those duplicate arguments from the notebook call.

### Results are flat or all policies behave the same

Check these first:

- Set `normalize_actions=True`.
- Reduce `max_order_quantity` to 100 or 200 for debugging.
- Use balanced weights such as `[0.6, 0.2, 0.2]`.
- Disable warm-starting first: `USE_RLLIB_WARM_START = False`.
- Log action statistics to check whether the policy is producing non-zero actions.
- Evaluate every 1,000 steps rather than every 250 steps when using PPO rollouts.

### Rewards look worse because they are negative

The raw vector is `[profit, -emission, -service_inequality]`. For the second and third objectives, values closer to zero are better. Large negative emissions or inequality values mean worse environmental or service-equity performance.

## Reproducibility notes

- The environment can run without `data_input_v2.xlsx` using embedded simple-network parameters.
- If the workbook is available, place it next to `simple_state.py` or pass `input_file` to `make_env`.
- Use fixed seeds for deterministic debugging.
- Meta-training checkpoints are stored under `CKPT_DIR` in the notebook.
- Fine-tuning CSVs are stored under `FT_DIR` and performance metrics under `PERF_DIR`.

