# OR-Gym-style Supply Chain Environments

This folder contains three standalone Python environment files that replace the previous Messiah-based supply-chain state generators with Gymnasium / OR-Gym-style environments.

The environments are built around your Excel workbook `data_input_v2.xlsx` and keep the same simple, moderate, and complex network structures used in your MIRACL experiments.

## Files

```text
.
├── data_input_v2.xlsx          # Excel workbook with costs, emissions, inventory, and demand
├── simple_state.py             # Simple supply-chain environment
├── moderate_state.py           # Moderate supply-chain environment
├── complex_state.py            # Complex supply-chain environment
└── README.md                   # This guide
```

Each `.py` file is standalone. You can copy only the environment file you need, but keeping `data_input_v2.xlsx` beside it is recommended so the latest Excel parameters are used.

## What each file provides

| File | Meta-training class | Fine-tuning class | Aliases | Action size | Default observation size |
|---|---:|---:|---|---:|---:|
| `simple_state.py` | `SimpleSupplyChainEnv` | `FTSimpleSupplyChainEnv` | `SimpleState`, `TestSimpleState` | 8 | 24 |
| `moderate_state.py` | `ModerateSupplyChainEnv` | `FTModerateSupplyChainEnv` | `ModerateState`, `TestModerateState` | 21 | 58 |
| `complex_state.py` | `ComplexSupplyChainEnv` | `FTComplexSupplyChainEnv` | `ComplexState`, `TestComplexState` | 59 | 150 |

The `SimpleState`, `ModerateState`, and `ComplexState` aliases are stochastic training variants. The `TestSimpleState`, `TestModerateState`, and `TestComplexState` aliases are fixed-demand / fixed-cost fine-tuning variants.

## Excel parameter mapping

The environments read the following Excel sheets:

| Environment | Parameter sheet | Demand sheet |
|---|---|---|
| Simple | `Parameters_simple` | `Data Demand` |
| Moderate | `Parameters` | `Data Demand` |
| Complex | `Parameters_complex` | `Data Demand` |

The code reads:

- `Cost_process` for edge movement / production costs
- `GHG_Unit` for edge and node emissions
- `Cost_Inv` for inventory holding costs
- `Initial_Inv` for initial finished-goods inventory
- `Demand A`, `Demand B`, `Demand C`, `Demand D`, `Demand E` from `Data Demand`

The number of demand columns used depends on the environment:

- Simple uses `Demand A` and `Demand B`
- Moderate uses `Demand A`, `Demand B`, and `Demand C`
- Complex uses `Demand A` through `Demand E`

If the Excel file is not found, each file falls back to embedded values extracted from the uploaded workbook. For reproducible experiments using updated data, keep `data_input_v2.xlsx` in the same folder as the `.py` files or pass its path explicitly with `input_file="..."`.

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install numpy pandas openpyxl gymnasium
```

For Stable-Baselines3 training, also install:

```bash
pip install stable-baselines3
```

For RLlib training, install the Ray version used by your project, for example:

```bash
pip install "ray[rllib]==2.3.1"
```

The files include a small import fallback for basic static checks, but real RL training should use `gymnasium`.

## Quick smoke test

From the folder containing the files, run:

```bash
python simple_state.py
python moderate_state.py
python complex_state.py
```

Expected output should look similar to:

```text
simple_state.py: action_space= (8,) observation_space= (24,)
step reward= ... mo_reward= ... terminated= False

moderate_state.py: action_space= (21,) observation_space= (58,)
step reward= ... mo_reward= ... terminated= False

complex_state.py: action_space= (59,) observation_space= (150,)
step reward= ... mo_reward= ... terminated= False
```

This checks that each file imports correctly, loads the Excel parameters or embedded fallback values, resets the environment, samples one action, and performs one step.

## Basic usage

### Simple environment

```python
from simple_state import SimpleState, TestSimpleState

# Stochastic meta-training variant
env = SimpleState(input_file="data_input_v2.xlsx", seed=0)

obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(obs.shape)
print(reward)
print(info["mo_reward"])
print(info["mo_reward_raw"])
```

### Moderate environment

```python
from moderate_state import ModerateState, TestModerateState

# Fixed fine-tuning variant
env = TestModerateState(input_file="data_input_v2.xlsx", seed=0)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(env.action_space.shape)       # (21,)
print(env.observation_space.shape)  # (58,)
```

### Complex environment

```python
from complex_state import ComplexState, TestComplexState

env = ComplexState(input_file="data_input_v2.xlsx", seed=0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(env.action_space.shape)       # (59,)
print(env.observation_space.shape)  # (150,)
```

## Factory usage

Each file includes a `make_env(...)` function. This is useful for RLlib or wrapper code where environments are constructed from a config dictionary.

```python
from simple_state import make_env

# Stochastic meta-training environment
env = make_env({
    "input_file": "data_input_v2.xlsx",
    "fixed_demand": False,
    "seed": 123,
})

# Fixed fine-tuning environment
env_ft = make_env({
    "input_file": "data_input_v2.xlsx",
    "fixed_demand": True,
    "num_periods": 100,
    "seed": 123,
})
```

`num_timesteps` is also accepted as an alias for `num_periods`:

```python
env = make_env({"num_timesteps": 100, "fixed_demand": True})
```

## Choosing training vs fine-tuning variants

Use the stochastic classes for meta-training:

```python
from simple_state import SimpleState
from moderate_state import ModerateState
from complex_state import ComplexState

env_simple = SimpleState(input_file="data_input_v2.xlsx")
env_moderate = ModerateState(input_file="data_input_v2.xlsx")
env_complex = ComplexState(input_file="data_input_v2.xlsx")
```

Use the fixed variants for fine-tuning / evaluation:

```python
from simple_state import TestSimpleState
from moderate_state import TestModerateState
from complex_state import TestComplexState

env_simple_ft = TestSimpleState(input_file="data_input_v2.xlsx")
env_moderate_ft = TestModerateState(input_file="data_input_v2.xlsx")
env_complex_ft = TestComplexState(input_file="data_input_v2.xlsx")
```

The fixed variants use the Excel demand series and turn off per-episode cost and demand randomisation.

## Important API differences from Messiah

These files are Gymnasium environments. They do **not** return a Messiah `State` object through `__call__()`.

Use this Gymnasium pattern instead:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

The `info` dictionary contains the multi-objective values and operational logs you usually need for MORL / MIRACL evaluation.

## Reward structure

The environment has three objectives:

```text
[profit, -emission, -service_inequality]
```

The scalar reward is computed as:

```python
scalar_reward = weights @ info["mo_reward"]
```

The default weights are uniform:

```python
[1/3, 1/3, 1/3]
```

You can set scalarisation weights using:

```python
env.set_scalarization_weights([0.6, 0.3, 0.1])
print(env.get_scalarization_weights())
```

Weights are clipped to be non-negative and normalised to sum to 1.

The step `info` dictionary includes:

```python
info["mo_reward"]                  # normalised vector reward
info["mo_reward_raw"]              # raw vector reward: [profit, -emission, -inequality]
info["weights"]                    # active scalarisation weights
info["profit"]                     # current-step profit
info["emission"]                   # current-step positive emission value
info["service_inequality"]         # current-step positive inequality value
info["cumulative_profit"]
info["cumulative_emission"]
info["cumulative_service_inequality"]
info["inventory"]
info["fulfilled"]
info["unfulfilled"]
info["edge_inputs"]
info["edge_outputs"]
```

## Action format

By default, actions are continuous order quantities with shape equal to the controllable edge count.

```python
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

The market-demand edges are **not** part of the action space. They are handled internally using the Excel or stochastic demand schedule.

You can also pass a dictionary keyed by controllable edge name:

```python
action = {
    "supplier0_factory1": 100,
    "supplier0_factory2": 80,
    "factory1_factory1": 40,
}
obs, reward, terminated, truncated, info = env.step(action)
```

Or by original edge index:

```python
action = {0: 100, 1: 80, 2: 40}
obs, reward, terminated, truncated, info = env.step(action)
```

Missing controllable edges are treated as zero orders.

## Normalised actions

If you prefer policy outputs in `[0, 1]`, create the environment with `normalize_actions=True`:

```python
env = SimpleState(
    input_file="data_input_v2.xlsx",
    normalize_actions=True,
    max_order_quantity=500,
)
```

The environment will multiply actions by `max_order_quantity` internally.

## Stable-Baselines3 example

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from simple_state import TestSimpleState

env = TestSimpleState(
    input_file="data_input_v2.xlsx",
    normalize_actions=True,
    max_order_quantity=500,
)

env.set_scalarization_weights([0.6, 0.3, 0.1])
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000)
```

For moderate or complex environments, import `TestModerateState` or `TestComplexState` instead.

## RLlib example

```python
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from moderate_state import make_env

register_env("moderate_sc_orgym", lambda env_config: make_env(env_config))

config = (
    PPOConfig()
    .environment(
        env="moderate_sc_orgym",
        env_config={
            "input_file": "data_input_v2.xlsx",
            "fixed_demand": False,
            "normalize_actions": True,
            "max_order_quantity": 500,
        },
    )
    .framework("torch")
)

algo = config.build()
result = algo.train()
print(result["episode_reward_mean"])
```

For fine-tuning, set `fixed_demand=True` in the `env_config`.

## Task sampling for meta-learning

The environments include lightweight task utilities:

```python
env = SimpleState(input_file="data_input_v2.xlsx", seed=0)

tasks = env.sample_tasks(5)
env.set_task(tasks[0])
```

Currently, `set_task(...)` applies:

- `task["demand"]`, if supplied
- `task["weights"]`, if supplied

This is useful for MIRACL-style task switching where each task corresponds to a demand schedule and/or scalarisation preference.

Example:

```python
task = {
    "demand": tasks[0]["demand"],
    "weights": [0.5, 0.4, 0.1],
}

env.set_task(task)
obs, info = env.reset()
```

## Accessing topology and Excel-derived state configuration

Each environment exposes a `get_state_config()` helper for backward-friendly access to topology arrays and parameters:

```python
env = TestSimpleState(input_file="data_input_v2.xlsx")
state_cfg = env.get_state_config()

print(state_cfg["names"]["node_names"])
print(state_cfg["edge_upstream_nodes"])
print(state_cfg["edge_cost"])
print(state_cfg["demand"])
```

This is not a Messiah `State`, but it gives you the main topology and parameter arrays if older wrappers need them.

## Reproducibility

Set seeds at construction and reset:

```python
env = ModerateState(input_file="data_input_v2.xlsx", seed=42)
obs, info = env.reset(seed=42)
```

For fixed evaluation, use the `Test*State` classes:

```python
env = TestModerateState(input_file="data_input_v2.xlsx", seed=42)
```

For stochastic meta-training, use the non-test classes:

```python
env = ModerateState(input_file="data_input_v2.xlsx", seed=42)
```

## Common configuration arguments

| Argument | Meaning | Default |
|---|---|---|
| `input_file` | Path to `data_input_v2.xlsx` | auto-detected / embedded fallback |
| `num_periods` | Episode length | `100` |
| `num_timesteps` | Alias for `num_periods` in `make_env` | `100` |
| `seed` | Random seed | `None` |
| `fixed_demand` | Use Excel demand and fixed costs | depends on class |
| `randomise_costs` | Randomise costs/emissions each episode | `True` for training, `False` for fixed variants |
| `randomise_demand` | Randomise demand each episode | `True` for training, `False` for fixed variants |
| `normalize_actions` | Use action range `[0, 1]` | `False` |
| `max_order_quantity` | Maximum order quantity per controllable edge | `500` |
| `backlog` | Carry unmet demand forward | `False` |
| `shortage_penalty` | Penalty per unfulfilled unit | `1.0` |
| `include_demand_in_obs` | Append next demand to observation | `False` |
| `include_metrics_in_obs` | Append cumulative emission and inequality to observation | `False` |
| `weights` | Scalarisation weights for the 3 objectives | uniform |
| `reward_clip` | Optional `(low, high)` scalar reward clipping | `None` |

## Troubleshooting

### `ModuleNotFoundError: No module named 'gymnasium'`

Install Gymnasium:

```bash
pip install gymnasium
```

### `ImportError: Missing optional dependency 'openpyxl'`

Install OpenPyXL so pandas can read `.xlsx` files:

```bash
pip install openpyxl
```

### Excel file not being used

Pass the workbook path explicitly:

```python
env = SimpleState(input_file="/full/path/to/data_input_v2.xlsx")
```

### Old code expects `env.reset()` to return only `obs`

Gymnasium returns two values:

```python
obs, info = env.reset()
```

If a legacy wrapper expects only the observation, use:

```python
obs = env.reset()[0]
```

### Old code expects four values from `step(...)`

Gymnasium returns five values:

```python
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

### Action shape error

Use the environment action space shape:

```python
print(env.action_space.shape)
action = env.action_space.sample()
```

Expected action sizes:

- Simple: `(8,)`
- Moderate: `(21,)`
- Complex: `(59,)`

### Negative emissions in `mo_reward_raw`

This is intentional. The vector reward uses a maximisation convention:

```text
[profit, -emission, -service_inequality]
```

The positive emission value is available separately as:

```python
info["emission"]
```

## Suggested project placement

A clean setup for your MIRACL experiments would be:

```text
_miracl/
├── data_input_v2.xlsx
├── envs/
│   ├── simple_state.py
│   ├── moderate_state.py
│   └── complex_state.py
└── train.py
```

Then import with:

```python
from envs.simple_state import SimpleState, TestSimpleState
from envs.moderate_state import ModerateState, TestModerateState
from envs.complex_state import ComplexState, TestComplexState
```

If `envs/` is not already a package, add an empty file:

```bash
touch envs/__init__.py
```

## Minimal end-to-end example

```python
from simple_state import TestSimpleState

weights = [0.5, 0.3, 0.2]

env = TestSimpleState(
    input_file="data_input_v2.xlsx",
    normalize_actions=True,
    max_order_quantity=500,
    seed=0,
)
env.set_scalarization_weights(weights)

obs, info = env.reset(seed=0)
done = False
episode_return = 0.0
vector_return = [0.0, 0.0, 0.0]

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_return += reward
    vector_return = [a + b for a, b in zip(vector_return, info["mo_reward_raw"])]

print("scalar return:", episode_return)
print("raw vector return [profit, -emission, -inequality]:", vector_return)
print("final cumulative profit:", info["cumulative_profit"])
print("final cumulative emission:", info["cumulative_emission"])
print("final cumulative inequality:", info["cumulative_service_inequality"])
```
