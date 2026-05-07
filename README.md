# MIRACL: Hierarchical Diversity-Aware Meta-Reinforcement Learning for Sustainable Multi-Echelon Supply Chain Optimisation

This repository contains the supplementary implementation for **MIRACL**: a hierarchical, diversity-aware meta-reinforcement learning framework for multi-objective combinatorial supply chain optimisation.

MIRACL combines gradient-based meta-reinforcement learning with multi-objective reward scalarisation and a Pareto Simulated Annealing (PSA)-based diversity mechanism. The framework is designed to learn a reusable meta-policy that can be quickly adapted to unseen supply chain tasks and different stakeholder preference weights.

## Overview

Multi-echelon supply chain optimisation is challenging because decisions across suppliers, manufacturers, distributors, and markets interact over time. Optimising such systems often requires balancing conflicting objectives, such as:

- profit or operational cost,
- greenhouse gas emissions,
- service-level equity or inequality.

MIRACL formulates this as a multi-objective reinforcement learning problem and uses meta-learning to improve adaptation across different supply chain configurations.

The repository includes:

- MIRACL and Meta-MORL algorithm components,
- a simple supply chain environment interface,
- random task/state generation utilities,
- meta-training and fine-tuning scripts,
- plotting and evaluation utilities.

## Important note on executability

> ⚠️ The supply chain environment depends on the proprietary `messiah` module, developed in collaboration with industrial partner. Due to confidentiality restrictions, this module cannot be released publicly.

As a result, the full supply chain experiments are **not directly executable** without access to `messiah`. Researchers may adapt the algorithm to open environments such as **OR-Gym (provided in this repository)** or other Gymnasium-compatible supply chain simulators.

However, the repository still provides the main algorithmic logic, environment interfaces, training structure, and evaluation workflow for inspection, adaptation, and reproducibility guidance. The simple environment is included as a template, while the moderate and complex environments follow the same structure with different network scales and parameter settings.



## Repository structure

```text
miracl/
├── algorithms/
│   ├── miracl.py
│   ├── maml_loc.py
│   ├── maml_tf_policy.py
│   ├── maml_torch_policy.py
│   └── maml_torch_policy_v2.py
│
├── environments/
│   ├── simple_environment.py
│   └── state_generator.py
│
├── fine_tuning/
│   ├── evaluate_normalise.py
│   ├── simple_test_env.py
│   └── test_state.py
│
├── metatraining/
│   └── train_simple.ipynb
│
├── or_gym_env/
│
├── utils/
│   └── plot_utils.py
│
├── data_input_v2.xlsx
├── LICENSE
├── README.md
└── __init__.py
```

## Main components

### `algorithms/`

Contains the MIRACL and MAML-style meta-reinforcement learning implementation. The main file is:

```text
algorithms/miracl.py
```

This includes:

- MIRACL configuration,
- meta-policy update logic,
- inner-loop adaptation,
- outer-loop meta-update,
- diversity-aware weight adjustment.

### `environments/`

Contains the supply chain environment interface and task generator.

```text
environments/simple_environment.py
environments/state_generator.py
```

The simple environment defines a Gymnasium-compatible supply chain environment with vector rewards for multi-objective optimisation. The state generator creates randomised supply chain task instances for meta-training.

### `metatraining/`

Contains the notebook for training the MIRACL/meta-policy on simple supply chain tasks.

```text
metatraining/train_simple.ipynb
```

### `fine_tuning/`

Contains scripts for adapting a trained meta-policy to a target task and evaluating the resulting solutions.

```text
fine_tuning/
```

These scripts are intended to adapt trained meta-policies to target supply chain tasks and evaluate multi-objective performance.

### `utils/`

Contains supporting plotting and analysis utilities.

```text
utils/plot_utils.py
```

## Method summary

MIRACL consists of three main stages:

1. **Meta-training**

   A meta-policy is trained across a distribution of supply chain tasks. Each task is decomposed into several scalarised subproblems using different objective-weight vectors.

2. **Diversity mechanism**

   Pareto Simulated Annealing is used to adjust scalarisation weights based on archived vector rewards. This encourages exploration of diverse Pareto-efficient regions.

3. **Fine-tuning**

   The trained meta-policy is adapted to an unseen supply chain task using multiple preference weights. The final solution set approximates the Pareto front for that task.

## Installation

Clone the repository:

```bash
git clone https://github.com/rifnyrachman/miracl.git
cd miracl
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install common dependencies:

```bash
pip install numpy pandas matplotlib gymnasium ray[rllib] stable-baselines3 openpyxl
```

If you have access to the proprietary `messiah` module, make sure it is installed or available in your `PYTHONPATH`.

Example:

```bash
export PYTHONPATH="/path/to/messiah:$PYTHONPATH"
```

## Usage

### Meta-training

The main meta-training workflow is provided in:

```text
metatraining/train_simple.ipynb
```

Open the notebook and run the cells sequentially after setting the correct paths and environment dependencies.

### Fine-tuning

Fine-tuning and evaluation scripts are provided in:

```text
fine_tuning/
```

These scripts are intended to adapt trained meta-policies to target supply chain tasks and evaluate multi-objective performance.

## Evaluation metrics

The MIRACL experiments evaluate Pareto-front approximation quality using common multi-objective indicators, including:

- **Hypervolume** — measures the dominated objective-space volume covered by the non-dominated solution set.
- **Sparsity** — measures the spread or spacing of solutions along the approximate Pareto front.
- **Expected Utility Metric (EUM)** — evaluates the expected utility of the solution set over a distribution of stakeholder preferences.

## Reproducibility limitations

The full industrial supply chain simulator cannot be released because of proprietary dependencies. This repository therefore provides:

- core MIRACL/meta-learning implementation,
- environment interface structure,
- simple environment template,
- task generation utilities,
- fine-tuning and evaluation workflow.

The moderate and complex supply chain settings used in the paper follow the same environment structure but differ in network size and configuration parameters.

## License

This repository is released under the MIT License. See:

```text
LICENSE
```

## Citation

If you use this repository or build upon MIRACL, please cite the associated paper once available.

```bibtex
@article{rachman2026miracl,
  title   = {MIRACL: Hierarchical Diversity-Aware Meta-Reinforcement Learning for Sustainable Multi-Echelon Supply Chain Optimisation},
  author  = {Rachman, Rifny and others},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Contact

For questions, issues, or collaboration inquiries, please open a GitHub issue or contact the repository maintainer.
