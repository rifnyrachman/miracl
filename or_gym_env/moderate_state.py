"""
OR-Gym-style Moderate supply-chain environment generated from data_input_v2.xlsx.

This file replaces the Messiah State/Generator dependency for the moderate
network.  It keeps the uploaded topology, reads monetary costs, GHG factors,
initial inventories, and fixed demand from the Excel workbook, and falls back to
embedded values extracted from the workbook uploaded in this ChatGPT session.

Key classes
-----------
- ModerateSupplyChainEnv: stochastic training variant, equivalent to the original ModerateState.
- FTModerateSupplyChainEnv: fixed-demand/fixed-cost fine-tuning variant, equivalent to TestModerateState.
- ModerateState: alias of ModerateSupplyChainEnv.
- TestModerateState: alias of FTModerateSupplyChainEnv.

The action space excludes demand/market edges and therefore has dimension
21.  Demand edges are handled internally using the Excel demand
series and market prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import random
import warnings

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except Exception:  # minimal fallback for static/import tests
        class _MiniBox:
            def __init__(self, low, high, dtype=np.float32):
                self.low = np.asarray(low, dtype=dtype)
                self.high = np.asarray(high, dtype=dtype)
                self.dtype = dtype
                self.shape = self.low.shape

            def sample(self):
                finite_low = np.where(np.isfinite(self.low), self.low, 0.0)
                finite_high = np.where(np.isfinite(self.high), self.high, 1.0)
                return np.random.uniform(finite_low, finite_high).astype(self.dtype)

        class _MiniSpaces:
            Box = _MiniBox

        class _MiniEnv:
            metadata = {}
            def reset(self, *args, **kwargs):
                return None

        class _MiniGym:
            Env = _MiniEnv

        gym = _MiniGym()
        spaces = _MiniSpaces()


PRODUCT_RAW = 0
PRODUCT_FINISHED = 1
NUM_PRODUCTS = 2
NUM_OBJECTIVES = 3
NUM_PERIODS_DEFAULT = 100
SHEET_NAME = 'Parameters'
NUM_SUPPLIERS = 2
NUM_MARKETS = 3
NUM_CONTROL_EDGES = 21

# Runtime lookup order: adjacent workbook, working directory, uploaded sandbox,
# and the path used in your existing MERLION/MIRACL scripts.
_DEFAULT_INPUT_CANDIDATES = [
    Path(__file__).with_name("data_input_v2.xlsx"),
    Path.cwd() / "data_input_v2.xlsx",
    Path("/mnt/data/data_input_v2.xlsx"),
    Path("/home/rifnyrachman7/_merlion/data_input_v2.xlsx"),
    Path("merlion/data_input_v2.xlsx"),
]

NODE_NAMES = ['supplier0', 'supplier1', 'factory2', 'factory3', 'factory4', 'warehouse5', 'warehouse6', 'retailer7', 'retailer8', 'retailer9', 'market10', 'market11', 'market12']
EDGE_NAMES = ['supplier0_factory2', 'supplier0_factory3', 'supplier0_factory4', 'supplier1_factory2', 'supplier1_factory3', 'supplier1_factory4', 'factory2_factory2', 'factory3_factory3', 'factory4_factory4', 'factory2_warehouse5', 'factory2_warehouse6', 'factory3_warehouse5', 'factory3_warehouse6', 'factory4_warehouse5', 'factory4_warehouse6', 'warehouse5_retailer7', 'warehouse5_retailer8', 'warehouse5_retailer9', 'warehouse6_retailer7', 'warehouse6_retailer8', 'warehouse6_retailer9', 'retailer7_market10', 'retailer8_market11', 'retailer9_market12']
NODE_TAGS = ['supplier', 'supplier', 'factory', 'factory', 'factory', 'warehouse', 'warehouse', 'retailer', 'retailer', 'retailer', 'market', 'market', 'market']
EDGE_TAGS = ['supply', 'supply', 'supply', 'supply', 'supply', 'supply', 'production', 'production', 'production', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'distribution', 'demand', 'demand', 'demand']
EDGE_UPSTREAM_NODES = np.array([0, 0, 0, 1, 1, 1, 2, 3, 4, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9], dtype=np.int64)
EDGE_DOWNSTREAM_NODES = np.array([2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6, 5, 6, 7, 8, 9, 7, 8, 9, 10, 11, 12], dtype=np.int64)
EDGE_INPUT_PRODUCTS = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64)
EDGE_OUTPUT_PRODUCTS = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64)
NODE_CONTROL = np.array([False, False, True, True, True, True, True, True, True, True, False, False, False], dtype=bool)
MARKET_PRICES = np.array([20.0, 21.0, 20.5], dtype=np.float64)

# Embedded fallback values extracted from the uploaded data_input_v2.xlsx.
EMBEDDED_EDGE_COST = np.array([0.22, 0.6900000000000001, 0.5650000000000001, 1.055, 0.65, 0.63, 2.0, 2.2, 2.3, 0.075, 0.43, 0.63, 0.23, 0.495, 0.075, 1.095, 0.625, 0.9500000000000001, 1.6400000000000001, 1.16, 0.58, 0.0, 0.0, 0.0], dtype=np.float64)
EMBEDDED_EDGE_EMISSION = np.array([0.12583999999999998, 0.39468, 0.32317999999999997, 0.6034599999999999, 0.37179999999999996, 0.36035999999999996, 5.012557999999999, 4.575442, 5.449104, 0.042899999999999994, 0.24595999999999998, 0.36035999999999996, 0.13155999999999998, 0.28313999999999995, 0.042899999999999994, 0.6263399999999999, 0.3575, 0.5434, 0.9380799999999999, 0.6635199999999999, 0.33175999999999994, 0.0, 0.0, 0.0], dtype=np.float64)
EMBEDDED_NODE_COST = np.array([0.0, 0.0, 0.11, 0.13, 0.12, 0.15, 0.2, 0.25, 0.3, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
EMBEDDED_NODE_EMISSION = np.array([0.0, 0.0, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019, 0.0, 0.0, 0.0], dtype=np.float64)
EMBEDDED_INITIAL_INVENTORY = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 380.0], [0.0, 350.0], [0.0, 400.0], [0.0, 80.0], [0.0, 110.0], [0.0, 100.0], [0.0, 80.0], [0.0, 120.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
EMBEDDED_DEMAND = np.array([[149.0, 232.0, 139.0, 176.0, 15.0, 49.0, 170.0, 158.0, 205.0, 81.0, 133.0, 0.0, 100.0, 124.0, 144.0, 117.0, 207.0, 63.0, 73.0, 178.0, 127.0, 230.0, 191.0, 170.0, 47.0, 113.0, 166.0, 111.0, 167.0, 168.0, 100.0, 83.0, 208.0, 153.0, 153.0, 208.0, 254.0, 0.0, 237.0, 210.0, 132.0, 221.0, 145.0, 246.0, 218.0, 175.0, 113.0, 155.0, 97.0, 108.0, 161.0, 142.0, 110.0, 249.0, 117.0, 148.0, 150.0, 19.0, 307.0, 115.0, 194.0, 21.0, 140.0, 52.0, 120.0, 226.0, 197.0, 187.0, 180.0, 45.0, 82.0, 183.0, 232.0, 177.0, 184.0, 183.0, 192.0, 179.0, 84.0, 208.0, 96.0, 62.0, 158.0, 70.0, 85.0, 156.0, 101.0, 97.0, 177.0, 217.0, 194.0, 204.0, 61.0, 189.0, 149.0, 107.0, 132.0, 127.0, 87.0, 33.0], [0.0, 0.0, 133.0, 37.0, 140.0, 129.0, 77.0, 127.0, 120.0, 140.0, 129.0, 156.0, 204.0, 64.0, 202.0, 158.0, 286.0, 154.0, 126.0, 26.0, 121.0, 56.0, 118.0, 106.0, 56.0, 47.0, 0.0, 0.0, 0.0, 0.0, 0.0, 89.0, 33.0, 103.0, 108.0, 88.0, 104.0, 86.0, 154.0, 97.0, 111.0, 276.0, 216.0, 232.0, 54.0, 180.0, 177.0, 116.0, 141.0, 116.0, 175.0, 70.0, 68.0, 126.0, 131.0, 113.0, 0.0, 0.0, 0.0, 0.0, 0.0, 81.0, 182.0, 82.0, 107.0, 56.0, 68.0, 98.0, 67.0, 105.0, 132.0, 210.0, 202.0, 298.0, 160.0, 112.0, 68.0, 13.0, 12.0, 140.0, 185.0, 85.0, 104.0, 82.0, 138.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 188.0, 154.0, 108.0, 0.0, 101.0, 130.0, 115.0, 74.0, 181.0], [165.0, 202.0, 191.0, 203.0, 173.0, 190.0, 188.0, 193.0, 212.0, 209.0, 179.0, 207.0, 222.0, 198.0, 166.0, 182.0, 193.0, 209.0, 220.0, 230.0, 164.0, 192.0, 210.0, 220.0, 203.0, 227.0, 191.0, 170.0, 182.0, 199.0, 209.0, 189.0, 198.0, 191.0, 195.0, 172.0, 221.0, 204.0, 225.0, 215.0, 199.0, 201.0, 225.0, 208.0, 208.0, 178.0, 197.0, 186.0, 195.0, 188.0, 199.0, 178.0, 194.0, 195.0, 178.0, 194.0, 213.0, 218.0, 198.0, 192.0, 197.0, 197.0, 219.0, 175.0, 206.0, 200.0, 195.0, 193.0, 236.0, 226.0, 191.0, 182.0, 199.0, 207.0, 191.0, 183.0, 208.0, 176.0, 218.0, 193.0, 172.0, 197.0, 227.0, 217.0, 197.0, 200.0, 175.0, 201.0, 212.0, 188.0, 202.0, 194.0, 202.0, 194.0, 207.0, 201.0, 212.0, 202.0, 214.0, 195.0]], dtype=np.float64)


@dataclass
class SCNetworkConfig:
    name: str = 'moderate'
    num_periods: int = NUM_PERIODS_DEFAULT
    input_file: Optional[str] = None
    fixed_demand: bool = False
    randomise_costs: bool = True
    randomise_demand: bool = True
    seed: Optional[int] = None
    max_order_quantity: float = 500.0
    backlog: bool = False
    shortage_penalty: float = 1.0
    include_demand_in_obs: bool = False
    include_metrics_in_obs: bool = False
    reward_scale_profit: Optional[float] = None
    reward_scale_emission: Optional[float] = None
    reward_scale_inequality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    node_names: List[str] = field(default_factory=lambda: list(NODE_NAMES))
    edge_names: List[str] = field(default_factory=lambda: list(EDGE_NAMES))
    node_tags: List[str] = field(default_factory=lambda: list(NODE_TAGS))
    edge_tags: List[str] = field(default_factory=lambda: list(EDGE_TAGS))
    edge_upstream_nodes: np.ndarray = field(default_factory=lambda: EDGE_UPSTREAM_NODES.copy())
    edge_downstream_nodes: np.ndarray = field(default_factory=lambda: EDGE_DOWNSTREAM_NODES.copy())
    edge_input_products: np.ndarray = field(default_factory=lambda: EDGE_INPUT_PRODUCTS.copy())
    edge_output_products: np.ndarray = field(default_factory=lambda: EDGE_OUTPUT_PRODUCTS.copy())
    node_control: np.ndarray = field(default_factory=lambda: NODE_CONTROL.copy())
    price_market: np.ndarray = field(default_factory=lambda: MARKET_PRICES.copy())
    edge_cost: np.ndarray = field(default_factory=lambda: EMBEDDED_EDGE_COST.copy())
    edge_emission: np.ndarray = field(default_factory=lambda: EMBEDDED_EDGE_EMISSION.copy())
    node_cost: np.ndarray = field(default_factory=lambda: EMBEDDED_NODE_COST.copy())
    node_emission: np.ndarray = field(default_factory=lambda: EMBEDDED_NODE_EMISSION.copy())
    initial_inventory: np.ndarray = field(default_factory=lambda: EMBEDDED_INITIAL_INVENTORY.copy())
    demand: Optional[np.ndarray] = None
    lead_times: np.ndarray = field(default_factory=lambda: np.full(len(EDGE_NAMES), 2, dtype=np.int64))

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return len(self.edge_names)

    @property
    def demand_edge_indices(self) -> np.ndarray:
        return np.array([i for i, tag in enumerate(self.edge_tags) if tag == "demand"], dtype=np.int64)

    @property
    def controllable_edge_indices(self) -> np.ndarray:
        return np.array([i for i, tag in enumerate(self.edge_tags) if tag != "demand"], dtype=np.int64)

    @property
    def inventory_node_indices(self) -> np.ndarray:
        return np.array(
            [i for i, tag in enumerate(self.node_tags) if tag not in {"supplier", "market"}],
            dtype=np.int64,
        )

    def validate(self) -> None:
        if self.edge_upstream_nodes.shape != (self.num_edges,):
            raise ValueError("edge_upstream_nodes length does not match edge_names")
        if self.edge_downstream_nodes.shape != (self.num_edges,):
            raise ValueError("edge_downstream_nodes length does not match edge_names")
        if self.edge_input_products.shape != (self.num_edges, NUM_PRODUCTS):
            raise ValueError("edge_input_products has wrong shape")
        if self.edge_output_products.shape != (self.num_edges, NUM_PRODUCTS):
            raise ValueError("edge_output_products has wrong shape")
        if self.node_control.shape != (self.num_nodes,):
            raise ValueError("node_control length does not match node_names")
        if len(self.demand_edge_indices) != NUM_MARKETS:
            raise ValueError("number of demand edges does not match NUM_MARKETS")
        if self.initial_inventory.shape != (self.num_nodes, NUM_PRODUCTS):
            raise ValueError("initial_inventory has wrong shape")
        if self.edge_cost.shape != (self.num_edges,):
            raise ValueError("edge_cost has wrong shape")
        if self.edge_emission.shape != (self.num_edges,):
            raise ValueError("edge_emission has wrong shape")
        if self.node_cost.shape != (self.num_nodes,):
            raise ValueError("node_cost has wrong shape")
        if self.node_emission.shape != (self.num_nodes,):
            raise ValueError("node_emission has wrong shape")


def _resolve_input_file(input_file: Optional[str]) -> Optional[Path]:
    if input_file:
        path = Path(input_file).expanduser()
        if path.exists():
            return path
        warnings.warn(f"Input workbook not found: {path}. Using embedded Excel-derived parameters.")
        return None
    for candidate in _DEFAULT_INPUT_CANDIDATES:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate
    return None


def _as_nonnegative_float_array(values: Sequence[Any], length: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    if length is not None:
        if len(arr) < length:
            arr = np.pad(arr, (0, length - len(arr)), constant_values=0.0)
        elif len(arr) > length:
            arr = arr[:length]
    return arr.astype(np.float64)


def _load_excel_parameters(input_file: Optional[str], num_periods: int, fixed_demand: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read costs, emissions, initial inventory, and demand from data_input_v2.xlsx.

    Falls back to embedded values extracted from the uploaded workbook if the
    Excel file or pandas/openpyxl stack is unavailable.
    """
    path = _resolve_input_file(input_file)
    if path is None or pd is None:
        demand = EMBEDDED_DEMAND[:, :num_periods].copy() if fixed_demand else None
        return (
            EMBEDDED_EDGE_COST.copy(),
            EMBEDDED_EDGE_EMISSION.copy(),
            EMBEDDED_NODE_COST.copy(),
            EMBEDDED_NODE_EMISSION.copy(),
            EMBEDDED_INITIAL_INVENTORY.copy(),
            demand,
        )

    try:
        df = pd.read_excel(path, sheet_name=SHEET_NAME)
        edge_cost = _as_nonnegative_float_array(df["Cost_process"].iloc[:NUM_CONTROL_EDGES], NUM_CONTROL_EDGES)
        edge_emission = _as_nonnegative_float_array(df["GHG_Unit"].iloc[:NUM_CONTROL_EDGES], NUM_CONTROL_EDGES)
        edge_cost = np.concatenate([edge_cost, np.zeros(NUM_MARKETS, dtype=np.float64)])
        edge_emission = np.concatenate([edge_emission, np.zeros(NUM_MARKETS, dtype=np.float64)])

        node_cost_mid = _as_nonnegative_float_array(df["Cost_Inv"].iloc[NUM_CONTROL_EDGES:])
        node_emission_mid = _as_nonnegative_float_array(df["GHG_Unit"].iloc[NUM_CONTROL_EDGES:])
        init_mid = _as_nonnegative_float_array(df["Initial_Inv"].iloc[NUM_CONTROL_EDGES:])

        node_cost = np.zeros(len(NODE_NAMES), dtype=np.float64)
        node_emission = np.zeros(len(NODE_NAMES), dtype=np.float64)
        initial_inventory = np.zeros((len(NODE_NAMES), NUM_PRODUCTS), dtype=np.float64)

        inventory_nodes = list(range(NUM_SUPPLIERS, len(NODE_NAMES) - NUM_MARKETS))
        for node_idx, inv, c_inv, e_inv in zip(inventory_nodes, init_mid, node_cost_mid, node_emission_mid):
            initial_inventory[node_idx, PRODUCT_FINISHED] = inv
            node_cost[node_idx] = c_inv
            node_emission[node_idx] = e_inv

        demand = None
        if fixed_demand:
            df_demand = pd.read_excel(path, sheet_name="Data Demand")
            demand_cols = ["Demand A", "Demand B", "Demand C", "Demand D", "Demand E"][:NUM_MARKETS]
            demand = np.zeros((NUM_MARKETS, num_periods), dtype=np.float64)
            for i, col in enumerate(demand_cols):
                if col in df_demand.columns:
                    vals = _as_nonnegative_float_array(df_demand[col].to_numpy())
                    if len(vals) < num_periods:
                        pad_value = vals[-1] if len(vals) else 0.0
                        vals = np.pad(vals, (0, num_periods - len(vals)), constant_values=pad_value)
                    demand[i, :] = vals[:num_periods]
        return edge_cost, edge_emission, node_cost, node_emission, initial_inventory, demand

    except Exception as exc:
        warnings.warn(f"Could not parse {path} for {SHEET_NAME}: {exc}. Using embedded Excel-derived parameters.")
        demand = EMBEDDED_DEMAND[:, :num_periods].copy() if fixed_demand else None
        return (
            EMBEDDED_EDGE_COST.copy(),
            EMBEDDED_EDGE_EMISSION.copy(),
            EMBEDDED_NODE_COST.copy(),
            EMBEDDED_NODE_EMISSION.copy(),
            EMBEDDED_INITIAL_INVENTORY.copy(),
            demand,
        )


def _stochastic_demand(num_markets: int, num_periods: int, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        lam = int(rng.integers(100, 200))
        demand = rng.poisson(lam=lam, size=(num_markets, num_periods))
    else:
        loc = int(rng.integers(100, 150))
        scale = int(rng.integers(40, 60))
        demand = rng.normal(loc=loc, scale=scale, size=(num_markets, num_periods))
    return np.clip(np.rint(demand), 0, None).astype(np.float64)


def _normal_factors(rng: np.random.Generator, periods: int, *, clip: bool = False) -> np.ndarray:
    factors = rng.normal(loc=1.0, scale=0.1, size=periods)
    if clip:
        factors = np.clip(factors, 0.1, 10.0)
    return factors.astype(np.float64)


def build_config(
    num_periods: int = NUM_PERIODS_DEFAULT,
    *,
    input_file: Optional[str] = None,
    fixed_demand: bool = False,
    seed: Optional[int] = None,
    randomise_costs: Optional[bool] = None,
    randomise_demand: Optional[bool] = None,
    include_demand_in_obs: bool = False,
    include_metrics_in_obs: bool = False,
    max_order_quantity: float = 500.0,
    backlog: bool = False,
    shortage_penalty: float = 1.0,
) -> SCNetworkConfig:
    if randomise_costs is None:
        randomise_costs = not fixed_demand
    if randomise_demand is None:
        randomise_demand = not fixed_demand

    edge_cost, edge_emission, node_cost, node_emission, initial_inventory, demand = _load_excel_parameters(
        input_file=input_file,
        num_periods=num_periods,
        fixed_demand=fixed_demand,
    )

    cfg = SCNetworkConfig(
        num_periods=num_periods,
        input_file=input_file,
        fixed_demand=fixed_demand,
        randomise_costs=bool(randomise_costs),
        randomise_demand=bool(randomise_demand),
        seed=seed,
        max_order_quantity=max_order_quantity,
        backlog=backlog,
        shortage_penalty=shortage_penalty,
        include_demand_in_obs=include_demand_in_obs,
        include_metrics_in_obs=include_metrics_in_obs,
        edge_cost=edge_cost,
        edge_emission=edge_emission,
        node_cost=node_cost,
        node_emission=node_emission,
        initial_inventory=initial_inventory,
        demand=demand,
    )
    cfg.validate()
    return cfg


class ORGymSupplyChainEnv(gym.Env):
    """Pure Gym/Gymnasium supply-chain environment with OR-Gym-style dynamics."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[SCNetworkConfig] = None,
        *,
        weights: Optional[Sequence[float]] = None,
        normalize_actions: bool = False,
        reward_clip: Optional[Tuple[float, float]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else build_config()
        self.config.validate()
        self.render_mode = render_mode
        self.normalize_actions = normalize_actions
        self.reward_clip = reward_clip
        self.rng = np.random.default_rng(self.config.seed)
        self.python_rng = random.Random(self.config.seed)

        self.control_edges = self.config.controllable_edge_indices
        self.demand_edges = self.config.demand_edge_indices
        self.inventory_nodes = self.config.inventory_node_indices
        self.num_control_edges = len(self.control_edges)
        self.num_market = NUM_MARKETS
        self.weights = self._normalise_weights(weights if weights is not None else np.ones(NUM_OBJECTIVES) / NUM_OBJECTIVES)

        action_high = np.ones(self.num_control_edges, dtype=np.float32)
        if not normalize_actions:
            action_high *= float(self.config.max_order_quantity)
        self.action_space = spaces.Box(low=np.zeros_like(action_high), high=action_high, dtype=np.float32)

        self.pipeline_slots = int(np.sum(self.config.lead_times[self.control_edges]))
        obs_dim = len(self.inventory_nodes) * NUM_PRODUCTS + self.pipeline_slots
        if self.config.include_demand_in_obs:
            obs_dim += self.num_market
        if self.config.include_metrics_in_obs:
            obs_dim += 2
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.period = 0
        self.inventory = np.zeros((self.config.num_nodes, NUM_PRODUCTS), dtype=np.float64)
        self.backlog = np.zeros(self.num_market, dtype=np.float64)
        self.current_demand = np.zeros(self.num_market, dtype=np.float64)
        self.demand_schedule = np.zeros((self.num_market, self.config.num_periods), dtype=np.float64)
        self.edge_costs_t = np.zeros((self.config.num_edges, self.config.num_periods), dtype=np.float64)
        self.edge_emissions_t = np.zeros((self.config.num_edges, self.config.num_periods), dtype=np.float64)
        self.node_costs_t = np.zeros((self.config.num_nodes, self.config.num_periods), dtype=np.float64)
        self.node_emissions_t = np.zeros((self.config.num_nodes, self.config.num_periods), dtype=np.float64)
        self.pipeline: Dict[int, List[Tuple[int, float, int]]] = {}
        self.action_log = np.zeros((self.config.num_periods, self.num_control_edges), dtype=np.float64)
        self.edge_inputs = np.zeros((self.config.num_edges, self.config.num_periods), dtype=np.float64)
        self.edge_outputs = np.zeros((self.config.num_edges, self.config.num_periods), dtype=np.float64)
        self.fulfilled = np.zeros((self.config.num_periods, self.num_market), dtype=np.float64)
        self.unfulfilled = np.zeros((self.config.num_periods, self.num_market), dtype=np.float64)
        self.reward_vectors_raw = np.zeros((self.config.num_periods, NUM_OBJECTIVES), dtype=np.float64)
        self.reward_vectors = np.zeros((self.config.num_periods, NUM_OBJECTIVES), dtype=np.float64)
        self.cumulative_profit = 0.0
        self.cumulative_emission = 0.0
        self.cumulative_inequality = 0.0

    @property
    def num_objectives(self) -> int:
        return NUM_OBJECTIVES

    @property
    def num_edges(self) -> int:
        return self.config.num_edges

    @property
    def num_nodes(self) -> int:
        return self.config.num_nodes

    def set_scalarization_weights(self, weights: Sequence[float]) -> None:
        self.weights = self._normalise_weights(weights)

    def get_scalarization_weights(self) -> np.ndarray:
        return self.weights.copy()

    def sample_tasks(self, n: int) -> List[Dict[str, Any]]:
        return [
            {
                "demand": _stochastic_demand(self.num_market, self.config.num_periods, self.rng),
                "cost_factor": _normal_factors(self.rng, self.config.num_periods),
                "emission_factor": _normal_factors(self.rng, self.config.num_periods),
            }
            for _ in range(n)
        ]

    def set_task(self, task: Mapping[str, Any]) -> None:
        if "demand" in task:
            demand = np.asarray(task["demand"], dtype=np.float64)
            if demand.shape != (self.num_market, self.config.num_periods):
                raise ValueError(f"task['demand'] must have shape {(self.num_market, self.config.num_periods)}, got {demand.shape}")
            self.config.demand = np.clip(demand, 0.0, None)
            self.config.fixed_demand = True
            self.config.randomise_demand = False
        if "weights" in task:
            self.set_scalarization_weights(task["weights"])

    def get_state_config(self) -> Dict[str, Any]:
        return {
            "num_timesteps": self.config.num_periods,
            "names": {
                "node_names": np.asarray(self.config.node_names),
                "edge_names": np.asarray(self.config.edge_names),
                "product_names": np.asarray(["raw", "finished"]),
                "cost_names": np.asarray(["monetary", "emission"]),
                "node_tags": self.config.node_tags,
                "edge_tags": self.config.edge_tags,
            },
            "edge_upstream_nodes": self.config.edge_upstream_nodes.copy(),
            "edge_downstream_nodes": self.config.edge_downstream_nodes.copy(),
            "edge_input_products": self.config.edge_input_products.copy(),
            "edge_output_products": self.config.edge_output_products.copy(),
            "edge_expected_lengths": self.config.lead_times.copy(),
            "edge_cost": self.config.edge_cost.copy(),
            "edge_emission": self.config.edge_emission.copy(),
            "node_cost": self.config.node_cost.copy(),
            "node_emission": self.config.node_emission.copy(),
            "node_control": self.config.node_control.copy(),
            "initial_inventory": self.config.initial_inventory.copy(),
            "demand": None if self.config.demand is None else self.config.demand.copy(),
        }

    @staticmethod
    def _normalise_weights(weights: Sequence[float]) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (NUM_OBJECTIVES,):
            raise ValueError(f"weights must have shape ({NUM_OBJECTIVES},), got {w.shape}")
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        return np.ones(NUM_OBJECTIVES, dtype=np.float64) / NUM_OBJECTIVES if s <= 0 else w / s

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        if seed is not None:
            try:
                super().reset(seed=seed)
            except TypeError:
                pass
            self.rng = np.random.default_rng(seed)
            self.python_rng = random.Random(seed)

        self.period = 0
        self.inventory = self.config.initial_inventory.copy().astype(np.float64)
        self.backlog = np.zeros(self.num_market, dtype=np.float64)
        self.pipeline = {int(e): [] for e in self.control_edges}
        self.action_log.fill(0.0)
        self.edge_inputs.fill(0.0)
        self.edge_outputs.fill(0.0)
        self.fulfilled.fill(0.0)
        self.unfulfilled.fill(0.0)
        self.reward_vectors_raw.fill(0.0)
        self.reward_vectors.fill(0.0)
        self.cumulative_profit = 0.0
        self.cumulative_emission = 0.0
        self.cumulative_inequality = 0.0

        self._sample_episode_parameters()
        self.current_demand = self.demand_schedule[:, 0].copy()
        obs = self._get_obs()
        info = self._get_info(np.zeros(NUM_OBJECTIVES), np.zeros(NUM_OBJECTIVES), scalar_reward=0.0)
        return obs, info

    def step(self, action: Union[np.ndarray, Sequence[float], Mapping[Union[int, str], float]]):
        if self.period >= self.config.num_periods:
            raise RuntimeError("Episode is done. Call reset() before step().")

        t = self.period
        action_vec = self._action_to_vector(action)
        if self.normalize_actions:
            action_vec = action_vec * float(self.config.max_order_quantity)
        action_vec = np.clip(action_vec, 0.0, float(self.config.max_order_quantity))
        self.action_log[t, :] = action_vec

        movement_cost = 0.0
        movement_emission = 0.0

        # 1) OR-Gym-style order placement.
        for local_i, edge_idx in enumerate(self.control_edges):
            requested = float(np.rint(action_vec[local_i]))
            shipped = self._process_control_edge(int(edge_idx), requested)
            self.edge_inputs[edge_idx, t] = requested
            self.edge_outputs[edge_idx, t] = shipped
            movement_cost += shipped * self.edge_costs_t[edge_idx, t]
            movement_emission += shipped * self.edge_emissions_t[edge_idx, t]

        # 2) Receive pipeline arrivals.
        self._receive_pipeline_arrivals(t)

        # 3) Realise demand at retailer-market edges.
        demand = self.demand_schedule[:, t].copy()
        if self.config.backlog:
            demand = demand + self.backlog
        fulfilled, unfulfilled, revenue = self._realise_market_demand(t, demand)
        self.current_demand = self.demand_schedule[:, min(t + 1, self.config.num_periods - 1)].copy()

        # 4) Reward components.
        inv_nodes = self.config.inventory_node_indices
        holding_cost = float(np.sum(self.inventory[inv_nodes, :] * self.node_costs_t[inv_nodes, t, None]))
        holding_emission = float(np.sum(self.inventory[inv_nodes, :] * self.node_emissions_t[inv_nodes, t, None]))
        shortage_cost = float(np.sum(unfulfilled) * self.config.shortage_penalty)
        service = np.divide(fulfilled, np.maximum(demand, 1.0), out=np.ones_like(fulfilled), where=np.maximum(demand, 1.0) > 0)
        inequality = float(np.std(service))

        profit = revenue - movement_cost - holding_cost - shortage_cost
        emission = movement_emission + holding_emission
        raw_vec = np.array([profit, -emission, -inequality], dtype=np.float64)
        norm_vec = self._normalise_reward(raw_vec)
        scalar_reward = float(np.dot(self.weights, norm_vec))
        if self.reward_clip is not None:
            scalar_reward = float(np.clip(scalar_reward, self.reward_clip[0], self.reward_clip[1]))

        self.reward_vectors_raw[t, :] = raw_vec
        self.reward_vectors[t, :] = norm_vec
        self.cumulative_profit += profit
        self.cumulative_emission += emission
        self.cumulative_inequality += inequality

        self.period += 1
        terminated = self.period >= self.config.num_periods
        truncated = False
        obs = self._get_obs()
        info = self._get_info(raw_vec, norm_vec, scalar_reward=scalar_reward)
        return obs, scalar_reward, terminated, truncated, info

    def render(self):
        msg = (
            f"{self.config.name} period={self.period}/{self.config.num_periods} "
            f"profit={self.cumulative_profit:.2f} emission={self.cumulative_emission:.2f} "
            f"ineq={self.cumulative_inequality:.4f}"
        )
        if self.render_mode == "human" or self.render_mode is None:
            print(msg)
        return msg

    def _sample_episode_parameters(self) -> None:
        c = self.config
        periods = c.num_periods

        if c.fixed_demand and c.demand is not None:
            self.demand_schedule = np.asarray(c.demand, dtype=np.float64).copy()
        elif c.randomise_demand:
            self.demand_schedule = _stochastic_demand(NUM_MARKETS, periods, self.rng)
        else:
            self.demand_schedule = np.zeros((NUM_MARKETS, periods), dtype=np.float64)

        self.edge_costs_t = np.repeat(c.edge_cost[:, None], periods, axis=1).astype(np.float64)
        self.edge_emissions_t = np.repeat(c.edge_emission[:, None], periods, axis=1).astype(np.float64)
        self.node_costs_t = np.repeat(c.node_cost[:, None], periods, axis=1).astype(np.float64)
        self.node_emissions_t = np.repeat(c.node_emission[:, None], periods, axis=1).astype(np.float64)

        if c.randomise_costs:
            clip = c.name == "complex"
            for i in range(c.num_edges):
                self.edge_costs_t[i, :] = c.edge_cost[i] * _normal_factors(self.rng, periods, clip=clip)
                self.edge_emissions_t[i, :] = c.edge_emission[i] * _normal_factors(self.rng, periods, clip=clip)
            for i in range(c.num_nodes):
                self.node_costs_t[i, :] = c.node_cost[i] * _normal_factors(self.rng, periods, clip=clip)
                self.node_emissions_t[i, :] = c.node_emission[i] * _normal_factors(self.rng, periods, clip=clip)

        # Demand edges store positive market prices, as in your Messiah state files.
        for market_i, edge_idx in enumerate(c.demand_edge_indices):
            if c.fixed_demand or not c.randomise_costs:
                factor = np.ones(periods, dtype=np.float64)
            else:
                factor = _normal_factors(self.rng, periods, clip=(c.name == "complex"))
            self.edge_costs_t[edge_idx, :] = c.price_market[market_i] * factor
            self.edge_emissions_t[edge_idx, :] = 0.0

    def _action_to_vector(self, action: Union[np.ndarray, Sequence[float], Mapping[Union[int, str], float]]) -> np.ndarray:
        if isinstance(action, Mapping):
            values = np.zeros(self.num_control_edges, dtype=np.float64)
            edge_name_to_local = {self.config.edge_names[e]: i for i, e in enumerate(self.control_edges)}
            edge_idx_to_local = {int(e): i for i, e in enumerate(self.control_edges)}
            for key, value in action.items():
                if isinstance(key, str):
                    if key not in edge_name_to_local:
                        raise KeyError(f"Unknown or non-controllable edge name {key!r}")
                    values[edge_name_to_local[key]] = value
                else:
                    if int(key) not in edge_idx_to_local:
                        raise KeyError(f"Unknown or non-controllable edge index {key!r}")
                    values[edge_idx_to_local[int(key)]] = value
            return values
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.shape != (self.num_control_edges,):
            raise ValueError(f"action must have shape {(self.num_control_edges,)}, got {arr.shape}")
        return arr

    def _process_control_edge(self, edge_idx: int, requested: float) -> float:
        if requested <= 0:
            return 0.0

        c = self.config
        upstream = int(c.edge_upstream_nodes[edge_idx])
        downstream = int(c.edge_downstream_nodes[edge_idx])
        in_product = int(np.argmax(c.edge_input_products[edge_idx]))
        out_product = int(np.argmax(c.edge_output_products[edge_idx]))
        upstream_tag = c.node_tags[upstream]

        if upstream_tag == "supplier":
            shipped = requested
        else:
            available = max(float(self.inventory[upstream, in_product]), 0.0)
            shipped = min(requested, available)
            self.inventory[upstream, in_product] -= shipped

        lead_time = int(c.lead_times[edge_idx])
        arrival_period = self.period + max(lead_time, 0)
        self.pipeline[edge_idx].append((arrival_period, shipped, out_product))
        return shipped

    def _receive_pipeline_arrivals(self, t: int) -> None:
        c = self.config
        for edge_idx in self.control_edges:
            edge_idx = int(edge_idx)
            downstream = int(c.edge_downstream_nodes[edge_idx])
            remaining: List[Tuple[int, float, int]] = []
            for arrival_period, qty, product in self.pipeline[edge_idx]:
                if arrival_period <= t:
                    self.inventory[downstream, product] += qty
                else:
                    remaining.append((arrival_period, qty, product))
            self.pipeline[edge_idx] = remaining

    def _realise_market_demand(self, t: int, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        c = self.config
        fulfilled = np.zeros(NUM_MARKETS, dtype=np.float64)
        unfulfilled = np.zeros(NUM_MARKETS, dtype=np.float64)
        revenue = 0.0
        for market_i, edge_idx in enumerate(c.demand_edge_indices):
            retailer = int(c.edge_upstream_nodes[edge_idx])
            available = max(float(self.inventory[retailer, PRODUCT_FINISHED]), 0.0)
            sale = min(float(demand[market_i]), available)
            self.inventory[retailer, PRODUCT_FINISHED] -= sale
            fulfilled[market_i] = sale
            unfulfilled[market_i] = max(float(demand[market_i]) - sale, 0.0)
            self.fulfilled[t, market_i] = sale
            self.unfulfilled[t, market_i] = unfulfilled[market_i]
            self.edge_outputs[edge_idx, t] = sale
            revenue += sale * self.edge_costs_t[edge_idx, t]
        self.backlog = unfulfilled if c.backlog else np.zeros_like(unfulfilled)
        return fulfilled, unfulfilled, revenue

    def _pipeline_vector(self) -> np.ndarray:
        values: List[float] = []
        t = self.period
        for edge_idx in self.control_edges:
            edge_idx = int(edge_idx)
            lead_time = int(self.config.lead_times[edge_idx])
            for offset in range(lead_time):
                target_period = t + offset
                qty = sum(q for arrival_period, q, _product in self.pipeline[edge_idx] if arrival_period == target_period)
                values.append(float(qty))
        return np.asarray(values, dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        pieces: List[np.ndarray] = []
        if self.config.include_demand_in_obs:
            pieces.append(self.current_demand.astype(np.float64))
        pieces.append(self.inventory[self.inventory_nodes, :].reshape(-1).astype(np.float64))
        pieces.append(self._pipeline_vector())
        if self.config.include_metrics_in_obs:
            pieces.append(np.asarray([self.cumulative_emission, self.cumulative_inequality], dtype=np.float64))
        return np.concatenate(pieces).astype(np.float32)

    def _normalise_reward(self, raw_vec: np.ndarray) -> np.ndarray:
        c = self.config
        if c.reward_scale_profit is None:
            max_revenue = max(float(np.max(c.price_market) * max(np.max(self.demand_schedule), 1.0) * NUM_MARKETS), 1.0)
            profit_scale = max_revenue
        else:
            profit_scale = float(c.reward_scale_profit)

        if c.reward_scale_emission is None:
            max_emission = max(float(np.max(c.edge_emission) * c.max_order_quantity * max(self.num_control_edges, 1)), 1.0)
        else:
            max_emission = float(c.reward_scale_emission)

        return np.array([
            raw_vec[0] / max(profit_scale, 1e-9),
            raw_vec[1] / max(max_emission, 1e-9),
            raw_vec[2] / max(float(c.reward_scale_inequality), 1e-9),
        ], dtype=np.float64)

    def _get_info(self, raw_vec: np.ndarray, norm_vec: np.ndarray, *, scalar_reward: float) -> Dict[str, Any]:
        row = max(self.period - 1, 0)
        return {
            "period": self.period,
            "weights": self.weights.copy(),
            "scalar_reward": float(scalar_reward),
            "mo_reward": norm_vec.astype(np.float32),
            "mo_reward_raw": raw_vec.astype(np.float32),
            "profit": float(raw_vec[0]),
            "emission": float(-raw_vec[1]),
            "service_inequality": float(-raw_vec[2]),
            "cumulative_profit": float(self.cumulative_profit),
            "cumulative_emission": float(self.cumulative_emission),
            "cumulative_service_inequality": float(self.cumulative_inequality),
            "inventory": self.inventory.copy(),
            "fulfilled": self.fulfilled[row].copy() if self.period > 0 else np.zeros(NUM_MARKETS),
            "unfulfilled": self.unfulfilled[row].copy() if self.period > 0 else np.zeros(NUM_MARKETS),
            "edge_inputs": self.edge_inputs[:, row].copy() if self.period > 0 else np.zeros(self.config.num_edges),
            "edge_outputs": self.edge_outputs[:, row].copy() if self.period > 0 else np.zeros(self.config.num_edges),
        }


class ModerateSupplyChainEnv(ORGymSupplyChainEnv):
    """Stochastic moderate environment for meta-training."""

    def __init__(self, num_periods: int = NUM_PERIODS_DEFAULT, **kwargs: Any) -> None:
        weights = kwargs.pop("weights", None)
        normalize_actions = kwargs.pop("normalize_actions", False)
        reward_clip = kwargs.pop("reward_clip", None)
        render_mode = kwargs.pop("render_mode", None)
        cfg = build_config(num_periods=num_periods, fixed_demand=False, **kwargs)
        super().__init__(cfg, weights=weights, normalize_actions=normalize_actions, reward_clip=reward_clip, render_mode=render_mode)


class FTModerateSupplyChainEnv(ORGymSupplyChainEnv):
    """Fixed-demand/fixed-cost moderate environment for fine-tuning."""

    def __init__(self, num_periods: int = NUM_PERIODS_DEFAULT, **kwargs: Any) -> None:
        weights = kwargs.pop("weights", None)
        normalize_actions = kwargs.pop("normalize_actions", False)
        reward_clip = kwargs.pop("reward_clip", None)
        render_mode = kwargs.pop("render_mode", None)
        cfg = build_config(num_periods=num_periods, fixed_demand=True, randomise_costs=False, randomise_demand=False, **kwargs)
        super().__init__(cfg, weights=weights, normalize_actions=normalize_actions, reward_clip=reward_clip, render_mode=render_mode)


# Backward-friendly aliases.
ModerateState = ModerateSupplyChainEnv
TestModerateState = FTModerateSupplyChainEnv


def make_env(env_config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> ORGymSupplyChainEnv:
    """Factory useful for RLlib/SB3.

    Example:
        env = make_env({"fixed_demand": True, "input_file": "data_input_v2.xlsx"})
    """
    cfg = dict(env_config or {})
    cfg.update(kwargs)
    fixed_demand = bool(cfg.pop("fixed_demand", False))
    num_periods = int(cfg.pop("num_periods", cfg.pop("num_timesteps", NUM_PERIODS_DEFAULT)))
    cls = FTModerateSupplyChainEnv if fixed_demand else ModerateSupplyChainEnv
    return cls(num_periods=num_periods, **cfg)


if __name__ == "__main__":
    env = make_env({"fixed_demand": True, "input_file": None})
    obs, info = env.reset(seed=0)
    print("moderate_state.py: action_space=", env.action_space.shape, "observation_space=", env.observation_space.shape)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("step reward=", reward, "mo_reward=", info["mo_reward"], "terminated=", terminated)
