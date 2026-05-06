"""
The simple supply chain environment simulated in this experiment. The other moderate and complex environments follow its stucture, with modified parameters as defined in the Appendix file.
"""

import gymnasium as gym
from gymnasium import spaces

import sys, os
from messiah.agents.naive import FixedQuantityProcessing
from messiah.agents.basic import MaxAvailableProcessing, MaxRequiredProcessing
from messiah.config import settings
from messiah.core.runner import Runner
from messiah.history.base import History
from messiah.state.base import State

from state_generator import generate_sc

import numpy as np
from math import floor
import random

supply_capacity = 200

agents = [
    MaxAvailableProcessing('manufacturer2', 'manufacturer2', 'manufacture', supply_capacity),
    MaxAvailableProcessing('manufacturer3', 'manufacturer3', 'manufacture', supply_capacity),
    MaxRequiredProcessing('retailer4', 'marketa', 'supply'),
    MaxRequiredProcessing('retailer5', 'marketb', 'supply')
]

#Set reward range for normalisation
min_profit = 0
max_profit = 4000
min_emission = 0
max_emission = 2000
max_equity = 1
min_equity = 0

goods = ['raw','product']

runner = Runner(state, agents)
run_state = runner.run(episodes=100, history=True)

class StateWithObjectives(State):
    def __init__(self, base_state, vector_reward=None):
        # Extract nodes and edges from the base_state
        random_sc = generate_sc() #Instantiate the class
        random_sc.simple_network() #Call method within the class
        nodes = random_sc.simple_sc_nodes
        edges = random_sc.simple_sc_edges

        if nodes is None or edges is None:
            raise ValueError("state must have 'simple_sc_nodes' and 'simple_sc_edges' attributes")

        #Call the parent class constructor with the required arguments
        super().__init__(nodes, edges)

        #Store the base state and additional objectives
        self.base_state = state
        self.vector_reward = vector_reward if vector_reward is not None else [0, 0, 0]


#Create a class for simple supply chain environment
class SimpleSC(gym.Env):
    
    def __init__(self):
        self.state = state
        default_weights = np.random.dirichlet(alpha=[1, 1, 1])
        self.set_task(random.choice(self.sample_tasks_with_objectives(99)), default_weights)

        self.transforms = agents
        self.max_steps = self.state.episode_length
        self.time_step = 0
        
        self.emission = 0
        self.inequality = 0
        
        self.reward_space = spaces.Box(low=np.array([0, -np.inf, -np.inf]), high=np.array([np.inf, 0, 0]), shape=(3,), dtype=np.float32)
        self.reward_dim = 3
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(56,), dtype=np.float32)
        
    def _get_obs(self, mynodes, edges):
        
        inv = {} #Inventories at each node
        flow = {} #Ordered products at each edge
        
        #Define observation for inventories
        for node in mynodes.keys():
            if 'farm' in node or 'market' in node: #Farm and market don't have inventories
                continue
            else:
                inv[node] = {} #Set empty dictionary to store each good at each node
                for item in goods:
                    try:
                        inv[node][item] = self.state.get_node(node).inventory[item][self.time_step]
                    except KeyError:
                        pass
                    
        #Define observation for flows
        time_span = 2
        for edge in edges.keys():
            source, destination, process = edge
            flow[edge] = {} #Set empty dictionary to store each good flow at each edge
            for item in goods:
                for time in range(time_span):
                    try:
                        flow[edge][(item,time)] = self.state.get_edge(source, destination, process).outputs[item][self.time_step-(time+1)] #Access ongoing order D-1
                    except KeyError:
                        pass
                    
        inv_list = [inv.get(node,{}).get(item,0) for node in mynodes.keys() for item in goods] #5000 represents capacity of inventory
        flow_list = [flow.get(edge,{}).get((item,time),0) for edge in edges.keys() for item in goods for time in range(time_span)]
        
        arr_inv = np.array(inv_list, dtype=np.float32)
        arr_inv = arr_inv/10000 #Normalisation into obs space
        arr_flow = np.array(flow_list, dtype=np.float32)
        arr_flow = arr_flow/200 #Normalisation into obs space
        #Observation: accumulated emission & inequality
        emission = min(max((self.emission/1e6)/1e6, 0), 1) #First 1e6 for decimal values, second for normalisation
        inequality = min(max((self.inequality)/(self.time_step+1), 0), 1)
        
        arr_obs = np.concatenate((arr_inv,arr_flow))
        arr_obs_final = np.concatenate((arr_obs,[emission,inequality]))
        arr_obs_final = np.clip(arr_obs_final, 0, 1) #Clip obs within abs space
        arr_obs_final = np.array(arr_obs_final, dtype=np.float32)
        return arr_obs_final
    
    def _step_state(self):
        '''to apply standard transform to the state'''
        for transform in self.transforms:
            self.state = transform.apply(self.time_step, self.state)
        self.state.count_node_costs(self.time_step)
        
    def set_task(self, task, weights, task_id=None):
        """Set task and associated weights from MAML."""
        self.state = task
        self.weights = np.array(weights)  #Store the weights for scalarisation
        self.task_id = task_id if task_id is not None else id(task)
 
            
    def get_task(self):
        return self.state
        
    def sample_tasks(self, n_tasks):
        return self.sample_tasks_with_objectives(n_tasks)
    
    def get_weights(self):
        """Return the current weights as a dictionary with task_id as the key."""
        return {self.task_id: np.array(self.weights).tolist()}
    
    def set_weights(self, updated_task_weights):
        """Update the weights in the environment."""
        if self.task_id in updated_task_weights:
            self.weights = np.array(updated_task_weights[self.task_id])
            print(f"Weights updated for task {self.task_id}: {self.weights}")
        else:
            raise KeyError(f"Task ID {self.task_id} not found in updated_task_weights")


    def get_metrics(self):
        """Return environment-specific metrics for MAML."""
        metrics = {
            "task_id": self.task_id,  #Include the task_id
            "emission": self.emission,  #Total accumulated emission
            "inequality": self.inequality,  #Total accumulated inequality
            "time_step": self.time_step,  #Current timestep
            "vector_reward": self.vector_reward.tolist() if hasattr(self, "vector_reward") else None,
        }
        return metrics

    
    def sample_tasks_with_objectives(self,n_tasks):
        simple_random_list = []
        for _ in range(n_tasks):
            base_state = random_sc.simple_network()
            #Define a placeholder vector_reward (replace with actual logic if needed)
            vector_reward = [0.0, 0.0, 0.0]  #Default or initial vector reward
            simple_random_list.append(StateWithObjectives(base_state, vector_reward=vector_reward))
        return np.array(simple_random_list)
        
    def reset(self, seed=None, options=None):
        self.state.reset()
        self.time_step = 0

        observation = self._get_obs(nodes,edges)
        return observation, {}
        
    def step(self, action):
        self._step_state()
        
        clipped_action = np.clip(action,self.action_space.low, self.action_space.high) #Clip action to prevent out of bound values
        scaled_actions = np.floor((clipped_action + 1)*(supply_capacity/2))
        action_edges = []
        for i, edge in enumerate(edges.keys()):
            source, destination, process = edge
            if process == 'transport':
                action_edges.append((source, destination, process))
        
        for k, edge in enumerate(action_edges):
            source, destination, process = edge
            self.state.start_process(self.time_step, source, destination, process, int(scaled_actions[k]))

        terminated = (self.time_step == (self.max_steps - 1))
        
        reward_1 = max(float(self.state.costs['monetary'][self.time_step])/10000, 0)
        reward_2 = max(float(self.state.costs['emission'][self.time_step])/10000, 0)
        
        #call demand variables
        demand_a = max(random_sc.demand['market_0'][self.time_step], 1e-6)
        demand_b = max(random_sc.demand['market_1'][self.time_step], 1e-6)
        
        sl = np.zeros(2)
        sl[0] = min((scaled_actions[2] + scaled_actions[4]) / demand_a, 1)
        sl[1] = min((scaled_actions[3] + scaled_actions[5]) / demand_b, 1)
        
        gap = sum(abs(sl[i] - sl[j]) for i in range(2) for j in range(2) if i != j)
        reward_3 = -0.5*gap
        
        #scalarise rewards       
        scaled_reward_1 = np.clip((reward_1-min_profit)/(max_profit-min_profit),0,1)
        scaled_reward_2 = np.clip((max_emission+reward_2)/(max_emission-min_emission),0,1)
        scaled_reward_3 = np.clip(max_equity + reward_3,0,1)
        self.vector_reward = np.array([scaled_reward_1,scaled_reward_2,scaled_reward_3]) #Create vector reward for multi-objective
        
        self.scalar_reward = np.dot(self.vector_reward,self.weights)
        
        observation = self._get_obs(nodes,edges)
        self.time_step += 1
        
        #Calculate accumulated emission and inequality
        self.emission += (((max_emission-min_emission)*self.vector_reward[1])-max_emission)
        self.inequality += self.vector_reward[2]
        
        return observation, self.scalar_reward, terminated, False, {}
