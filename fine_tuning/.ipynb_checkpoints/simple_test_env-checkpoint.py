'''In here, we configure the simple SC environment to train MetaLearning'''

import gymnasium as gym
from gymnasium import spaces

import sys, os
from messiah.agents.naive import FixedQuantityProcessing
from messiah.agents.basic import MaxAvailableProcessing, MaxRequiredProcessing
from messiah.config import settings
from messiah.core.runner import Runner
from messiah.history.base import History
from messiah.state.base import State

from fine_tuning.test_state import test_sc

import numpy as np
from math import floor
import random

settings.episode_length = 100

supply_capacity = 200

#set reward range
min_profit = 0
max_profit = 4000
min_emission = 0
max_emission = 2000
max_equity = 1
min_equity = 0

agents = [
    MaxAvailableProcessing('manufacturer2', 'manufacturer2', 'manufacture', supply_capacity),
    MaxAvailableProcessing('manufacturer3', 'manufacturer3', 'manufacture', supply_capacity),
    MaxRequiredProcessing('retailer4', 'marketa', 'supply'),
    MaxRequiredProcessing('retailer5', 'marketb', 'supply')
]

test_sc = test_sc() #Instantiate the class
test_sc.simple_network() #Call method within the class
nodes = test_sc.simple_sc_nodes
edges = test_sc.simple_sc_edges

goods = ['raw','product']

state = test_sc.simple_network()
runner = Runner(state, agents)
run_state = runner.run(episodes=100, history=True)

#create a class for simple SC env
class TestSimpleSC(gym.Env):
    
    def __init__(self):
        self.state = state #previously None
        self.transforms = agents
        self.max_steps = self.state.episode_length
        self.time_step = 0
        self.weights = np.array([1/3,1/3,1/3])
        
        self.emission = 0
        self.inequality = 0
        
        self.reward_space = spaces.Box(low=np.array([0, -np.inf, -np.inf]), high=np.array([np.inf, 0, 0]), shape=(3,), dtype=np.float32)
        self.reward_dim = 3
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(56,), dtype=np.float32)
        
    def _get_obs(self, mynodes, edges):
        
        inv = {} #inventories at each node
        flow = {} #ordered products at each edge
        
        #define observation for inventories
        for node in mynodes.keys():
            if 'farm' in node or 'market' in node: #farm and market don't have inventories
                continue
            else:
                inv[node] = {} #set empty dictionary to store each good at each node
                for item in goods:
                    try:
                        inv[node][item] = self.state.get_node(node).inventory[item][self.time_step]
                    except KeyError:
                        pass
                    
        #define observation for flows
        time_span = 2
        for edge in edges.keys():
            source, destination, process = edge
            flow[edge] = {} #set empty dictionary to store each good flow at each edge
            for item in goods:
                for time in range(time_span):
                    try:
                        flow[edge][(item,time)] = self.state.get_edge(source, destination, process).outputs[item][self.time_step-(time+1)] #access ongoing order D-1
                    except KeyError:
                        pass
                    
        inv_list = [inv.get(node,{}).get(item,0) for node in mynodes.keys() for item in goods] #5000 represents capacity of inventory
        flow_list = [flow.get(edge,{}).get((item,time),0) for edge in edges.keys() for item in goods for time in range(time_span)]
        
        arr_inv = np.array(inv_list, dtype=np.float32)
        arr_inv = arr_inv/10000 #normalisation into obs space
        arr_flow = np.array(flow_list, dtype=np.float32)
        arr_flow = arr_flow/200 #normalisation into obs space
        #observation: accumulated emission & inequality
        emission = (self.emission/1e6)/1e6 #first 1e6 for decimal values, second for normalisation
        inequality = (self.inequality)/(self.time_step+1)
        
        arr_obs = np.concatenate((arr_inv,arr_flow))
        arr_obs_final = np.concatenate((arr_obs,[emission,inequality]))
        arr_obs_final = np.clip([_ for _ in arr_obs_final], 0, 1) #clip obs within abs space
        arr_obs_final = np.array(arr_obs_final, dtype=np.float32)
        return arr_obs_final
    
    def _step_state(self):
        '''to apply standard transform to the state'''
        for transform in self.transforms:
            self.state = transform.apply(self.time_step, self.state)
        self.state.count_node_costs(self.time_step)
        
    def reset(self, seed=None, options=None):
        self.state.reset()
        self.time_step = 0

        observation = self._get_obs(nodes, edges)
        return observation, {}
    
    '''def set_weights(self):
        self.weights = np.array(np.random.dirichlet(alpha=[1,1,1], size=1)[0])
        #print('weights:',self.weights) #for checking only
        return self.weights'''
        
    def step(self, action):
        self._step_state()

        scaled_actions = np.floor((action + 1)*(supply_capacity/2))
        action_edges = []
        for i, edge in enumerate(edges.keys()):
            source, destination, process = edge
            k = 0
            if process == 'transport':
                k += 1
                action_edges.append((source, destination, process))
        
        for k, edge in enumerate(action_edges):
            source, destination, process = edge
            self.state.start_process(self.time_step, source, destination, process, int(scaled_actions[k]))

        terminated = (self.time_step == (self.max_steps - 1))
        
        reward_1 = float(self.state.costs['monetary'][self.time_step])/10000
        reward_2 = float(self.state.costs['emission'][self.time_step])/10000
        
        #call demand variables
        demand_a = test_sc.demand_a
        demand_b = test_sc.demand_b
        
        #set reward 3
        sl = np.zeros(2)

        sl[0] = min(int(scaled_actions[2]+scaled_actions[4])/(demand_a[self.time_step]+1),1) #make it automatic later
        sl[1] = min(int(scaled_actions[3]+scaled_actions[5])/(demand_b[self.time_step]+1),1)
        
        gap = 0
        for i in range(0,2):
            for j in range(0,2):
                if (j != i):
                    gap += abs(sl[i]-sl[j])
                else:
                    pass
        reward_3 = -0.5*gap
        
        #scalarise rewards       
        scaled_reward_1 = (reward_1-min_profit)/(max_profit-min_profit)
        scaled_reward_2 = (max_emission+reward_2)/(max_emission-min_emission)
        scaled_reward_3 = max_equity + reward_3
        self.vector_reward = np.array([scaled_reward_1,scaled_reward_2,scaled_reward_3]) #Create vector reward for multi-objective
        
        #Set random weights
        self.scalar_reward = np.dot(self.vector_reward,self.weights)
        
        observation = self._get_obs(nodes, edges)
        self.time_step += 1
        
        #calculate accumulated emission and inequality
        self.emission += (((max_emission-min_emission)*self.vector_reward[1])-max_emission)
        self.inequality += self.vector_reward[2]
        
        return observation, self.scalar_reward, terminated, False, {}
    
    def count_reward(self):        
        if self.vector_reward is not None:
            vector_reward_origin = (((max_profit-min_profit)*self.vector_reward[0]+min_profit),
                                    (((max_emission-min_emission)*self.vector_reward[1])-max_emission),
                                    self.vector_reward[2]-max_equity)
            return vector_reward_origin
        else:
            pass
