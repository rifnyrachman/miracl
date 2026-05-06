"""
Random state generator for meta-training sampling. Here, supply chain environment parameters are randomly generated within defined probability distribution.
"""

import sys
import os

# Import the required modules
from messiah import State, Runner, History
from messiah.state.components import Node, Edge
from messiah.series.basic import ConstantSeries, RandomSeries, DataSeries
from messiah.agents.naive import FixedQuantityProcessing
from messiah.agents.basic import MaxAvailableProcessing, MaxRequiredProcessing
from messiah.config import settings

import numpy as np
import random
from random import randrange
import pandas as pd
import itertools
import time

import matplotlib.pyplot as plt

class generate_sc:
    
    def __init__(self):
        self.demand = {}
        settings.episode_length = 100 #Simulated for 100 days
        self.file_id = 1 #State identifier
        
    def random_demand(self):
        random_dist = random.choice(['poisson','normal'])
    
        if random_dist == 'poisson':
            lam = randrange(100,200)
            random_series = RandomSeries(random_dist, lam=lam, is_integer=True)
        else:
            loc = randrange(100,150)
            scale = randrange(40,60)
            random_series = RandomSeries(random_dist, loc=loc, scale=scale, is_integer=True)

        return np.array(random_series)
    
    def define_demand(self, num_market: int) -> dict:
        """
        Define random demand for multiple markets.

        Parameters:
        ----------
        num_market : int
            Number of markets to generate random demand for.

        Returns:
        -------
        dict
            A dictionary where keys are market identifiers and values are the generated demand arrays.
        """
        self.num_market = num_market

        df = pd.DataFrame()
        for i in range(self.num_market):
            # Generate random demand and store it in the dictionary
            self.demand[f'market_{i}'] = self.random_demand()
            df[f'market_{i}'] = self.demand[f'market_{i}']
        
        # Uncomment to save the demand data to a CSV file
        #df.to_csv(f'random_demand{self.file_id}.csv', index=False) #uncomment to export demand data
        self.file_id += 1

        return self.demand
    
    def random_param(self, loc=0, scale=0): #Randomise parameters
        if loc >= 0:
            loc = randrange(int(0.9*loc),int(1.1*loc))
        else:
            loc = randrange(int(1.1*loc),int(0.9*loc))
            
        if scale >= 0:
            scale = randrange(int(0.9*scale),int(1.1*scale))
        else:
            scale = randrange(int(1.1*scale),int(0.9*scale))
            
        random_param = np.random.normal(loc,scale,settings.episode_length)
            
        return np.array(random_param)
        
    def simple_network(self): #Define the supply chain network configuration
        num_market = 2
        self.define_demand(num_market)
        settings.episode_length = 100
        #Define nodes that represent facilities in the supply chain network
        self.simple_sc_nodes = {
            'farm1': Node(
                control             = False #Do we have control over the facility?
            ),
            'manufacturer2': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 380},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-1100,110))},
                                      'emission':{'product': DataSeries(self.random_param(-2,1))}}
            ),
            'manufacturer3': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 350},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-1300,130))},
                                      'emission':{'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer4': Node(
                control             = True,
                initial_inventory   = {'product': 400},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-1200,120))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer5': Node(
                control             = True,
                initial_inventory   = {'product': 80},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-1500,150))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'marketa': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_0'])}
            ),
            'marketb': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_1'])}
            )
        }
        
        #Define the edges that represent routes in supply chain network
        self.simple_sc_edges = {
            ('farm1','manufacturer2','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-2200,220)),'emission': DataSeries(self.random_param(-1258,125))} #Ordering cost/unit + transport cost/unit/leadtime
            ),
            ('farm1','manufacturer3','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-6900,690)),'emission': DataSeries(self.random_param(-3947,394))}
            ),
            ('manufacturer2','manufacturer2','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-20000,2000)),'emission': DataSeries(self.random_param(-50126,5012))}
            ),
            ('manufacturer3','manufacturer3','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-22000,2200)),'emission': DataSeries(self.random_param(-45754,4575))}
            ),
            ('manufacturer2','retailer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-10550,1055)),'emission': DataSeries(self.random_param(-6035,603))}
            ),
            ('manufacturer2','retailer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-4300,430)),'emission': DataSeries(self.random_param(-2460,246))}
            ),
            ('manufacturer3','retailer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-4850,485)),'emission': DataSeries(self.random_param(-2774,277))}
            ),
            ('manufacturer3','retailer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-7500,750)),'emission': DataSeries(self.random_param(-4290,429))}
            ),
            ('retailer4','marketa','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(200000,20000)),'emission': ConstantSeries(0)}
            ),
            ('retailer5','marketb','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(200000,20000)),'emission': ConstantSeries(0)} #Set price here
            )
        }
        
        return State(self.simple_sc_nodes, self.simple_sc_edges)
        
    def moderate_network(self):
        num_market = 3
        self.define_demand(num_market)
        settings.episode_length = 100
        self.moderate_sc_nodes = {
            'farm1': Node(
                control             = False
            ),
            'farm2': Node(
                control             = False
            ),
            'manufacturer3': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 380},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-1100,110))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'manufacturer4': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 350},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-1300,130))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'manufacturer5': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 400},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-1200,120))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'distributor6': Node(
                control             = True,
                initial_inventory   = {'product': 80},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-1500,150))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'distributor7': Node(
                control             = True,
                initial_inventory   = {'product': 110},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-2000,200))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'retailer8': Node(
                control             = True,
                initial_inventory   = {'product': 0},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-2500,250))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer9': Node(
                control             = True,
                initial_inventory   = {'product': 0},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-3000,300))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer10': Node(
                control             = True,
                initial_inventory   = {'product': 0},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-2000,200))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'marketa': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_0'])}
            ),
            'marketb': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_1'])}
            ),
            'marketc': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_2'])}
            )
        }

        self.moderate_sc_edges = {
            ('farm1','manufacturer3','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)), #Transport lead time
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-2200,220)),'emission': DataSeries(self.random_param(-1258,126))}
            ),
            ('farm1','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-6900,690)),'emission': DataSeries(self.random_param(-3947,395))} #(p/(L+1))+g
            ),
            ('farm1','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-5650,565)),'emission': DataSeries(self.random_param(-3232,323))} #(p/(L+1))+g
            ),
            ('farm2','manufacturer3','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-10550,1055)),'emission': DataSeries(self.random_param(-6035,604))}
            ),
            ('farm2','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-6500,650)),'emission': DataSeries(self.random_param(-3718,372))}
            ),
            ('farm2','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-6300,630)),'emission': DataSeries(self.random_param(-3604,360))}
            ),
            ('manufacturer3','manufacturer3','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-20000,2000)),'emission': DataSeries(self.random_param(-50126,5012))}
            ),
            ('manufacturer4','manufacturer4','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-22000,2200)),'emission': DataSeries(self.random_param(-45754,4575))}
            ),
            ('manufacturer5','manufacturer5','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-23000,2300)),'emission': DataSeries(self.random_param(-54491,5449))}
            ),
            ('manufacturer3','distributor6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-750,75)),'emission': DataSeries(self.random_param(-429,43))}
            ),
            ('manufacturer3','distributor7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-4300,430)),'emission': DataSeries(self.random_param(-2460,246))}
            ),
            ('manufacturer4','distributor6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6300,630)),'emission': DataSeries(self.random_param(-3604.,360))}
            ),
            ('manufacturer4','distributor7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-2300,230)),'emission': DataSeries(self.random_param(-1316,132))}
            ),
            ('manufacturer5','distributor6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-4950,495)),'emission': DataSeries(self.random_param(-2831,283))}
            ),
            ('manufacturer5','distributor7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-750,75)),'emission': DataSeries(self.random_param(-429,43))}
            ),
            ('distributor6','retailer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-10950,1095)),'emission': DataSeries(self.random_param(-6263,626))}
            ),
            ('distributor6','retailer9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6250,625)),'emission': DataSeries(self.random_param(-3575,356))}
            ),
            ('distributor6','retailer10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-9500,950)),'emission': DataSeries(self.random_param(-5434,543))}
            ),
            ('distributor7','retailer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-16400,1640)),'emission': DataSeries(self.random_param(-9381,938))}
            ),
            ('distributor7','retailer9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-11600,1160)),'emission': DataSeries(self.random_param(-6635,6636))}
            ),
            ('distributor7','retailer10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-5800,580)),'emission': DataSeries(self.random_param(-3318,332))}
            ),
            ('retailer8','marketa','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(200000,20000)),'emission': ConstantSeries(0)}
            ),
            ('retailer9','marketb','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(210000,21000)),'emission': ConstantSeries(0)}
            ),
            ('retailer10','marketc','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(205000,20500)),'emission': ConstantSeries(0)}
            )
        }

        return State(self.moderate_sc_nodes, self.moderate_sc_edges)
    
    def complex_network(self):
        num_market = 5
        self.define_demand(num_market)
        settings.episode_length = 100
        self.complex_sc_nodes = {
            'farm1': Node(
                control             = False
            ),
            'farm2': Node(
                control             = False
            ),
            'farm3': Node(
                control             = False
            ),
            'manufacturer4': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 155},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-2300,230))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'manufacturer5': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 267},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-3500,350))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'manufacturer6': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 342},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-2200,220))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'manufacturer7': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 211},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-1100,110))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'manufacturer8': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 162},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': DataSeries(self.random_param(-2900,280))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'warehouse9': Node(
                control             = True,
                initial_inventory   = {'product': 195},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-3700,370))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'warehouse10': Node(
                control             = True,
                initial_inventory   = {'product': 333},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-1100,110))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'warehouse11': Node(
                control             = True,
                initial_inventory   = {'product': 96},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-3600,360))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'distributor12': Node(
                control             = True,
                initial_inventory   = {'product': 285},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-3300,330))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'distributor13': Node(
                control             = True,
                initial_inventory   = {'product': 68},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-2600,260))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'distributor14': Node(
                control             = True,
                initial_inventory   = {'product': 379},
                costs               = {'monetary':{'product': DataSeries(self.random_param(-3000,300))},
                                      'emission':{'product':DataSeries(self.random_param(-2,1))}}
            ),
            'retailer15': Node(
                control             = True,
                initial_inventory   = {'product': 344},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-1700,170))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer16': Node(
                control             = True,
                initial_inventory   = {'product': 66},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-2900,290))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer17': Node(
                control             = True,
                initial_inventory   = {'product': 356},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-2700,270))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer18': Node(
                control             = True,
                initial_inventory   = {'product': 382},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-2300,230))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'retailer19': Node(
                control             = True,
                initial_inventory   = {'product': 362},
                costs               = {'monetary': {'product': DataSeries(self.random_param(-3700,370))},
                                      'emission': {'product': DataSeries(self.random_param(-2,1))}}
            ),
            'marketa': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_0'])}
            ),
            'marketb': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_1'])}
            ),
            'marketc': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_2'])}
            ),
            'marketd': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_3'])}
            ),
            'markete': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand['market_4'])}
            )
        }

        self.complex_sc_edges = {
            ('farm1','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-5350,535)),'emission': DataSeries(self.random_param(-3060,306))}
            ),
            ('farm1','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-2650,265)),'emission': DataSeries(self.random_param(-1516,152))}
            ),
            ('farm1','manufacturer6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-18450,1845)),'emission': DataSeries(self.random_param(-10553,1055))}
            ),
            ('farm1','manufacturer7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-16000,1600)),'emission': DataSeries(self.random_param(-9152,913))}
            ),
            ('farm1','manufacturer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-14400,1440)),'emission': DataSeries(self.random_param(-8237,824))}
            ),
           ('farm2','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6000,600)),'emission': DataSeries(self.random_param(-3432,343))}
            ),
            ('farm2','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-1750,175)),'emission': DataSeries(self.random_param(-1001,100))}
            ),
            ('farm2','manufacturer6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-7450,745)),'emission': DataSeries(self.random_param(-4261,426))}
            ),
            ('farm2','manufacturer7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-13300,1330)),'emission': DataSeries(self.random_param(-7608,760))}
            ),
            ('farm2','manufacturer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-1700,170)),'emission': DataSeries(self.random_param(-972,97))}
            ),
            ('farm3','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-3600,360)),'emission': DataSeries(self.random_param(-2059,206))}
            ),
            ('farm3','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-2950,295)),'emission': DataSeries(self.random_param(-1687,169))}
            ),
            ('farm3','manufacturer6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-12350,1235)),'emission': DataSeries(self.random_param(-7064,706))}
            ),
            ('farm3','manufacturer7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-6250,625)),'emission': DataSeries(self.random_param(-3575,358))}
            ),
            ('farm3','manufacturer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': DataSeries(self.random_param(-18550,1855)),'emission': DataSeries(self.random_param(-10611,1061))}
            ),
            ('manufacturer4','manufacturer4','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-20000,2000)),'emission': DataSeries(self.random_param(-50126,5013))}
            ),
            ('manufacturer5','manufacturer5','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-22000,2200)),'emission': DataSeries(self.random_param(-45754,4575))}
            ),
            ('manufacturer6','manufacturer6','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-21000,2100)),'emission': DataSeries(self.random_param(-54491,5449))}
            ),
            ('manufacturer7','manufacturer7','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-20000,2000)),'emission': DataSeries(self.random_param(-61232,6123))}
            ),
            ('manufacturer8','manufacturer8','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-23000,2300)),'emission': DataSeries(self.random_param(-55157,5516))}
            ),
            ('manufacturer4','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-19900,1990)),'emission': DataSeries(self.random_param(-11383,1138))}
            ),
            ('manufacturer4','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-3400,340)),'emission': DataSeries(self.random_param(-1945,195))}
            ),
            ('manufacturer4','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-8100,810)),'emission': DataSeries(self.random_param(-4633,463))}
            ),
            ('manufacturer5','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-15150,1515)),'emission': DataSeries(self.random_param(-8666,867))}
            ),
            ('manufacturer5','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6600,660)),'emission': DataSeries(self.random_param(-3775,378))}
            ),
            ('manufacturer5','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6450,645)),'emission': DataSeries(self.random_param(-3689,369))}
            ),
            ('manufacturer6','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-16950,1695)),'emission': DataSeries(self.random_param(-9695,970))}
            ),
            ('manufacturer6','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-15800,1580)),'emission': DataSeries(self.random_param(-9038,904))}
            ),
            ('manufacturer6','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-8150,816)),'emission': DataSeries(self.random_param(-4662,466))}
            ),
            ('manufacturer7','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-16150,1615)),'emission': DataSeries(self.random_param(-9238,924))}
            ),
            ('manufacturer7','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-12600,1260)),'emission': DataSeries(self.random_param(-7207,721))}
            ),
            ('manufacturer7','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6750,675)),'emission': DataSeries(self.random_param(-3861,386))}
            ),
            ('manufacturer8','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-10300,1030)),'emission': DataSeries(self.random_param(-5892,589))}
            ),
            ('manufacturer8','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-10900,1090)),'emission': DataSeries(self.random_param(-6235,624))}
            ),
            ('manufacturer8','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-16300,1630)),'emission': DataSeries(self.random_param(-9324,932))}
            ),
            ('warehouse9','distributor12','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-19650,1965)),'emission': DataSeries(self.random_param(-11240,1124))}
            ),
            ('warehouse9','distributor13','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-19250,1925)),'emission': DataSeries(self.random_param(-11011,1101))}
            ),
            ('warehouse9','distributor14','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-16200,1620)),'emission': DataSeries(self.random_param(-9266,927))}
            ),
            ('warehouse10','distributor12','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-14900,1490)),'emission': DataSeries(self.random_param(-8523,852))}
            ),
            ('warehouse10','distributor13','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-19600,1960)),'emission': DataSeries(self.random_param(-11211,1121))}
            ),
            ('warehouse10','distributor14','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6350,635)),'emission': DataSeries(self.random_param(-3632,363))}
            ),
            ('warehouse11','distributor12','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-18700,1870)),'emission': DataSeries(self.random_param(-10696,1070))}
            ),
            ('warehouse11','distributor13','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-2000,200)),'emission': DataSeries(self.random_param(-1144,114))}
            ),
            ('warehouse11','distributor14','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-18550,1855)),'emission': DataSeries(self.random_param(-10611,1061))}
            ),
            ('distributor12','retailer15','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-19450,1945)),'emission': DataSeries(self.random_param(-11125,1112))}
            ),
            ('distributor12','retailer16','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-9650,965)),'emission': DataSeries(self.random_param(-5520,552))}
            ),
            ('distributor12','retailer17','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-19050,1905)),'emission': DataSeries(self.random_param(-10897,1090))}
            ),
            ('distributor12','retailer18','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-9000,900)),'emission': DataSeries(self.random_param(-5148,515))}
            ),
            ('distributor12','retailer19','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-6900,690)),'emission': DataSeries(self.random_param(-3947,394))}
            ),
            ('distributor13','retailer15','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-8050,805)),'emission': DataSeries(self.random_param(-4605,460))}
            ),
            ('distributor13','retailer16','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-10650,1065)),'emission': DataSeries(self.random_param(-6092,609))}
            ),
            ('distributor13','retailer17','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-18400,1840)),'emission': DataSeries(self.random_param(-10525,1052))}
            ),
            ('distributor13','retailer18','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-8300,830)),'emission': DataSeries(self.random_param(-4748,475))}
            ),
            ('distributor13','retailer19','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-18850,1885)),'emission': DataSeries(self.random_param(-10782,1078))}
            ),
            ('distributor14','retailer15','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-16600,1660)),'emission': DataSeries(self.random_param(-9495,950))}
            ),
            ('distributor14','retailer16','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-15100,1510)),'emission': DataSeries(self.random_param(-8637,864))}
            ),
            ('distributor14','retailer17','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-5900,590)),'emission': DataSeries(self.random_param(-3375,338))}
            ),
            ('distributor14','retailer18','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-4000,400)),'emission': DataSeries(self.random_param(-2288,229))}
            ),
            ('distributor14','retailer19','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = DataSeries(self.random_param(2,1)),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(-13950,1395)),'emission': DataSeries(self.random_param(-7979,798))}
            ),
            ('retailer15','marketa','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(1000000,100000)),'emission': ConstantSeries(0)}
            ),
            ('retailer16','marketb','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(1010000,101000)),'emission': ConstantSeries(0)}
            ),
            ('retailer17','marketc','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(1050000,105000)),'emission': ConstantSeries(0)}
            ),
            ('retailer18','marketd','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(1030000,103000)),'emission': ConstantSeries(0)}
            ),
            ('retailer19','markete','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': DataSeries(self.random_param(1040000,104000)),'emission': ConstantSeries(0)}
            )
        }

        return State(self.complex_sc_nodes, self.complex_sc_edges)
