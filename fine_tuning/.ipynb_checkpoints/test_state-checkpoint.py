'''A method to generate fixed SC network, consisting of simple, moderate, and complex network, with random generated demand'''
#pip install git+https://github.com/LucasAlegre/morl-baselines.git
import sys
import os

print(sys.path)

#TODO add hardcoded complex supply chain with randomly extracted demand
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

#generate random demand parameters following demand distribution currently used in MORL project
class test_sc:
    
    def __init__(self):
        settings.episode_length = 100# set episode length to be 100
        demand = pd.read_excel('/home/rifnyrachman7/_metamorl/fine_tuning/Data Input.xlsx', sheet_name = 'Data Demand') #Change with actuall file repository
        self.demand_a = np.array(demand['Demand A'])
        self.demand_b = np.array(demand['Demand B'])
        self.demand_c = np.array(demand['Demand C'])
        self.demand_d = np.array(demand['Demand D'])
        self.demand_e = np.array(demand['Demand E'])
        
    def simple_network(self): #Use DataSeries to read parameter data from data input
        settings.episode_length = 100 #Set episode length to be 100
        self.simple_sc_nodes = {
            'farm1': Node(
                control             = False
            ),
            'manufacturer2': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 380},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-1100)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'manufacturer3': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 350},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-1300)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'retailer4': Node(
                control             = True,
                initial_inventory   = {'product': 400},
                costs               = {'monetary': {'product': ConstantSeries(-1200)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'retailer5': Node(
                control             = True,
                initial_inventory   = {'product': 80},
                costs               = {'monetary': {'product': ConstantSeries(-1500)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'marketa': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_a)}
            ),
            'marketb': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_b)}
            )
        }
        
        self.simple_sc_edges = {
            ('farm1','manufacturer2','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, #Q: is it an initial value?
                costs       = {'monetary': ConstantSeries(-2200),'emission': ConstantSeries(-1258)} #ordering cost/unit + transport cost/unit/leadtime
            ),
            ('farm1','manufacturer3','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-6900),'emission': ConstantSeries(-3947)} #(p/(L+1))+g
            ),
            ('manufacturer2','manufacturer2','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-20000),'emission': ConstantSeries(-50126)}
            ),
            ('manufacturer3','manufacturer3','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-22000),'emission': ConstantSeries(-45754)}
            ),
            ('manufacturer2','retailer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-10550),'emission': ConstantSeries(-6035)}
            ),
            ('manufacturer2','retailer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-4300),'emission': ConstantSeries(-2460)}
            ),
            ('manufacturer3','retailer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-4850),'emission': ConstantSeries(-2774)}
            ),
            ('manufacturer3','retailer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-7500),'emission': ConstantSeries(-4290)}
            ),
            ('retailer4','marketa','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(200000),'emission': ConstantSeries(0)}
            ),
            ('retailer5','marketb','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(200000),'emission': ConstantSeries(0)} #set price here
            )
        }
        
        return State(self.simple_sc_nodes, self.simple_sc_edges)
        
    def moderate_network(self):
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
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-1100)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'manufacturer4': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 350},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-1300)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'manufacturer5': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 400},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-1200)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'distributor6': Node(
                control             = True,
                initial_inventory   = {'product': 80},
                costs               = {'monetary':{'product': ConstantSeries(-1500)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'distributor7': Node(
                control             = True,
                initial_inventory   = {'product': 110},
                costs               = {'monetary':{'product': ConstantSeries(-2000)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'retailer8': Node(
                control             = True,
                initial_inventory   = {'product': 0},
                costs               = {'monetary': {'product': ConstantSeries(-2500)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'retailer9': Node(
                control             = True,
                initial_inventory   = {'product': 0},
                costs               = {'monetary': {'product': ConstantSeries(-3000)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'retailer10': Node(
                control             = True,
                initial_inventory   = {'product': 0},
                costs               = {'monetary': {'product': ConstantSeries(-2000)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'marketa': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_a)}
            ),
            'marketb': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_b)}
            ),
            'marketc': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_c)}
            )
        }


        self.moderate_sc_edges = {
            ('farm1','manufacturer3','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, #Q: is it an initial value?
                costs       = {'monetary': ConstantSeries(-2200),'emission': ConstantSeries(-1258)} #ordering cost/unit + transport cost/unit/leadtime
            ),
            ('farm1','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-6900),'emission': ConstantSeries(-3947)}
            ),
            ('farm1','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-5650),'emission': ConstantSeries(-3232)} 
            ),
            ('farm2','manufacturer3','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, #Q: is it an initial value?
                costs       = {'monetary': ConstantSeries(-10550),'emission': ConstantSeries(-6035)} 
            ),
            ('farm2','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-6500),'emission': ConstantSeries(-3718)} 
            ),
            ('farm2','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-6300),'emission': ConstantSeries(-3604)} 
            ),
            ('manufacturer3','manufacturer3','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-20000),'emission': ConstantSeries(-50126)}
            ),
            ('manufacturer4','manufacturer4','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-22000),'emission': ConstantSeries(-45754)}
            ),
            ('manufacturer5','manufacturer5','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-23000),'emission': ConstantSeries(-54491)}
            ),
            ('manufacturer3','distributor6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-750),'emission': ConstantSeries(-429)}
            ),
            ('manufacturer3','distributor7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-4300),'emission': ConstantSeries(-2460)}
            ),
            ('manufacturer4','distributor6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-6300),'emission': ConstantSeries(-3604)}
            ),
            ('manufacturer4','distributor7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-2300),'emission': ConstantSeries(-1316)}
            ),
            ('manufacturer5','distributor6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-4950),'emission': ConstantSeries(-2831)}
            ),
            ('manufacturer5','distributor7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-750),'emission': ConstantSeries(-429)}
            ),
            ('distributor6','retailer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-10950),'emission': ConstantSeries(-6263)}
            ),
            ('distributor6','retailer9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-6250),'emission': ConstantSeries(-3575)}
            ),
            ('distributor6','retailer10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-9500),'emission': ConstantSeries(-5434)}
            ),
            ('distributor7','retailer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-16400),'emission': ConstantSeries(-9381)}
            ),
            ('distributor7','retailer9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-11600),'emission': ConstantSeries(-6635)}
            ),
            ('distributor7','retailer10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-5800),'emission': ConstantSeries(-3318)}
            ),
            ('retailer8','marketa','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(200000),'emission': ConstantSeries(0)}
            ),
            ('retailer9','marketb','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(210000),'emission': ConstantSeries(0)}
            ),
            ('retailer10','marketc','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(205000),'emission': ConstantSeries(0)} #set price here
            )
        }

        return State(self.moderate_sc_nodes, self.moderate_sc_edges)
    
    def complex_network(self):
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
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-2300)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'manufacturer5': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 267},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-3500)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'manufacturer6': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 342},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-2200)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'manufacturer7': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 211},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-1100)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'manufacturer8': Node(
                control             = True,
                initial_inventory   = {'raw': 0, 'product': 162},
                costs               = {'monetary': {'raw': ConstantSeries(0), 'product': ConstantSeries(-2900)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'warehouse9': Node(
                control             = True,
                initial_inventory   = {'product': 195},
                costs               = {'monetary':{'product': ConstantSeries(-3700)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'warehouse10': Node(
                control             = True,
                initial_inventory   = {'product': 333},
                costs               = {'monetary':{'product': ConstantSeries(-1100)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'warehouse11': Node(
                control             = True,
                initial_inventory   = {'product': 96},
                costs               = {'monetary':{'product': ConstantSeries(-3600)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'distributor12': Node(
                control             = True,
                initial_inventory   = {'product': 285},
                costs               = {'monetary':{'product': ConstantSeries(-3300)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'distributor13': Node(
                control             = True,
                initial_inventory   = {'product': 68},
                costs               = {'monetary':{'product': ConstantSeries(-2600)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'distributor14': Node(
                control             = True,
                initial_inventory   = {'product': 379},
                costs               = {'monetary':{'product': ConstantSeries(-3000)},
                                      'emission':{'product':ConstantSeries(-2)}}
            ),
            'retailer15': Node(
                control             = True,
                initial_inventory   = {'product': 344},
                costs               = {'monetary': {'product': ConstantSeries(-1700)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'retailer16': Node(
                control             = True,
                initial_inventory   = {'product': 66},
                costs               = {'monetary': {'product': ConstantSeries(-2900)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'retailer17': Node(
                control             = True,
                initial_inventory   = {'product': 356},
                costs               = {'monetary': {'product': ConstantSeries(-2700)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'retailer18': Node(
                control             = True,
                initial_inventory   = {'product': 382},
                costs               = {'monetary': {'product': ConstantSeries(-2300)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'retailer19': Node(
                control             = True,
                initial_inventory   = {'product': 362},
                costs               = {'monetary': {'product': ConstantSeries(-3700)},
                                      'emission': {'product': ConstantSeries(-2)}}
            ),
            'marketa': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_a)}
            ),
            'marketb': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_b)}
            ),
            'marketc': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_c)}
            ),
            'marketd': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_d)}
            ),
            'markete': Node(
                control             = False,
                demand              = {'product': DataSeries(self.demand_e)}
            )
        }


        self.complex_sc_edges = {
            ('farm1','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-5350),'emission': ConstantSeries(-3060)} #ordering cost/unit + transport cost/unit/leadtime
            ),
            ('farm1','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-2650),'emission': ConstantSeries(-1516)} 
            ),
            ('farm1','manufacturer6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-18450),'emission': ConstantSeries(-10553)} 
            ),
            ('farm1','manufacturer7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-16000),'emission': ConstantSeries(-9152)} 
            ),
            ('farm1','manufacturer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-14400),'emission': ConstantSeries(-8237)} 
            ),
           ('farm2','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, #Q: is it an initial value?
                costs       = {'monetary': ConstantSeries(-6000),'emission': ConstantSeries(-3432)} #ordering cost/unit + transport cost/unit/leadtime
            ),
            ('farm2','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-1750),'emission': ConstantSeries(-1001)} 
            ),
            ('farm2','manufacturer6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-7450),'emission': ConstantSeries(-4261)} 
            ),
            ('farm2','manufacturer7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-13300),'emission': ConstantSeries(-7608)} 
            ),
            ('farm2','manufacturer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-1700),'emission': ConstantSeries(-972)} 
            ),
            ('farm3','manufacturer4','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-3600),'emission': ConstantSeries(-2059)} 
            ),
            ('farm3','manufacturer5','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-2950),'emission': ConstantSeries(-1687)} 
            ),
            ('farm3','manufacturer6','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-12350),'emission': ConstantSeries(-7064)} 
            ),
            ('farm3','manufacturer7','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-6250),'emission': ConstantSeries(-3575)}
            ),
            ('farm3','manufacturer8','transport'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'raw': 1}},
                length      = ConstantSeries(2),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'raw': ConstantSeries(0)}, 
                costs       = {'monetary': ConstantSeries(-18550),'emission': ConstantSeries(-10611)}
            ),
            ('manufacturer4','manufacturer4','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-20000),'emission': ConstantSeries(-50126)}
            ),
            ('manufacturer5','manufacturer5','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-22000),'emission': ConstantSeries(-45754)}
            ),
            ('manufacturer6','manufacturer6','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-21000),'emission': ConstantSeries(-54491)}
            ),
            ('manufacturer7','manufacturer7','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-20000),'emission': ConstantSeries(-61232)}
            ),
            ('manufacturer8','manufacturer8','manufacture'): Edge(
                control     = True,
                process     = {'inputs': {'raw': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'raw': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-23000),'emission': ConstantSeries(-55157)}
            ),
            ('manufacturer4','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-19900),'emission': ConstantSeries(-11383)}
            ),
            ('manufacturer4','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-3400),'emission': ConstantSeries(-1945)}
            ),
            ('manufacturer4','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-8100),'emission': ConstantSeries(-4633)}
            ),
            ('manufacturer5','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-15150),'emission': ConstantSeries(-8666)}
            ),
            ('manufacturer5','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-6600),'emission': ConstantSeries(-3775)}
            ),
            ('manufacturer5','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-6450),'emission': ConstantSeries(-3689)}
            ),
            ('manufacturer6','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-16950),'emission': ConstantSeries(-9695)}
            ),
            ('manufacturer6','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-15800),'emission': ConstantSeries(-9038)}
            ),
            ('manufacturer6','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-8150),'emission': ConstantSeries(-4662)}
            ),
            ('manufacturer7','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-16150),'emission': ConstantSeries(-9238)}
            ),
            ('manufacturer7','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-12600),'emission': ConstantSeries(-7207)}
            ),
            ('manufacturer7','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-6750),'emission': ConstantSeries(-3861)}
            ),
            ('manufacturer8','warehouse9','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-10300),'emission': ConstantSeries(-5892)}
            ),
            ('manufacturer8','warehouse10','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-10900),'emission': ConstantSeries(-6235)}
            ),
            ('manufacturer8','warehouse11','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-16300),'emission': ConstantSeries(-9324)}
            ),
            ('warehouse9','distributor12','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-19650),'emission': ConstantSeries(-11240)}
            ),
            ('warehouse9','distributor13','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-19250),'emission': ConstantSeries(-11011)}
            ),
            ('warehouse9','distributor14','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-16200),'emission': ConstantSeries(-9266)}
            ),
            ('warehouse10','distributor12','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-14900),'emission': ConstantSeries(-8523)}
            ),
            ('warehouse10','distributor13','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-19600),'emission': ConstantSeries(-11211)}
            ),
            ('warehouse10','distributor14','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-6350),'emission': ConstantSeries(-3632)}
            ),
            ('warehouse11','distributor12','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-18700),'emission': ConstantSeries(-10696)}
            ),
            ('warehouse11','distributor13','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-2000),'emission': ConstantSeries(-1144)}
            ),
            ('warehouse11','distributor14','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-18550),'emission': ConstantSeries(-10611)}
            ),
            ('distributor12','retailer15','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-19450),'emission': ConstantSeries(-11125)}
            ),
            ('distributor12','retailer16','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-9650),'emission': ConstantSeries(-5520)}
            ),
            ('distributor12','retailer17','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-19050),'emission': ConstantSeries(-10897)}
            ),
            ('distributor12','retailer18','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-9000),'emission': ConstantSeries(-5148)}
            ),
            ('distributor12','retailer19','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-6900),'emission': ConstantSeries(-3947)}
            ),
            ('distributor13','retailer15','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-8050),'emission': ConstantSeries(-4605)}
            ),
            ('distributor13','retailer16','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-10650),'emission': ConstantSeries(-6092)}
            ),
            ('distributor13','retailer17','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-18400),'emission': ConstantSeries(-10525)}
            ),
            ('distributor13','retailer18','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-8300),'emission': ConstantSeries(-4748)}
            ),
            ('distributor13','retailer19','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-18850),'emission': ConstantSeries(-10782)}
            ),
            ('distributor14','retailer15','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-16600),'emission': ConstantSeries(-9495)}
            ),
            ('distributor14','retailer16','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-15100),'emission': ConstantSeries(-8637)}
            ),
            ('distributor14','retailer17','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-5900),'emission': ConstantSeries(-3375)}
            ),
            ('distributor14','retailer18','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-4000),'emission': ConstantSeries(-2288)}
            ),
            ('distributor14','retailer19','transport'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(2),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(-13950),'emission': ConstantSeries(-7979)}
            ),
            ('retailer15','marketa','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(1000000),'emission': ConstantSeries(0)}
            ),
            ('retailer16','marketb','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(1010000),'emission': ConstantSeries(0)}
            ),
            ('retailer17','marketc','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(1050000),'emission': ConstantSeries(0)}
            ),
            ('retailer18','marketd','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(1030000),'emission': ConstantSeries(0)}
            ),
            ('retailer19','markete','supply'): Edge(
                control     = True,
                process     = {'inputs': {'product': 1}, 'outputs': {'product': 1}},
                length      = ConstantSeries(0),
                inputs      = {'product': ConstantSeries(0)},
                outputs     = {'product': ConstantSeries(0)},
                costs       = {'monetary': ConstantSeries(1040000),'emission': ConstantSeries(0)} 
            )
        }

        return State(self.complex_sc_nodes, self.complex_sc_edges)
