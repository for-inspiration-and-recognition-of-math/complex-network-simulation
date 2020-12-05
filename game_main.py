import networkx as nx
import matplotlib.pyplot as plt
from simulations import *
from cleaning import *
from visualization import *

if __name__ == '__main__':
        num_nodes = 50
        numIterations = 50
        
        # gneerate network
        nodes, adj = generate_random_network("ER", num_nodes, p=0.6)
        

        
        # run simulation
        saveRate = 1
        simulation = Simulation(numIterations, saveRate, strategy= 2) # choose from 0, 1, 2
        
        
        nodes_list, adj_list = simulation(nodes, adj)

        
        # visualize
        
        visualize_list(nodes_list, adj_list, numIterations, "ER+strat2")        # 4th parameter (model name) is for bookkeeping purposes
        

        

