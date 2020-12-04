import networkx as nx
import matplotlib.pyplot as plt
from simulations import *
from cleaning import *
from visualization import *

if __name__ == '__main__':
        num_nodes = 50
        numIterations = 100
        
        # gneerate network
        nodes, adj = generate_random_network("ER", num_nodes, p=0.8)
        

        
        # run simulation
        saveRate = 1
        simulation_simplest = Simulation(numIterations, saveRate, strategy= 2) # choose from 0, 1, 2
        
        
        nodes_list, adj_list = simulation_simplest(nodes, adj)

        
        # visualize
        
        visualize_list(nodes_list, adj_list, numIterations)
        generate_gif()

        

