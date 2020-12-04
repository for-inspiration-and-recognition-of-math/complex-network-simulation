from simulations import *
from cleaning import *
from visualization import *

if __name__ == '__main__':
        num_nodes = 10
        numIterations = 20
        
        # generate network
        nodes, adj = generate_random_network("ER", num_nodes, p=0.8)
        

        
        # run simulation
        saveRate = 1
        simulation_simplest = Simulation(numIterations, saveRate, strategy= 1) # choose from 0, 1, 2
        
        
        nodes_list, adj_list = simulation_simplest(nodes, adj)

        print(adj_list[0])
        print(adj_list[1])
        print(adj_list[-1])
        print([[node.wealth for node in nodes.values()] for nodes in nodes_list])

        print([[node.status for node in nodes.values()] for nodes in nodes_list])

        #


        
        # visualize
        
        # visualize_list(nodes_list, adj_list, numIterations)
        # generate_gif()

        

