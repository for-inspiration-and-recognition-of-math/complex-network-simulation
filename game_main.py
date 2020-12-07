from simulations import *
from cleaning import *
from visualization import *
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import time

def complexity_graph():
        # timer
        node_count = [i for i in range (500, 2000, 500)]
        time_elapsed = []

        for i in (node_count):
                start = time.time()
                
                num_nodes = i
                numIterations = 1
                
                # gneerate network
                nodes, adj = generate_random_network("BA", num_nodes, p=0.9)
                # nodes, adj = facebook_clean()

                
                # run simulation
                saveRate = 1

                simulation = Simulation(numIterations, saveRate, strategy= 2) # choose from 0, 1, 2
                
                nodes_list, adj_list = simulation(nodes, adj)


                # calculations
                # G = nx.convert_matrix.from_numpy_matrix(adj)
                # shortest = nx.average_shortest_path_length(G)
                # closeness = nx.closeness_centrality(G)
                # print("Closeness: " + str(sum(closeness.values())/len(adj)))
                
                # print("Clustering: " + str(nx.average_clustering(G)))

                end = time.time()
                elapsed = abs(end-start)
                time_elapsed.append(elapsed)
                print("Node count: {0}, Time Elapsed: {1} seconds".format(i, elapsed))            


                # convert to np array
                x = np.array(node_count)
                y = np.array(time_elapsed)    
                
                # plot with line of best fit
                #a, b, c = np.polyfit(node_count, time_elapsed, 2)
                plt.plot(x, y, 'o')
                #plt.plot(node_count, a*time_elapsed**2 + b*time_elapsed + c)
                coefs = poly.polyfit(x, y, 2)
                x_new = np.linspace(x[0], x[-1], num=len(x)*10)
                print(x_new)
                ffit = poly.polyval(x_new, coefs)
                
                sig_figs = 7
                # formula = "{0}x^3 + {1}x^2 + {2}x + {3}".format(round(coefs[0],sig_figs), round(coefs[1],sig_figs), round(coefs[2], sig_figs), round(coefs[3], sig_figs))
                formula = "{0}x^2 + {1}x+ {2}".format(round(coefs[0],sig_figs), round(coefs[1],sig_figs), round(coefs[2], sig_figs))
                print(formula)
                plt.plot(x_new, ffit, label=formula)
                
                # plt.text(0, 1,'matplotlib', horizontalalignment='center',
                #      verticalalignment='center'))
                plt.legend()
                plt.show()             
                plt.close()




if __name__ == '__main__':

        num_nodes = 600
        numIterations = 15
        
        # gneerate network
        nodes, adj = generate_random_network("BA", num_nodes, p=0.9)
        # nodes, adj = facebook_clean()

        
        # run simulation
        saveRate = 1
        payoff = 1
        strategy = 2

        simulation = Simulation(numIterations, saveRate, strategy, payoff)
        nodes_list, adj_list = simulation(nodes, adj)

        
        # visualize
        visualize_list(nodes_list, adj_list, numIterations, "ER+strat2+{0}".format(payoff), True)
         # 4th parameter (model name) is for bookkeeping purposes
         # 5th parameter (defaulted to True) means position is LOCKED for future iteration 
         # choose False to recalculate the position of Nodes every iteration (which significantly slows down the process)
        

