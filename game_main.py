from simulations_updated import *
from cleaning import *
from visualization import *
from analysis import *
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import time

def complexity_graph():
        # timer
        node_count = [i for i in range (500, 2500, 500)]
        time_elapsed = []

        for i in (node_count):
                start = time.time()
                
                num_nodes = i
                numIterations = 1
                
                # generate network
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
        print("raw coefs: " + repr(coefs))

        x_new = np.linspace(x[0], x[-1], num=len(x)*10)
        ffit = poly.polyval(x_new, coefs)
        
        sig_figs = 7
        # formula = "{0}x^3 + {1}x^2 + {2}x + {3}".format(round(coefs[0],sig_figs), round(coefs[1],sig_figs), round(coefs[2], sig_figs), round(coefs[3], sig_figs))
        formula = "{0}x^2 + {1}x+ {2}".format(round(coefs[2],sig_figs), round(coefs[1],sig_figs), round(coefs[0], sig_figs))
        print("formatted formula: " + repr(formula))
        plt.plot(x_new, ffit, label=formula)
        
        # plt.text(0, 1,'matplotlib', horizontalalignment='center',
        #      verticalalignment='center'))
        plt.legend()
        plt.show()             
        plt.close()


def load_data(strat, payoff, saveRate):
        start = time.perf_counter()
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        fd = open(f'{dir_path}/savedInfo/adjMat_strategy{strat}payoff{payoff}saveRate{saveRate}.pickle', 'rb')
        adj_list = pickle.load(fd)

        fd = open(f'{dir_path}/savedInfo/nodesDict_strategy{strat}payoff{payoff}saveRate{saveRate}.pickle', 'rb')
        nodes_list = pickle.load(fd)
        
        numIterations = len(nodes_list)-1       # -1 since it included the initial, uniterated graph as well
        
        print(f'time used to unpickle : {time.perf_counter()-start}s')
        
        return nodes_list, adj_list, numIterations


if __name__ == '__main__':
        start = time.perf_counter()
        
        num_nodes = int(input("Node # (default: 200): ") or "200")
        numIterations = int(input("Iteration # (default: 5): ",) or "5")
        model = str(input("Model (default: ER): ") or "ER")
        strat = int(input("Strat # (default: 3): ") or "3")
        payoff = int(input("Payoff # (default: 0): ") or "0")
        saveRate = int(input("Saverate # (default: 1): ") or "1")
        
        load_flag = int(input("Run(0) or Load(1) (default: 0): ") or "0")
        
        # Step 1. generate network
        nodes, adj = generate_random_network(model, num_nodes, p=0.2)
        # nodes, adj = facebook_clean()
        # nodes, adj = karate_clean()
        
        
        # Step 2. run simulation
        if load_flag:
                nodes_list, adj_list, numIterations = load_data(strat, payoff, saveRate)              # Load pickles to save time
        else:
                simulation = Simulation(numIterations, saveRate, strat, payoff) # choose from 0, 1, 2
                nodes_list, adj_list = simulation(nodes, adj)
        
        
        print(f'time used to simulate: {time.perf_counter()-start}s')
        
        
        
        # Step 3. run all measurements
        measures_list = all_measures_master(nodes_list, adj_list, "{0}+s{1}+p{2}".format(model, strat, payoff))
        
        
        
        
        # Step 4. Visualize
        
        start = time.perf_counter()
        
        visualize(nodes_list, adj_list, measures_list, numIterations, "{0}+s{1}+p{2}".format(model, strat, payoff), pos_lock=True)
         # 4th parameter (model name) is for bookkeeping purposes
         # 5th parameter (defaulted to True) means position is LOCKED for future iteration
         # choose False to recalculate the position of Nodes every iteration (which significantly slows down the process)
        
        
        print(f'time used to visualize: {time.perf_counter()-start}s \n')

        # complexity_graph()