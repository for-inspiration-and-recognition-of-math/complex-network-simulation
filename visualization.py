import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import collections
import os
import imageio
import time
import concurrent.futures
from multiprocessing import Process, Pipe
import threading
from tqdm import tqdm


good_good_edges_list, mixed_edges_list, bad_bad_edges_list = {}, {}, {}
# no these are not misnomers, they will eventually be sorted into lists

# helper function: create directory if not already exists
def creat_dir(folder_name, catagory="visualization"):
        dir_path = os.path.dirname(os.path.realpath(__file__))          # NOT os.getcwd() <——> this incantation is faulty
        path = "{directory}/{catagory}/{subdirectory}/".format(directory = dir_path, catagory=catagory, subdirectory=folder_name)
        mode = 0o755
        # output graph png
        try:  
                os.makedirs(path, mode)
        except OSError as error:  
                pass
        return path

def commandline_decorator(interface):
        def inner(*args, **kwargs):     # must have inner function to take and transfer the proper arguments
                print("\n----------------------------------")
                print("      Begin Visualization...")
                print("----------------------------------")
                
                interface(*args, **kwargs)
                
                dir_path = os.path.dirname(os.path.realpath(__file__))
                print("\n----------------------------------")
                print("     Visualization Complete!")
                print("----------------------------------\n")
                print(f'View Visualizations in: \n{dir_path}/visualization\n\n')
        return inner

'''
# INTERFACE (i.e. the only function you need to use): 
        # if input is a list of node lists and list of adj matricies, 
        # outputs graph given folder_name and generates gif
        # 4th parameter (model name) is for bookkeeping purposes
        # 5th parameter (defaulted to True) means position is LOCKED for future iteration 
        # 6th parameter (continuation=False) set to true if continuing a previous simulation using pickled data
        # choose False to recalculate the position of Nodes every iteration (which significantly slows down the process)
'''
@commandline_decorator
def visualize_list(nodesDict_list, adjMatrix_list, iterations, model_name, pos_lock=True, continuation=False):
        print("\nGenerating graphs...")
        # create directories and generate correct absolute path name
        path_network, path_node_histogram, path_edge_histogram, path_animation, path_evolution = creat_dir(model_name + " (network)"), creat_dir(model_name + " (node-histogram)"), creat_dir(model_name + " (edge-histogram)"), creat_dir("animation"), creat_dir("evolution")
        
        # cleans directory containing the network and histogram visualizations
        for model_path in [path_network, path_node_histogram, path_edge_histogram]:
                for root, dirs, files in os.walk(model_path, onerror=lambda err: print("OSwalk error: " + repr(err))):
                        for file in files:
                                os.remove(os.path.join(root, file))
        
        # ---------------------------------------------------------
        # generating graphs using multiple subprocesses 
        #       (use previous releases for non-concurrent version)
        # ---------------------------------------------------------
        global good_good_edges_list, mixed_edges_list, bad_bad_edges_list
        runs = []

        def update_bar(pbar, total):
                cur = 0
                while len(runs) < total:
                        x_sync = len(runs)
                        pbar.update(x_sync - cur)
                        cur=x_sync
                pbar.update(len(runs) - cur)

        
        with concurrent.futures.ProcessPoolExecutor() as executor:
                pbar = tqdm(total = iterations+1, unit='graphs')
                
                t1 = threading.Thread(target=update_bar, args=[pbar, iterations+1])
                t1.start()
                
                for i in range(0, iterations + 1):
                        f = executor.submit(visualization, nodesDict_list[i], adjMatrix_list[i], 
                                        path_network, path_node_histogram, path_edge_histogram, i, pos_lock)
                        f.add_done_callback(lambda x: print(f'{x.result()[0]} ', end='', flush=True))
                        runs.append(f)

                t1.join()
                pbar.close()
                print("all subprocesses queued, waiting to complete...\n\nAwaiting subprocess completion... \n>", end = ' ', flush=True)
                
                
                for run in concurrent.futures.as_completed(runs):
                        index = run.result()[0]
                        good_good_edges_list[index] = run.result()[1]
                        mixed_edges_list[index] = run.result()[2]
                        bad_bad_edges_list[index] = run.result()[3]
                print("\n<--- all subprocesses completed --->")
        
        ### convert from dictionary into lists sorted by dictionary key ###

        good_good_edges_list = sorted(good_good_edges_list.items())     # coverts into sorted tuples
        good_good_edges_list = [ x[1] for x in good_good_edges_list ]    # converts into lists of lists (removes the index => tuple[0])
        
        mixed_edges_list = sorted(mixed_edges_list.items())     
        mixed_edges_list = [ x[1] for x in mixed_edges_list ] 
        
        bad_bad_edges_list = sorted(bad_bad_edges_list.items())     
        bad_bad_edges_list = [ x[1] for x in bad_bad_edges_list ] 
        
        
        # compile PNGs into gif (for both network and histogram)
        print("\nCompiling GIF...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(generate_gif, model_name + " (network)", path_network, path_animation)
                executor.submit(generate_gif, model_name + " (edge-histogram)", path_edge_histogram, path_animation)
                executor.submit(generate_gif, model_name + " (node-histogram)", path_node_histogram, path_animation)
        
        # # generating graph over time for node status + edge type
        plot_status_over_time(nodesDict_list, model_name, path_evolution, 
                              good_good_edges_list, mixed_edges_list, bad_bad_edges_list, )
        
def test():
        global good_good_edges_list
        print(f'the length is: {len(good_good_edges_list)}')

'''
func generate_png_csv (the alternate interface for 1 time visualizations)
@param 
        nodes: dictionary of nodes
        adj: adjacency matrix (numpy 2D array)
        abs_path: absolute path to the subdirectory in which all graphs will be stored (defaults to "graph")
        index (optional): current simulation iteration (needed to generate png, else just oopens for 1 time viewing)
        color_edges (true by default): colors edges depending on the nodes attached
@return
        none
@usage
        by passing in a dictionary of nodes along with its adjacency matrix
        a visualization of the data will be generated
        if an optional index is passed, a PNG named: "[index].png" will be generated 
        in "graph" subdirectory of current working directory
        E.g. index of 1 will generate "1.png" in CWD/graph
        if an optional color_edges is passed, then edges will be colored with this rule:
                if both stubs connect to cooperating nodes, edge = dark green
                if both stubs connect to defecting nodes, edge = red
                if one stub connects to cooperating node and the other defecting node, edge = orange (leaning yellow)
'''

# node colors [0,1] : green, red (orangish), 
# edge colors [2,3,4] : dark green, orange(leaning yellow), red (leaning magenta)
color = ['#03b500', '#b52a00', "#077d11", "#ffbb3d", "#db0231"]    
optimized_pos, position_lock = False, False            #setting position variables

def visualization(nodes, adj, path_network, path_node_histogram, path_edge_histogram, index=-1, pos_lock = True, color_edges=True):
        global color

        G = nx.convert_matrix.from_numpy_matrix(adj)
        nodes_color = [color[1] if (nodes[i].status == 1) else color[0] for i in range (len(nodes))]

        # node positioning
        if pos_lock:
                global optimized_pos, position_lock
                if not position_lock:
                        optimized_pos = nx.spring_layout(G)
                        position_lock = True
        else: optimized_pos = nx.spring_layout(G)  
        # optimized_pos = nx.shell_layout(G) / nx.spiral_layout(G) / nx.spectral_layout(G)
        
        plt.figure(figsize = (10, 10))
        plt.title("Iteration={0}".format(index))
        
        
        # ---------------------------------------------------------
        # generating network visualization
        # ---------------------------------------------------------
        
        edge_width = 0.1
        edge_alpha = 0.7
        nx.draw(G, optimized_pos, with_labels=False, node_color = nodes_color, node_size = 15, width=edge_width)

        # Add relationship-sensitive coloring to edges
        if color_edges:
                good_good_edges = []
                mixed_edges = []
                bad_bad_edges = []
                
                for i in range (len(adj)):
                        for j in range (len(adj)):
                                if adj[i][j] > 0:
                                        if nodes[i].status == 0 and nodes[j].status == 0: good_good_edges.append((i,j))
                                        elif nodes[i].status == 1 and nodes[j].status == 1: bad_bad_edges.append((i,j))
                                        else: mixed_edges.append((i,j))

                nx.draw_networkx_edges(G,optimized_pos,
                        edgelist=good_good_edges,
                        width=edge_width,alpha=edge_alpha,edge_color=color[2])
                nx.draw_networkx_edges(G,optimized_pos,
                        edgelist=mixed_edges,
                        width=edge_width,alpha=edge_alpha,edge_color=color[3])
                nx.draw_networkx_edges(G,optimized_pos,
                        edgelist=bad_bad_edges,
                        width=edge_width,alpha=edge_alpha,edge_color=color[4])
        
        # saving network graphs and histograms as PNGs in separate folders
        if index != -1: plt.savefig(path_network + "net-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        # ---------------------------------------------------------
        # plotting degree distribution histogram
        # ---------------------------------------------------------
        
        # 1. edge histogram

        heights = [len(good_good_edges), len(mixed_edges), len(bad_bad_edges)]
        edge_types = ("good-good", "mixed", "bad-bad")
        
        bar_spacing = np.arange(len(edge_types))
        plt.bar(bar_spacing, heights, width=0.80, color=[color[2], color[3], color[4]])    # generate the histogram
        
        # setting attributes of bar graph
        plt.title("Edge Type Distribution (iter={0})".format(index))
        plt.ylabel("Number of Edges")
        plt.xlabel("Edge Type")
        
        plt.xticks(bar_spacing, edge_types)

        top = len(good_good_edges) + len(bad_bad_edges) + len(mixed_edges)
        if top <= 0: return index, good_good_edges, mixed_edges, bad_bad_edges            
                # since no more interactions (because no more edges), simulation is finished (just return empty lists)
        
        plt.ylim([0, top])
        plt.grid(True, axis='y')
        
        if index != -1: plt.savefig(path_edge_histogram + "edge-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        # 2. node histogram
        good_nodes_count, bad_nodes_count = 0, 0
        for col in nodes_color:
                if col == color[0]: good_nodes_count += 1
                elif col == color[1]: bad_nodes_count += 1
        
        heights = [good_nodes_count, bad_nodes_count]
        edge_types = ("Cooperator", "Defector")
        
        bar_spacing = np.arange(len(edge_types))
        plt.bar(bar_spacing, heights, width=0.80, color=[color[0], color[1]])    # generate the histogram
        
        # setting attributes of bar graph
        plt.title("Node Type Distribution (iter={0})".format(index))
        plt.ylabel("Number of Nodes")
        plt.xlabel("Node Type")
        
        plt.xticks(bar_spacing, edge_types)
        plt.ylim([0, len(nodes_color)])
        plt.grid(True, axis='y')
        
        if index != -1: plt.savefig(path_node_histogram + "/" +"node-" + repr(index) + ".png", format="PNG")      # output graph png 
        else: plt.show()
        plt.close()
        
        
        # enable parallel concurrency
        return index, good_good_edges, mixed_edges, bad_bad_edges


# graph the status of all nodes over time (proportion normalized by total)
def plot_status_over_time(nodesDict_list, model_name, path_evolution, good_good_edges_list, mixed_edges_list, bad_bad_edges_list):
        print("\nPlotting node status evolution...")
        
        num_defect = []
        num_nodes = len(nodesDict_list[0])
        for nodeDict in nodesDict_list:
                def_count = 0
                for nodes in nodeDict.values():
                        def_count += nodes.status         # assuming 1 represents defect, no multiedges, and nodes have binary status (0/1)
                num_defect.append(def_count/num_nodes)
        
        num_coop = [1 - def_count for def_count in num_defect]      # populate cooperating array with the 'complement' of num_defect
        
        # convert to np array
        iteration_count = [ i for i in range (0, len(nodesDict_list)) ]
        x = np.array(iteration_count)
        y1 = np.array(num_coop)    
        y2 = np.array(num_defect)    
        
        # plot with line of best fit
        plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x, y1, '-o', color=color[0], label="total cooperators")
        plt.plot(x, y2, '-o', color=color[1], label="total defectors")

        # plotting reference lines (.25, .5, .75)
        plt.plot([-1, len(iteration_count)], [0.5, 0.5], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.25, 0.25], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.75, 0.75], 'k-', lw=1,dashes=[2, 2])
        plt.ylim(0, 1)
        
        plt.xlabel('Iteration #')
        plt.ylabel('Proportion')
        plt.title("Proportion of Cooperators to Defectors over Iterations")
        # plt.text(0, 1,'matplotlib', horizontalalignment='center',
        #      verticalalignment='center'))
        plt.legend()
        plt.savefig(path_evolution + model_name + "-Evolution-Nodes.png", format="PNG")
        plt.close()
        
        
        ##### graph edge type evolution #####
        
        print("Plotting edge type evolution...\n")
        
        # get edge counts
        total = [ len(good_good_edges_list[i]) + len(mixed_edges_list[i]) + len(bad_bad_edges_list[i]) 
                 if len(good_good_edges_list[i]) + len(mixed_edges_list[i]) + len(bad_bad_edges_list[i]) > 0
                 else -1
                 for i in range (len(iteration_count)) ]
        
        good_edges_count = [ len(good_good_edges_list[i])/total[i] for i in range(len(iteration_count)) ]
        mixed_edges_count = [ len(mixed_edges_list[i])/total[i] for i in range(len(iteration_count)) ]
        bad_edge_count = [ len(bad_bad_edges_list[i])/total[i] for i in range(len(iteration_count)) ]

        y1 = np.array(good_edges_count)    
        y2 = np.array(mixed_edges_count)    
        y3 = np.array(bad_edge_count)    
        
        # plot with line of best fit
        plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(x, y1, '-o', color=color[2], label="total good-good edges")
        plt.plot(x, y2, '-o', color=color[3], label="total mixed edges")
        plt.plot(x, y3, '-o', color=color[4], label="total bad-bad edges")

        # plotting reference lines (.25, .5, .75)
        plt.plot([-1, len(iteration_count)], [0.5, 0.5], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.25, 0.25], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.75, 0.75], 'k-', lw=1,dashes=[2, 2])
        plt.ylim(0, 1)
        
        plt.xlabel('Iteration #')
        plt.ylabel('Proportion')
        plt.title("Proportion of Edge Types")

        plt.legend()
        plt.savefig(path_evolution + model_name + "--Evolution-Edges.png", format="PNG")
        plt.close()
        
        
        #--------------#
        # graphing properties in analysis.py (centralities, mean geodesic distance, clustering coefficient) #
        #--------------#
        '''
                # grabbing all measures (somehow... we'll sort that out later)
                # also take average centrality? or just node
                # NOTE: are nodes returns by centrality measure in the same order as the nodes from our Nodes list?
        labels = ['degree', 'eigenvector', 'katz', 'closeness', 'betweeness', 'avg geodesic']
        results = [{}]

        collected = zip(results, labels)

        plt.figure(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
        for y, label in collected:
                plt.plot(x, y, '-o', label=label)

        # plotting reference lines (.25, .5, .75)
        plt.plot([-1, len(iteration_count)], [0.5, 0.5], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.25, 0.25], 'k-', lw=1,dashes=[2, 2])
        plt.plot([-1, len(iteration_count)], [0.75, 0.75], 'k-', lw=1,dashes=[2, 2])
        # plt.ylim(0, 1)
        
        plt.xlabel('Iteration #')
        plt.ylabel('What is this scale???')
        plt.title("Analysis calculations over iterations")

        plt.legend()
        plt.savefig(path_evolution + model_name + "--Analysis.png", format="PNG")
        plt.close()
        '''

'''
func generate_gif
@param 
        input_path: directory path to image folder
        index (optional): current gif iteration (recommended if more than 1 gif is generated)
@return none
@output compiles all images by index into "animated.gif", and outputs gif into /animation
Note: only call when all graphs have been generated using func visualization(...)
'''
def generate_gif(model_name, input_path, output_path):
        my_path = input_path
        # basically the python version of regular expression search of list segments
        # you can also use globbing and bash commands, but apparently wildcard symbols can be unstable across different shell versions
        only_PNGs = [os.path.join(my_path, f) for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f)) and f.endswith(".png")]
        
        # encapsulated helper function to parse unconventional file names
        def digit_grub(f):
                digits = list(filter(str.isdigit, f))
                list_to_string = ''.join(map(str, digits))
                return int(list_to_string)
        
        only_PNGs.sort(key=digit_grub, reverse=False) 

        sizecounter = 0
        for filepath in only_PNGs:
                sizecounter += os.stat(filepath).st_size
        
        with tqdm(total=sizecounter, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                with imageio.get_writer(os.path.join(output_path, model_name + "-animated.gif"), mode='I', duration=0.3) as writer:
                        for pic in only_PNGs:
                                image = imageio.imread(pic)
                                writer.append_data(image)
                                pbar.set_postfix(file=os.path.basename(pic), refresh=True)             # does NOT work on a POSIX system to get the base name from a Windows styled path
                                pbar.update(os.stat(pic).st_size)
        print("GIF Generated: stored in {0}".format(output_path))