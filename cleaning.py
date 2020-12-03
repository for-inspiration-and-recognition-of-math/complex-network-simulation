import csv
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import random

def twitter_random_subgraph( directory, num_subgraphs =100):
	subgraph_names = os.listdir(directory)
	subgraph_set = set()
	for name in subgraph_names:
		subgraph_id = re.findall("^\d+", name)
		subgraph_set.add(subgraph_id[0])

	subgraph_set = list(subgraph_set)
	subgraph_set = [ name + '.edges' for name in subgraph_set]
	subgraph_random_set = random.sample(subgraph_set, num_subgraphs)
	combined = []
	for subgraph in subgraph_random_set:
		network = open(subgraph, 'r').read().splitlines()
		combined.extend(network)

	
	sources = [int(L.split()[0]) for L in combined]
	targets = [int(L.split()[1]) for L in combined]
	G = nx.Graph()
	E = len(sources)    # Number of edges
	for i in range(E):
		G.add_edge(sources[i], targets[i])
	return nx.to_numpy_matrix(G)


def read_edge_list( filename ): #reads in an edgelist and returns the corresponding numpy adjacnecy matrix
	network1 = open(filename, 'r')
	network1 = network1.read().splitlines()
	sources = [int(L.split()[0]) for L in network1]
	targets = [int(L.split()[1]) for L in network1]
	G = nx.Graph()
	E = len(sources)    # Number of edges
	for i in range(E):
		G.add_edge(sources[i], targets[i])
	return nx.to_numpy_matrix(G)

def generate_random_network(type, num_nodes , p =0.5, m =3, k =3 ):
	#Enter type of graph and corresponding parameters and returns randomly generated graph's numpy adjacency matrix
	if type == 'ER':
		graph = nx.erdos_renyi_graph(num_nodes, p)
		return nx.to_numpy_matrix(graph)
	elif type == 'BA':
		graph = nx.barabasi_albert_graph(num_nodes, m)
		return nx.to_numpy_matrix(graph)
	elif type == 'Watts Strogatz':
		graph = nx.watts_strogatz_graph(num_nodes, k, p)
		return nx.to_numpy_matrix(graph)
	else:
		return "Enter a valid graph type : ER, configuration, BA, or Watts Strogatz"

if __name__ == '__main__':
	os.chdir('/Users/louiszhao/Documents/GitHub/M168-simulation/resources/twitter')
	#print(twitter_random_subgraph( '/Users/louiszhao/Documents/GitHub/M168-simulation/resources/twitter',100 ).shape)
	print(len(np.unique(random.sample(range(1,100), 50))))
	#print(read_edge_list('facebook_combined.txt').shape)
	#print(read_edge_list('soc-karate.txt'))
	#print(read_edge_list('256497288.edges').shape)
	#G = nx.convert_matrix.from_numpy_matrix(generate_random_network('BA',100,m=10)) 
	#nx.draw(G)
	#plt.show()






# Make the network
