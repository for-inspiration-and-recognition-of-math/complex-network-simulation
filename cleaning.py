import csv
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import random

def twitter_random_subgraph( directory, num_subgraphs =100, seed =100):
	random.seed(seed)
	os.chdir(directory)
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


def read_edge_list(directory, filename ): #reads in an edgelist and returns the corresponding numpy adjacnecy matrix
	os.chdir(directory)
	network1 = open(filename, 'r')
	network1 = network1.read().splitlines()
	sources = [int(L.split()[0]) for L in network1]
	targets = [int(L.split()[1]) for L in network1]
	G = nx.Graph()
	E = len(sources)    # Number of edges
	for i in range(E):
		G.add_edge(sources[i], targets[i])
	return nx.to_numpy_matrix(G)

def generate_random_network(type, num_nodes , p =0.5, m =3, k =3, seed =100):
	#Enter type of graph and corresponding parameters and returns randomly generated graph's numpy adjacency matrix
	if type == 'ER':
		graph = nx.erdos_renyi_graph(num_nodes, p, seed= seed)
		return nx.to_numpy_matrix(graph)
	elif type == 'BA':
		graph = nx.barabasi_albert_graph(num_nodes, m, seed = seed)
		return nx.to_numpy_matrix(graph)
	elif type == 'WS':
		graph = nx.watts_strogatz_graph(num_nodes, k, p, seed= seed)
		return nx.to_numpy_matrix(graph)
	else:
		return "Enter a valid graph type : ER, configuration, BA, or Watts Strogatz"

if __name__ == '__main__':

	twitter_directory = '/Users/louiszhao/Documents/GitHub/M168-simulation/resources/twitter'
	facebook_directory = '/Users/louiszhao/Documents/GitHub/M168-simulation/resources'
	karate_directory = '/Users/louiszhao/Documents/GitHub/M168-simulation/resources'
	# print(twitter_random_subgraph(twitter_directory , num_subgraphs =100, seed = 100 ).shape)
	# print(read_edge_list(facebook_directory, 'facebook_combined.txt').shape)

	# print(read_edge_list(karate_directory, 'soc-karate.txt').shape)


	# G = nx.convert_matrix.from_numpy_matrix(read_edge_list(karate_directory, 'soc-karate.txt')) 
	# G1 = nx.convert_matrix.from_numpy_matrix(read_edge_list(facebook_directory, 'facebook_combined.txt')) 
	# G2 = nx.convert_matrix.from_numpy_matrix(twitter_random_subgraph(twitter_directory, num_subgraphs = 100, seed = 100))
	ER = nx.convert_matrix.from_numpy_matrix(generate_random_network('ER', 50, p = 0.2, seed = 100))
	BA = nx.convert_matrix.from_numpy_matrix(generate_random_network('BA', 50, m = 3, seed = 100))
	WS = nx.convert_matrix.from_numpy_matrix(generate_random_network('WS', 50, k = 4, p = 0.5, seed = 100))
	nx.draw(WS)
	plt.show()







# Make the network
