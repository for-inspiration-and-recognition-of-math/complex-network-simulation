import networkx as nx 
import numpy as np
from cleaning import *

def centrality(adj , type):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	if type == 'degree':
		return nx.degree_centrality(G)
	if type == 'eigenvector':
		return nx.eigenvector_centrality(G)
	if type =='katz':
		return nx.katz_centrality(G)
	if type == 'closeness':
		return nx.closeness_centrality(G)
	if type == 'betweenness':
		return nx.betweenness_centrality(G)

def geodesic(adj):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	return nx.average_shortest_path_length(G)

def average_clustering_coefficient(adj):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	return nx.average_clustering(G)


if __name__ == '__main__':
	adj = karate_clean()[1]
	adj2 = generate_random_network('ER', 500, p =0.05)[1]
	#print(centrality(karate_clean()[1], 'betweenness'))
	print(geodesic(adj2))