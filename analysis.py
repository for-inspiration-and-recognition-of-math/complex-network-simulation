import networkx as nx 
import numpy as np
from cleaning import *
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
import os 
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import itertools


def centrality(adj , node_dict, type):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	value_dict = {}
	return_dict = {}
	if type == 'degree':
		value_dict =  nx.degree_centrality(G)
	if type == 'eigenvector':
		value_dict =  nx.eigenvector_centrality(G)
	if type =='katz':
		value_dict = nx.katz_centrality(G)
	if type == 'closeness':
		value_dict =  nx.closeness_centrality(G)
	if type == 'betweenness':
		value_dict = nx.betweenness_centrality(G)
	for (index1, node), (index2, value) in zip(nodes.items(), value_dict.items()):
		return_dict[node] = value
	return return_dict


def all_measures( node_dict, adj, iteration, alpha = 0.9):
	#pool = mp.Pool(mp.cpu_count())
	G = nx.convert_matrix.from_numpy_matrix(adj)

	df = pd.DataFrame.from_dict(node_dict, orient = 'index')
	df.columns = ['Node']
	df['iteration'] = iteration
	df['status'] = pd.Series([val.status for val in node_dict.values()])
	df['degree'] = pd.Series(nx.degree_centrality(G))
	df['eigenvector'] = pd.Series(nx.eigenvector_centrality(G))
	#df['katz'] = pd.Series(nx.katz_centrality(G))
	#df['closeness'] = pd.Series(nx.closeness_centrality(G))
	#df['betweenness'] = pd.Series(nx.betweenness_centrality(G))
	df['pagerank'] = pd.Series(nx.pagerank(G, alpha))
	df['local_clustering_coefficients'] = pd.Series(nx.clustering(G))

	return( df)

def all_measures_master(node_dict_list, adj_list, name ):
	master_df = pd.DataFrame()
	master_list = []
	it = range(0, len(adj_list))
	f_list = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		for node_dict, adj, iterator in zip(node_dict_list, adj_list, it):
			f = executor.submit(all_measures, node_dict, adj, iterator)
			f_list.append(f)
			print(iterator, end=' ')
			#values = all_measures(node_dict,adj,iterator,0.9)
		print()
		it = 0
		for f in f_list:
			print(it, end=' ')
			master_df = pd.concat([master_df, f.result()])
			master_list.append(f.result())
			it = it+1

	if not os.path.exists('analysis'):
		os.makedirs('analysis')
	directory = os.path.dirname(__file__) + '/analysis' + '/centrality_values_' + name + '-'+ str(iterator) +  '.csv' 
	master_df.to_csv(directory)
	return master_list

def community_detection(node_dict_list, adj_list ):
	iteration_list = []
	it = 0
	print('iteration:', end=' ') 
	for node_dict, adj in zip(node_dict_list, adj_list):
		id_status_list = []
		G = nx.from_numpy_matrix(adj)
		c = list(greedy_modularity_communities(G))

		print(str(it), end=' ')
		for val in c:
			community = sorted(val)
			community_status = [node_dict.get(item,item).status for item in sorted(val)]
			id_status_list.append(community)
			id_status_list.append(community_status)
		iteration_list.append(id_status_list)
		it = it +1
	return iteration_list


def geodesic(adj):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	return nx.average_shortest_path_length(G)

def average_clustering_coefficient(adj):
	G = nx.convert_matrix.from_numpy_matrix(adj)
	return nx.average_clustering(G)


