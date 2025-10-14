#####################################
#----- Preamble
#####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig 
ig.config['plotting.backend'] = 'matplotlib'

import utils
from utils import data_folder
from utils import graphs_folder
from utils import dags_folder_notears

import notears
from notears import linear





#####################################
#----- Loading our data sets
#####################################

# load the file where there is the info about the events
editing_sites =  pd.read_csv(data_folder/f'edition_sep_col.csv')

# we use the data sets that we have already imputed in 'imputation.py'
ndhB = pd.read_csv(data_folder/'ndhB_imputed_IterativeImputer_LogisticReg.csv', index_col=0) 
ndhB_intron = pd.read_csv(data_folder/'ndhB_imputed_IterativeImputer_LogisticReg_with_intron.csv', index_col=0) 
ndhD = pd.read_csv(data_folder/'ndhD_imputed_IterativeImputer_LogisticReg.csv', index_col=0)

# get the names of variables of our data sets
events_ndhB = ndhB.columns
events_ndhB_intron = ndhB_intron.columns
events_ndhD = ndhD.columns

# get the genomic positions of the events
# with intron
events_sites_ndhB_intron = {}
pos_intron_ndhB = events_ndhB_intron.get_loc('intron_2_16')
for k in range(len(events_ndhB_intron)):
    if k == pos_intron_ndhB:
        events_sites_ndhB_intron[events_ndhB_intron[k]] = str(f'ndhB_intron')
    elif k < pos_intron_ndhB:
        events_sites_ndhB_intron[events_ndhB_intron[k]] = str(f'ndhB_{editing_sites.iloc[k + 24, 3]}')
    elif k > pos_intron_ndhB:
        events_sites_ndhB_intron[events_ndhB_intron[k]] = str(f'ndhB_{editing_sites.iloc[k - 1 + 24, 3]}')
# without intron
events_sites_ndhB = {}
for k in range(len(events_ndhB)):
    events_sites_ndhB[events_ndhB[k]] = str(f'ndhB_{editing_sites.iloc[k + 24, 3]}')

events_sites_ndhD = {}
for k in range(len(events_ndhD)):
    events_sites_ndhD[events_ndhD[k]] = str(f'ndhD_{editing_sites.iloc[k + 37, 3]}')

print("\n Genomic position of the events for ndhB (with intron):\n", events_sites_ndhB_intron)
print("\n Genomic position of the events for ndhB (without intron):\n", events_sites_ndhB)
print("\n Genomic position of the events for ndhD:\n", events_sites_ndhD)





###################################################################
#----- Reference Graphs: graphs in Guillem's paper (cf. Fig. 5)
###################################################################

# Ref. Graph for ndhB
gen_pos_ndhB = list(events_sites_ndhB_intron.values())

G_ref_ndhB = ig.Graph(n = len(gen_pos_ndhB), directed = False) # create a undirected graph 
G_ref_ndhB.vs["name"] = gen_pos_ndhB # the vertices of the graph represents the events of Guillem et al. paper
G_ref_ndhB.vs["label"] = G_ref_ndhB.vs["name"] # define the labels of the vertex, that is, we print the names of the vertex in the graphical representation

# define the edges following the paper
G_ref_ndhB.add_edges([(3, 1), (3, 4), (3, 5), (3, 6), (3, 7), (3, 10), (3, 11),
                (4, 1), (4, 6),
                (5, 1), (5, 4), (5, 6), (5, 7), (5, 10), (5, 11), (5, 12),
                (7, 6), (7, 10), (7, 11), (7, 12),
                (10, 4), (10, 6), (10, 11), (10, 12),
                (11, 4), (11, 6),
                (12, 11)])

W_ref_ndhB = G_ref_ndhB.get_adjacency() # obtain the adjacency matrix
W_ref_ndhB = pd.DataFrame(W_ref_ndhB, columns=gen_pos_ndhB, index=gen_pos_ndhB)
W_ref_ndhB = W_ref_ndhB.astype(int)

# ig.plot(G_ref_ndhB, vertex_color="#8D5E01", vertex_size=75, edge_arrow_size=50, edge_width=0.5, layout="kamada_kawai", vertex_label_size=10, vertex_label_dist=0.5)
# plt.savefig(graphs_folder/f"Graph_ndhB_Fig5.png", format="PNG", dpi=500)
# plt.close()

# # Dependent events in ndhB according to Fig. 5
# W_ref_ndhB_upper = np.triu(W_ref_ndhB, k=0)
# rows, cols = np.where(W_ref_ndhB != 0)
# coord_depend_ref_ndhB = list(zip(rows, cols))
# graph_depend_ref_ndhB = []
# for k in range(len(coord_depend_ref_ndhB)):
#     pair_events = (gen_pos_ndhB[coord_depend_ref_ndhB[k][0]], gen_pos_ndhB[coord_depend_ref_ndhB[k][1]])
#     graph_depend_ref_ndhB.append(pair_events)

# print(f"\n There are {len(graph_depend_ref_ndhB)} dependent events for ndhB according to Fig. 5:\n {graph_depend_ref_ndhB}")

# W_ref_ndhB.to_csv(graphs_folder/f'AdjacencyMatrix_ndhB_Fig5.csv')
# W_ref_ndhB = pd.read_csv(graphs_folder/f'AdjacencyMatrix_ndhB_Fig5.csv', sep=',', header=None)


# Ref. Graph for ndhD
gen_pos_ndhD = list(events_sites_ndhD.values())

G_ref_ndhD = ig.Graph(n = len(gen_pos_ndhD), directed = False) # create a undirected graph 
G_ref_ndhD.vs["name"] = gen_pos_ndhD # the vertices of the graph represents the events of Guillem et al. paper
G_ref_ndhD.vs["label"] = G_ref_ndhD.vs["name"] # define the labels of the vertex, that is, we print the names of the vertex in the graphical representation

# define the edges following the paper
G_ref_ndhD.add_edges([(0, 1), (0, 2), (0, 3), (0, 4),
                (1, 2), (1, 3), (1, 4),
                (2, 3), (2, 4)])

W_ref_ndhD = G_ref_ndhD.get_adjacency() # obtain the adjacency matrix
W_ref_ndhD = pd.DataFrame(W_ref_ndhD, columns=gen_pos_ndhD, index=gen_pos_ndhD)
W_ref_ndhD = W_ref_ndhD.astype(int)

# ig.plot(G_ref_ndhD, vertex_color="#8D5E01", vertex_size=75, edge_arrow_size=20, edge_width=0.5, layout="kamada_kawai", vertex_label_size=10, vertex_label_dist=0.5)
# plt.savefig(graphs_folder/f"Graph_ndhD_Fig5.png", format="PNG", dpi=500)
# plt.close()

# # Dependent events in ndhD according to Fig. 5
# W_ref_ndhD_upper = np.triu(W_ref_ndhD, k=0)
# rows, cols = np.where(W_ref_ndhD_upper != 0)
# coord_depend_ref_ndhD = list(zip(rows, cols))
# graph_depend_ref_ndhD = []
# for k in range(len(coord_depend_ref_ndhD)):
#     pair_events = (gen_pos_ndhD[coord_depend_ref_ndhD[k][0]], gen_pos_ndhD[coord_depend_ref_ndhD[k][1]])
#     graph_depend_ref_ndhD.append(pair_events)

# print(f"\n There are {len(graph_depend_ref_ndhD)} dependent events for ndhD according to Fig. 5:\n {graph_depend_ref_ndhD}")

# W_ref_ndhD.to_csv(graphs_folder/f'AdjacencyMatrix_ndhD_Fig5.csv')
# W_ref_ndhD = pd.read_csv(graphs_folder/f'AdjacencyMatrix_ndhD_Fig5.csv', sep=',', header=None)


# com_dep_events = [tuple_dep_test for tuple_dep_test in dep_events_after if set(tuple_dep_test[:2]) in [set(tuple_graph) for tuple_graph in graph_depend]]
# print(f"\n The Chi2 test for these events yields the following {len(com_dep_events)}:\n {com_dep_events}")

# indep_events_test = [tuple_dep_test for tuple_dep_test in indep_events_after if set(tuple_dep_test[:2]) in [set(tuple_graph) for tuple_graph in graph_depend]]
# print(f"\n The following events are dependent according to Fig. 5, but not according to the Chi2 test {len(indep_events_test)}:\n {indep_events_test}")





#################################
#----- DAGs with NOTEARS
#################################

# # DAG for ndhB
# gen_pos_ndhB = list(events_sites_ndhB.values())
# ndhB_notears = np.loadtxt(data_folder/'ndhB_imputed_IterativeImputer_LogisticReg.csv', delimiter=',', skiprows=1, usecols=range(1, len(events_ndhB)+1))
# loss_type = 'logistic'

# # # Preliminary analysis: 'loss curve for the sparcity parameter', 'sparsity models', 'sparsity models through stability selection'
# # # 'loss curve for the sparcity parameter'
# # utils.loss_values_lamb(ndhB_notears, loss_type, sample_lambda=100, name_chart='ndhB_loss_curve_lambda')
# # # sparsity models
# # utils.connex_lamb(ndhB_notears, loss_type, sample_lambda=100, name_chart='ndhB_sparsity_models')
# # # sparsity models through Stability Selection
# # utils.connex_lamb_StabSelec(ndhB_notears, loss_type, stab_freq=0.75, sample_lambda=100, num_iter=5, name_chart='ndhB_sparsity_models_StabSelec')

# # Choosing the sparsity parameter
# # # Cross-Validation wrt 'loss' function
# # lamb = utils.CV_notears_lamb(ndhB_notears, loss_type, K_folds=5, sample_lambda=100, name_chart='ndhB_loss_curve_lambda_CV')
# # model_selector = 'CV with loss'
# # # Cross-Validation wrt 'score_connexions' function by means of StabSelec
# # lamb = utils.CVStab_notears_lamb(ndhB_notears, loss_type, K_folds=3, stab_freq=0.75, sample_lambda=100, num_iter=3, name_chart='ndhB_score_curve_lambda_CV_StabSelec')
# # model_selector = 'CV and StabSelec with score_connexions'
# # Stability Selection of the sparsity model within the whole space of parameters lambda (we do not choose a specific labmda here!)
# W_ndhB = utils.StabSelec_notears_sparsity(ndhB_notears, loss_type, stab_freq=0.75, sample_lambda=100, num_iter=5)[0]
# lamb = 'None'
# model_selector = 'StablSelec'

# # # Constructing the DAG
# # W_ndhB = linear.notears_linear(ndhB_notears, lamb, loss_type)

# G_ndhB = ig.Graph.Weighted_Adjacency(W_ndhB)
# G_ndhB.vs['name'] = gen_pos_ndhB
# G_ndhB.vs['label'] = G_ndhB.vs['name']

# fig_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_ndhB, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=7, vertex_label_dist=0.5)

# param_alg = [lamb, loss_type, model_selector]
# utils.save_results_notears(figure=fig_ndhB, adj_matrix=W_ndhB, vertex=gen_pos_ndhB, dataset_name='ndhB', param_alg=param_alg)
# plt.close(fig_ndhB)


# # Dependent events in ndhB according to NOTEARS
# W_ndhB = pd.read_csv(dags_folder_notears/f'NOTEARS_ndhB_2025_09_09_at_00_20_00/AdjacencyMatrix_ndhB.csv', sep=',', header=None)
# W_ndhB = W_ndhB.iloc[1:, 1:]
# W_ndhB = W_ndhB.astype(int)
# rows, cols = np.where(W_ndhB != 0)
# coord_depend_notears_ndhB = list(zip(rows, cols))
# graph_depend_notears_ndhB = []
# for k in range(len(coord_depend_notears_ndhB)):
#     pair_events = (gen_pos_ndhB[coord_depend_notears_ndhB[k][0]], gen_pos_ndhB[coord_depend_notears_ndhB[k][1]])
#     graph_depend_notears_ndhB.append(pair_events)

# print(f"\n There are {len(graph_depend_notears_ndhB)} dependent events for ndhB according to NOTEARS:\n {graph_depend_notears_ndhB}")


# # Comparing with the reference graph (DAG version of it!)
# W_ndhB = pd.read_csv(dags_folder_notears/f'NOTEARS_ndhB_2025_09_09_at_00_20_00/AdjacencyMatrix_ndhB.csv', sep=',', header=None)
# W_ndhB = W_ndhB.iloc[1:, 1:]
# W_ndhB = W_ndhB.where(W_ndhB == 0, 1)
# W_ndhB = W_ndhB.astype(int)
# W_ndhB = W_ndhB.to_numpy()

# W_ref_ndhB = pd.read_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhB_Fig6.csv', sep=',', header=None)
# W_ref_ndhB = W_ref_ndhB.iloc[1:, 1:]
# W_ref_ndhB = W_ref_ndhB.astype(int)
# W_ref_ndhB = W_ref_ndhB.to_numpy()

# SHD = utils.shd(W_ndhB, W_ref_ndhB)

# print(f"\n The SHD between G_notears_ndhB and G_ref_ndhB is {SHD}")


# # Counting the degrees of the edges
# pointed_nodes_out_ndhB, pointed_nodes_in_ndhB, source_nodes_ndhB, target_nodes_ndhB, mixed_nodes_ndhB = utils.deg_edges(adj_matrix=W_ndhB, vertex=gen_pos_ndhB)

# print("\n Nodes that are sources in G_notears_ndhB:")
# print(source_nodes_ndhB)

# print("\n Nodes that are targets in G_notears_ndhB:")
# print(target_nodes_ndhB)

# print("\n Nodes that are both sources and targets in G_notears_ndhB:")
# print(mixed_nodes_ndhB)





# # DAG for ndhD
# ndhD_notears = np.loadtxt(data_folder/'ndhD_imputed_IterativeImputer_LogisticReg.csv', delimiter=',', skiprows=1, usecols=range(1, len(events_ndhD)+1))
# loss_type = 'logistic'

# # # Preliminary analysis: 'loss curve for the sparcity parameter', 'sparsity models', 'sparsity models through stability selection'
# # # 'loss curve for the sparcity parameter'
# # utils.loss_values_lamb(ndhD_notears, loss_type, sample_lambda=100, name_chart='ndhD_loss_curve_lambda')
# # # sparsity models
# # utils.connex_lamb(ndhD_notears, loss_type, sample_lambda=100, name_chart='ndhD_sparsity_models')
# # # sparsity models through Stability Selection
# # utils.connex_lamb_StabSelec(ndhD_notears, loss_type, stab_freq=0.75, sample_lambda=100, num_iter=5, name_chart='ndhD_sparsity_models_StabSelec')

# # # Choosing the sparsity parameter
# # # Cross-Validation wrt 'loss' function
# # lamb = utils.CV_notears_lamb(ndhD_notears, loss_type, K_folds=5, sample_lambda=100, name_chart='ndhD_loss_curve_lambda_CV')
# # model_selector = 'CV with loss'
# # # Cross-Validation wrt 'score_connexions' function by means of StabSelec
# # lamb = utils.CVStab_notears_lamb(ndhD_notears, loss_type, K_folds=3, stab_freq=0.75, sample_lambda=100, num_iter=3, name_chart='ndhD_score_curve_lambda_CV_StabSelec')
# # model_selector = 'CV and StabSelec with score_connexions'
# # Stability Selection of the sparsity model within the whole space of parameters lambda (we do not choose a specific lambda here!)
# W_ndhD = utils.StabSelec_notears_sparsity(ndhD_notears, loss_type, stab_freq=0.75, sample_lambda=100, num_iter=5)[0]
# lamb = 'None'
# model_selector = 'StablSelec'

# # # Constructing the DAG
# # W_ndhD = linear.notears_linear(ndhD_notears, lamb, loss_type)

# G_ndhD = ig.Graph.Weighted_Adjacency(W_ndhD)
# G_ndhD.vs['name'] = gen_pos_ndhD
# G_ndhD.vs['label'] = G_ndhD.vs['name']

# fig_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_ndhD, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=10, vertex_label_dist=0.5)

# param_alg = [lamb, loss_type, model_selector]
# utils.save_results_notears(figure=fig_ndhD, adj_matrix=W_ndhD, vertex=gen_pos_ndhD, dataset_name='ndhD', param_alg=param_alg)
# plt.close(fig_ndhD)


# # Dependent events in ndhD according to NOTEARS
# W_ndhD = pd.read_csv(dags_folder_notears/f'NOTEARS_ndhD_2025_09_08_at_22_29_07/AdjacencyMatrix_ndhD.csv', sep=',', header=None)
# W_ndhD = W_ndhD.iloc[1:, 1:]
# W_ndhD = W_ndhD.astype(int)
# rows, cols = np.where(W_ndhD != 0)
# coord_depend_notears_ndhD = list(zip(rows, cols))
# graph_depend_notears_ndhD = []
# for k in range(len(coord_depend_notears_ndhD)):
#     pair_events = (gen_pos_ndhD[coord_depend_notears_ndhD[k][0]], gen_pos_ndhD[coord_depend_notears_ndhD[k][1]])
#     graph_depend_notears_ndhD.append(pair_events)

# print(f"\n There are {len(graph_depend_notears_ndhD)} dependent events for ndhD according to NOTEARS:\n {graph_depend_notears_ndhD}")


# # Comparing with the reference graph (DAG version of it!)
# W_ndhD = pd.read_csv(dags_folder_notears/f'NOTEARS_ndhD_2025_09_08_at_22_29_07/AdjacencyMatrix_ndhD.csv', sep=',', header=None)
# W_ndhD = W_ndhD.iloc[1:, 1:]
# W_ndhD = W_ndhD.where(W_ndhD == 0, 1)
# W_ndhD = W_ndhD.astype(int)
# W_ndhD = W_ndhD.to_numpy()

# W_ref_ndhD = pd.read_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhD_Fig6.csv', sep=',', header=None)
# W_ref_ndhD = W_ref_ndhD.iloc[1:, 1:]
# W_ref_ndhD = W_ref_ndhD.astype(int)
# W_ref_ndhD = W_ref_ndhD.to_numpy()

# SHD = utils.shd(W_ndhD, W_ref_ndhD)

# print(f"\n The SHD between G_notears_ndhD and G_ref_ndhD is {SHD}")


# # Counting the degrees of the edges
# pointed_nodes_out_ndhD, pointed_nodes_in_ndhD, source_nodes_ndhD, target_nodes_ndhD, mixed_nodes_ndhD = utils.deg_edges(adj_matrix=W_ndhD, vertex=gen_pos_ndhD)

# print("\n Nodes that are sources in G_notears_ndhD:")
# print(source_nodes_ndhD)

# print("\n Nodes that are targets in G_notears_ndhD:")
# print(target_nodes_ndhD)

# print("\n Nodes that are both sources and targets in G_notears_ndhD:")
# print(mixed_nodes_ndhD)





########################################################################################################
#----- Testing Stability Selection strategy to choose the sparsity model: Sachs dataset.
########################################################################################################

# # load the Sachs dataset
# # Here we use the first 9 files of the original supplementary material in order to have 7466 observations as the dataset used by Zheng et al. in NOTEARS paper
# sachs = pd.read_csv(data_folder/'sachs_data.csv', sep=';')

# #get the names of variables of Sachs dataset
# proteins = sachs.columns


# # Ref. Graph for Sachs
# G_ref_sachs = ig.Graph(n = len(proteins), directed = True)
# G_ref_sachs.vs["name"] = proteins 
# G_ref_sachs.vs["label"] = G_ref_sachs.vs["name"]

# # define the edges following the paper
# G_ref_sachs.add_edges([(0, 1),
#                        (1, 5),
#                        (2,3), (2, 4), (2, 8),
#                        (3, 8),
#                        (4, 3), (4, 6),
#                        (5, 6),
#                        (7, 10), (7, 9), (7, 6), (7, 5), (7, 1), (7, 0),
#                        (8,10), (8, 9), (8, 7), (8, 0), (8, 1)])

# # W_ref_sachs = G_ref_sachs.get_adjacency() # obtain the adjacency matrix
# # W_ref_sachs = pd.DataFrame(W_ref_sachs, columns=proteins, index=proteins)
# # W_ref_sachs = W_ref_sachs.astype(int)
# # W_ref_sachs.to_csv(graphs_folder/f'AdjacencyMatrix_sachs_ref.csv')
# # # W_ref_sachs = pd.read_csv(graphs_folder/f'AdjacencyMatrix_sachs_ref.csv', sep=',', header=None)
# # # W_ref_sachs = W_ref_sachs.iloc[1:, 1:]

# ig.plot(G_ref_sachs, vertex_color="#9D0A31", vertex_size=75, edge_arrow_size=45, edge_arrow_width=30, edge_width=0.5, layout="rt", vertex_label_size=12, vertex_label_dist=0.5)
# plt.savefig(graphs_folder/f"Graph_sachs_ref.png", format="PNG", dpi=500)
# plt.close()


# # DAG for Sachs with NOTEARS + Stability Selection for the sparsity model
# sachs = np.loadtxt(data_folder/'sachs_data_decimal_point.csv', delimiter=';', skiprows=1)
# loss_type = 'l2'

# # # Preliminary analysis: 'sparsity models'
# # # sparsity models (here the interval should be [0, 100 000]?)
# # utils.connex_lamb(sachs, loss_type, sample_lambda=1000, name_chart='sachs_sparsity_models')

# # Choosing the sparsity parameter
# # Stability Selection of the sparsity model (we do not choose a specific lambda here! The interval should be [0, 100 000]?)
# W_sachs = utils.StabSelec_notears_sparsity(sachs, loss_type, stab_freq=0.75, sample_lambda=75, interval_lambda=[0,100], num_iter=5)[0]
# lamb = 'None'
# model_selector = 'StablSelec'

# # Constructing the DAG
# G_sachs = ig.Graph.Weighted_Adjacency(W_sachs)
# G_sachs.vs['name'] = proteins
# G_sachs.vs['label'] = G_sachs.vs['name']

# fig_sachs, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_sachs, vertex_color = "#F5265D", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, layout = "rt", vertex_label_size=12, vertex_label_dist=0.5)

# param_alg = [lamb, loss_type, model_selector]
# utils.save_results_notears(figure=fig_sachs, adj_matrix=W_sachs, vertex=proteins, dataset_name='sachs', param_alg=param_alg)
# plt.close(fig_sachs)


# # Comparing with the reference graph
# W_sachs = pd.read_csv(dags_folder_notears/f'NOTEARS_sachs_2025_10_07_at_10_24_38/AdjacencyMatrix_sachs.csv', sep=',', header=None)
# W_sachs = W_sachs.iloc[1:, 1:]
# W_sachs = W_sachs.where(W_sachs == 0, 1)
# W_sachs = W_sachs.astype(int)
# W_sachs = W_sachs.to_numpy()

# W_ref_sachs = pd.read_csv(graphs_folder/f'AdjacencyMatrix_sachs_ref.csv', sep=',', header=None)
# W_ref_sachs = W_ref_sachs.iloc[1:, 1:]
# W_ref_sachs = W_ref_sachs.astype(int)
# W_ref_sachs = W_ref_sachs.to_numpy()

# SHD = utils.shd(W_sachs, W_ref_sachs)

# print(f"\n The SHD between G_notears_sachs and G_ref_sachs is {SHD}")