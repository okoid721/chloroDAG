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

from pgmpy.sampling import BayesianModelSampling





#####################################
#----- Loading our data sets
#####################################

# load the file where there is the info about the events
editing_sites =  pd.read_csv(data_folder/f'edition_sep_col.csv')

# we use the data sets that we have already imputed in 'imputation.py'
ndhB = pd.read_csv(data_folder/'ndhB_imputed_IterativeImputer_LogisticReg.csv', index_col=0) 
ndhD = pd.read_csv(data_folder/'ndhD_imputed_IterativeImputer_LogisticReg.csv', index_col=0)

# we also need the NaN positions of these data sets in order to apply the EM-alg
ndhB_notimp = pd.read_csv('Data/ndhB_sorted.csv', index_col=0)
pos_NA_ndhB = np.where(np.isnan(ndhB_notimp))
pos_NA_ndhB = list(zip(pos_NA_ndhB[0], pos_NA_ndhB[1])) # put the positions in coordinate form
ndhD_notimp = pd.read_csv('Data/ndhD_sorted.csv', index_col=0)
pos_NA_ndhD = np.where(np.isnan(ndhD_notimp))
pos_NA_ndhD = list(zip(pos_NA_ndhD[0], pos_NA_ndhD[1])) # put the positions in coordinate form

# get the names of variables of our data sets
events_ndhB = ndhB.columns
events_ndhD = ndhD.columns

# get the genomic positions of the events
events_sites_ndhB = {}
for k in range(len(events_ndhB)):
    events_sites_ndhB[events_ndhB[k]] = str(f'ndhB_{editing_sites.iloc[k + 24, 3]}')

events_sites_ndhD = {}
for k in range(len(events_ndhD)):
    events_sites_ndhD[events_ndhD[k]] = str(f'ndhD_{editing_sites.iloc[k + 37, 3]}')

print("\n Genomic position of the events for ndhB:\n", events_sites_ndhB)
print("\n Genomic position of the events for ndhD:\n", events_sites_ndhD)

gen_pos_ndhB = list(events_sites_ndhB.values())
gen_pos_ndhD = list(events_sites_ndhD.values())




######################################################################
#----- Application of EM with Hill Climbing
######################################################################

# # EM for ndhB
# data = ndhB
# pos_NA = pos_NA_ndhB
# pos_gen = list(events_sites_ndhB.values())

# # define the main parameters for the algorithm
# param_alg = ['hill_climbing', 'bic-d', 'BayesianEstimator - K2']

# EM_ndhB_hc = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen)
# EM_data_ndhB_hc = EM_ndhB_hc[2]
# EM_bn_ndhB_hc = EM_ndhB_hc[1]
# EM_model_ndhB_hc = EM_ndhB_hc[0]
# print("The EM model for ndhB with Hill Climbing is a", EM_model_ndhB_hc)

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhB_hc) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = list(EM_model_ndhB_hc.nodes())
# edges = list(EM_model_ndhB_hc.edges())

# G_EM_ndhB = ig.Graph(n = len(nodes), directed=True)
# G_EM_ndhB.vs["label"] = G_EM_ndhB.vs["name"] = nodes
# G_EM_ndhB.add_edges(edges)

# W_EM_ndhB = G_EM_ndhB.get_adjacency() # obtain the adjacency matrix

# fig_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhB, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=7, vertex_label_dist=0.5)

# # save the results
# utils.save_results_EM(figure=fig_ndhB, 
#                       adj_matrix=W_EM_ndhB, 
#                       vertex=nodes, 
#                       dataset_name='ndhB', 
#                       EM_bn=EM_bn_ndhB_hc, 
#                       EM_data=EM_data_ndhB_hc, 
#                       EM_syn=synthetic_data, 
#                       param_alg=param_alg)
# plt.close(fig_ndhB)



# # EM for ndhD
# data = ndhD
# pos_NA = pos_NA_ndhD
# pos_gen = list(events_sites_ndhD.values())

# # define the main parameters for the algorithm
# param_alg = ['hill_climbing', 'bic-d', 'BayesianEstimator - K2']

# EM_ndhD_hc = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen)
# EM_data_ndhD_hc = EM_ndhD_hc[2]
# EM_bn_ndhD_hc = EM_ndhD_hc[1]
# EM_model_ndhD_hc = EM_ndhD_hc[0]
# print("The EM model for ndhD with Hill Climbing is a", EM_model_ndhD_hc)

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhD_hc) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = list(EM_model_ndhD_hc.nodes())
# edges = list(EM_model_ndhD_hc.edges())

# G_EM_ndhD = ig.Graph(n=len(nodes), directed=True)
# G_EM_ndhD.vs["label"] = G_EM_ndhD.vs["name"] = nodes
# G_EM_ndhD.add_edges(edges)

# W_EM_ndhD = G_EM_ndhD.get_adjacency() # obtain the adjacency matrix

# fig_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhD, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=8, vertex_label_dist=0.5)

# # save the results
# utils.save_results_EM(figure=fig_ndhD, 
#                     adj_matrix=W_EM_ndhD, 
#                     vertex=nodes, 
#                     dataset_name='ndhD', 
#                     EM_bn=EM_bn_ndhD_hc, 
#                     EM_data=EM_data_ndhD_hc, 
#                     EM_syn=synthetic_data, 
#                     param_alg=param_alg)
# plt.close(fig_ndhD)





# ######################################################################
# #----- Application of EM with PC
# ######################################################################

# # EM for ndhB
# data = ndhB
# pos_NA = pos_NA_ndhB
# pos_gen = list(events_sites_ndhB.values())

# # define the main parameters for the algorithm
# param_alg = ['pc', 'chi-square', 'BayesianEstimator - K2']

# EM_ndhB_pc = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen)
# EM_data_ndhB_pc = EM_ndhB_pc[2]
# EM_bn_ndhB_pc = EM_ndhB_pc[1]
# EM_model_ndhB_pc = EM_ndhB_pc[0]
# print("The EM model for ndhB with PC is a", EM_model_ndhB_pc)

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhB_pc) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = list(EM_model_ndhB_pc.nodes())
# edges = list(EM_model_ndhB_pc.edges())

# G_EM_ndhB = ig.Graph(n=len(nodes), directed=True)
# G_EM_ndhB.vs["label"] = G_EM_ndhB.vs["name"] = nodes
# G_EM_ndhB.add_edges(edges)

# W_EM_ndhB = G_EM_ndhB.get_adjacency() # obtain the adjacency matrix

# fig_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhB, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)

# #  save the results
# utils.save_results_EM(figure=fig_ndhB, 
#                       adj_matrix=W_EM_ndhB, 
#                       vertex=nodes, 
#                       dataset_name='ndhB', 
#                       EM_bn=EM_bn_ndhB_pc, 
#                       EM_data=EM_data_ndhB_pc, 
#                       EM_syn=synthetic_data, 
#                       param_alg=param_alg)
# plt.close(fig_ndhB)



# # EM for ndhD
# data = ndhD
# pos_NA = pos_NA_ndhD
# pos_gen = list(events_sites_ndhD.values())

# # define the main parameters for the algorithm
# param_alg = ['pc', 'chi-square', 'BayesianEstimator - K2']

# EM_ndhD_pc = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen)
# EM_data_ndhD_pc = EM_ndhD_pc[2]
# EM_bn_ndhD_pc = EM_ndhD_pc[1]
# EM_model_ndhD_pc = EM_ndhD_pc[0]
# print("The EM model for ndhD with PC is a", EM_model_ndhD_pc)

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhD_pc) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = list(EM_model_ndhD_pc.nodes())
# edges = list(EM_model_ndhD_pc.edges())

# G_EM_ndhD = ig.Graph(n=len(nodes), directed=True)
# G_EM_ndhD.vs["label"] = G_EM_ndhD.vs["name"] = nodes
# G_EM_ndhD.add_edges(edges)

# W_EM_ndhD = G_EM_ndhD.get_adjacency() # obtain the adjacency matrix

# fig_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhD, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=8, vertex_label_dist=0.5)

# # save the results
# utils.save_results_EM(figure=fig_ndhD, 
#                       adj_matrix=W_EM_ndhD, 
#                       vertex=nodes, 
#                       dataset_name='ndhD', 
#                       EM_bn=EM_bn_ndhD_pc, 
#                       EM_data=EM_data_ndhD_pc, 
#                       EM_syn=synthetic_data, 
#                       param_alg=param_alg)
# plt.close(fig_ndhD)





# ######################################################################
# #----- Application of EM with LiNGAM
# ######################################################################

# # EM for ndhB
# data = ndhB
# pos_NA = pos_NA_ndhB
# pos_gen = list(events_sites_ndhB.values())

# # define the main parameters for the algorithm
# param_alg = ['lingam', 'ICA', 'BayesianEstimator - K2']

# EM_ndhB_lingam = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen)
# EM_data_ndhB_lingam = EM_ndhB_lingam[2]
# EM_bn_ndhB_lingam = EM_ndhB_lingam[1]
# EM_model_ndhB_lingam = EM_ndhB_lingam[0]
# print("The EM model for ndhB with LiNGAM is a", EM_model_ndhB_lingam)

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhB_lingam) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = EM_model_ndhB_lingam['nodes']
# edges = EM_model_ndhB_lingam['edges']

# G_EM_ndhB = ig.Graph(n=len(nodes), directed=True)
# G_EM_ndhB.vs["label"] = G_EM_ndhB.vs["name"] = nodes
# G_EM_ndhB.add_edges(edges)

# W_EM_ndhB = G_EM_ndhB.get_adjacency() # obtain the adjacency matrix

# fig_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhB, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=8, vertex_label_dist=0.5)

# #  save the results
# utils.save_results_EM(figure=fig_ndhB, 
#                       adj_matrix=W_EM_ndhB, 
#                       vertex=nodes, 
#                       dataset_name='ndhB', 
#                       EM_bn=EM_bn_ndhB_lingam, 
#                       EM_data=EM_data_ndhB_lingam, 
#                       EM_syn=synthetic_data, param_alg=param_alg)
# plt.close(fig_ndhB)



# # EM for ndhD
# data = ndhD
# pos_NA = pos_NA_ndhD
# pos_gen = list(events_sites_ndhD.values())

# # define the main parameters for the algorithm
# param_alg = ['lingam', 'ICA', 'BayesianEstimator - K2']

# EM_ndhD_lingam = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen)
# EM_data_ndhD_lingam = EM_ndhD_lingam[2]
# EM_bn_ndhD_lingam = EM_ndhD_lingam[1]
# EM_model_ndhD_lingam = EM_ndhD_lingam[0]
# print("The EM model for ndhD with LiNGAM is a", EM_model_ndhD_lingam)

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhD_lingam) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = EM_model_ndhD_lingam['nodes']
# edges = EM_model_ndhD_lingam['edges']

# G_EM_ndhD = ig.Graph(n=len(nodes), directed=True)
# G_EM_ndhD.vs["label"] = G_EM_ndhD.vs["name"] = nodes
# G_EM_ndhD.add_edges(edges)

# W_EM_ndhD = G_EM_ndhD.get_adjacency() # obtain the adjacency matrix

# fig_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhD, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=8, vertex_label_dist=0.5)

# #  save the results
# utils.save_results_EM(figure=fig_ndhD, 
#                       adj_matrix=W_EM_ndhD, 
#                       vertex=nodes, dataset_name='ndhD', 
#                       EM_bn=EM_bn_ndhD_lingam, 
#                       EM_data=EM_data_ndhD_lingam, 
#                       EM_syn=synthetic_data, 
#                       param_alg=param_alg)
# plt.close(fig_ndhD)





######################################################################
#----- Application of EM with NOTEARS
######################################################################

# # EM for ndhB
# data = ndhB
# pos_NA = pos_NA_ndhB
# pos_gen = list(events_sites_ndhB.values())

# # define the main parameters for the algorithm
# param_alg = ['notears', 'notears_optimization', 'BayesianEstimator - K2']

# EM_ndhB_notears = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen, notears_loss_type='logistic')
# EM_data_ndhB_notears = EM_ndhB_notears[2]
# EM_bn_ndhB_notears = EM_ndhB_notears[1]
# EM_model_ndhB_notears = EM_ndhB_notears[0]
# print(f"The EM model for ndhB with NOTEARS is a DAG with {len(EM_model_ndhB_notears['nodes'])} nodes and {len(EM_model_ndhB_notears['edges'])} edges")

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhB_notears) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = EM_model_ndhB_notears['nodes']
# edges = EM_model_ndhB_notears['edges']

# G_EM_ndhB = ig.Graph(n=len(nodes), directed=True)
# G_EM_ndhB.vs["label"] = G_EM_ndhB.vs["name"] = nodes
# G_EM_ndhB.add_edges(edges)

# W_EM_ndhB = G_EM_ndhB.get_adjacency() # obtain the adjacency matrix

# fig_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhB, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)

# # save the results
# utils.save_results_EM(figure=fig_ndhB, 
#                       adj_matrix=W_EM_ndhB, 
#                       vertex=nodes, 
#                       dataset_name='ndhB', 
#                       EM_bn=EM_bn_ndhB_notears, 
#                       EM_data=EM_data_ndhB_notears, 
#                       EM_syn=synthetic_data, 
#                       param_alg=param_alg)
# plt.close(fig_ndhB)



# # EM for ndhD
# data = ndhD
# pos_NA = pos_NA_ndhD
# pos_gen = list(events_sites_ndhD.values())

# # define the main parameters for the algorithm
# param_alg = ['notears', 'notears_optimization', 'BayesianEstimator - K2']

# EM_ndhD_notears = utils.EM_dag(data, pos_NA, param_alg[0], name_nodes=pos_gen, notears_loss_type='logistic')
# EM_data_ndhD_notears = EM_ndhD_notears[2]
# EM_bn_ndhD_notears = EM_ndhD_notears[1]
# EM_model_ndhD_notears = EM_ndhD_notears[0]
# print(f"The EM model for ndhD with NOTEARS is a DAG with {len(EM_model_ndhD_notears['nodes'])} nodes and {len(EM_model_ndhD_notears['edges'])} edges")

# #synthetic data from the learned Bayesian Network
# syn_bn = BayesianModelSampling(EM_bn_ndhD_notears) # class for sampling methods from a Bayesian Network
# synthetic_data = syn_bn.forward_sample(size=data.shape[0]) # generate the synthetic data

# # define the graphical representation of the model (from the DAG model!)
# nodes = EM_model_ndhD_notears['nodes']
# edges = EM_model_ndhD_notears['edges']

# G_EM_ndhD = ig.Graph(n=len(nodes), directed=True)
# G_EM_ndhD.vs["label"] = G_EM_ndhD.vs["name"] = nodes
# G_EM_ndhD.add_edges(edges)

# W_EM_ndhD = G_EM_ndhD.get_adjacency() # obtain the adjacency matrix

# fig_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(G_EM_ndhD, vertex_color = "#034730", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=8, vertex_label_dist=0.5)

# # save the results
# utils.save_results_EM(figure=fig_ndhD, 
#                       adj_matrix=W_EM_ndhD, 
#                       vertex=nodes, 
#                       dataset_name='ndhD', 
#                       EM_bn=EM_bn_ndhD_notears, 
#                       EM_data=EM_data_ndhD_notears, 
#                       EM_syn=synthetic_data, 
#                       param_alg=param_alg)
# plt.close(fig_ndhD)