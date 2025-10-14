#####################################
#----- Preamble
#####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig 
ig.config['plotting.backend'] = 'matplotlib'

from pathlib import Path
import pickle

import utils
from utils import data_folder
from utils import graphs_folder
from utils import dags_folder_EM
from utils import dags_folder_causality

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.metrics import structure_score
from pgmpy.metrics import log_likelihood_score





#####################################
#----- Loading our data sets
#####################################

# load the file where there is the info about the events
editing_sites =  pd.read_csv(data_folder/f'edition_sep_col.csv')

# we use the data sets that we have already imputed in 'imputation.py'
ndhB = pd.read_csv(data_folder/'ndhB_imputed_IterativeImputer_LogisticReg.csv', index_col=0)
ndhB_intron = pd.read_csv(data_folder/'ndhB_imputed_IterativeImputer_LogisticReg_with_intron.csv', index_col=0)  
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





#############################################################################
#----- Reference Chronology: chronology in Guillem's paper (cf. Fig. 6)
#############################################################################

'''# Ref. Chronology for ndhB: WITH intron
gen_pos_ndhB = list(events_sites_ndhB_intron.values())
ndhB.columns = gen_pos_ndhB

G_chron_ref_ndhB = ig.Graph(n=len(gen_pos_ndhB), directed=True) # create a directed graph
G_chron_ref_ndhB.vs["name"] = gen_pos_ndhB # the vertices of the graph represents the events of Guillem et al. paper
G_chron_ref_ndhB.vs["label"] = G_chron_ref_ndhB.vs["name"] # define the labels of the vertex, that is, we print the names of the vertex in the graphical representation

# define the edges following the paper
edges = [(G_chron_ref_ndhB.vs[1]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[1]["name"], G_chron_ref_ndhB.vs[7]["name"]),
         (G_chron_ref_ndhB.vs[1]["name"], G_chron_ref_ndhB.vs[11]["name"]),
         (G_chron_ref_ndhB.vs[3]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[3]["name"], G_chron_ref_ndhB.vs[7]["name"]),
         (G_chron_ref_ndhB.vs[3]["name"], G_chron_ref_ndhB.vs[11]["name"]),
         (G_chron_ref_ndhB.vs[4]["name"], G_chron_ref_ndhB.vs[6]["name"]),
         (G_chron_ref_ndhB.vs[5]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[5]["name"], G_chron_ref_ndhB.vs[7]["name"]),
         (G_chron_ref_ndhB.vs[5]["name"], G_chron_ref_ndhB.vs[11]["name"]),
         (G_chron_ref_ndhB.vs[7]["name"], G_chron_ref_ndhB.vs[6]["name"]),
         (G_chron_ref_ndhB.vs[10]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[10]["name"], G_chron_ref_ndhB.vs[7]["name"]),
         (G_chron_ref_ndhB.vs[10]["name"], G_chron_ref_ndhB.vs[11]["name"]),
         (G_chron_ref_ndhB.vs[11]["name"], G_chron_ref_ndhB.vs[6]["name"]),
         (G_chron_ref_ndhB.vs[12]["name"], G_chron_ref_ndhB.vs[1]["name"]), 
         (G_chron_ref_ndhB.vs[12]["name"], G_chron_ref_ndhB.vs[3]["name"]), 
         (G_chron_ref_ndhB.vs[12]["name"], G_chron_ref_ndhB.vs[5]["name"]), 
         (G_chron_ref_ndhB.vs[12]["name"], G_chron_ref_ndhB.vs[10]["name"])]
G_chron_ref_ndhB.add_edges(edges)

W_chron_ref_ndhB = G_chron_ref_ndhB.get_adjacency()
W_chron_ref_ndhB = pd.DataFrame(W_chron_ref_ndhB, columns=gen_pos_ndhB, index=gen_pos_ndhB)
# W_chron_ref_ndhB.to_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhB_Fig6.csv')

# ig.plot(G_chron_ref_ndhB, vertex_color = "#F6AD1A", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=8, vertex_label_dist=0.5)
# plt.savefig(graphs_folder/f"chron_ndhB_Fig6.png", format="PNG", dpi=500)
# plt.close()'''

# Ref. Chronology for ndhB: WITHOUT intron
gen_pos_ndhB = list(events_sites_ndhB.values())
ndhB.columns = gen_pos_ndhB

G_chron_ref_ndhB = ig.Graph(n=len(gen_pos_ndhB), directed=True) # create a directed graph
G_chron_ref_ndhB.vs["name"] = gen_pos_ndhB # the vertices of the graph represents the events of Guillem et al. paper
G_chron_ref_ndhB.vs["label"] = G_chron_ref_ndhB.vs["name"] # define the labels of the vertex, that is, we print the names of the vertex in the graphical representation


# define the edges following the paper
edges = [(G_chron_ref_ndhB.vs[1]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[1]["name"], G_chron_ref_ndhB.vs[6]["name"]),
         (G_chron_ref_ndhB.vs[1]["name"], G_chron_ref_ndhB.vs[10]["name"]),
         (G_chron_ref_ndhB.vs[3]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[3]["name"], G_chron_ref_ndhB.vs[6]["name"]),
         (G_chron_ref_ndhB.vs[3]["name"], G_chron_ref_ndhB.vs[10]["name"]),
         (G_chron_ref_ndhB.vs[5]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[5]["name"], G_chron_ref_ndhB.vs[6]["name"]),
         (G_chron_ref_ndhB.vs[5]["name"], G_chron_ref_ndhB.vs[10]["name"]),
         (G_chron_ref_ndhB.vs[9]["name"], G_chron_ref_ndhB.vs[4]["name"]),
         (G_chron_ref_ndhB.vs[9]["name"], G_chron_ref_ndhB.vs[6]["name"]),
         (G_chron_ref_ndhB.vs[9]["name"], G_chron_ref_ndhB.vs[10]["name"]),
         (G_chron_ref_ndhB.vs[11]["name"], G_chron_ref_ndhB.vs[1]["name"]), 
         (G_chron_ref_ndhB.vs[11]["name"], G_chron_ref_ndhB.vs[3]["name"]), 
         (G_chron_ref_ndhB.vs[11]["name"], G_chron_ref_ndhB.vs[5]["name"]), 
         (G_chron_ref_ndhB.vs[11]["name"], G_chron_ref_ndhB.vs[9]["name"])]
G_chron_ref_ndhB.add_edges(edges)

W_chron_ref_ndhB = G_chron_ref_ndhB.get_adjacency()
W_chron_ref_ndhB = pd.DataFrame(W_chron_ref_ndhB, columns=gen_pos_ndhB, index=gen_pos_ndhB)
# W_chron_ref_ndhB.to_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhB_Fig6_without_intron.csv')

# ig.plot(G_chron_ref_ndhB, vertex_color = "#F6AD1A", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=8, vertex_label_dist=0.5)
# plt.savefig(graphs_folder/f"chron_ndhB_Fig6_withouot_intron.png", format="PNG", dpi=500)
# plt.close()


# create the Discrete Bayesian Network model toghether with its parameters (CPDs) associated to the Ref. chronology DAG
chronology_ref_bn = DiscreteBayesianNetwork(edges)
dag_estimator = BayesianEstimator(chronology_ref_bn, ndhB) # initialize a Bayesian Estimator for computing the parameters of the model (CPDs)
dag_cpds = [] # store the CPDs for each variable
for col in chronology_ref_bn.nodes: # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
    cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
    chronology_ref_bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
    dag_cpds.append(cpd_col)

# Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
filtered_ndhB = ndhB[list(chronology_ref_bn.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
BIC_score_chron_ref_ndhB = structure_score(chronology_ref_bn, filtered_ndhB, scoring_method="bic-d")
log_likeli_score_chron_ref_ndhB = log_likelihood_score(chronology_ref_bn, filtered_ndhB)
scores_chron_ref_ndhB = [BIC_score_chron_ref_ndhB, log_likeli_score_chron_ref_ndhB]

# # Falsification of Ref. chronology DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_ndhB_Fig6 = utils.falsify_dag(ndhB, G_chron_ref_ndhB)
# # save the falsification summary
# fals_summary = {
#     "Falsification Summary" : falsification_summary_chron_ndhB_Fig6[0],
#     "DAG is Falsifiable" : falsification_summary_chron_ndhB_Fig6[1],
#     "DAG is Falsified" : falsification_summary_chron_ndhB_Fig6[2]
# }
# with open(graphs_folder/f"falsification_summary_chron_ndhB_Fig6_without_intron.txt", "w") as file:
#     for fals, value in fals_summary.items():
#         file.write(f"{fals}: {value}\n")


'''# # Completing the Ref. chronology DAG into a full DAG
# # with Hill-Climbing
# hc_completion_ref_chron_ndhB = utils.completion_chron_dag(ndhB, chron_ref_ndhB, 'hill_climbing')
# hc_completed_ref_chron_model_ndhB = hc_completion_ref_chron_ndhB[0]
# nodes_completed = hc_completed_ref_chron_model_ndhB.vs["name"]
# W_hc_completed_ref_chron_model_ndhB = hc_completed_ref_chron_model_ndhB.get_adjacency()
# W_hc_completed_ref_chron_model_ndhB = pd.DataFrame(W_hc_completed_ref_chron_model_ndhB, columns=nodes_completed, index=nodes_completed)
# # W_hc_completed_ref_chron_model_ndhB.to_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhB_Fig5_completed_hc.csv')
# # ig.plot(hc_completed_ref_chron_model_ndhB, vertex_color = "lightcoral", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)
# # plt.savefig(graphs_folder/f"chron_ndhB_Fig6_completed_hc.png", format="PNG", dpi=500)
# # plt.close()

# # with PC
# pc_completion_ref_chron_ndhB = utils.completion_chron_dag(ndhB, chron_ref_ndhB, 'pc')
# pc_completed_ref_chron_model_ndhB = hc_completion_ref_chron_ndhB[0]
# nodes_completed = pc_completed_ref_chron_model_ndhB.vs["name"]
# W_pc_completed_ref_chron_model_ndhB = pc_completed_ref_chron_model_ndhB.get_adjacency()
# W_pc_completed_ref_chron_model_ndhB = pd.DataFrame(W_pc_completed_ref_chron_model_ndhB, columns=nodes_completed, index=nodes_completed)
# # W_pc_completed_ref_chron_model_ndhB.to_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhB_Fig5_completed_pc.csv')
# # ig.plot(pc_completed_ref_chron_model_ndhB, vertex_color = "lightcoral", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)
# # plt.savefig(graphs_folder/f"chron_ndhB_Fig6_completed_pc.png", format="PNG", dpi=500)
# # plt.close()


# #synthetic data from the completed Bayesian Network
# # with Hill-Climbing
# hc_bn_completed_ref_chron_model_ndhB = hc_completion_ref_chron_ndhB[1]
# hc_bn_completed_syn = BayesianModelSampling(hc_bn_completed_ref_chron_model_ndhB)
# hc_completed_ref_chron_model_ndhB_synthetic_data = hc_bn_completed_syn.forward_sample(size=ndhB.shape[0])
# hc_completed_ref_chron_model_ndhB_synthetic_data.to_csv(graphs_folder/f'chron_ndhB_syn_data_Fig6_completed_hc.csv')

# # with PC
# pc_bn_completed_ref_chron_model_ndhB = pc_completion_ref_chron_ndhB[1]
# pc_bn_completed_syn = BayesianModelSampling(pc_bn_completed_ref_chron_model_ndhB)
# pc_completed_ref_chron_model_ndhB_synthetic_data = pc_bn_completed_syn.forward_sample(size=ndhB.shape[0])
# pc_completed_ref_chron_model_ndhB_synthetic_data.to_csv(graphs_folder/f'chron_ndhB_syn_data_Fig6_completed_pc.csv')'''




# Ref. Chronology for ndhD
gen_pos_ndhD = list(events_sites_ndhD.values())
ndhD.columns = gen_pos_ndhD

G_chron_ref_ndhD = ig.Graph(n=len(gen_pos_ndhD), directed=True) # create a directed graph
G_chron_ref_ndhD.vs["name"] = gen_pos_ndhD # the vertices of the graph represents the events of Guillem et al. paper
G_chron_ref_ndhD.vs["label"] = G_chron_ref_ndhD.vs["name"] # define the labels of the vertex, that is, we print the names of the vertex in the graphical representation

# define the edges following the paper
edges = [(G_chron_ref_ndhD.vs[0]["name"], G_chron_ref_ndhD.vs[4]["name"]),
         (G_chron_ref_ndhD.vs[1]["name"], G_chron_ref_ndhD.vs[4]["name"]),
         (G_chron_ref_ndhD.vs[2]["name"], G_chron_ref_ndhD.vs[0]["name"]),
         (G_chron_ref_ndhD.vs[2]["name"], G_chron_ref_ndhD.vs[1]["name"]),
         (G_chron_ref_ndhD.vs[3]["name"], G_chron_ref_ndhD.vs[2]["name"])]
G_chron_ref_ndhD.add_edges(edges)

W_chron_ref_ndhD = G_chron_ref_ndhD.get_adjacency()
W_chron_ref_ndhD = pd.DataFrame(W_chron_ref_ndhD, columns=gen_pos_ndhD, index=gen_pos_ndhD)
# W_chron_ref_ndhD.to_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhD_Fig6.csv')

# ig.plot(G_chron_ref_ndhD, vertex_color = "#F6AD1A", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=10, vertex_label_dist=0.5)
# plt.savefig(graphs_folder/f"chron_ndhD_Fig6.png", format="PNG", dpi=500)
# plt.close()


# create the Discrete Bayesian Network model toghether with its parameters (CPDs) associated to the Ref. chronology DAG
chronology_ref_bn = DiscreteBayesianNetwork(edges)
dag_estimator = BayesianEstimator(chronology_ref_bn, ndhD) # initialize a Bayesian Estimator for computing the parameters of the model (CPDs)
dag_cpds = [] # store the CPDs for each variable
for col in chronology_ref_bn.nodes: # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
    cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
    chronology_ref_bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
    dag_cpds.append(cpd_col)

# Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
filtered_ndhD = ndhD[list(chronology_ref_bn.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
BIC_score_chron_ref_ndhD = structure_score(chronology_ref_bn, filtered_ndhD, scoring_method="bic-d")
log_likeli_score_chron_ref_ndhD = log_likelihood_score(chronology_ref_bn, filtered_ndhD)
scores_chron_ref_ndhD = [BIC_score_chron_ref_ndhD, log_likeli_score_chron_ref_ndhD]

# # Falsification of Ref. chronology DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_ndhD_Fig6 = utils.falsify_dag(ndhD, chron_ref_ndhD)
# # save the falsification summary
# fals_summary = {
#     "Falsification Summary" : falsification_summary_chron_ndhD_Fig6[0],
#     "DAG is Falsifiable" : falsification_summary_chron_ndhD_Fig6[1],
#     "DAG is Falsified" : falsification_summary_chron_ndhD_Fig6[2]
# }
# with open(graphs_folder/f"falsification_summary_chron_ndhD_Fig6.txt", "w") as file:
#     for fals, value in fals_summary.items():
#         file.write(f"{fals}: {value}\n")


'''# # Completing the Ref. chronology DAG into a full DAG
# # with Hill-Climbing
# hc_completion_ref_chron_ndhD = utils.completion_chron_dag(ndhD, chron_ref_ndhD, 'hill_climbing')
# hc_completed_ref_chron_model_ndhD = hc_completion_ref_chron_ndhD[0]
# nodes_completed = hc_completed_ref_chron_model_ndhD.vs["name"]
# W_hc_completed_ref_chron_model_ndhB = hc_completed_ref_chron_model_ndhD.get_adjacency()
# W_hc_completed_ref_chron_model_ndhB = pd.DataFrame(W_hc_completed_ref_chron_model_ndhB, columns=nodes_completed, index=nodes_completed)
# # W_hc_completed_ref_chron_model_ndhB.to_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhD_Fig5_completed_hc.csv')
# # ig.plot(hc_completed_ref_chron_model_ndhD, vertex_color = "lightcoral", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)
# # plt.savefig(graphs_folder/f"chron_ndhD_Fig6_completed_hc.png", format="PNG", dpi=500)
# # plt.close()

# # with PC
# pc_completion_ref_chron_ndhD = utils.completion_chron_dag(ndhD, chron_ref_ndhD, 'pc')
# pc_completed_ref_chron_model_ndhD = pc_completion_ref_chron_ndhD[0]
# nodes_completed = pc_completed_ref_chron_model_ndhD.vs["name"]
# W_pc_completed_ref_chron_model_ndhD = pc_completed_ref_chron_model_ndhD.get_adjacency()
# W_pc_completed_ref_chron_model_ndhD = pd.DataFrame(W_pc_completed_ref_chron_model_ndhD, columns=nodes_completed, index=nodes_completed)
# # W_pc_completed_ref_chron_model_ndhD.to_csv(graphs_folder/f'AdjacencyMatrix_chron_ndhD_Fig5_completed_pc.csv')
# # ig.plot(pc_completed_ref_chron_model_ndhD, vertex_color = "lightcoral", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)
# # plt.savefig(graphs_folder/f"chron_ndhD_Fig6_completed_pc.png", format="PNG", dpi=500)
# # plt.close()


# #synthetic data from the completed Bayesian Network
# # with Hill-Climbing
# hc_bn_completed_ref_chron_model_ndhD = hc_completion_ref_chron_ndhD[1]
# hc_bn_completed_syn = BayesianModelSampling(hc_bn_completed_ref_chron_model_ndhD)
# hc_completed_ref_chron_model_ndhD_synthetic_data = hc_bn_completed_syn.forward_sample(size=ndhD.shape[0])
# hc_completed_ref_chron_model_ndhD_synthetic_data.to_csv(graphs_folder/f'chron_ndhD_syn_data_Fig6_completed_hc.csv')

# # with PC
# pc_bn_completed_ref_chron_model_ndhD = pc_completion_ref_chron_ndhD[1]
# pc_bn_completed_syn = BayesianModelSampling(pc_bn_completed_ref_chron_model_ndhD)
# pc_completed_ref_chron_model_ndhD_synthetic_data = pc_bn_completed_syn.forward_sample(size=ndhD.shape[0])
# pc_completed_ref_chron_model_ndhD_synthetic_data.to_csv(graphs_folder/f'chron_ndhD_syn_data_Fig6_completed_pc.csv')'''





# #################################################################
# #----- Causality from Hill Climbing DAG discovery
# #################################################################

# # Causality for ndhB

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhB_hill_climbing_2025_10_03_at_01_44_29')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhB.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhB.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhB.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhB

# # load the data obtained after the EM implementation
# ndhB_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhB = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhB = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhB = ig.Graph.Adjacency(W_EM_ndhB)
# G_EM_ndhB.vs['label'] = G_EM_ndhB.vs["name"] = list(W_EM_ndhB.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhB_EM, G_EM_ndhB, bn_EM_ndhB)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhB = utils.chronology_events(causality_summary, G_EM_ndhB, original_data)
# chronology_dag_ndhB = chronology_ndhB[0]
# chronology_nodes = chronology_dag_ndhB.vs["name"]
# W_chronology_ndhB = chronology_dag_ndhB.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhB = pd.DataFrame(W_chronology_ndhB, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhB = chronology_ndhB[1]
# causal_rel_topo_level_ndhB = chronology_ndhB[2]
# causal_relations_ndhB = chronology_ndhB[3]
# refutation_summary_causal_rel_ndhB = chronology_ndhB[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhB, vertex_color = "#0ABE8B", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=8, vertex_label_dist=0.5)

# # Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# filtered_original_data = original_data[list(chronology_bn_ndhB.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
# BIC_score_chron_ndhB = structure_score(chronology_bn_ndhB, filtered_original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhB = log_likelihood_score(chronology_bn_ndhB, filtered_original_data)
# scores_chron_ndhB = [BIC_score_chron_ndhB, log_likeli_score_chron_ndhB]

# # Falsification of chronology (completed) DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhB = utils.falsify_dag(ndhB, chronology_dag_ndhB)


'''# # # SANITY CHECK: Completing the chronology DAG into a full DAG
# #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # completion_chron_ndhB = utils.completion_chron_dag(ndhB, chronology_dag_ndhB, EM_param_alg[0])
# # completed_chron_model_ndhB = completion_chron_ndhB[0]
# # nodes_completed = completed_chron_model_ndhB.vs["name"]
# # W_completed_chron_model_ndhB = completed_chron_model_ndhB.get_adjacency()
# # W_completed_chron_model_ndhB = pd.DataFrame(W_completed_chron_model_ndhB, columns=nodes_completed, index=nodes_completed)
# # # define the graphical representation for this completion
# # fig_completed_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# # ig.plot(completed_chron_model_ndhB, vertex_color = "red", vertex_size = 55, edge_arrow_size = 25, edge_width = 0.5, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhB,
#                              adj_matrix_chron = W_chronology_ndhB,
#                              dataset_name='ndhB',
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhB,
#                              causality_summary=causality_summary, 
#                              causal_relations=causal_relations_ndhB,
#                              refutation_summary=refutation_summary_causal_rel_ndhB,
#                              param_alg=EM_param_alg, 
#                              scores_chron=[scores_chron_ref_ndhB, scores_chron_ndhB],
#                              falsification_summary=falsification_summary_chron_model_ndhB,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhB)
# # plt.close(fig_completed_chron_ndhB)



# # Causality for ndhD

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhD_hill_climbing_2025_10_03_at_01_46_50')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhD.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhD.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhD.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhD

# # load the data obtained after the EM implementation
# ndhD_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhD = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhD = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhD = ig.Graph.Adjacency(W_EM_ndhD)
# G_EM_ndhD.vs['label'] = G_EM_ndhD.vs["name"] = list(W_EM_ndhD.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhD_EM, G_EM_ndhD, bn_EM_ndhD)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhD = utils.chronology_events(causality_summary, G_EM_ndhD, original_data)
# chronology_dag_ndhD = chronology_ndhD[0]
# chronology_nodes = chronology_dag_ndhD.vs["name"]
# W_chronology_ndhD = chronology_dag_ndhD.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhD = pd.DataFrame(W_chronology_ndhD, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhD = chronology_ndhD[1]
# causal_rel_topo_level_ndhD = chronology_ndhD[2]
# causal_relations_ndhD = chronology_ndhD[3]
# refutation_summary_causal_rel_ndhD = chronology_ndhD[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhD, vertex_color = "#0ABE8B", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=10, vertex_label_dist=0.5)

# Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# BIC_score_chron_ndhD = structure_score(chronology_bn_ndhD, original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhD = log_likelihood_score(chronology_bn_ndhD, original_data)
# scores_chron_ndhD = [BIC_score_chron_ndhD, log_likeli_score_chron_ndhD]

# # Falsification of chronology DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhD = utils.falsify_dag(original_data, chronology_dag_ndhD)


'''# # # # SANITY CHECK: Completing the chronology DAG into a full DAG
# #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # # completion_chron_ndhD = utils.completion_chron_dag(original_data, chronology_dag_ndhD, EM_param_alg[0])
# # completed_chron_model_ndhD = completion_chron_ndhD[0]
# # nodes_completed = completed_chron_model_ndhD.vs["name"]
# # W_completed_chron_model_ndhD = completed_chron_model_ndhD.get_adjacency()
# # W_completed_chron_model_ndhD = pd.DataFrame(W_completed_chron_model_ndhD, columns=nodes_completed, index=nodes_completed)
# # # define the graphical representation for this completion
# # fig_completed_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# # ig.plot(completed_chron_model_ndhD, vertex_color = "red", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhD, 
#                              adj_matrix_chron = W_chronology_ndhD,
#                              dataset_name='ndhD', 
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhD,
#                              causality_summary=causality_summary,
#                              refutation_summary=refutation_summary_causal_rel_ndhD, 
#                              causal_relations=causal_relations_ndhD, 
#                              param_alg=EM_param_alg,
#                              scores_chron=[scores_chron_ref_ndhD, scores_chron_ndhD],
#                              falsification_summary=falsification_summary_chron_model_ndhD,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhD)
# # plt.close(fig_completed_chron_ndhD)





# #################################################################
# #----- Causality from PC DAG discovery
# #################################################################

# # Causality for ndhB

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhB_pc_2025_10_03_at_01_49_32')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhB.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhB.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhB.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhB

# # load the data obtained after the EM implementation
# ndhB_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhB = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhB = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhB = ig.Graph.Adjacency(W_EM_ndhB)
# G_EM_ndhB.vs['label'] = G_EM_ndhB.vs["name"] = list(W_EM_ndhB.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhB_EM, G_EM_ndhB, bn_EM_ndhB)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhB = utils.chronology_events(causality_summary, G_EM_ndhB, original_data)
# chronology_dag_ndhB = chronology_ndhB[0]
# chronology_nodes = chronology_dag_ndhB.vs["name"]
# W_chronology_ndhB = chronology_dag_ndhB.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhB = pd.DataFrame(W_chronology_ndhB, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhB = chronology_ndhB[1]
# causal_rel_topo_level_ndhB = chronology_ndhB[2]
# causal_relations_ndhB = chronology_ndhB[3]
# refutation_summary_causal_rel_ndhB = chronology_ndhB[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhB, vertex_color = "#0ABE8B", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=6, vertex_label_dist=0.5)

# # Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# filtered_original_data = original_data[list(chronology_bn_ndhB.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
# BIC_score_chron_ndhB = structure_score(chronology_bn_ndhB, filtered_original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhB = log_likelihood_score(chronology_bn_ndhB, filtered_original_data)
# scores_chron_ndhB = [BIC_score_chron_ndhB, log_likeli_score_chron_ndhB]

# # Falsification of chronology (completed) DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhB = utils.falsify_dag(original_data, chronology_dag_ndhB)


'''# # # SANITY CHECK: Completing the chronology DAG into a full DAG
# # #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # # completion_chron_ndhB = utils.completion_chron_dag(original_data, chronology_dag_ndhB, EM_param_alg[0])
# # # completed_chron_model_ndhB = completion_chron_ndhB[0]
# # # nodes_completed = completed_chron_model_ndhB.vs["name"]
# # # W_completed_chron_model_ndhB = completed_chron_model_ndhB.get_adjacency()
# # # W_completed_chron_model_ndhB = pd.DataFrame(W_completed_chron_model_ndhB, columns=nodes_completed, index=nodes_completed)
# # # # define the graphical representation for this completion
# # # fig_completed_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# # # ig.plot(completed_chron_model_ndhB, vertex_color = "red", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhB,
#                              adj_matrix_chron = W_chronology_ndhB,
#                              dataset_name='ndhB',
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhB,
#                              causality_summary=causality_summary, 
#                              causal_relations=causal_relations_ndhB,
#                              refutation_summary=refutation_summary_causal_rel_ndhB,
#                              param_alg=EM_param_alg, 
#                              scores_chron=[scores_chron_ref_ndhB, scores_chron_ndhB],
#                              falsification_summary=falsification_summary_chron_model_ndhB,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhB)
# # plt.close(fig_completed_chron_ndhB)



# # Causality for ndhD

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhD_pc_2025_10_03_at_01_51_14')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhD.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhD.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhD.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhD

# # load the data obtained after the EM implementation
# ndhD_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhD = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhD = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhD = ig.Graph.Adjacency(W_EM_ndhD)
# G_EM_ndhD.vs['label'] = G_EM_ndhD.vs["name"] = list(W_EM_ndhD.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhD_EM, G_EM_ndhD, bn_EM_ndhD)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhD = utils.chronology_events(causality_summary, G_EM_ndhD, original_data)
# chronology_dag_ndhD = chronology_ndhD[0]
# chronology_nodes = chronology_dag_ndhD.vs["name"]
# W_chronology_ndhD = chronology_dag_ndhD.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhD = pd.DataFrame(W_chronology_ndhD, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhD = chronology_ndhD[1]
# causal_rel_topo_level_ndhD = chronology_ndhD[2]
# causal_relations_ndhD = chronology_ndhD[3]
# refutation_summary_causal_rel_ndhD = chronology_ndhD[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhD, vertex_color = "#0ABE8B", vertex_size =75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=9, vertex_label_dist=0.5)

# # Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# filtered_original_data = original_data[list(chronology_bn_ndhD.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
# BIC_score_chron_ndhD = structure_score(chronology_bn_ndhD, filtered_original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhD = log_likelihood_score(chronology_bn_ndhD, filtered_original_data)
# scores_chron_ndhD = [BIC_score_chron_ndhD, log_likeli_score_chron_ndhD]

# # Falsification of chronology (completed) DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhD = utils.falsify_dag(original_data, chronology_dag_ndhD)


'''# # # # SANITY CHECK: Completing the chronology DAG into a full DAG
# # #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # # completion_chron_ndhD = utils.completion_chron_dag(original_data, chronology_dag_ndhD, EM_param_alg[0])
# # # completed_chron_model_ndhD = completion_chron_ndhD[0]
# # # nodes_completed = completed_chron_model_ndhD.vs["name"]
# # # W_completed_chron_model_ndhD = completed_chron_model_ndhD.get_adjacency()
# # # W_completed_chron_model_ndhD = pd.DataFrame(W_completed_chron_model_ndhD, columns=nodes_completed, index=nodes_completed)
# # # # define the graphical representation for this completion
# # # fig_completed_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# # # ig.plot(completed_chron_model_ndhD, vertex_color = "red", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhD, 
#                              adj_matrix_chron = W_chronology_ndhD,
#                              dataset_name='ndhD', 
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhD,
#                              causality_summary=causality_summary,
#                              refutation_summary=refutation_summary_causal_rel_ndhD,
#                              causal_relations=causal_relations_ndhD, 
#                              param_alg=EM_param_alg,
#                              scores_chron=[scores_chron_ref_ndhD, scores_chron_ndhD],
#                              falsification_summary=falsification_summary_chron_model_ndhD,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhD)
# # plt.close(fig_completed_chron_ndhD)





# #################################################################
# #----- Causality from LiNGAM DAG discovery
# #################################################################

# # Causality for ndhB

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhB_lingam_2025_10_03_at_01_52_17')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhB.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhB.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhB.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhB

# # load the data obtained after the EM implementation
# ndhB_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhB = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhB = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhB = ig.Graph.Adjacency(W_EM_ndhB)
# G_EM_ndhB.vs['label'] = G_EM_ndhB.vs["name"] = list(W_EM_ndhB.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhB_EM, G_EM_ndhB, bn_EM_ndhB)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhB = utils.chronology_events(causality_summary, G_EM_ndhB, original_data)
# chronology_dag_ndhB = chronology_ndhB[0]
# chronology_nodes = chronology_dag_ndhB.vs["name"]
# W_chronology_ndhB = chronology_dag_ndhB.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhB = pd.DataFrame(W_chronology_ndhB, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhB = chronology_ndhB[1]
# causal_rel_topo_level_ndhB = chronology_ndhB[2]
# causal_relations_ndhB = chronology_ndhB[3]
# refutation_summary_causal_rel_ndhB = chronology_ndhB[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhB, vertex_color = "#0ABE8B", vertex_size =75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=8, vertex_label_dist=0.5)

# # Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# filtered_original_data = original_data[list(chronology_bn_ndhB.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
# BIC_score_chron_ndhB = structure_score(chronology_bn_ndhB, filtered_original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhB = log_likelihood_score(chronology_bn_ndhB, filtered_original_data)
# scores_chron_ndhB = [BIC_score_chron_ndhB, log_likeli_score_chron_ndhB]

# # Falsification of chronology (completed) DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhB = utils.falsify_dag(original_data, chronology_dag_ndhB)


'''# # # SANITY CHECK: Completing the chronology DAG into a full DAG
# #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # completion_chron_ndhB = utils.completion_chron_dag(original_data, chronology_dag_ndhB, EM_param_alg[0])
# # completed_chron_model_ndhB = completion_chron_ndhB[0]
# # nodes_completed = completed_chron_model_ndhB.vs["name"]
# # W_completed_chron_model_ndhB = completed_chron_model_ndhB.get_adjacency()
# # W_completed_chron_model_ndhB = pd.DataFrame(W_completed_chron_model_ndhB, columns=nodes_completed, index=nodes_completed)
# # # define the graphical representation for this completion
# # fig_completed_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# # ig.plot(completed_chron_model_ndhB, vertex_color = "red", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhB,
#                              adj_matrix_chron = W_chronology_ndhB,
#                              dataset_name='ndhB',
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhB,
#                              causality_summary=causality_summary, 
#                              causal_relations=causal_relations_ndhB,
#                              refutation_summary=refutation_summary_causal_rel_ndhB,
#                              param_alg=EM_param_alg, 
#                              scores_chron=[scores_chron_ref_ndhB, scores_chron_ndhB],
#                              falsification_summary=falsification_summary_chron_model_ndhB,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhB)
# # plt.close(fig_completed_chron_ndhB)



# # Causality for ndhD

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhD_lingam_2025_10_03_at_01_53_09')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhD.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhD.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhD.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhD

# # load the data obtained after the EM implementation
# ndhD_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhD = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhD = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhD = ig.Graph.Adjacency(W_EM_ndhD)
# G_EM_ndhD.vs['label'] = G_EM_ndhD.vs["name"] = list(W_EM_ndhD.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhD_EM, G_EM_ndhD, bn_EM_ndhD)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhD = utils.chronology_events(causality_summary, G_EM_ndhD, original_data)
# chronology_dag_ndhD = chronology_ndhD[0]
# chronology_nodes = chronology_dag_ndhD.vs["name"]
# W_chronology_ndhD = chronology_dag_ndhD.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhD = pd.DataFrame(W_chronology_ndhD, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhD = chronology_ndhD[1]
# causal_rel_topo_level_ndhD = chronology_ndhD[2]
# causal_relations_ndhD = chronology_ndhD[3]
# refutation_summary_causal_rel_ndhD = chronology_ndhD[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhD, vertex_color = "#0ABE8B", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=9, vertex_label_dist=0.5)

# # # Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# filtered_original_data = original_data[list(chronology_bn_ndhD.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
# BIC_score_chron_ndhD = structure_score(chronology_bn_ndhD, filtered_original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhD = log_likelihood_score(chronology_bn_ndhD, filtered_original_data)
# scores_chron_ndhD = [BIC_score_chron_ndhD, log_likeli_score_chron_ndhD]

# # Falsification of chronology (completed) DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhD = utils.falsify_dag(original_data, chronology_dag_ndhD)


'''# # # SANITY CHECK: Completing the chronology DAG into a full DAG
# #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # completion_chron_ndhD = utils.completion_chron_dag(original_data, chronology_dag_ndhD, EM_param_alg[0])
# # completed_chron_model_ndhD = completion_chron_ndhD[0]
# # nodes_completed = completed_chron_model_ndhD.vs["name"]
# # W_completed_chron_model_ndhD = completed_chron_model_ndhD.get_adjacency()
# # W_completed_chron_model_ndhD = pd.DataFrame(W_completed_chron_model_ndhD, columns=nodes_completed, index=nodes_completed)
# # # define the graphical representation for this completion
# # fig_completed_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# # ig.plot(completed_chron_model_ndhD, vertex_color = "red", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhD, 
#                              adj_matrix_chron = W_chronology_ndhD,
#                              dataset_name='ndhD', 
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhD,
#                              causality_summary=causality_summary,
#                              refutation_summary=refutation_summary_causal_rel_ndhD, 
#                              causal_relations=causal_relations_ndhD, 
#                              param_alg=EM_param_alg,
#                              scores_chron=[scores_chron_ref_ndhD, scores_chron_ndhD],
#                              falsification_summary=falsification_summary_chron_model_ndhD,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhD)
# # plt.close(fig_completed_chron_ndhD)





# #################################################################
# #----- Causality from NOTEARS DAG discovery
# #################################################################

# # Causality for ndhB

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhB_notears_2025_10_03_at_02_11_04')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhB.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhB.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhB.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhB

# # load the data obtained after the EM implementation
# ndhB_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhB = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhB = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhB = ig.Graph.Adjacency(W_EM_ndhB)
# G_EM_ndhB.vs['label'] = G_EM_ndhB.vs["name"] = list(W_EM_ndhB.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhB_EM, G_EM_ndhB, bn_EM_ndhB)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhB = utils.chronology_events(causality_summary, G_EM_ndhB, original_data)
# chronology_dag_ndhB = chronology_ndhB[0]
# chronology_nodes = chronology_dag_ndhB.vs["name"]
# W_chronology_ndhB = chronology_dag_ndhB.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhB = pd.DataFrame(W_chronology_ndhB, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhB = chronology_ndhB[1]
# causal_rel_topo_level_ndhB = chronology_ndhB[2]
# causal_relations_ndhB = chronology_ndhB[3]
# refutation_summary_causal_rel_ndhB = chronology_ndhB[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhB, vertex_color = "#0ABE8B", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=5, vertex_label_dist=0.5)

# # Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# filtered_original_data = original_data[list(chronology_bn_ndhB.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
# BIC_score_chron_ndhB = structure_score(chronology_bn_ndhB, filtered_original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhB = log_likelihood_score(chronology_bn_ndhB, filtered_original_data)
# scores_chron_ndhB = [BIC_score_chron_ndhB, log_likeli_score_chron_ndhB]

# # Falsification of chronology (completed) DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhB = utils.falsify_dag(original_data, chronology_dag_ndhB)

'''# # # SANITY CHECK: Completing the chronology DAG into a full DAG
# #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # # completion_chron_ndhB = utils.completion_chron_dag(original_data, chronology_dag_ndhB, EM_param_alg[0], notears_loss_type='logistic')
# # # completed_chron_model_ndhB = completion_chron_ndhB[0]
# # # nodes_completed = completed_chron_model_ndhB.vs["name"]
# # # W_completed_chron_model_ndhB = completed_chron_model_ndhB.get_adjacency()
# # # W_completed_chron_model_ndhB = pd.DataFrame(W_completed_chron_model_ndhB, columns=nodes_completed, index=nodes_completed)
# # # define the graphical representation for this completion
# # # fig_completed_chron_ndhB, ax = plt.subplots(figsize=(5, 5))
# # # ig.plot(completed_chron_model_ndhB, vertex_color = "red", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhB,
#                              adj_matrix_chron = W_chronology_ndhB,
#                              dataset_name='ndhB',
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhB,
#                              causality_summary=causality_summary, 
#                              causal_relations=causal_relations_ndhB,
#                              refutation_summary=refutation_summary_causal_rel_ndhB,
#                              param_alg=EM_param_alg, 
#                              scores_chron=[scores_chron_ref_ndhB, scores_chron_ndhB],
#                              falsification_summary=falsification_summary_chron_model_ndhB,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhB)
# # plt.close(fig_completed_chron_ndhB)



# # Causality for ndhD

# # Loading data
# # file paths
# EM_folder_name = Path('EM_ndhD_notears_2025_10_03_at_02_17_45')
# EM_data_path = dags_folder_EM/EM_folder_name/'EM_data_ndhD.csv'
# EM_bn_path = dags_folder_EM/EM_folder_name/'EM_bn_ndhD.pkl'
# EM_adj_path = dags_folder_EM/EM_folder_name/'AdjacencyMatrix_ndhD.csv'
# EM_param_path = dags_folder_EM/EM_folder_name/'config_param.txt'

# # load the original data (= only imputation was applied)
# original_data = ndhD

# # load the data obtained after the EM implementation
# ndhD_EM = pd.read_csv(EM_data_path, index_col=0)

# # load the Bayesian Network obtained after the EM implementation
# with open(EM_bn_path, "rb") as f:
#     bn_EM_ndhD = pickle.load(f)

# # load the adjacency matrix of the DAG obtained in this model
# W_EM_ndhD = pd.read_csv(EM_adj_path, index_col=0)
# # define the corresponding graph ('igraph' is convenient for large graphs!)
# G_EM_ndhD = ig.Graph.Adjacency(W_EM_ndhD)
# G_EM_ndhD.vs['label'] = G_EM_ndhD.vs["name"] = list(W_EM_ndhD.columns)

# # recall here the main parameters for the algorithm for this model
# param_file = {}
# with open(EM_param_path, "r") as file:
#     for line in file:
#         if ":" in line:
#             key, val = line.strip().split(":", 1)
#             param_file[key.strip()] = val.strip()
# EM_param_alg = [param_file.get("discovery_alg"), param_file.get("estimator_structure"), param_file.get("estimator_parameters")]


# # Causal Inference: interventions and causal effects between nodes of the given DAG
# causality_summary = utils.causal_inf(ndhD_EM, G_EM_ndhD, bn_EM_ndhD)

# # Chronology: the biggest is the causal effect, the earliest appear in the event chronology (respecting the topological order of the given DAG)
# chronology_ndhD = utils.chronology_events(causality_summary, G_EM_ndhD, original_data)
# chronology_dag_ndhD = chronology_ndhD[0]
# chronology_nodes = chronology_dag_ndhD.vs["name"]
# W_chronology_ndhD = chronology_dag_ndhD.get_adjacency() # obtain the adjacency matrix of the DAG representing the chronology
# W_chronology_ndhD = pd.DataFrame(W_chronology_ndhD, columns=chronology_nodes, index=chronology_nodes)
# chronology_bn_ndhD = chronology_ndhD[1]
# causal_rel_topo_level_ndhD = chronology_ndhD[2]
# causal_relations_ndhD = chronology_ndhD[3]
# refutation_summary_causal_rel_ndhD = chronology_ndhD[4]
# # define the graphical representation for the chronological DAG
# fig_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# ig.plot(chronology_dag_ndhD, vertex_color = "#0ABE8B", vertex_size = 75, edge_arrow_size = 45, edge_arrow_width=30, edge_width = 0.5, edge_curved=False, layout = "rt", vertex_label_size=10, vertex_label_dist=0.5)

# # Scoring of the chronology DAG: BIC and Log-likelihood scores (cf. 'pgmpy' documentation)
# filtered_original_data = original_data[list(chronology_bn_ndhD.nodes())] # filter the data to keep only the columns corresponding to nodes in the Bayesian Network
# BIC_score_chron_ndhD = structure_score(chronology_bn_ndhD, filtered_original_data, scoring_method="bic-d")
# log_likeli_score_chron_ndhD = log_likelihood_score(chronology_bn_ndhD, filtered_original_data)
# scores_chron_ndhD = [BIC_score_chron_ndhD, log_likeli_score_chron_ndhD]

# # Falsification of chronology (completed) DAG (cf. 'DoWhy' documentation)
# falsification_summary_chron_model_ndhD = utils.falsify_dag(original_data, chronology_dag_ndhD)


'''# # # SANITY CHECK: Completing the chronology DAG into a full DAG
# #     # in order to compare with the Ref. chronology DAG, use the original dataset! (just the imputed one)
# # # completion_chron_ndhD = utils.completion_chron_dag(original_data, chronology_dag_ndhD, EM_param_alg[0], notears_loss_type='logistic')
# # # completed_chron_model_ndhD = completion_chron_ndhD[0]
# # # nodes_completed = completed_chron_model_ndhD.vs["name"]
# # # W_completed_chron_model_ndhD = completed_chron_model_ndhD.get_adjacency()
# # # W_completed_chron_model_ndhD = pd.DataFrame(W_completed_chron_model_ndhD, columns=nodes_completed, index=nodes_completed)
# # # define the graphical representation for this completion
# # # fig_completed_chron_ndhD, ax = plt.subplots(figsize=(5, 5))
# # # ig.plot(completed_chron_model_ndhD, vertex_color = "red", vertex_size = 50, edge_arrow_size = 10, edge_width = 0.2, edge_curved=True, layout = "sugiyama", vertex_label_size=5, vertex_label_dist=0.5)'''


# # save the results
# utils.save_results_causality(figure_chron=fig_chron_ndhD, 
#                              adj_matrix_chron = W_chronology_ndhD,
#                              dataset_name='ndhD', 
#                              version_EM_impl=str(EM_folder_name),
#                              chron_bn=chronology_bn_ndhD,
#                              causality_summary=causality_summary,
#                              refutation_summary=refutation_summary_causal_rel_ndhD, 
#                              causal_relations=causal_relations_ndhD, 
#                              param_alg=EM_param_alg,
#                              scores_chron=[scores_chron_ref_ndhD, scores_chron_ndhD],
#                              falsification_summary=falsification_summary_chron_model_ndhD,
#                              figure_completed_chron=None, 
#                              adj_matrix_completed_chron=None)
# plt.close(fig_chron_ndhD)
# # plt.close(fig_completed_chron_ndhD)





# #############################################################################
# #----- Cross-analysis of the four models: identifying consistent arrows
# #############################################################################

# # create the folder to store the results of the cross-analysis
# folder_cross_analysis = f"cross-analysis"
# folder_results = Path(dags_folder_causality/folder_cross_analysis)
# # folder_results.mkdir(parents=True, exist_ok=False)


# # Causality for ndhB

# print("\n Consistent arrows for ndhB:")

# # load the 4 adjacency matrix obtained for ndhB
# W_chron_HC = pd.read_csv(dags_folder_causality/f'Chron_ndhB_hill_climbing_2025_10_06_at_16_39_27/AdjacencyMatrix_chron_ndhB.csv', sep=',', index_col=0)

# W_chron_PC = pd.read_csv(dags_folder_causality/f'Chron_ndhB_pc_2025_10_06_at_11_23_18/AdjacencyMatrix_chron_ndhB.csv', sep=',', index_col=0)

# W_chron_LiNGAM = pd.read_csv(dags_folder_causality/f'Chron_ndhB_lingam_2025_10_06_at_11_44_41/AdjacencyMatrix_chron_ndhB.csv', sep=',', index_col=0)

# W_chron_NOTEARS = pd.read_csv(dags_folder_causality/f'Chron_ndhB_notears_2025_10_06_at_12_36_40/AdjacencyMatrix_chron_ndhB.csv', sep=',', index_col=0)

# graph_models = [W_chron_HC, W_chron_PC, W_chron_LiNGAM, W_chron_NOTEARS]

# # rearrange the columns (and the rows) so that all matrices have the same order for the entries
# graph_models_sorted_nodes =[]
# for model in graph_models:
#     model = model.loc[gen_pos_ndhB, gen_pos_ndhB]
#     graph_models_sorted_nodes.append(model)


# # select the consistet arrows I: those appearing in all models
# intersection_arrows = graph_models_sorted_nodes[0] & graph_models_sorted_nodes[1] & graph_models_sorted_nodes[2] & graph_models_sorted_nodes[3]

# consistent_arrows_all = intersection_arrows[intersection_arrows == 1].stack().index.tolist()
# print("\n The following arrows appear in all models:", consistent_arrows_all)

# # select the consistet arrows II: those appearing in at least 3 models
# models = {"HC": graph_models_sorted_nodes[0], 
#           "PC": graph_models_sorted_nodes[1],
#           "LiNGAM": graph_models_sorted_nodes[2],
#           "NOTEARS": graph_models_sorted_nodes[3]}

# counting_arrows = graph_models_sorted_nodes[0] + graph_models_sorted_nodes[1] + graph_models_sorted_nodes[2] + graph_models_sorted_nodes[3]
# intersection_arrows_at_three = (counting_arrows >= 3)

# consistent_arrows_three = []
# for (idx, col) in intersection_arrows_at_three.stack()[intersection_arrows_at_three.stack()].index:
#     appearing = [name for name, df in models.items() if df.at[idx, col] == 1]
#     consistent_arrows_three.append({'Source': idx,
#                                     'Target': col,
#                                     'Model': appearing
#                                     })

# consistent_arrows_at3_summary = pd.DataFrame(consistent_arrows_three)
# print("\n The following arrows appear in at least 3 models:\n",consistent_arrows_at3_summary)
# consistent_arrows_at3_summary.to_csv(folder_results/f'consistent_arrows_at3_ndhB.csv')


# # select the consistet arrows III: those NO DIRECTED appearing in at least 3 models

# # convert matrices to non-directed (1 if (i,j) or (j,i) == 1)
# models_sym = {name: ((df + df.T) > 0).astype(int) for name, df in models.items()}
# counting_arrows = sum(models_sym.values())
# intersection_arrows_at_three = (counting_arrows >= 3)

# consistent_arrows_three = []
# seen = set()
# for (idx, col) in intersection_arrows_at_three.stack()[intersection_arrows_at_three.stack()].index:
#     if idx == col:
#         continue  # ignore auto-cycles

#     pair = tuple(sorted((idx, col))) # couple as non-directed
#     if pair in seen:
#         continue # check whether the pair has been added with a different order

#     appearing = [name for name, df in models.items() if df.at[idx, col] == 1 or df.at[col, idx] == 1]

#     consistent_arrows_three.append({'Edge': (pair[0], pair[1]),
#                                     'Model': appearing
#                                     })
#     seen.add(pair)

# consistent_arrows_at3_summary = pd.DataFrame(consistent_arrows_three)
# print("\n The following non-directed arrows appear in at least 3 models:\n",consistent_arrows_at3_summary)
# consistent_arrows_at3_summary.to_csv(folder_results/f'consistent_edges_at3_ndhB.csv')



# # Causality for ndhD

# print("\n Consistent arrows for ndhD:")

# # load the 4 adjacency matrix obtained for ndhD
# W_chron_HC = pd.read_csv(dags_folder_causality/f'Chron_ndhD_hill_climbing_2025_09_10_at_20_46_15/AdjacencyMatrix_chron_ndhD.csv', sep=',', index_col=0)

# W_chron_PC = pd.read_csv(dags_folder_causality/f'Chron_ndhD_pc_2025_09_10_at_23_30_21/AdjacencyMatrix_chron_ndhD.csv', sep=',', index_col=0)

# W_chron_LiNGAM = pd.read_csv(dags_folder_causality/f'Chron_ndhD_lingam_2025_09_10_at_23_47_21/AdjacencyMatrix_chron_ndhD.csv', sep=',', index_col=0)

# W_chron_NOTEARS = pd.read_csv(dags_folder_causality/f'Chron_ndhD_notears_2025_09_11_at_00_05_15/AdjacencyMatrix_chron_ndhD.csv', sep=',', index_col=0)

# graph_models = [W_chron_HC, W_chron_PC, W_chron_LiNGAM, W_chron_NOTEARS]

# # rearrange the columns (and the rows) so that all matrices have the same order for the entries
# graph_models_sorted_nodes =[]
# for model in graph_models:
#     model = model.loc[gen_pos_ndhD, gen_pos_ndhD]
#     graph_models_sorted_nodes.append(model)


# # select the consistet arrows I: those appearing in all models

# intersection_arrows = graph_models_sorted_nodes[0] & graph_models_sorted_nodes[1] & graph_models_sorted_nodes[2] & graph_models_sorted_nodes[3]

# consistent_arrows_all = intersection_arrows[intersection_arrows == 1].stack().index.tolist()
# print("\n The following arrows appear in all models:", consistent_arrows_all)


# # select the consistet arrows II: those appearing in at least 3 models

# models = {"HC": graph_models_sorted_nodes[0], 
#           "PC": graph_models_sorted_nodes[1],
#           "LiNGAM": graph_models_sorted_nodes[2],
#           "NOTEARS": graph_models_sorted_nodes[3]}

# counting_arrows = graph_models_sorted_nodes[0] + graph_models_sorted_nodes[1] + graph_models_sorted_nodes[2] + graph_models_sorted_nodes[3]
# intersection_arrows_at_three = (counting_arrows >= 3)

# consistent_arrows_three = []
# for (idx, col) in intersection_arrows_at_three.stack()[intersection_arrows_at_three.stack()].index:
#     appearing = [name for name, df in models.items() if df.at[idx, col] == 1]
#     consistent_arrows_three.append({'Source': idx,
#                                     'Target': col,
#                                     'Model': appearing
#                                     })

# consistent_arrows_at3_summary = pd.DataFrame(consistent_arrows_three)
# print("\n The following arrows appear in at least 3 models:\n",consistent_arrows_at3_summary)
# # consistent_arrows_at3_summary.to_csv(folder_results/f'consistent_arrows_at3_ndhD.csv')


# # select the consistet arrows III: those NO DIRECTED appearing in at least 3 models

# # convert matrices to non-directed (1 if (i,j) or (j,i) == 1)
# models_sym = {name: ((df + df.T) > 0).astype(int) for name, df in models.items()}
# counting_arrows = sum(models_sym.values())
# intersection_arrows_at_three = (counting_arrows >= 3)

# consistent_arrows_three = []
# seen = set()
# for (idx, col) in intersection_arrows_at_three.stack()[intersection_arrows_at_three.stack()].index:
#     if idx == col:
#         continue  # ignore auto-cycles

#     pair = tuple(sorted((idx, col))) # couple as non-directed
#     if pair in seen:
#         continue # check whether the pair has been added with a different order

#     appearing = [name for name, df in models.items() if df.at[idx, col] == 1 or df.at[col, idx] == 1]

#     consistent_arrows_three.append({'Edge': (pair[0], pair[1]),
#                                     'Model': appearing
#                                     })
#     seen.add(pair)

# consistent_arrows_at3_summary = pd.DataFrame(consistent_arrows_three)
# print("\n The following non-directed arrows appear in at least 3 models:\n",consistent_arrows_at3_summary)
# # consistent_arrows_at3_summary.to_csv(folder_results/f'consistent_edges_at3_ndhD.csv')