#####################################
#----- Preamble
#####################################

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
ig.config['plotting.backend'] = 'matplotlib'

import math

from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

from sklearn.model_selection import KFold
from sklearn.utils import resample

import random
from collections import Counter
from collections import defaultdict
from itertools import zip_longest

from pathlib import Path
import datetime
import pickle

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import PC
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import CausalInference

from causallearn.search.FCMBased import lingam

from notears import linear

import dowhy
from dowhy import CausalModel
import dowhy.causal_estimators.linear_regression_estimator
from dowhy.gcm.falsify import falsify_graph





# #####################################
# #----- Folder paths
# #####################################

data_folder = Path('Data')
visu_folder = Path('Results/visu')
charts_folder = Path('Results/charts')
graphs_folder = Path('Results/graphs')
dags_folder_notears = Path ('Results/graphs/dags_notears')
dags_folder_EM = Path ('Results/graphs/dags_EM')
dags_folder_causality = Path ('Results/graphs/dags_causality')





# #####################################
# #----- Studying the NA values
# #####################################

# Counting how NA are distributed within the data
def count_NA(data):
    data_NA = data.fillna('NA')

    count_only_NA = 0
    rows_only_NA = []

    count_mix_NA = 0
    rows_mix_NA = []
    
    count_no_NA = 0
    rows_no_NA = []

    num_row = data.shape[0]

    for i in range(num_row):
        var_NA = data_NA.iloc[i].eq("NA")
        if not var_NA.any():
            # print('In row', i, 'there is no NA value')
            count_no_NA += 1
            rows_no_NA.append(i)
            continue

        var_NA = data_NA.iloc[i].eq("NA").idxmax()
        col_var_NA = data_NA.columns.get_loc(var_NA)

        if all(data_NA.iloc[i][col_var_NA + 1:] == "NA"):
            # print('In row ', i, ' there are no reads after the first NA:\n')
            # print(ndhB_NA.iloc[i, :])
            count_only_NA += 1
            rows_only_NA.append(i)
        else:
            # print('In row ', i, ' there are non-trivial reads after the first NA:\n')
            # print(ndhB_NA.iloc[i, :])
            count_mix_NA += 1
            rows_mix_NA.append(i)
    return count_no_NA, count_only_NA, count_mix_NA


# Counting blocks with values different from NA
def count_NA_blocks(data):
    data_NA = data.fillna('NA')

    init_read_var=0
    final_read_var=0
    col_init=[]
    col_fin=[]
    col_final_read_block=[]
    col_blocks=[]
    col_isolated=[]

    num_row = data_NA.shape[0]

    for i in range(num_row):
        row = data_NA.iloc[i]
        read_variables = row[row.ne('NA')]

        if not read_variables.empty:
            init_read_var = data_NA.columns.get_loc(read_variables.index[0])
            final_read_var = data_NA.columns.get_loc(read_variables.index[-1])
            col_init.append(init_read_var)
            col_fin.append(final_read_var)
        else:
            # print('\n There are no values different from NA in row ', i)
            col_init.append(0)
            col_fin.append(0)

        pos_read_variables=[]

        for pos in range(len(read_variables.index)): 
            pos_read_variables.append(data_NA.columns.get_loc(read_variables.index[pos]))
        
        pos_read_variables = pd.Series(pos_read_variables)
        differences = pos_read_variables.diff()
        blocks = (differences != 1).cumsum()
        consec_blocks = pos_read_variables.groupby(blocks).apply(list)
        col_blocks.append(len(consec_blocks))
        
        if len(consec_blocks)>1:
            col_final_read_block.append(consec_blocks.iloc[0][-1])
        elif len(consec_blocks)<2:
            col_final_read_block.append(final_read_var)
        
        num_isolated_reads = 0
        for block in consec_blocks:
            if len(block) == 1 and len(consec_blocks)>1:
                num_isolated_reads += 1
        col_isolated.append(num_isolated_reads)
    
    return col_init, col_final_read_block, col_fin, col_blocks, col_isolated





##########################################
#----- Visualising the data sets
##########################################

# the visu function
def visu_data(data, color_code, color_imputed=None, pos_NA=None, imputed=False):
    # create a figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_aspect('auto')

    # create an empty (white) table within the figure created above
    data_visu = ax.table(cellText = [['' for _ in range(data.shape[1])] for _ in range(data.shape[0])], 
                    colLabels = None, 
                    cellLoc = 'center',
                    loc = 'center',
                    colColours = ['white']*data.shape[1]
                    )
    for (i, j), cell in data_visu.get_celld().items():
        cell.set_linewidth(0)  # remove lines within the table
        cell.set_edgecolor('none') # remove the edge of the table

    data_visu.scale(10, 0.2)

    # Color each cell following 'color_code' and/or 'color_imputed'
    if imputed == False:
        for (i, j), word in np.ndenumerate(data.values):
            color = color_code.get(word, 'white') # default = 'white' (in case word is not in the dictionary 'color_code')
            data_visu[(i+1, j)].set_facecolor(color)
    else:
        for (i, j), word in np.ndenumerate(data.values):
            if (i, j) in zip(pos_NA[0], pos_NA[1]):
                color = color_imputed.get(word, 'white') # default = 'white' (in case word is not in the dictionary 'color_code')
            else:
                color = color_code.get(word, 'white') # default = 'white' (in case word is not in the dictionary 'color_code')
            
            data_visu[(i+1, j)].set_facecolor(color)

    # Save the table as an image
    if imputed == False:
        plt.savefig(visu_folder/f"data_visu.png", bbox_inches='tight', dpi=50)
    else:
        plt.savefig(visu_folder/f"data_visu_imputed.png", bbox_inches='tight', dpi=50)
    plt.close(fig)


# Ordering the rows: we re-arrange the lines of the data set in such a way to put together all NA according to their position within the table. The goal is to produce a more visual image to identify better the distribution of the NA values. In our case, we will get a matrix forming stairs by blocks.

# Arrange rows according to the pressence of NAs at the end (or at the begging) of the rows. (This is equivalent to look at the presence of 'ed' values from right to left (or from left to right).)
def reorder_NA_sides(data, val, side):
    NA_side = pd.DataFrame()
    data_sorted = data.copy()
    cols = data.columns.tolist()  # list of columns

    for j in range(len(cols), 0, -1): # reverse order in the loop
        if side == 'right':
            cols_subset = cols[:j]
        elif side == 'left':
            cols_subset = cols [-j:]
        else:
            raise ValueError("The value of 'side' must be 'right' or 'left'")
        
        sorting_condition = data_sorted[cols_subset].isin(val).all(axis=1) # look at all rows (--> 'axis=1') such that all columns in 'cols_subset' contains the value 'val'
        NA_side = pd.concat([NA_side, data_sorted[sorting_condition]], ignore_index=True)
        data_sorted = data_sorted[~sorting_condition] # update the data set by removing the selected rows
    
    data_without_NA_side = data_sorted

    return NA_side, data_without_NA_side


# Arrange rows according to the presence of NAs both at the beginning and at the end of the rows.
def reorder_NA_rl(data):
    NA_left_right = pd.DataFrame()
    data_sorted = data.copy()
    cols = data.columns.tolist()  # list of columns
    
    for j_left in range(0, len(cols)-1): # first, select the rows with 'n' on the left increasingly
        NA_left = pd.DataFrame()
        sorting_condition_left = (data_sorted[cols[j_left]] == 'NA') & (data_sorted[cols[j_left+1]] != 'NA')
        NA_left = pd.concat([NA_left, data_sorted[sorting_condition_left]], ignore_index=True)
        data_sorted = data_sorted[~sorting_condition_left] # update the data set by removing the selected rows
    
        for j_right in range(len(cols)-1, 1, -1): # second, select the rows with 'n' on the right increasingly
            sorting_condition_right = (NA_left[cols[j_right]] == 'NA') & (NA_left[cols[j_right-1]] != 'NA')
            NA_left_right = pd.concat([NA_left_right, NA_left[sorting_condition_right]], ignore_index=True)
            NA_left = NA_left[~sorting_condition_right] # update the data set by removing the selected rows

    return NA_left_right, data_sorted


# Coding the data sets
code_global = {'NA' : 'NA', 'True' : 'ed', 'False' : 'ed', 'Err' : 'ed'} # encode each read (i.e. 'True', 'False', 'Err') as something edited 'ed' for a global visualisation
color_global = {'NA' : 'beige', 'ed' : 'khaki'}

code_binary = {'NA' : 'NA', 'True' : 1, 'False' : 0, 'Err' : 0} # encode each read (i.e. 'True', 'False', 'Err') as a binary entry for a detailed visualisation
color_binary = {'NA' : "#CBD8DE", 1 : "#1387B9", 0 : "#083787"}

color_imputed = {1 : "#1387B957", 0 : "#08378766"}

color_freq_charts_before_imp = ['#083787', '#1387B9']
color_freq_charts_after_imp = ['#083787', '#1387B9']





#######################################
#----- Dependence tests
#######################################

def dep_tests_events(data, events, test, alpha=0.05, imputed=False):
    alpha = alpha

    contingency_pairs={}
    indep_events=[]
    dep_events=[]
    p_values_chi2=[]
    p_values_fisher=[]

    if imputed == False:
        data_cat = data.replace(np.nan, 'NA', inplace=False)
    else:
        data_cat = data
    
    for j in range(len(events)):
        for k in range(j+1, len(events)):
            col1 = events[j]
            col2 = events[k]
            contingency_pairs[(col1, col2)] = pd.crosstab(data_cat[col1], data_cat[col2], rownames=[col1], colnames=[col2])

    for pair, cont_matrix in contingency_pairs.items():
        if imputed == False:
            cont_matrix = (cont_matrix.drop('NA', axis=0)).drop('NA', axis=1)
        # print(f"\n Contingency matrix between {pair[0]} and {pair[1]}:")
        # print(cont_matrix)
        
        num_phi = cont_matrix.iloc[0,0] * cont_matrix.iloc[1,1] - cont_matrix.iloc[0,1] * cont_matrix.iloc[1,0]
        den_phi = np.sqrt(cont_matrix.iloc[0].sum() * cont_matrix.iloc[0].sum() * cont_matrix.iloc[:, 0].sum() * cont_matrix.iloc[:, 1].sum())
        phi = num_phi / den_phi

        num_odds_ratio = cont_matrix.iloc[0,0] * cont_matrix.iloc[1,1]
        den_odds_ratio = cont_matrix.iloc[0,1] * cont_matrix.iloc[1,0]
        odds_ratio = num_odds_ratio / den_odds_ratio
        
        if test == 'chi2':
            chi2, p_chi2, dof, expected = chi2_contingency(cont_matrix)
            p_values_chi2.append(p_chi2)
            # print(f"\n Chi2 p-value: {p_chi2}")

        elif test == 'fisher':
            odds_r, p_Fisher = fisher_exact(cont_matrix)
            p_values_fisher.append(p_Fisher)
            # print(f"\n Fisher p-value: {p_Fisher}")

    # Adjusted p-values using the Benjamini-Hochberg method
    if test == 'chi2':
        p_values_chi2 = np.array(p_values_chi2)
        decisions_chi2_adj, p_values_chi2_adj, _, _ = multipletests(p_values_chi2, alpha=alpha, method='fdr_bh')
    elif test == 'fisher':
        p_values_fisher = np.array(p_values_fisher)
        decisions_fisher_adj, p_values_fisher_adj, _, _ = multipletests(p_values_fisher, alpha=alpha, method='fdr_bh')

    for i, (pair, cont_matrix) in enumerate(contingency_pairs.items()):
        if test == 'chi2':
            if decisions_chi2_adj[i] == True: # equivalent to p_values_chi2_adj[i] < alpha
                # print(f"\n The events {pair[0]} and {pair[1]} are Chi2 NOT independent")
                pair = (pair[0], pair[1], p_values_chi2_adj[i], round(phi, 3), round(odds_ratio, 3))
                dep_events.append(pair)
            else:
                # print(f"\n The events {pair[0]} and {pair[1]} are Chi2 independent")
                pair = (pair[0], pair[1], p_values_chi2_adj[i], round(phi, 3), round(odds_ratio, 3))
                indep_events.append(pair)
        elif test == 'fisher':
            if decisions_fisher_adj[i] == True: # equivalent to p_values_fisher_adj[i] < alpha
                # print(f"\n The events {pair[0]} and {pair[1]} are Fisher NOT independent")
                pair = (pair[0], pair[1], p_values_fisher_adj[i], round(phi, 3), round(odds_ratio, 3))
                dep_events.append(pair)
            else:
                # print(f"\n The events {pair[0]} and {pair[1]} are Fisher independent")
                pair = (pair[0], pair[1], p_values_fisher_adj[i], round(phi, 3), round(odds_ratio, 3))
                indep_events.append(pair)

    # if test == 'chi2':
    #     print(f"\n There are {len(dep_events)} Chi2 dependent events")
    #     print(f"\n There are {len(indep_events)} Chi2 independent events")
    # elif test == 'fisher':
    #     print(f"\n There are {len(dep_events)} Fisher dependent events")
    #     print(f"\n There are {len(indep_events)} Fisher independent events")
    
    return dep_events, indep_events





#######################################
#----- Saving results from NOTEARS
#######################################

def save_results_notears(figure, adj_matrix, vertex, dataset_name, param_alg):
    # Generate a unique name based on date and time of implementation of the algorithm
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_at_%H_%M_%S")

    # Define the name of the folder where stocking the results
    folder_impl = f"NOTEARS_{dataset_name}_{time_stamp}"

    # Create a folder (if it does not exist) to store all the results of the implementations
    folder_results = Path(dags_folder_notears/folder_impl)
    folder_results.mkdir(parents=True, exist_ok=False) # 'parents=True' allows to create subfolders and 'exist_ok=False' gives an error if the target directory already exists ('FileExistsError')

    # Save the (main) parameters of the implemented configuration
    config_impl = {
        "dataset_name" : dataset_name,
        "lambda" : param_alg[0],
        "loss_type" : param_alg[1],
        "model_selector" : param_alg[2]
    }
    with open(folder_results/f"config_param.txt", "w") as file:
        for param, value in config_impl.items():
            file.write(f"{param}: {value}\n")

    # Save the DAG as a png image
    figure.savefig(folder_results/f"DAG_plot_{dataset_name}.png", format="PNG", dpi=500)

    # Save the Adjacency matrix as a csv file
    W = pd.DataFrame(adj_matrix, columns=vertex, index=vertex)
    W.to_csv(folder_results/f'AdjacencyMatrix_{dataset_name}.csv')





#######################################
#----- Saving results from EM
#######################################

def save_results_EM(figure, adj_matrix, vertex, dataset_name, EM_bn, EM_data, EM_syn, param_alg):
    # Generate a unique name based on date and time of implementation of the algorithm
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_at_%H_%M_%S")

    # Define the name of the folder where stocking the results
    folder_impl = f"EM_{dataset_name}_{param_alg[0]}_{time_stamp}"

    # Create a folder (if it does not exist) to store all the results of the implementations
    folder_results = Path(dags_folder_EM/folder_impl)
    folder_results.mkdir(parents=True, exist_ok=False) # 'parents=True' allows to create subfolders and 'exist_ok=False' gives an error if the target directory already exists ('FileExistsError')

    # Save the (main) parameters of the implemented configuration as a txt file
    config_impl = {
        "dataset_name" : dataset_name,
        "discovery_alg" : param_alg[0],
        "estimator_structure" : param_alg[1],
        "estimator_parameters" : param_alg[2]
    }
    with open(folder_results/f"config_param.txt", "w") as file:
        for param, value in config_impl.items():
            file.write(f"{param}: {value}\n")

    # Save the DAG as a png image
    figure.savefig(folder_results/f"DAG_plot_{dataset_name}.png", format="PNG", dpi=500)

    # Save the Adjacency matrix as a csv file
    W = pd.DataFrame(adj_matrix, columns=vertex, index=vertex)
    W.to_csv(folder_results/f'AdjacencyMatrix_{dataset_name}.csv')

    # Save the EM bayesian network as a pickle object for further use
    with open(folder_results/f"EM_bn_{dataset_name}.pkl", "wb") as f:
        pickle.dump(EM_bn, f)

    # Save the EM modified dataset as a csv file
    EM_data.to_csv(folder_results/f'EM_data_{dataset_name}.csv')

    # Save the synthetic data generated from the EM model as a csv file
    EM_syn.to_csv(folder_results/f'EM_synthetic_{dataset_name}.csv')





##############################################
#----- Saving results from Causal Inference
##############################################

def save_results_causality(figure_chron, adj_matrix_chron, dataset_name, version_EM_impl, chron_bn, causality_summary, refutation_summary, causal_relations, param_alg, scores_chron, falsification_summary=None, figure_completed_chron=None, adj_matrix_completed_chron=None):
    # Generate a unique name based on date and time of implementation of the algorithm
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_at_%H_%M_%S")

    # Define the name of the folder where stocking the results
    folder_impl = f"Chron_{dataset_name}_{param_alg[0]}_{time_stamp}"

    # Create a folder (if it does not exist) to store all the results of the implementations
    folder_results = Path(dags_folder_causality/folder_impl)
    folder_results.mkdir(parents=True, exist_ok=False) # 'parents=True' allows to create subfolders and 'exist_ok=False' gives an error if the target directory already exists ('FileExistsError')

    # Save the (main) parameters of the implemented configuration as a txt file
    config_impl = {
        "dataset_name" : dataset_name,
        "version_EM_implementation": version_EM_impl,
        "discovery_alg" : param_alg[0],
        "estimator_structure" : param_alg[1],
        "estimator_parameters" : param_alg[2]
    }
    with open(folder_results/f"config_param.txt", "w") as file:
        for param, value in config_impl.items():
            file.write(f"{param}: {value}\n")

    # Save the chronology DAG as a png image
    figure_chron.savefig(folder_results/f"chron_DAG_plot_{dataset_name}.png", format="PNG", dpi=500)

    # Save the Adjacency matrix of the chronology DAG as a csv file
    adj_matrix_chron.to_csv(folder_results/f'AdjacencyMatrix_chron_{dataset_name}.csv')

    # Save the chronology bayesian network as a pickle object for further use
    with open(folder_results/f"chron_bn_{dataset_name}.pkl", "wb") as f:
        pickle.dump(chron_bn, f)

    # Save the causal inference summary as a csv file
    causality_summary.to_csv(folder_results/f'causality_summary_{dataset_name}.csv')

    # Save the causal relations (simplified causality summary) as a csv file
    causal_relations.to_csv(folder_results/f'causal_relations_{dataset_name}.csv')

    # Save the refutation tests summary for the causal relations (simplified causality summary) as a csv file
    refutation_summary.to_csv(folder_results/f'refutation_summary_causal_rel_{dataset_name}.csv')

    # Save the scoring of the chronology DAGs as a txt file
    if scores_chron[0][0] > scores_chron[1][0]:
        best_bic = f'Ref. Chronology_{dataset_name}_Fig6'
    else:
        best_bic = f'Causal Chronology_{dataset_name}'
    if scores_chron[0][1] > scores_chron[1][1]:
        best_log_likeli = f'Ref. Chronology_{dataset_name}_Fig6'
    else:
        best_log_likeli = f'Causal Chronology_{dataset_name}'
    scoring_summary = {
        "dataset_name" : dataset_name,
        "BIC_scores": {
            'BIC_ref_chron': scores_chron[0][0],
            'BIC_causal_chron': scores_chron[1][0],
            'Difference': abs(scores_chron[0][0] - scores_chron[1][0]),
            'Best_BIC': best_bic
        },
        "Log-likelihood_scores": {
            'Log-likelihood_ref_chron': scores_chron[0][1],
            'Log-likelihood_causal_chron': scores_chron[1][1],
            'Difference': abs(scores_chron[0][1] - scores_chron[1][1]),
            'Best_Log-likelihood': best_log_likeli
        }
    }
    with open(folder_results/f"scoring_summary_chron_{dataset_name}.txt", "w") as file:
        for score, value in scoring_summary.items():
            if isinstance(value, dict):  
                file.write(f"{score}:\n")
                for sub_key, sub_value in value.items():
                    file.write(f"\t{sub_key}: [{sub_value}]\n")  # indentations for the elements within the sub-dictionary
            else:
                file.write(f"{score}: {value}\n")

    # Save the falsification summary as a txt file
    if falsification_summary == None:
        fals_summary = {
        "Falsification Summary" : 'None',
        "DAG is Falsifiable" : 'None',
        "DAG is Falsified" : 'None'
        }
        with open(folder_results/f"falsification_summary_chron_{dataset_name}.txt", "w") as file:
            for fals, value in fals_summary.items():
                file.write(f"{fals}: {value}\n")
    else:
        fals_summary = {
        "Falsification Summary" : falsification_summary[0],
        "DAG is Falsifiable" : falsification_summary[1],
        "DAG is Falsified" : falsification_summary[2]
        }
        with open(folder_results/f"falsification_summary_chron_{dataset_name}.txt", "w") as file:
            for fals, value in fals_summary.items():
                file.write(f"{fals}: {value}\n")

    if figure_completed_chron != None and adj_matrix_completed_chron != None:
        # Save the completed chronology DAG as a png image
        figure_completed_chron.savefig(folder_results/f"completed_chron_DAG_plot_{dataset_name}.png", format="PNG", dpi=500)

        # Save the Adjacency matrix of the completed chronology DAG as a csv file
        adj_matrix_completed_chron.to_csv(folder_results/f'AdjacencyMatrix_completed_chron_{dataset_name}.csv')






#######################################
#----- Some tools for studying DAGs
#######################################

# Counting the degrees of the edges
def deg_edges(adj_matrix, vertex):
    num_arrows_out = {}
    pointed_nodes_out = {}
    for i in range(adj_matrix.shape[0]):
        num_out = 0
        list_target_nodes = []
        for j in range(adj_matrix.shape[1]):
            if adj_matrix.iloc[i, j] != 0:
                num_out += 1
                list_target_nodes.append(vertex[j])
        pointed_nodes_out[vertex[i]] = list_target_nodes
        num_arrows_out[vertex[i]] = num_out
   
    num_arrows_in = {}
    pointed_nodes_in = {}
    for j in range(adj_matrix.shape[1]):
        num_in = 0
        list_source_nodes = []
        for i in range(adj_matrix.shape[0]):
            if adj_matrix.iloc[i, j] != 0:
                num_in += 1
                list_source_nodes.append(vertex[i])
        pointed_nodes_in[vertex[j]] = list_source_nodes
        num_arrows_in[vertex[j]] = num_in

    set_events = set(vertex)
    source_nodes = {node: (num_arrows_out[node], num_arrows_in[node]) 
                        for node in set_events 
                        if num_arrows_out[node] != 0 and num_arrows_in[node] == 0}
    source_nodes_MaxMin = dict(sorted(source_nodes.items(), key=lambda item: item[1], reverse=True))
    target_nodes = {node: (num_arrows_out[node], num_arrows_in[node]) 
                        for node in set_events 
                        if num_arrows_out[node] == 0 and num_arrows_in[node] != 0}
    target_nodes_MaxMin = dict(sorted(target_nodes.items(), key=lambda item: item[1], reverse=True))
    mixed_nodes = {node: (num_arrows_out[node], num_arrows_in[node]) 
                        for node in set_events 
                        if num_arrows_out[node] != 0 and num_arrows_in[node] != 0}
    mixed_nodes_MaxMin = dict(sorted(mixed_nodes.items(), key=lambda item: (item[1][0], -item[1][1]), reverse=True))

    return [pointed_nodes_out, num_arrows_out], [pointed_nodes_in, num_arrows_in], source_nodes_MaxMin, target_nodes_MaxMin, mixed_nodes_MaxMin

# Computing the Structural Hamming Distance (SHD)
def shd(A, B):
    # Check whether the matrices have the same
    assert A.shape == B.shape, "Matrices must have the same dimension"
    assert A.shape[0] == B.shape[1], "Matrices must be square"

    n = A.shape[0]
    shd_count = 0

    for i in range(n):
        for j in range(i+1, n): # we avoid counting the inversions twice
            # case edge i->j
            a_ij = A[i, j]
            b_ij = B[i, j]
            # case edge j->i
            a_ji = A[j, i]
            b_ji = B[j, i]

            # If the arrow appears in both DAGs with the same directioin, no change
            if a_ij == b_ij and a_ji == b_ji:
                continue
            # If the arrow is reversed between the DAGs
            if a_ij == 1 and b_ji == 1 and a_ji == 0 and b_ij == 0:
                shd_count += 1  # one inversion
            elif a_ji == 1 and b_ij == 1 and a_ij == 0 and b_ji == 0:
                shd_count += 1  # one inversoin
            else:
                # count the direct differences
                shd_count += abs(a_ij - b_ij) + abs(a_ji - b_ji)

    return shd_count





###############################
#----- Charts
###############################

# Distribution of 0s and 1s per column in the data set
def distr_values_var(data, name_cols=None, imputed=False):
    # give a particular name to the columns (variables) of the dataset instead of the default column names of 'data'
    if name_cols != None:
        data.columns = name_cols
    # We plot each chart in a single figure. We fix 5 charts per column --> how many rows are necessary to plot all the charts?
    num_charts_col = 5 
    num_rows = (len(data.columns) // num_charts_col) + (len(data.columns) % num_charts_col > 0)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_charts_col, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    if imputed == False:
        bar_colors = color_freq_charts_before_imp
    else:
        bar_colors = color_freq_charts_after_imp
    
    for i, col in enumerate(data.columns):
        counts = data[col].value_counts().sort_index()  # count the number of 0s and 1s and guarantee that the order of these counts is num_0s --> num_1s
        
        axes[i].bar(counts.index, counts.values, color=bar_colors, width=0.8)  # create a bar chart
        axes[i].set_title(f'{col}')
        # axes[i].set_xlabel('Value')
        # axes[i].set_ylabel('Frequency')
        axes[i].set_xticks([0, 1]) # establish only the values 0 and 1 in the X-axis
        axes[i].set_xticklabels([0, 1])  # establish the values 0 and 1 as the labels underneath the bars

    # Hide charts that are empty (in case they exist)
    for k in range(len(data.columns), len(axes)):
        axes[k].axis('off')  # Desactiva los subgráficos que no se usan

    plt.tight_layout()
    if imputed == False:
        plt.savefig(charts_folder/f'distr_val_variables.png')
    else:
        plt.savefig(charts_folder/f'distr_val_variables_imputed.png')
    plt.close(fig)


# Loss curve for the sparsity parameter
def loss_values_lamb(X, loss_type, sample_lambda=100, name_chart='loss_curve_lambda'):
    # lambda_values = np.linspace(0, 1, sample_lambda)
    np.random.seed(42)
    lambda_values = np.random.uniform(low=0, high=1, size=sample_lambda)

    loss_values = []
    for lamb in lambda_values:
        W_lambda = linear.notears_linear(X, lamb, loss_type)
        loss_lambda = loss_notears(W_lambda, X, loss_type)
        loss_values.append((lamb, loss_lambda))

    # Graphical representattion 
    x, y = zip(*loss_values)
    plt.scatter(x, y, color="red", marker=".", label='Loss values')

    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('loss_value')

    plt.title('Loss curve for the sparsity parameter')
    # plt.legend()

    plt.savefig(charts_folder/f'{name_chart}.png')
    plt.close()


# Representation of the number of connexions in the DAG for each value of lambda: sparsity model
def connex_lamb(X, loss_type, sample_lambda=100, name_chart='sparsity_models'):
    # lambda_values = np.linspace(0, 1, sample_lambda)
    np.random.seed(42)
    lambda_values = np.random.uniform(low=0, high=1, size=sample_lambda)

    connex_values = []
    for lamb in lambda_values:
        W_lambda = linear.notears_linear(X, lamb, loss_type)
        num_arrows_lambda = np.count_nonzero(W_lambda)
        connex_values.append((lamb, num_arrows_lambda))
   
    # Graphical representattion
    x, y = zip(*connex_values)
    plt.scatter(x, y, color="#155c90", marker=".")

    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('No. arrows')

    plt.title(f'{name_chart}')

    plt.savefig(charts_folder/f'{name_chart}.png')
    plt.close()


# Representation of the number of connexions in the DAG for each value of lambda II: sparsity model by Stability Selection
def connex_lamb_StabSelec(X, loss_type, stab_freq=0.75, sample_lambda=100, num_iter=5, name_chart='sparsity_models_StabSelec'):
    # lambda_values = np.linspace(0, 1, sample_lambda)
    np.random.seed(42)
    lambda_values = np.random.uniform(low=0, high=1, size=sample_lambda)

    stab_connex_values = []
    for lamb in lambda_values:    
        arrows_lambda = []

        for iter in range(num_iter):
            X_resampled = resample(X, random_state=iter)
            W_lambda = linear.notears_linear(X_resampled, lamb, loss_type)

            rows_arrows_iter, cols_arrows_iter = np.where(W_lambda != 0)
            arrows_lambda_iter = list(zip(rows_arrows_iter, cols_arrows_iter))

            arrows_lambda.extend(arrows_lambda_iter)

        arrows_lambda_count = Counter(arrows_lambda)
        arrows_lambda_freq = {arrow : count/num_iter for arrow, count in arrows_lambda_count.items()}

        stability_lambda = {arrow : freq for arrow, freq in arrows_lambda_freq.items() if freq >= stab_freq}
        stable_arrows_lambda = list(stability_lambda.keys())
        num_stable_arrows_lambda = len(stable_arrows_lambda)
        stab_connex_values.append((lamb, num_stable_arrows_lambda))

    # Graphical representattion 
    x, y = zip(*stab_connex_values)
    plt.scatter(x, y, color="#155c90", marker=".", label= f'Number of stable arrows (>= {stab_freq})')

    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('No. stable arrows')

    plt.title(f'{name_chart}')
    # plt.legend()

    plt.savefig(charts_folder/f'{name_chart}.png')
    plt.close()





#######################################
#----- Technical tools: NOTEARS
#######################################

# Partition the dataset into K-folds in order to obtain 'train_subsets' and 'test_subsets' for each fold (e.g. for Cross-Validation)
def partition_KFolds(X, K=5):
    if isinstance(X, np.ndarray): # note that the folds will be created as DataFrames
        X = pd.DataFrame(X)
    
    kf = KFold(n_splits=K, shuffle=True, random_state=42) # object KFold from 'sklearn'

    splits = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] # partitions of the dataset: 'train_subset' and 'test_subset'

        splits.append((X_train, X_test)) 

    return splits


# Extract the loss function from the main function of NOTEARS (cf. Zheng et al. 2018)
def loss_notears(W, X, loss_type):
    M = X @ W
    if loss_type == 'l2':
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
    elif loss_type == 'logistic':
        loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
    elif loss_type == 'poisson':
        S = np.exp(M)
        loss = 1.0 / X.shape[0] * (S - X * M).sum()
    else:
        raise ValueError('unknown loss type')
    return loss


# Stability Selection for choosing the arrows in the DAG: for a given 'lambda'
def StabSelec_arrows(X, lamb, loss_type, stab_freq=0.75, num_iter=10):
    arrows_lambda_count = Counter()
    for iter in range(num_iter):
        X_resampled = resample(X, random_state=iter)
        W_lambda = linear.notears_linear(X_resampled, lamb, loss_type)

        rows_arrows_iter, cols_arrows_iter = np.where(W_lambda != 0)
        arrows_lambda_iter = list(zip(rows_arrows_iter, cols_arrows_iter))

        arrows_lambda_count.update(arrows_lambda_iter)

        # Early stopping: break if no new arrow can become stable and no currently stable arrow can loose its stability in future iterations
        if iter >= 1:
            actual_iter = iter + 1

            # Current observed frequency
            stability_iter = {arrow : count/actual_iter for arrow, count in arrows_lambda_count.items()}
            
            # Max possible frequency if it appears in the remaining iterations
            max_possible_freq = {arrow: (count + (num_iter - actual_iter))/num_iter for arrow, count in arrows_lambda_count.items()}
            # Min possible frequency if it does not appear in the remaining iterations
            min_possible_freq = {arrow: count/num_iter for arrow, count in arrows_lambda_count.items()}

            # Can any arrow become stable?
            can_become_stable = any(freq >= stab_freq and stability_iter.get(arrow, 0) < stab_freq for arrow, freq in max_possible_freq.items())
            # Can any currently stable arrow loose its stability?
            stable_now_could_become_unstable = any(stability_iter.get(arrow, 0) >= stab_freq and min_possible_freq.get(arrow, 0) < stab_freq for arrow in arrows_lambda_count)

            if not can_become_stable and not stable_now_could_become_unstable:
                break
    final_iters = iter + 1

    arrows_lambda_freq = {arrow : count/final_iters for arrow, count in arrows_lambda_count.items()}
    stability_lambda = {arrow : freq for arrow, freq in arrows_lambda_freq.items() if freq >= stab_freq}
    stable_arrows_lambda = list(stability_lambda.keys())

    d = X.shape[1]
    W_stab_lamb = np.zeros((d,d), dtype=int)
    for i, j in stable_arrows_lambda:
        W_stab_lamb[i,j] = 1

    return W_stab_lamb, stable_arrows_lambda

# Delete arrows that generate cycles to ensure DAG structure in 'StabSelec_notears_sparsity()' below
def enforce_acyclicity(stable_arrows, d):
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for i, j in stable_arrows:
        G.add_edge(i, j)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(i, j)
    return list(G.edges())


# Stability Selection for choosing the sparsity model (in this case we do not choose a specific parameter lambda!)
def StabSelec_notears_sparsity(X, loss_type, stab_freq=0.75, sample_lambda=100, interval_lambda=[0,1], num_iter=10):
    # lambda_values = np.linspace(0, 1, sample_lambda)
    np.random.seed(42)
    lambda_values = np.random.uniform(low=interval_lambda[0], high=interval_lambda[1], size=sample_lambda)
    d = X.shape[1]
    
    max_arrows = (d*(d-1))/2
    stable_arrows = []
    new_stab_arr_lamb = []
    num_non_trivial_lamb = 0
    for idx, lamb in enumerate(lambda_values):
        old_stab_arr_lamb = new_stab_arr_lamb

        W_lambda = linear.notears_linear(X, lamb, loss_type)
        row_arrow_lamb, col_arrow_lamb = np.where(W_lambda != 0)
        notears_arrows_lambda = list(zip(row_arrow_lamb, col_arrow_lamb))

        new_stab_arr_lamb = notears_arrows_lambda
        if set(old_stab_arr_lamb) == set (new_stab_arr_lamb):
            continue
        else:
            stable_arrows_lambda = StabSelec_arrows(X, lamb, loss_type, stab_freq, num_iter)[1]
            stable_arrows.extend(stable_arrows_lambda)
            if 0 < len(stable_arrows_lambda) < max_arrows: # count the number of 'lambdas' that give non-trivial stable arrows
                num_non_trivial_lamb += 1
            stab_freq_lamb = num_non_trivial_lamb/sample_lambda
            
            # Early stopping: break if no arrow can become stable in future iterations
            # Current number of copies
            counts = Counter(stable_arrows)
            # Remaining iterations
            num_remaining_lambs = sample_lambda - (idx + 1)

            # Max possible frequency if it appears in the remaining iterations
            can_still_be_stable = False
            for arrow, count in counts.items():    
                max_possible_freq = (count + num_remaining_lambs) / (num_non_trivial_lamb + num_remaining_lambs)
                if max_possible_freq >= stab_freq_lamb:
                    can_still_be_stable = True
                    break
            if not can_still_be_stable:
                print(f"Early stopping at iteration {idx}")
                break

    stab_freq_lamb = num_non_trivial_lamb/sample_lambda

    stable_arrows_count = Counter(stable_arrows)
    stable_arrows_freq = {arrow : count/num_non_trivial_lamb for arrow, count in stable_arrows_count.items()}
    stability = {arrow : freq for arrow, freq in stable_arrows_freq.items() if freq >= stab_freq_lamb}
    dag_stable_arrows = enforce_acyclicity(list(stability.keys()), d)

    W_stable = np.zeros((d,d), dtype=int)
    for pos in dag_stable_arrows:
        i, j = pos
        W_stable[i,j] = 1

    return W_stable, dag_stable_arrows


# Cross-Validation combined with Stabtility Selection for choosing the sparsity parameter
def CVStab_notears_lamb(X, loss_type, K_folds=5, stab_freq=0.75, sample_lambda=100, num_iter=5, name_chart='score_curve_lambda_CV_StabSelec'):
    # lambda_values = np.linspace(0, 1, sample_lambda)
    np.random.seed(42)
    lambda_values = np.random.uniform(low=0, high=1, size=sample_lambda)
    splits = partition_KFolds(X, K_folds)

    perform_values = []
    for lamb in lambda_values:
        perform_lambda = 0
        perform_fold = []

        for k, (X_train, X_test) in enumerate(splits):
            X_train=X_train.values
            X_test=X_test.values
            
            W_lambda_hat = linear.notears_linear(X_train, lamb, loss_type)
            num_arrows_lamb_hat = np.count_nonzero(W_lambda_hat)

            stable_arrows_test = StabSelec_notears_sparsity(X_test, loss_type, stab_freq=stab_freq, sample_lambda=sample_lambda, num_iter=num_iter)[1]
            num_stable_arrows_test = len(stable_arrows_test)
            if num_stable_arrows_test > 0:
                connex_StabSelec_test = num_stable_arrows_test
            else:
                connex_StabSelec_test = 0

            if max(num_arrows_lamb_hat, num_stable_arrows_test) > 0:
                score_connex_fold = np.abs(num_arrows_lamb_hat - connex_StabSelec_test)/max(num_arrows_lamb_hat, connex_StabSelec_test)
            else:
                score_connex_fold = 0
        
            perform_fold.append(score_connex_fold)

        perform_lambda = np.mean(perform_fold)
        perform_values.append((lamb, perform_lambda))
    
    values_points = np.array(perform_values)
    best_perform = np.argmin(values_points[:, 1]) # in our case 'best_perform = min_score_connex'
    lambda_best_perform = values_points[best_perform, 0]

    # Graphical representattion 
    x, y = zip(*perform_values)
    plt.scatter(x, y, color="#155c90", marker=".", label='Score-connexion values in CV-StabSelec')

    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('score_connexions_StabSelec')

    plt.title('Score-connexion curve for the sparsity parameter in CV-StabSelec')
    # plt.legend()

    plt.savefig(charts_folder/f'{name_chart}.png')
    plt.close()

    return lambda_best_perform


# Cross-Validation for choosing the sparsity parameter
def CV_notears_lamb(X, loss_type, K_folds=5, sample_lambda=100, name_chart='loss_curve_lambda_CV'):
    # lambda_values = np.linspace(0, 1, sample_lambda)
    np.random.seed(42)
    lambda_values = np.random.uniform(low=0, high=1, size=sample_lambda)
    splits = partition_KFolds(X, K_folds)

    perform_values = []
    for lamb in lambda_values:
        perform_lambda = 0
        perform_fold = []

        for k, (X_train, X_test) in enumerate(splits):
            X_train=X_train.values
            X_test=X_test.values
            
            W_lambda_hat = linear.notears_linear(X_train, lamb, loss_type)
            loss_evalu_test = loss_notears(W_lambda_hat, X_test, loss_type)
        
            perform_fold.append(loss_evalu_test)

        perform_lambda = np.mean(perform_fold)
        perform_values.append((lamb, perform_lambda))
    
    values_points = np.array(perform_values)
    best_perform = np.argmin(values_points[:, 1]) # in our case 'best_perform = min_loss'
    lambda_best_perform = values_points[best_perform, 0]
    
    # Graphical representattion 
    x, y = zip(*perform_values)
    plt.scatter(x, y, color="#155c90", marker=".", label='Loss values in CV')

    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('loss_value_CV')

    plt.title('Loss curve for the sparsity parameter in CV')
    # plt.legend()

    plt.savefig(charts_folder/f'{name_chart}.png')
    plt.close()

    return lambda_best_perform





###################################################################
#----- Technical tools: EM, imputation and Bayesian Network
###################################################################

# EM algorithm for DAG/Bayesian Network discovery and imputation
    # supported discovery algorithms: 'hill_climbing', 'pc', 'lingam', 'notears'
def EM_dag(data, pos_NA, discovery_alg, name_nodes=None, notears_loss_type=None, max_iter = 50, epsilon = 0.01):

    if name_nodes != None: # give a particular name to the vertex of the final DAG instead of the default column names of 'data'
        data.columns = name_nodes

    new_Z = {} # store the values of the NaN positions at iteration n
    previous_edges = [] # store the edges of the discovered DAG at iteration n-1
    for iter in range(max_iter):
        # E-step: computation of the Bayesian Network; DAG structure and probability distribution (parameters)

        if discovery_alg == 'hill_climbing':
            for col in data.columns:
                data[col] = data[col].astype('category')

            hc = HillClimbSearch(data) # class for heuristic Hill Climbing searches for DAGs
            dag_model = hc.estimate(scoring_method='bic-d') # locall hill-climbing to estimate the DAG structure (dependencies without parameters!) that has optimal score
            dag_model_edges = dag_model.edges()
            bn = DiscreteBayesianNetwork(dag_model_edges) # initialize a Discrete Bayesian Network, i.e. a DAG where we will incorporate a Conditional Probability Distribution
            dag_estimator = BayesianEstimator(bn, data) # initialize a Bayesian Estimator for computing the parameters of the model (CPDs)

            dag_cpds = [] # store the CPDs for each variable
            index_node = {} # store the index of the nodes appearing in the Bayesian Network
            for idx, col in enumerate(bn.nodes): # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
                cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
                bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
                dag_cpds.append(cpd_col)
                index_node[col] = idx

        elif discovery_alg == 'pc':
            for col in data.columns:
                data[col] = data[col].astype('category')

            pc = PC(data) # class for constraint-based estimation of DAGs through Peter-Clark algorithm
            dag_model = pc.estimate(variant='orig', ci_test='chi_square', return_type='dag') # estimate the DAG structure through statistical independence tests. Moreover, we return a fully directed structure if it is possible to orient all the edges
            dag_model_edges = dag_model.edges()
            bn = DiscreteBayesianNetwork(dag_model_edges) # initialize a Discrete Bayesian Network, i.e. a DAG where we will incorporate a Conditional Probability Distribution
            dag_estimator = BayesianEstimator(bn, data) # initialize a Bayesian Estimator for computing the parameters of the model (CPDs)

            dag_cpds = [] # store the CPDs for each variable
            index_node = {} # store the index of the nodes appearing in the Bayesian Network
            for idx, col in enumerate(bn.nodes): # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
                cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
                bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
                dag_cpds.append(cpd_col)
                index_node[col] = idx

        elif discovery_alg == 'lingam':
            for col in data.columns:
                data[col] = data[col].astype('category')

            ling = lingam.ICALiNGAM() # class for causal discovery through LiNGAM algorithm
            lingam_model = ling.fit(data) # estimate the DAG structure through LiNGAM algorithm
            W_dag_model = lingam_model.adjacency_matrix_

            # obtain the edges from the adjacency matrix in order to create the Discrete Bayesian Network with the library 'pgmpy'
            dag_model_nodes = list(data.columns)
            dag_model_edges = []
            n = len(W_dag_model)
            for i in range(n):
                for j in range(n):
                    if W_dag_model[i][j] != 0:
                        dag_model_edges.append((dag_model_nodes[i], dag_model_nodes[j]))
            dag_model ={'nodes': dag_model_nodes, 'edges': dag_model_edges}

            bn = DiscreteBayesianNetwork(dag_model_edges) # initialize a Discrete Bayesian Network, i.e. a DAG where we will incorporate a Conditional Probability Distribution
            dag_estimator = BayesianEstimator(bn, data) # initialize a Bayesian Estimator for computing the parameters of the model (CPDs)

            dag_cpds = [] # store the CPDs for each variable
            index_node = {} # store the index of the nodes appearing in the Bayesian Network
            for idx, col in enumerate(bn.nodes): # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
                cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
                bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
                dag_cpds.append(cpd_col)
                index_node[col] = idx

        elif discovery_alg == 'notears':
            data_array = data.to_numpy()
            if notears_loss_type == None:
                print("Error: indicate a loss function to apply NOTEARS alg. Process interrupted!")
                break
            else:
                W_dag_model = StabSelec_notears_sparsity(data_array, notears_loss_type, stab_freq=0.75, sample_lambda=100, num_iter=5)[0] # estimates the DAG structure (dependencies without parameters!) using the optimization given by NOTEARS. This line is analogous to the both 'hc = HillClimbSearch(data)' and 'model = hc.estimate(scoring_method='bic-d')' in the Hill-Climbining case!

                # obtain the edges from the adjacency matrix in order to create the Discrete Bayesian Network with the library 'pgmpy'
                dag_model_nodes = list(data.columns)
                dag_model_edges = []
                n = len(W_dag_model)
                for i in range(n):
                    for j in range(n):
                        if W_dag_model[i][j] != 0:
                            dag_model_edges.append((dag_model_nodes[i], dag_model_nodes[j]))
                dag_model ={'nodes': dag_model_nodes, 'edges': dag_model_edges}

                bn = DiscreteBayesianNetwork(dag_model_edges) # initialize a Discrete Bayesian Network, i.e. a DAG using the learned model above 
                dag_estimator = BayesianEstimator(bn, data) # compute parameters for a model (CPDs)

                dag_cpds = [] # store the CPDs for each variable
                index_node = {} # store the index of the nodes appearing in the Bayesian Network
                for idx, col in enumerate(bn.nodes): # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
                    cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
                    bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
                    dag_cpds.append(cpd_col)
                    index_node[col] = idx

        # M-step: update the NaN values by maximizing the probability of its value according to the learned Bayesian Network 

        for i, j in pos_NA:
            if data.columns[j] not in bn.nodes:
                new_Z[(i,j)] = data.iloc[i,j]
                continue
            else:
                idx_node = index_node[data.columns[j]]
                parents = dag_cpds[idx_node].get_evidence()# get the parents of each NaN variable within the learned DAG. And this for each line
            
                if len(parents) > 0:
                    marginalized_cpd = dag_cpds[idx_node].marginalize(parents, False) # obtain the marginalized CPD with respect to the parents
                    marginal_prob0_NA = marginalized_cpd.values[0] # marginal probability of this NaN value to be 0 (with respect to the learned model)
                    marginal_prob1_NA = marginalized_cpd.values[1] # marginal probability of this NaN value to be 1 (with respect to the learned model)

                    if marginal_prob0_NA > marginal_prob1_NA:
                        new_Z[(i,j)] = 0
                    elif marginal_prob0_NA < marginal_prob1_NA:
                        new_Z[(i,j)] = 1
                    elif marginal_prob0_NA == marginal_prob1_NA:
                        new_Z[(i,j)] = random.randint(0,1)
                elif len(parents) == 0: # in this case the learned model already provides these probabilities
                    prob0_NA = dag_cpds[idx_node].values[0]
                    prob1_NA = dag_cpds[idx_node].values[1]

                    if prob0_NA > prob1_NA:
                        new_Z[(i,j)] = 0
                    elif prob0_NA < prob1_NA:
                        new_Z[(i,j)] = 1
                    elif prob0_NA == prob1_NA:
                        new_Z[(i,j)] = random.randint(0,1)

        for (i,j), estimated_val_NA in new_Z.items(): # update the data values at the NaN positions 
            data.iloc[i,j] = estimated_val_NA

        if previous_edges: # check the stop condition for EM
            num_changes = sum(prev != new for prev, new in zip_longest(previous_edges, dag_model_edges))
            if num_changes /max(len(previous_edges), len(dag_model_edges))  < epsilon:
                print(f'EM using the {discovery_alg} method stops at iteration {iter}') # stop when the edges of the model don't change any more
                break
        previous_edges = dag_model_edges # store the edges of the discovered DAG
    
    return dag_model, bn, data





#########################################################
#----- Technical tools: Causality for chronology
#########################################################

# Causal inference from a DAG
def causal_inf(data, dag_model, bn_model):

    # in order to use the DoWhy library ('CausalModel()'), we need the graph to be a 'networkx' object
    dag_model_nx = dag_model.to_networkx()
    name_nodes = {v: dag_model.vs[v]["name"] for v in range(len(dag_model.vs))}
    dag_model_nx = nx.relabel_nodes(dag_model_nx, name_nodes)
    # create a 'CausalInference' class in order to use the causal inference functions from 'pgmpy'
    bn_model_CausalInf = CausalInference(bn_model)

    causality_summary = [] # store the causal inference results
    variables = list(bn_model.nodes)
    for X in variables:
        for Y in variables:
            path_XY = dag_model.get_shortest_paths(X, Y, mode='out')[0]
            if X == Y or len(path_XY) == 0: # either when X==Y or there is no directed path from X to Y, no causal inference!
                continue
            
            # default values
            estimate_TE_val = float('nan')
            estimate_NDE_val = float('nan')
            estimate_NIE_val = float('nan')
            portion_direct_effect = float('nan')
            portion_indirect_effect = float('nan')
            causality = 'NA'
            refutation_summary = [] # store the TE-refutation results

            try:
                # 1. Modeling

                potential_CauseEffect = CausalModel(
                data=data, 
                treatment = X, 
                outcome = Y, 
                graph = dag_model_nx
                )
                

                # 2. Identification of the estimand

                # Total Effect (= Average Causal Effect)
                estimand_TE = potential_CauseEffect.identify_effect()
                backdoor_var = estimand_TE.get_backdoor_variables('backdoor')
                # Natural Direct Effect
                estimand_NDE = potential_CauseEffect.identify_effect(estimand_type="nonparametric-nde")
                # Natural Indirect Effect
                estimand_NIE = potential_CauseEffect.identify_effect(estimand_type="nonparametric-nie")
                mediator_NIE_var = estimand_NIE.get_mediator_variables()


                # 3. Estimation

                estimate_TE = potential_CauseEffect.estimate_effect(
                    estimand_TE, 
                    target_units = "ate", 
                    method_name="backdoor.linear_regression", 
                    )
                estimate_TE_val = estimate_TE.value
                if not math.isnan(estimate_TE_val)  and estimate_TE_val > 0:
                    causality = 'Yes'
                    # for the records: compute the interventions on X
                    do1 = bn_model_CausalInf.query(variables=[Y], do={X:1}, adjustment_set=backdoor_var, inference_algo='ve')
                    do0 = bn_model_CausalInf.query(variables=[Y], do={X:0}, adjustment_set=backdoor_var, inference_algo='ve')
                    # the ATE might be in [-1, 1] by definition-our variables are binary-. If not, there might be a problem with the 'linear_regression' estimation --> in this case, use the direct estimates provided by the 'pgmpy' library
                    if abs(estimate_TE_val) > 1:
                        estimate_TE_val = do1.values[1] - do0.values[1]
                else:
                    causality = 'No'

                estimate_NDE = potential_CauseEffect.estimate_effect(
                    estimand_NDE, 
                    method_name="mediation.two_stage_regression",
                    method_params = {
                    'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
                    'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
                    }
                    )
                estimate_NDE_val = estimate_NDE.value
                if not math.isnan(estimate_NDE_val) and not math.isnan(estimate_TE_val):
                    portion_direct_effect = estimate_NDE_val/estimate_TE_val * 100

                estimate_NIE = potential_CauseEffect.estimate_effect(
                    estimand_NIE, 
                    method_name="mediation.two_stage_regression",
                    method_params = {
                    'first_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator,
                    'second_stage_model': dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
                    }
                    )
                estimate_NIE_val = estimate_NIE.value
                if not math.isnan(estimate_NIE_val) and not math.isnan(estimate_TE_val):
                    portion_indirect_effect = estimate_NIE_val/estimate_TE_val * 100


                # 4. Refutation/Validaton of the estimates

                refuters = ["data_subset_refuter", "random_common_cause", "placebo_treatment_refuter"]
                for refuter in refuters:
                    try:
                        refutation = potential_CauseEffect.refute_estimate(estimand_TE, estimate_TE, method_name=refuter)
                        refutation_summary.append(refutation)
                    except Exception as e_refu:
                        print(f"Refutation '{refuter}' failed for {X} -> {Y}: {e_refu}")
                        refutation_summary.append('NA')
                    
            except Exception as e:
                print(f"Error occured when estimating the causal effect of {X} on {Y}: {e}")

            while len(refutation_summary) < 3:
                refutation_summary.append('NA')

            # summarizing the results
            causality_summary.append({
                'Cause (X)': X,
                'Outcome (Y)': Y,
                'Confounders (Z)': backdoor_var,
                'Mediators (M)': mediator_NIE_var,
                'P(Y=1|do(X=1))': do1.values[1],
                'P(Y=1|do(X=0))': do0.values[1],
                'Total Effect (TE)': estimate_TE_val,
                # 'Relative Improvement': (do1.values[1]-do0.values[1])/do0.values[1]*100,
                'Direct Effect (NDE)': estimate_NDE_val,
                'Indirect Effect (NIE)': estimate_NIE_val,
                'NDE/TE': portion_direct_effect,
                'NIE/TE': portion_indirect_effect,
                'Causality': causality,
                'Refutation TE_DataSubset': refutation_summary[0],
                'Refutation TE_RandCommCause': refutation_summary[1],
                'Refutation TE_Placebo': refutation_summary[2]
            })

    causality_summary = pd.DataFrame(causality_summary)

    return causality_summary


# Chronological order of the events based on the causality inference theory of Pearl
def chronology_events(causality_summary, dag_model, original_data):
    
    # simplified causality summary: focus on causal relations within the DAG and the refutation tests
    simp_causality_summary = causality_summary[causality_summary['Causality'] == 'Yes'][['Cause (X)', 'Outcome (Y)', 'Total Effect (TE)', 'Direct Effect (NDE)', 'Refutation TE_DataSubset', 'Refutation TE_RandCommCause', 'Refutation TE_Placebo']]
    
    # sort 'Cause(X)' by greatest (direct) effect
    simp_causality_summary['Actual Effect'] = simp_causality_summary['Direct Effect (NDE)'].fillna(simp_causality_summary['Total Effect (TE)']) # the actual effect is the NDE (which = TE when no mediators!)
    simp_causality_summary = simp_causality_summary.sort_values(by='Actual Effect', ascending=False).reset_index(drop=True) # sort the rows by descending order wrt AE
    
    # Quatitative causal relations summary
    causal_relations = simp_causality_summary
    causal_relations = causal_relations.drop(['Refutation TE_DataSubset', 'Refutation TE_RandCommCause', 'Refutation TE_Placebo'], axis=1)
    strong_causal_relations = causal_relations.drop_duplicates(subset='Outcome (Y)', keep='first') # among the possible causes for the same 'Outcome(Y)' keep only the one with greatest AE

    # refutation tests summary for the (relevant) causal relations selected above
    refutation_summary_causal_rel = simp_causality_summary
    refutation_summary_causal_rel = refutation_summary_causal_rel.drop(['Direct Effect (NDE)', 'Actual Effect'], axis=1)


    topo_order = dag_model.topological_sorting(mode="OUT") # get the topological order in the DAG (starting from a node with in-degree 0)
    names_mapping = {i: dag_model.vs[i]["name"] for i in range(len(dag_model.vs))} # dict. of the form {i : name_node_i}
    topo_names = [names_mapping[i] for i in topo_order] # get the names of the nodes in the topological order
    idx_from_name = {name: i for i, name in names_mapping.items()} # dict. of the form {name_node_i : i}

    # compute the levels of the topological order for each node
    node_level = {}
    for node in topo_order:
        parents_node = dag_model.predecessors(node)
        if not parents_node: # when there is no parents
            node_level[node] = 0
        else:
            node_level[node] = 1 + max(node_level[p] for p in parents_node)

    # group the nodes by their topological level
    levels_info = defaultdict(list) # initialize the dictionnary with empy lists for each key
    for node_topo_idx, lvl in node_level.items():
        name = dag_model.vs[node_topo_idx]["name"]
        levels_info[lvl].append(name)

    # sort the edges from 'strong_causal_relations' respecting the level in the topological order
    edges_strong_causal_rel = list(strong_causal_relations[['Cause (X)', 'Outcome (Y)']].itertuples(index=False, name=None)) # get the 'Cause(X)' and the 'Outcome(Y)' from 'strong_causal_relations' as a tuple (X, Y)
    edges_strong_causal_rel_topo = sorted(edges_strong_causal_rel, key=lambda edge: node_level[idx_from_name[edge[0]]])

    causal_rel_topo_level = defaultdict(list)
    for edge in edges_strong_causal_rel_topo:
        level = node_level[idx_from_name[edge[0]]]
        key_level = f"level {level}"
        causal_rel_topo_level[key_level] = edge

    # create the DAG representing the chronology
    chronology_dag = ig.Graph(directed=True)
    nodes = sorted(set(v for edge in edges_strong_causal_rel_topo for v in edge)) 
    chronology_dag.add_vertices(nodes)
    chronology_dag.add_edges(edges_strong_causal_rel_topo)
    chronology_dag.vs['label'] = chronology_dag.vs['name'] = nodes

    # check whether all roots of the original DAG are in the chronology. If not, add them as isolated nodes!
        # note that root nodes can dissapear from the previous causality analysis according to the causality path construction
        # even if in this case these nodes are not "causes", they have been seen, so they should appear in the chronology at time 0
    root_nodes = [names_mapping[i] for i in topo_order if node_level[i] == 0]
    chronology_nodes = chronology_dag.vs["name"] if chronology_dag.vcount() > 0 else []
    missing_roots = [root for root in root_nodes if root not in chronology_nodes]
    chronology_dag.add_vertices(missing_roots)
    for root in missing_roots:
        r = chronology_dag.vs.find(name=root)
        r['label'] = root
    # add also add all nodes that were already isolated in the original DAG
    isolated_nodes = [v["name"] for v in dag_model.vs if dag_model.degree(v.index, mode="all") == 0]
    chronology_nodes = chronology_dag.vs["name"] if chronology_dag.vcount() > 0 else []
    missing_isolated = [isol for isol in isolated_nodes if isol not in chronology_nodes]
    chronology_dag.add_vertices(missing_isolated)
    for isol in missing_isolated:
        iso = chronology_dag.vs.find(name=isol)
        iso['label'] = isol
    # other nodes can dissapear: if no entering/sorting arrow gives a positive causal effect. Add them too!
    no_causality_nodes = causality_summary[causality_summary['Causality'] == 'No'][['Cause (X)', 'Outcome (Y)', 'Mediators (M)']]
    existing_nodes = set(chronology_dag.vs["name"]) if chronology_dag.vcount() > 0 else set()
    missing_no_causality = [row['Outcome (Y)'] for _, row in no_causality_nodes.iterrows() 
                            if row['Mediators (M)'] == [] and row['Outcome (Y)'] not in existing_nodes]
    chronology_dag.add_vertices(missing_no_causality)
    for no_causality in missing_no_causality:
        v = chronology_dag.vs.find(name=no_causality)
        v['label'] = no_causality

    # create the corresponding Discrete Bayesian Network model toghether with its parameters (CPDs)
    chronology_dag_edges = [(chronology_dag.vs[src]['name'], chronology_dag.vs[tgt]['name']) for src, tgt in chronology_dag.get_edgelist()]
    chronology_bn = DiscreteBayesianNetwork(chronology_dag_edges)
    dag_estimator = BayesianEstimator(chronology_bn, original_data)
    dag_cpds = []
    for col in chronology_bn.nodes:
        cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2')
        chronology_bn.add_cpds(cpd_col)
        dag_cpds.append(cpd_col)

    return chronology_dag, chronology_bn, causal_rel_topo_level, causal_relations, refutation_summary_causal_rel, strong_causal_relations


'''# SANITY CHECK: Completion of a given chronological DAG wrt a given dataset
    # supported discovery algorithms: 'hill_climbing', 'pc', 'notears'. for 'notears' further improvement is needed: future research!
def completion_chron_dag(data, chronology_dag, discovery_alg, name_nodes=None, black_list_edges=None, notears_loss_type=None):
    
    if name_nodes != None: # give a particular name to the vertex of the final DAG instead of the default column names of 'data'
        data.columns = name_nodes
    
    white_list_edges = [(chronology_dag.vs[src]["name"], chronology_dag.vs[tgt]["name"]) for src, tgt in chronology_dag.get_edgelist()]
    black_list_edges = black_list_edges
    expert_knowledge = ExpertKnowledge(required_edges=white_list_edges, fobidden_edges=black_list_edges)

    if discovery_alg == 'hill_climbing':
        for col in data.columns:
            data[col] = data[col].astype('category')

        hc = HillClimbSearch(data) # class for heuristic Hill Climbing searches for DAGs
        dag_model = hc.estimate(scoring_method='bic-d', expert_knowledge=expert_knowledge) # locall hill-climbing to estimate the DAG structure (dependencies without parameters!) that has optimal score
        dag_model_nodes = dag_model.nodes()
        dag_model_edges = dag_model.edges()
        bn = DiscreteBayesianNetwork(dag_model_edges) # initialize a Discrete Bayesian Network, i.e. a DAG where we will incorporate a Conditional Probability Distribution
        dag_estimator = BayesianEstimator(bn, data) # initialize a Bayesian Estimator for computing the parameters of the model (CPDs)

        dag_cpds = [] # store the CPDs for each variable
        index_node = {} # store the index of the nodes appearing in the Bayesian Network
        for idx, col in enumerate(bn.nodes): # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
            cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
            bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
            dag_cpds.append(cpd_col)
            index_node[col] = idx

    elif discovery_alg == 'pc':
        for col in data.columns:
            data[col] = data[col].astype('category')

        pc = PC(data) # class for constraint-based estimation of DAGs through Peter-Clark algorithm
        dag_model = pc.estimate(variant='orig', ci_test='chi_square', return_type='dag', expert_knowledge=expert_knowledge) # estimate the DAG structure through statistical independence tests. Moreover, we return a fully directed structure if it is possible to orient all the edges
        dag_model_nodes = dag_model.nodes()
        dag_model_edges = dag_model.edges()
        bn = DiscreteBayesianNetwork(dag_model_edges) # initialize a Discrete Bayesian Network, i.e. a DAG where we will incorporate a Conditional Probability Distribution
        dag_estimator = BayesianEstimator(bn, data) # initialize a Bayesian Estimator for computing the parameters of the model (CPDs)

        dag_cpds = [] # store the CPDs for each variable
        index_node = {} # store the index of the nodes appearing in the Bayesian Network
        for idx, col in enumerate(bn.nodes): # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
            cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
            bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
            dag_cpds.append(cpd_col)
            index_node[col] = idx

    elif discovery_alg == 'notears':
        data_array = data.to_numpy()
        if notears_loss_type == None:
            print("Error: indicate a loss function to apply NOTEARS alg. Process interrupted!")
        else:
            W_dag_model = StabSelec_notears_sparsity(data_array, notears_loss_type, stab_freq=0.75, sample_lambda=25, num_iter=3)[0] # estimates the DAG structure (dependencies without parameters!) using the optimization given by NOTEARS. This line is analogous to the both 'hc = HillClimbSearch(data)' and 'model = hc.estimate(scoring_method='bic-d')' in the Hill-Climbining case!

            # obtain the edges from the adjacency matrix in order to create the Discrete Bayesian Network with the library 'pgmpy'
            dag_model_nodes = list(data.columns)
            dag_model_edges = []
            n = len(W_dag_model)
            for i in range(n):
                for j in range(n):
                    if W_dag_model[i, j] != 0:
                        dag_model_edges.append((dag_model_nodes[i], dag_model_nodes[j]))
            # use the whitelist of edges to force the learned DAG to have those edges (= 'expert_knowledge' within NOTEARS)
                # check whether this process creates cycles in the graph!
            W_dag_model_df = pd.DataFrame(W_dag_model, columns=dag_model_nodes, index=dag_model_nodes)
            G_expert_knowledge = nx.DiGraph()
            G_expert_knowledge.add_nodes_from(dag_model_nodes)
            G_expert_knowledge.add_edges_from(dag_model_edges)
            for src, tgt in white_list_edges:
                G_expert_knowledge.add_edge(src, tgt)
                if not nx.is_directed_acyclic_graph(G_expert_knowledge):
                    G_expert_knowledge.remove_edge(src, tgt) 
            # for src, tgt in white_list_edges:
            #     i_whitelist = W_dag_model_df.index.get_loc(src)
            #     j_whitelist = W_dag_model_df.columns.get_loc(tgt)
            #     W_dag_model[i_whitelist, j_whitelist] = 1
            dag_model_edges = list(G_expert_knowledge.edges())
            dag_model ={'nodes': dag_model_nodes, 'edges': dag_model_edges}
            bn = DiscreteBayesianNetwork(dag_model_edges) # initialize a Discrete Bayesian Network, i.e. a DAG using the learned model above 
            dag_estimator = BayesianEstimator(bn, data) # compute parameters for a model (CPDs)

            dag_cpds = [] # store the CPDs for each variable
            index_node = {} # store the index of the nodes appearing in the Bayesian Network
            for idx, col in enumerate(bn.nodes): # compute the parameters for the model ('bn') accoding to the dataset ('data') using the estimator 'dag_estimator'
                cpd_col = dag_estimator.estimate_cpd(col, prior_type='K2') # estimate the CPD for each variable ('col') of the dataset appearing in the Bayesian Network model
                bn.add_cpds(cpd_col) # incorporate the estimated CPD to the Bayesian Network 'bn'
                dag_cpds.append(cpd_col)
                index_node[col] = idx

    # define the completed dag model as an 'igraph' object
    nodes = list(dag_model_nodes)
    edges = list(dag_model_edges)
    completed_chron_model = ig.Graph(n=len(nodes), directed=True)
    completed_chron_model.vs['label'] = completed_chron_model.vs["name"] = nodes
    completed_chron_model.add_edges(edges)

    bn_completed_chron_model = bn

    return completed_chron_model, bn_completed_chron_model'''


# Falsification of a given DAG in terms of a dataset (cf. 'DoWhy' documentation)
    # in our case, we use the chronology DAG obtained by causal inference or the Ref. chronology DAG suggested in Guillem's paper
    # Indeed, recall the 'causal minimality principle': if the ACTUAL Markov joint prob. distr. of G is P, then P is NOT the Markov joint prob. distr. of any proper subgraph of G
    # i.e. if the causal Markov condition is true, then the true causal structure is a minimal structure that is Markov to P.
    # Note: the causal minimality condition is also about the ACTUAL prob. distr., and hence is inferentially relevant in the sense that the causal Markov condition is.
    # cf. "Intervention, determinism, and the causal minimality condition" (by Zhang-Spirtes), for instance
def falsify_dag(data, dag_model):
    
    # in order to use the DoWhy library ('Falsify'), we need the graph to be a 'networkx' object
    dag_model_nx = dag_model.to_networkx()
    name_nodes = {v: dag_model.vs[v]["name"] for v in range(len(dag_model.vs))}
    dag_model_nx = nx.relabel_nodes(dag_model_nx, name_nodes)
    
    # apply the 'falsify_graph' funcion (cf. 'DoWhy' documentation)
    fals_result = falsify_graph(dag_model_nx, data, plot_histogram=False, suggestions=False)

    # summarizing the results
    falsification_summary = [repr(fals_result), fals_result.falsifiable, fals_result.falsified]

    return falsification_summary