#####################################
#----- Preamble
#####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from utils import data_folder
from utils import charts_folder
from utils import color_binary
from utils import color_imputed

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression





############################################
#----- Imputation with IterativeImputer
############################################

# we use the data sets that we have already treated and sorted in 'data_management.py'
ndhB = pd.read_csv(data_folder/f'ndhB_sorted.csv', index_col=0)
pos_NA_ndhB = np.where(np.isnan(ndhB)) # save the positions where there are NA values after ordering the rows
ndhD = pd.read_csv(data_folder/f'ndhD_sorted.csv', index_col=0)
pos_NA_ndhD = np.where(np.isnan(ndhD)) # save the positions where there are NA values after ordering the rows

# replace the 'NA' values by 'np.nan' values in order to apply the IterativeImputer module
ndhB = ndhB.replace('NA', np.nan, inplace=False)
ndhD = ndhD.replace('NA', np.nan, inplace=False)

# get the names of variables of our data sets
events_ndhB = ndhB.columns
events_ndhD = ndhD.columns

# get the genomic positions of the events

# load the file where there is the info about the events
editing_sites =  pd.read_csv(data_folder/f'edition_sep_col.csv')

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





# default estimator = BayesianRidge(). For our problem, it is better to use the LogisticRegression() estimator
imputer = IterativeImputer(max_iter=10, random_state=0, estimator=LogisticRegression())


# ndhB
ndhB_imputed = imputer.fit_transform(ndhB)
ndhB_imputed = pd.DataFrame(ndhB_imputed, columns=ndhB.columns)
ndhB_imputed = ndhB_imputed.astype(int)
ndhB_imputed.to_csv(data_folder/f"ndhB_imputed_IterativeImputer_LogisticReg.csv")
# utils.visu_data(ndhB_imputed, color_binary, color_imputed, pos_NA_ndhB, imputed=True)


#ndhD
ndhD_imputed = imputer.fit_transform(ndhD)
ndhD_imputed = pd.DataFrame(ndhD_imputed, columns=ndhD.columns)
ndhD_imputed = ndhD_imputed.astype(int)
ndhD_imputed.to_csv(data_folder/f"ndhD_imputed_IterativeImputer_LogisticReg.csv")
# utils.visu_data(ndhD_imputed, color_binary, color_imputed, pos_NA_ndhD, imputed=True)





# Compare the distribution of 0s and 1s by column of the data sets before and after imputation

# Before imputation
# utils.distr_values_var(ndhB, name_cols=gen_pos_ndhB)
# utils.distr_values_var(ndhD, name_cols=gen_pos_ndhD)

# After imputation
# utils.distr_values_var(ndhB_imputed, name_cols=gen_pos_ndhB, imputed=True)
# utils.distr_values_var(ndhD_imputed, name_cols=gen_pos_ndhD, imputed=True)