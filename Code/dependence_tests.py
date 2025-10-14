#####################################
#----- Preamble
#####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from utils import data_folder




############################################
#----- Dependence tests: Fisher and Chi2
############################################

# we use the data sets that we have already treated and sorted in 'data_management.py'
ndhB = pd.read_csv(data_folder/f'ndhB_sorted.csv', index_col=0) 
ndhD = pd.read_csv(data_folder/f'ndhD_sorted.csv', index_col=0)

events_ndhB = ndhB.columns
events_ndhD = ndhD.columns

# we compare with the imputed data sets
ndhB_imp = pd.read_csv(data_folder/f'ndhB_imputed_IterativeImputer_LogisticReg.csv', index_col=0)
ndhD_imp = pd.read_csv(data_folder/f'ndhD_imputed_IterativeImputer_LogisticReg.csv', index_col=0)





# Before imputation
print(f"\n Dependencies before imputation for ndhB:")
# print("\n Test Chi2.")
# ndhB_dep_events_before_chi2, ndhB_indep_events_before_chi2 = utils.dep_tests_events(ndhB, events_ndhB, test = 'chi2', alpha=0.005)
# print(f"\n {len(ndhB_dep_events_before_chi2)} Dependent events:\n", ndhB_dep_events_before_chi2)
# print(f"\n {len(ndhB_indep_events_before_chi2)} Independent events:\n", ndhB_indep_events_before_chi2)
print("\n Test Fisher.")
ndhB_dep_events_before_fisher, ndhB_indep_events_before_fisher = utils.dep_tests_events(ndhB, events_ndhB, test = 'fisher', alpha=0.005)
print(f"\n {len(ndhB_dep_events_before_fisher)} Dependent events:\n", ndhB_dep_events_before_fisher)
print(f"\n {len(ndhB_indep_events_before_fisher)} Independent events:\n", ndhB_indep_events_before_fisher)

# print(f"\n Dependencies before imputation for ndhD:")
# print("\n Test Chi2.")
# ndhD_dep_events_before_chi2, ndhD_indep_events_before_chi2 = utils.dep_tests_events(ndhD, events_ndhD, test = 'chi2', alpha=0.005)
# print(f"\n {len(ndhD_dep_events_before_chi2)} Dependent events:\n", ndhD_dep_events_before_chi2)
# print(f"\n {len(ndhD_indep_events_before_chi2)} Independent events:\n", ndhD_indep_events_before_chi2)
# print("\n Test Fisher.")
# ndhD_dep_events_before_fisher, ndhD_indep_events_before_fisher = utils.dep_tests_events(ndhD, events_ndhD, test = 'fisher', alpha=0.005)
# print(f"\n {len(ndhD_dep_events_before_fisher)} Dependent events:\n", ndhD_dep_events_before_fisher)
# print(f"\n {len(ndhD_indep_events_before_fisher)} Independent events:\n", ndhD_indep_events_before_fisher)





# After imputation
print(f"\n Dependencies after imputation for ndhB:")
# print("\n Test Chi2.")
# ndhB_dep_events_after_chi2, ndhB_indep_events_after_chi2 = utils.dep_tests_events(ndhB_imp, events_ndhB, test = 'chi2', alpha=0.005, imputed = True)
# print(f"\n {len(ndhB_dep_events_after_chi2)} Dependent events:\n", ndhB_dep_events_after_chi2)
# print(f"\n {len(ndhB_indep_events_after_chi2)} Independent events:\n", ndhB_indep_events_after_chi2)
print("\n Test Fisher.")
ndhB_dep_events_after_fisher, ndhB_indep_events_after_fisher = utils.dep_tests_events(ndhB_imp, events_ndhB, test = 'fisher', alpha=0.005, imputed = True)
print(f"\n {len(ndhB_dep_events_after_fisher)} Dependent events:\n", ndhB_dep_events_after_fisher)
print(f"\n {len(ndhB_indep_events_after_fisher)} Independent events:\n", ndhB_indep_events_after_fisher)

# print(f"\n Dependencies after imputation for ndhD:")
# print("\n Test Chi2.")
# ndhD_dep_events_after_chi2, ndhD_indep_events_after_chi2 = utils.dep_tests_events(ndhD_imp, events_ndhD, test = 'chi2', alpha=0.005, imputed = True)
# print(f"\n {len(ndhD_dep_events_after_chi2)} Dependent events:\n", ndhD_dep_events_after_chi2)
# print(f"\n {len(ndhD_indep_events_after_chi2)} Independent events:\n", ndhD_indep_events_after_chi2)
# print("\n Test Fisher.")
# ndhD_dep_events_after_fisher, ndhD_indep_events_after_fisher = utils.dep_tests_events(ndhD_imp, events_ndhD, test = 'fisher', alpha=0.005, imputed = True)
# print(f"\n {len(ndhD_dep_events_after_fisher)} Dependent events:\n", ndhD_dep_events_after_fisher)
# print(f"\n {len(ndhD_indep_events_after_fisher)} Independent events:\n", ndhD_indep_events_after_fisher)





# # Relative comparaisons
# print("\n Tests Chi2 for ndhB.")
# print(f"\n There are {len(ndhB_dep_events_before_chi2)} Chi2 dependent events before imputation. There are {len(ndhB_dep_events_after_chi2)} Chi2 dependent events after imputation.")
# print(f"\n There are {len(ndhB_indep_events_before_chi2)} Chi2 independent events before imputation. There are {len(ndhB_indep_events_after_chi2)} Chi2 independent events after imputation.")
# print(f"\n Ratio dep/indep before: {round(len(ndhB_dep_events_before_chi2)/len(ndhB_indep_events_before_chi2), 2)}")
# print(f"\n Ratio dep/indep after: {round(len(ndhB_dep_events_after_chi2)/len(ndhB_indep_events_after_chi2), 2)}")
# print("\n Tests Fisher for ndhB.")
# print(f"\n There are {len(ndhB_dep_events_before_fisher)} Chi2 dependent events before imputation. There are {len(ndhB_dep_events_after_fisher)} Chi2 dependent events after imputation.")
# print(f"\n There are {len(ndhB_indep_events_before_fisher)} Chi2 independent events before imputation. There are {len(ndhB_indep_events_after_fisher)} Chi2 independent events after imputation.")
# print(f"\n Ratio dep/indep before imputation: {round(len(ndhB_dep_events_before_fisher)/len(ndhB_indep_events_before_fisher), 2)}")
# print(f"\n Ratio dep/indep after imputation: {round(len(ndhB_dep_events_after_fisher)/len(ndhB_indep_events_after_fisher), 2)}")

# print("\n Tests Chi2 for ndhD.")
# print(f"\n There are {len(ndhD_dep_events_before_chi2)} Chi2 dependent events before imputation. There are {len(ndhD_dep_events_after_chi2)} Chi2 dependent events after imputation.")
# print(f"\n There are {len(ndhD_indep_events_before_chi2)} Chi2 independent events before imputation. There are {len(ndhD_indep_events_after_chi2)} Chi2 independent events after imputation.")
# # print(f"\n Ratio dep/indep before imputation: {round(len(ndhD_dep_events_before_chi2)/len(ndhD_indep_events_before_chi2), 2)}")
# # print(f"\n Ratio dep/indep after imputation: {round(len(ndhD_dep_events_after_chi2)/len(ndhD_indep_events_after_chi2), 2)}")
# print("\n Tests Fisher for ndhD.")
# print(f"\n There are {len(ndhD_dep_events_before_fisher)} Chi2 dependent events before imputation. There are {len(ndhD_dep_events_after_fisher)} Chi2 dependent events after imputation.")
# print(f"\n There are {len(ndhD_indep_events_before_fisher)} Chi2 independent events before imputation. There are {len(ndhD_indep_events_after_fisher)} Chi2 independent events after imputation.")
# # print(f"\n Ratio dep/indep before imputation: {round(len(ndhD_dep_events_before_fisher)/len(ndhD_indep_events_before_fisher), 2)}")
# # print(f"\n Ratio dep/indep after imputation: {round(len(ndhD_dep_events_after_fisher)/len(ndhD_indep_events_after_fisher), 2)}")