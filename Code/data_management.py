#####################################
#----- Preamble
#####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from utils import data_folder
from utils import code_global
from utils import color_global
from utils import code_binary
from utils import color_binary





#########################################
#----- Pre-processing of the data
#########################################

# Separate the columns from the original data files

edition = pd.read_csv(data_folder/f'edition.csv', header=None)
edition_sep_col = edition[0].str.split(' ', expand=True) # create the columns in the data set
edition_sep_col.to_csv(data_folder/f'edition_sep_col.csv', index=False)

intron = pd.read_csv(data_folder/f'intron.csv', header=None)
intron_sep_col = intron[0].str.split(' ', expand=True) # create the columns in the data set
intron_sep_col.to_csv(data_folder/f'intron_sep_col.csv', index=False)

ndhB = pd.read_csv(data_folder/f'ndhB.csv', header=None)
ndhB_sep_col = ndhB[0].str.split(' ', expand=True) # create the columns in the data set
ndhB_sep_col = ndhB_sep_col.applymap(lambda x: x.strip('"').strip("'") if isinstance(x, str) else x) # remove quotation marks from strings in the data set
ndhB_sep_col.to_csv(data_folder/f'ndhB_sep_col.csv', index=False)

ndhD = pd.read_csv(data_folder/f'ndhD.csv', header=None)
ndhD_sep_col = ndhD[0].str.split(' ', expand=True) # create the columns in the data set
ndhD_sep_col = ndhD_sep_col.applymap(lambda x: x.strip('"').strip("'") if isinstance(x, str) else x) # remove quotation marks from strings in the data set
ndhD_sep_col.to_csv(data_folder/f'ndhD_sep_col.csv', index=False)


# Load the data for the genes ndhB and ndhD. Note that the position of the intron_2_16 must be reordered within ndhB!

ndhB =  pd.read_csv(data_folder/f'ndhB_sep_col.csv', sep=',', header=1)
intron_2_16 = ndhB.pop('intron_2_16')
ndhB.insert(ndhB.columns.get_loc('editing_1_30')+1, 'intron_2_16', intron_2_16)
ndhB.to_csv(data_folder/f'ndhB_intron_16_ord.csv', index=False)
ndhB = ndhB.drop('rep', axis=1) # remove the first column, which is not an event
ndhB = ndhB.drop('intron_2_16', axis=1) # remove the intron column
num_row_ndhB, num_col_ndhB = ndhB.shape
ndhD = pd.read_csv(data_folder/f'ndhD_sep_col.csv', sep=',', header=1)
ndhD = ndhD.drop('rep', axis=1) # remove the first column, which is not an event
num_row_ndhD, num_col_ndhD = ndhD.shape


print(f'\n There are {num_row_ndhB} observations for ndhB. There are {len(ndhB.columns)} maturation events for the gene ndhB:')
print('\n', ndhB.columns)
print(f'\n There are {num_row_ndhD} observations for ndhD. There are {len(ndhD.columns)} maturation events for the gene ndhD:')
print('\n', ndhD.columns)





# #####################################
# #----- Studying the NA values
# #####################################

# Are there values different from NA after the first NA? Yes!
# Are there NA values at the beginning and at the end of a row? Yes!
# Are there values different from NA isolated among blocks of values different from NA? No!


print("\n Counting how NA are distributed within the data.")

ndhB = ndhB.fillna('NA')
count_no_NA, count_only_NA, count_mix_NA = utils.count_NA(ndhB)
print('\n In ndhB there are ', count_no_NA, ' rows with no NA at all')
print('\n In ndhB there are ', count_only_NA, ' rows with only NA after the first NA')
print('\n In ndhB there are ', count_mix_NA, ' rows with non-trivial reads after the first NA')

ndhD = ndhD.fillna('NA')
count_no_NA, count_only_NA, count_mix_NA = utils.count_NA(ndhD)
print('\n In ndhD there are ', count_no_NA, ' rows with no NA at all')
print('\n In ndhD there are ', count_only_NA, ' rows with only NA after the first NA')
print('\n In ndhD there are ', count_mix_NA, ' rows with non-trivial reads after the first NA')


print("\n Counting blocks with values different from NA.")

col_init, col_final_read_block, col_fin, col_blocks, col_isolated = utils.count_NA_blocks(ndhB)

# Enlarge the original data sets with the counting of blocks made with the function 'count_NA_blocks()'
init_fin_reads=pd.DataFrame(
    {
        'initial_read': col_init,
        'final_read_block': col_final_read_block,
        'final_read': col_fin,
        'consec_reads_blocks': col_blocks,
        'isolated_reads_between_NAs': col_isolated
    })

ndhB_init_fin=pd.concat([ndhB, init_fin_reads], axis=1)
print(ndhB_init_fin)

if (ndhB_init_fin.iloc[:, -1]).all() == 0:
    print(f"\n Between NAs there is no isolated reads, i.e. in each row there is a unique block with consecutive reads (either at the beginning of the row or at the end of the row)")
else:
    print(f"\n There exist isolated reads between NAs:\n")
    print(ndhB_init_fin[ndhB_init_fin.iloc[:, -1]])


col_init, col_final_read_block, col_fin, col_blocks, col_isolated = utils.count_NA_blocks(ndhD)

init_fin_reads=pd.DataFrame(
    {
        'initial_read': col_init,
        'final_read_block': col_final_read_block,
        'final_read': col_fin,
        'consec_reads_blocks': col_blocks,
        'isolated_reads_between_NAs': col_isolated
    })

ndhD_init_fin=pd.concat([ndhD, init_fin_reads], axis=1)
print(ndhD_init_fin)

if (ndhD_init_fin.iloc[:, -1]).all() == 0:
    print(f"\n Between NAs there is no isolated reads, i.e. in each row there is a unique block with consecutive reads (either at the beginning of the row or at the end of the row)")
else:
    print(f"\n There exist isolated reads between NAs:\n")
    print(ndhD_init_fin[ndhD_init_fin.iloc[:, -1]])





##########################################
#----- Visualising the data sets
##########################################

# Coding the data sets

ndhB_global = ndhB.replace(code_global)
ndhD_global = ndhD.replace(code_global)

ndhB_binary = ndhB.replace(code_binary)
ndhD_binary = ndhD.replace(code_binary)


# Visualisation of the raw data 

# utils.visu_data(ndhB_global, color_global)
# utils.visu_data(ndhB_binary, color_binary)

# utils.visu_data(ndhD_global, color_global)
# utils.visu_data(ndhD_binary, color_binary)


val_code = [1, 0] # values to code the entries of the variables in order to use the the function 'reorder_NA_sides()'

block_NA_right, data_without_NA_right = utils.reorder_NA_sides(ndhB_binary, val_code, 'right') # get blocks with NA at the end of the rows
block_NA_left, data_without_NA_left_right = utils.reorder_NA_sides(data_without_NA_right, val_code, 'left') # get blocks with NA at the beginning of the rows 
block_NA_left_right, data_only_NA = utils.reorder_NA_rl(data_without_NA_left_right) # get blocks with NA both at the beginning and at the end of the rows

ndhB_sorted = pd.concat([block_NA_right, block_NA_left, block_NA_left_right, data_only_NA], ignore_index=True)
ndhB_sorted.to_csv(data_folder/f'ndhB_sorted.csv')

block_NA_right, data_without_NA_right = utils.reorder_NA_sides(ndhD_binary, val_code, 'right') # get blocks with NA at the end of the rows
block_NA_left, data_without_NA_left_right = utils.reorder_NA_sides(data_without_NA_right, val_code, 'left') # get blocks with NA at the beginning of the rows 
block_NA_left_right, data_only_NA = utils.reorder_NA_rl(data_without_NA_left_right) # get blocks with NA both at the beginning and at the end of the rows

ndhD_sorted = pd.concat([block_NA_right, block_NA_left, block_NA_left_right, data_only_NA], ignore_index=True)
# ndhD_sorted.to_csv(data_folder/f'ndhD_sorted.csv')


# Visualisation of the sorted data 

# utils.visu_data(ndhB_sorted, color_binary)
# utils.visu_data(ndhD_sorted, color_binary)





# Visualisation of the global data

# val_code = ['ed']
# block_NA_right, data_without_NA_right = reorder_NA_sides(ndhB_global, val_code, 'right') # get blocks with NA at the end of the rows
# block_NA_left, data_without_NA_left_right = reorder_NA_sides(data_without_NA_right, val_code, 'left') # get blocks with NA at the beginning of the rows 
# block_NA_left_right, data_only_NA = reorder_NA_rl(data_without_NA_left_right) # get blocks with NA both at the beginning and at the end of the rows

# ndhB_sorted_glob = pd.concat([block_NA_right, block_NA_left, block_NA_left_right, data_only_NA], ignore_index=True)
# ndhB_sorted_glob.to_csv(data_folder/f'ndhB_sorted_glob.csv')