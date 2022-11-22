# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:41:59 2022

@author: artem belopolsky

SPLOT re-analysis for Vehlen, Belopolsky & Domes https://osf.io/x92ds/
"""




import pandas as pd
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42  # necessary to be able to edit figure text in Illustrator


from splot_utils import smooth_sqwave, extract_conditions, two_sample_ttest, \
                        two_sample_permutation


#================================================================================================s
# Read and organize the data in a dataframe
#================================================================================================
WHERE= 'VU'

if WHERE == 'VU':
    path_to_data = 'D:/'
else:
    path_to_data = 'C:/Users/artem'
    
path_to_data = os.path.join(path_to_data, 'Dropbox/papers_to_write/VehlenBelopolskyDomes/SPLOT/Data')

dfs = []
for aoi in os.listdir(path_to_data):
    for condition in os.listdir(os.path.join(path_to_data, aoi)):
        for fname in os.listdir(os.path.join(path_to_data, aoi, condition)):
            subj_nr = int(fname.split('_')[1])
            df = pd.read_csv(os.path.join(path_to_data, aoi, condition, fname), sep=';')
            df = df.T
            df['sqwave'] = [x for x in df[df.columns].to_numpy(dtype=float)]
            df = df[['sqwave']]
            df['condition'] = condition
            df['area_of_interest'] = aoi
            df['subject_nr'] = subj_nr            
            
            
            dfs.append(df)

df = pd.concat(dfs)
df = df.reset_index()
df = df.sort_values('subject_nr')

#================================================================================================
# Smooth SQUARE WAVES for every trial
#================================================================================================

df = smooth_sqwave(df, depend_var='sqwave',sigma=12)


#================================================================================================
# Select which AOI you want to analyse
#================================================================================================

df = df[df.area_of_interest == 'Face']


#================================================================================================
# Extracting and plotting data for Cond1 and Cond2 
#================================================================================================

cond1, cond2 = extract_conditions(df, depend_var='proportion', ind_var='condition', 
                               conditions=['Cond1', 'Cond2'], to_plot='yes', 
                               per_subj=False, title='Face')

#================================================================================================
# Statistics for COND1 and COND2 conditions
#================================================================================================    
#
# Two-sample t-test for COND1 and COND2, cluster sizes and optionally add errobars 
#                                                             and clusters to the existing figure
#
#================================================================================================  


clusters_rep_mis = two_sample_ttest(cond1, cond2, confidence=0.975, color1='#5e3c99',\
                color2='#e66101',to_plot='yes', cond_names= ['negative', 'positive'])

#================================================================================================ 
#
# Permutation testing: Two-sample for COND1 vs COND2, plots cluster distribution, returns 95th percentile
#
#================================================================================================  
print('\n Starting permutation for TWO-SAMPLE tests: cond1 vs cond2...\n')

cutoff_2sampl, clusters_all = two_sample_permutation(df, num_perm=1000, obs_clusters=clusters_rep_mis, 
                                       label_to_shuffle='condition', \
                                       to_plot='yes', title='Negative vs Positive')                 



