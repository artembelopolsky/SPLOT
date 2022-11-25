# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:46:25 2022

@author: artem

SPLOT for simulated independent samples data
"""



import pandas as pd
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42  # necessary to be able to edit figure text in Illustrator


from splot_utils import smooth_sqwave, extract_conditions, two_sample_ttest, \
                        two_sample_independent_permutation


#================================================================================================s
# Read and organize the data in a dataframe
#================================================================================================
WHERE= 'notVU'

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

sampling_Hz = 120. 
sigma_ms = 100.
sigma_pts = sigma_ms/(1000/sampling_Hz) 

df = smooth_sqwave(df, depend_var='sqwave',sigma=sigma_pts)


#================================================================================================
# Select which AOI you want to analyse
#================================================================================================
AOI = 'Face'
df = df[df.area_of_interest == AOI]


#================================================================================================ 
#
# Simulate a dataframe for independent samples
#
#================================================================================================  
cond1 = df[(df.condition=='Cond1') & (df.subject_nr%2 == 0)]
cond2 = df[(df.condition=='Cond2') & (df.subject_nr%2 != 0)]
df = pd.concat([cond1, cond2])


#================================================================================================
# Extracting and plotting data for Cond1 and Cond2 
#================================================================================================
cond1, cond2 = extract_conditions(df, design='independent', depend_var='proportion', ind_var='condition', 
                               conditions=['Cond1', 'Cond2'], to_plot='yes', 
                               per_subj=False, title=AOI)


#================================================================================================
# Statistics for COND1 and COND2 conditions
#================================================================================================    
#
# Two-sample independent t-test for COND1 and COND2, cluster sizes and optionally add errobars 
#                                                             and clusters to the existing figure
#
#================================================================================================  
clusters_cond1_cond2 = two_sample_ttest(cond1, cond2, ttest_type='independent', confidence=0.975, color1='#5e3c99',\
                color2='#e66101',to_plot='yes', cond_names= ['negative', 'positive'])
    

#================================================================================================ 
#
# Permutation testing: Two-sample independent for COND1 vs COND2, plots cluster distribution, returns 95th percentile
#
#================================================================================================      
cutoff_2sampl, clusters_all = two_sample_independent_permutation(df, subj_label='subject_nr', cond_label='condition', obs_clusters=clusters_cond1_cond2,
                                       num_perm=1000, to_plot='yes')
