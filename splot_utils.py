# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:47:57 2022

@author: Artem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel, ttest_ind, sem, t, ttest_1samp, percentileofscore


 #================================================================================================
 # Smoothing each trial in df
 #================================================================================================

def smooth_sqwave(df, depend_var='counts2', sigma=100):
    
    start_time = time.process_time()

   
                
    ys = []
    for trial in df.index:
        y = gaussian_filter1d(df[depend_var][trial], sigma=sigma) # smooth each trial
        ys.append(y) # make a list of all smoothed trials
                      
    print("Finished smoothing, time elapsed: ", time.perf_counter() - start_time)
        
    # Put the smoothed data in the data frame
    df['proportion'] = pd.Series(ys, index=df.index)    
    
    

    return df


#================================================================================================
# Extracting and plotting data for Cond1 and Cond2 
#================================================================================================
def extract_conditions(df, depend_var, ind_var, conditions, design='paired', to_plot='no', per_subj=True, title=''):
    
    print(f'\nExtracting conditions {conditions} and averaging over trials and then over participants...')       


    cond1 = []
    cond2 = []
    
    assert(len(conditions) == 2)
    
    
    if design == 'paired':
        for subj_nr in df.subject_nr.unique():
            cond1_df = df[(df.subject_nr == subj_nr) & (df[ind_var] == conditions[0])] # select the right slice of df
            cond1.append(np.vstack(cond1_df[depend_var]).mean(axis=0)) # accumulating across trials
            cond2_df = df[(df.subject_nr == subj_nr) & (df[ind_var] == conditions[1])] # select the right slice of df
            cond2.append(np.vstack(cond2_df[depend_var]).mean(axis=0)) # accumulating across trials
            
    elif design == 'independent':
        for name, group in df.groupby(df.subject_nr):
            if (group.condition == conditions[0]).all():
                cond1.append(group.proportion.mean())
            elif (group.condition == conditions[1]).all():
                cond2.append(group.proportion.mean()) 
    
    cond1 = np.array(cond1)
    cond2 = np.array(cond2) 
    
    print('Plotting time-courses...')
    
    # Plotting
    fig, ax = plt.subplots()
    
    
    ax.set_title(title, fontsize=35)
    ax.plot(cond1.mean(axis=0), color='#5e3c99', label= conditions[0])
    ax.plot(cond2.mean(axis=0), color='#e66101', label= conditions[1])    
    ax.legend(fontsize=32, frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.set_ylabel('Look probability', fontsize=32)
    ax.set_xlabel('Time (ms)', fontsize=32)
    ax.set_ylim((0,1))
    
    
    return cond1, cond2


#=========================================================================================================
# Two sample t-test, with optional plotting of error bars and clusters
# print cluster strengths       
#=========================================================================================================
def two_sample_ttest(data1, data2, ttest_type='paired', confidence=0.975, bonferroni=False, to_plot='no', color1='#2c7bb6', color2='#d7191c', cond_names=[]):
    """
    Computes two-sample t-test against the overall mean
       
        
    Parameters
    ----------
    data : 2D numpy array of shape (Nsubj x Timepoints)
     
    Returns
    -------      
    
    """
                       
    alpha_level = (1 - confidence)*2
    if bonferroni == True:
        alpha_level = alpha_level/data1.shape[1]
        confidence = (1-alpha_level)
        dot_size = 100
        
    N = data1.shape[0]
    
    if ttest_type == 'paired':
        stats = ttest_rel(data1, data2)
        sems = sem(data1-data2)/2.
    elif ttest_type == 'independent':
        stats = ttest_ind(data1, data2)
        sems = np.sqrt(sem(data1)**2 + sem(data2)**2)
        
    stats = np.array(stats)
    
    
    sems = np.array(sems).T 
    CIs = sems.astype('float') * (t.ppf((1 + confidence) / 2., N-1))
    
    t_stats = stats[0] # extract t-values
    t_stats = t_stats.T
       
    sign_mask = stats[1] < alpha_level  # extract p-values and make a mask of significant values
    sign_mask = sign_mask.T
    
    # Split into significant clusters
    # https://stackoverflow.com/questions/43385877/efficient-numpy-subarrays-extraction-from-a-mask
    clusters_strength = []    
    clusters = np.split(t_stats, np.flatnonzero(np.diff(sign_mask)) + 1)[1 - sign_mask[0]::2]
    for i in range(len(clusters)):
        clusters_strength.append(np.abs(clusters[i]).sum())
    print(f'Cluster strength(s) as sum of t-values for 2sample {ttest_type} t-test' + str(cond_names) + ' : ', str(clusters_strength))
        
    if to_plot == 'yes': 
        print('Adding error-bars and cluster sizes to the existing figure...')
        #for i in np.arange(idx.size):
        x = np.arange(data1.shape[1])   
        
        if bonferroni == True:
                                                
            plt.errorbar(x, data1.mean(axis=0), yerr=CIs, elinewidth=1, capsize=5, color=color1, fmt='o', mec='white')    
            plt.errorbar(x, data2.mean(axis=0), yerr=CIs, elinewidth=1, capsize=5, color=color2, fmt='o', mec='white')
            
            plt.scatter(x[sign_mask], data1.mean(axis=0)[sign_mask], color=color1, s=dot_size)  
            plt.scatter(x[sign_mask], data2.mean(axis=0)[sign_mask], color=color2, s=dot_size)
           
        else:
            plt.scatter(x[sign_mask], data1.mean(axis=0)[sign_mask], color=color1)  
            plt.scatter(x[sign_mask], data2.mean(axis=0)[sign_mask], color=color2)
            
            plt.fill_between(x, data1.mean(axis=0)-CIs, data1.mean(axis=0)+CIs,
                alpha=0.2, edgecolor=color1, facecolor=color1,
                linewidth=1, antialiased=True)
            
            plt.fill_between(x, data2.mean(axis=0)-CIs, data2.mean(axis=0)+CIs,
                alpha=0.2, edgecolor=color2, facecolor=color2,
                linewidth=1, antialiased=True)      
        
    return clusters_strength    

#=========================================================================================================
# 
# Two-sample paired permutation test 
#
#=========================================================================================================
def two_sample_paired_permutation(df, label_to_shuffle, depend_var='proportion', num_perm=1000, 
                                  obs_clusters=None, to_plot='no', title='', log_yaxis=True):
    """
    Only paired test permutation for now.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    label_to_shuffle : TYPE
        DESCRIPTION.
    depend_var : TYPE, optional
        DESCRIPTION. The default is 'proportion'.
    num_perm : TYPE, optional
        DESCRIPTION. The default is 1000.
    obs_clusters : TYPE, optional
        DESCRIPTION. The default is None.
    to_plot : TYPE, optional
        DESCRIPTION. The default is 'no'.
    title : TYPE, optional
        DESCRIPTION. The default is ''.
    log_yaxis : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    cutoff : TYPE
        DESCRIPTION.
    clusters_all : TYPE
        DESCRIPTION.

    """
    
        
    
    start_time = time.perf_counter()
    
    cond1_perm = []
    cond2_perm = []
    labels = df[label_to_shuffle].unique()
    print(labels)
    if labels.size !=2:
        print('Use only two labels...')
    
    for perm in np.arange(num_perm):
        groups_list = []
        for name, group in df.groupby(df.subject_nr): # loop thru every subject and shuffle the labels
            np.random.shuffle(group[label_to_shuffle].values)
            groups_list.append(group) # put together all subjects again
            
        df_new = pd.concat(groups_list) # make a new data frame with labels shuffled for each subject
           
        cond1 = []
        cond2 = []
        for subj_nr in df_new.subject_nr.unique():
            
            cond1_df = df_new[(df_new.subject_nr == subj_nr) & (df_new[label_to_shuffle] == labels[0])]
            cond1.append(np.array(cond1_df[depend_var]).mean()) # average time-courses across trials
            cond2_df = df_new[(df_new.subject_nr == subj_nr) & (df_new[label_to_shuffle] == labels[1])]
            cond2.append(np.array(cond2_df[depend_var]).mean()) # average time-courses across trials
            
        
        cond1 = np.array(cond1)
        cond2 = np.array(cond2)
        
        cond1_perm.append(cond1)
        cond2_perm.append(cond2)
    
    cond1_perm = np.dstack(cond1_perm)
    cond2_perm = np.dstack(cond2_perm)
    
    print('Generating permutations took: ', time.perf_counter() - start_time)
        
    # Cluster extraction
    start_time = time.perf_counter()

    stats = ttest_rel(cond1_perm, cond2_perm)
    stats = np.array(stats)
    
    mask_nan = np.isnan(stats[0]) # get index of nan values
    stats[0][mask_nan] = 0 # replace nan t-values with zeros
    
    sems = sem(cond1_perm - cond2_perm)/2. 
    sems = np.array(sems)               
    
    t_stats = stats[0].T    
    sign_mask = stats[1] < 0.05
    sign_mask = sign_mask.T
    
    print('Paired T-test took: ', time.perf_counter() - start_time)
    
    # Split into significant clusters
    # https://stackoverflow.com/questions/43385877/efficient-numpy-subarrays-extraction-from-a-mask
    start_time = time.perf_counter()
    clusters_all = []
    for perm in range(t_stats.shape[0]):
        clusters = np.split(t_stats[perm], np.flatnonzero(np.diff(sign_mask[perm])) + 1)[1 - sign_mask[perm][0]::2]
        clusters_distrib = []
        for i in range(len(clusters)):
            clusters_distrib.append(np.abs(clusters[i].sum()))
        if len(clusters_distrib) > 0:
            clusters_all.append(np.array(clusters_distrib).max())
        else:
            clusters_all.append(t_stats[perm].max())
        
    
    print('Extracting clusters took: ', time.perf_counter() - start_time)
    
    cutoff = np.percentile(clusters_all,95)
    print('Significance Cutoff is: ', cutoff)
    
    if obs_clusters:
        percentile_values = [percentileofscore(clusters_all, obs_cluster) for obs_cluster in obs_clusters]
        p_values = 1 - np.array(percentile_values)/100.
        print('Cluster p-values: ', p_values)
        
    if to_plot == 'yes':
        plt.figure()        
        plt.title(title, fontsize=35)       
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.ylabel(f'Frequency (log={log_yaxis})', fontsize=32)
        plt.xlabel('Cluster strength (sum of t-values)', fontsize=32)   
        
        hist_info = plt.hist(clusters_all, log=log_yaxis)
        #cutoff = np.percentile(clusters_all,95)
        #print('Significance Cutoff is: ', cutoff)
        plt.text(0,0, 'Significance Cutoff is: ' + str(cutoff))
        
        max_freq = int(np.max(hist_info[0]))
        plt.plot(cutoff *np.ones(max_freq+1), np.arange(max_freq+1), color='red', lw=10)
        
        if obs_clusters: # plot observed cluster strengths            
            obs_clusters = [obs_cluster if obs_cluster<35000 else 35000 for obs_cluster in obs_clusters] #set a max of 35000 to fit in the plot
            [plt.plot(obs_cluster *np.ones(max_freq+1), np.arange(max_freq+1), color='black', lw=10) for obs_cluster in obs_clusters]

    return cutoff, clusters_all

#=========================================================================================================
# 
# Two-sample permutation test independent samples
#
#=========================================================================================================
def two_sample_independent_permutation(df, subj_label='subject_nr', cond_label='conditions', depend_var='proportion',
                                       num_perm=1000, obs_clusters=None, to_plot='no', title='', log_yaxis=True):
    """
    Under development: Independent test permutation

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    label_to_shuffle : TYPE
        DESCRIPTION.
    depend_var : TYPE, optional
        DESCRIPTION. The default is 'proportion'.
    num_perm : TYPE, optional
        DESCRIPTION. The default is 1000.
    obs_clusters : TYPE, optional
        DESCRIPTION. The default is None.
    to_plot : TYPE, optional
        DESCRIPTION. The default is 'no'.
    title : TYPE, optional
        DESCRIPTION. The default is ''.
    log_yaxis : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    cutoff : TYPE
        DESCRIPTION.
    clusters_all : TYPE
        DESCRIPTION.

    """
    
        
    
    start_time = time.perf_counter()
    
    cond1_perm = []
    cond2_perm = []
    cond_names = df[cond_label].unique()
    print(cond_names)
    if cond_names.size !=2:
        print('Use only two condition labels...')
    
    for perm in np.arange(num_perm):
        groups_list = []
        for name, group in df.groupby(df[cond_label]): # loop thru every condition and shuffle the subjects
            np.random.shuffle(group[subj_label].values)
            groups_list.append(group) # put together all subjects again
            
        df_new = pd.concat(groups_list) # make a new data frame with labels shuffled for each subject     
        
        cond1 = []
        cond2 = []
        for name, group in df_new.groupby(df_new.subject_nr):
            if (group.condition == cond_names[0]).all():
                cond1.append(group.proportion.mean())
            elif (group.condition == cond_names[1]).all():
                cond2.append(group.proportion.mean())  
        
        cond1_perm.append(cond1)
        cond2_perm.append(cond2)
    
    cond1_perm = np.dstack(cond1_perm)
    cond2_perm = np.dstack(cond2_perm)
    
    print('Generating permutations took: ', time.perf_counter() - start_time)
        
    # Cluster extraction
    start_time = time.perf_counter()

    stats = ttest_ind(cond1_perm, cond2_perm)
    stats = np.array(stats)
    
    mask_nan = np.isnan(stats[0]) # get index of nan values
    stats[0][mask_nan] = 0 # replace nan t-values with zeros
    
    sems = np.sqrt(sem(cond1_perm)**2 + sem(cond2_perm)**2)
    sems = np.array(sems)               
    
    t_stats = stats[0].T    
    sign_mask = stats[1] < 0.05
    sign_mask = sign_mask.T
    
    print('Paired T-test took: ', time.perf_counter() - start_time)
    
    # Split into significant clusters
    # https://stackoverflow.com/questions/43385877/efficient-numpy-subarrays-extraction-from-a-mask
    start_time = time.perf_counter()
    clusters_all = []
    for perm in range(t_stats.shape[0]):
        clusters = np.split(t_stats[perm], np.flatnonzero(np.diff(sign_mask[perm])) + 1)[1 - sign_mask[perm][0]::2]
        clusters_distrib = []
        for i in range(len(clusters)):
            clusters_distrib.append(np.abs(clusters[i].sum()))
        if len(clusters_distrib) > 0:
            clusters_all.append(np.array(clusters_distrib).max())
        else:
            clusters_all.append(t_stats[perm].max())
        
    
    print('Extracting clusters took: ', time.perf_counter() - start_time)
    
    cutoff = np.percentile(clusters_all,95)
    print('Significance Cutoff is: ', cutoff)
    
    if obs_clusters:
        percentile_values = [percentileofscore(clusters_all, obs_cluster) for obs_cluster in obs_clusters]
        p_values = 1 - np.array(percentile_values)/100.
        print('Cluster p-values: ', p_values)
        
    if to_plot == 'yes':
        plt.figure()        
        plt.title(title, fontsize=35)       
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.ylabel(f'Frequency (log={log_yaxis})', fontsize=32)
        plt.xlabel('Cluster strength (sum of t-values)', fontsize=32)   
        
        hist_info = plt.hist(clusters_all, log=log_yaxis)
        #cutoff = np.percentile(clusters_all,95)
        #print('Significance Cutoff is: ', cutoff)
        plt.text(0,0, 'Significance Cutoff is: ' + str(cutoff))
        
        max_freq = int(np.max(hist_info[0]))
        plt.plot(cutoff *np.ones(max_freq+1), np.arange(max_freq+1), color='red', lw=10)
        
        if obs_clusters: # plot observed cluster strengths            
            obs_clusters = [obs_cluster if obs_cluster<35000 else 35000 for obs_cluster in obs_clusters] #set a max of 35000 to fit in the plot
            [plt.plot(obs_cluster *np.ones(max_freq+1), np.arange(max_freq+1), color='black', lw=10) for obs_cluster in obs_clusters]

    return cutoff, clusters_all