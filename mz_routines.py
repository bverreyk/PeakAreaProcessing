# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:13:07 2024

@author: bertve
"""

import numpy as np

###########################
#### Cluster selection ####
###########################
def unpack_df_clusters(df_clusters):
    peaks = df_clusters.index
    limits = [[df_clusters.cluster_min[index], df_clusters.cluster_max[index]] for index in df_clusters.index]
    
    return peaks, limits

def get_selected_fFWHM(df_clusters,resolving=4000,fFWHM=1.):
    '''
    resolving: resolving power of the instrument
    fraction of full width half maximum (based on resolving power) above which clusters are cut from the selection
    '''
    peaks, limits = unpack_df_clusters(df_clusters)
    
    speaks = [] # Selected peaks
    slimits = [] # Selected limits
    for pi in np.arange(len(peaks)):
        [ll, ul] = limits[pi]
        peak = peaks[pi]
        if (ul-ll) <= fFWHM*(peak/resolving):
            # cluster width is smaller than expected FWHM according to resolving power, accept the cluster
            speaks.append(peak)
            slimits.append([ll, ul])

    df_clusters['stable'] = [peak in speaks for peak in peaks]

    return df_clusters

def manually_add_clusters(df_cluster, mz_exact=[21.020,38.033],how='inCluster',tol = 0.001):
    if how not in ('inCluster','nearest'):
        print('Error, method not recognised, defaulting to inCluster')
        raise ValueError
    
    df_cluster['selected'] = df_cluster['stable']

    for mz in mz_exact:
        matched = False
        i = np.argmin(abs(df_cluster.index-mz))
        
        if how == 'inCluster':
            # check nearest and neighbouring clusters
            for j in [0,-1,1]:
                ll, ul = df_cluster.cluster_min.iloc[i+j], df_cluster.cluster_max.iloc[i+j]
                if (mz <= ul) and (mz >= ll):
                    idx = df_cluster.index[i+j]
                    df_cluster.loc[idx, 'selected'] = True
                    matched = True
                    break

        elif how == 'nearest':
            matched = (abs(df_cluster.index[i]-mz) <= tol)
            idx = df_cluster.index[i]
            df_cluster.loc[idx,'selected'] = matched
        
        if not matched:
            print('Warning: mz {} has not been matched to any cluster.'.format(mz))
        
    added = df_cluster.index.values[df_cluster.stable != df_cluster.selected]
    print('Cluster(s) at {} was(/were) not stable but has(/have) been added.'.format(added))
        
    return df_cluster

def select_mz_cluster_from_exact(mz_exact,df_clusters,tol=0.005,mute=False):
    peaks, limits = unpack_df_clusters(df_clusters)
    
    i = np.argmin(abs(np.array(peaks)-mz_exact))
    mz_cluster = peaks[i]
    [min_cluster, max_cluster] = limits[i]
    if (mz_exact > max_cluster) or (mz_exact < min_cluster):
        if not mute:
            print('WARNING: Exact mass {} not in cluster limits.'.format(mz_exact))
        if abs(mz_cluster - mz_exact) >= tol:
            if not mute:
                print('WARNING: Cluster at exact mass {} outside tolerance! Returning NaN!'.format(mz_exact))
            mz_cluster = np.NaN
    
    return mz_cluster

def select_mz_clusters_from_exact(mz_exact,df_clusters,tol=0.005,mute=False):
    '''
    Get selected columns from peaks/clusters using a list of exact mz values
    '''
    selected = []
    for mz in mz_exact:
        selected.append(select_mz_cluster_from_exact(mz,df_clusters,tol=tol,mute=mute))

    return selected

#####################################################################
## Do not consider clusters but only the exact mz in the dataframe ##
#####################################################################
def select_mz_col_from_exact(mz_col,mz_exact,eps=0.005,verbose=True):
    '''
    Select column in description array from time_series output obtained out of IDA based on series of exact m/z values
    Note that the method looks for matches within an m/z range (eps), if a match is unique the column is selected.
    '''
    sel = np.where(np.abs(mz_col-mz_exact)<=eps,True,False)
    if sel.sum() > 1:
        if verbose:
            print('ERROR: More than one one experimental mz value within range:')
            print('Exact mz = {} not selected, returned nan instead'.format(mz_exact))
        selected = np.nan
        
    elif sel.sum() == 0:
        if verbose:
            print('ERROR: No experimental mz value within range:')
            print('Exact mz = {} not selected, returned nan instead'.format(mz_exact))
        selected = np.nan
        
    else:
        selected = mz_col[np.where(sel==1)[0][0]]
        
    return selected

def select_mz_cols_from_exact(mz_col,mz_exact,eps=0.005,verbose=True):
    '''
    Select columns in description array from time_series output obtained out of IDA based on series of exact m/z values
    Note that the method looks for matches within an m/z range (eps), if a match is unique the column is selected.
    '''
    selected = []
    for mz in mz_exact:
        selected.append(select_mz_col_from_exact(mz_col,mz,eps=eps,verbose=verbose))

    return selected
