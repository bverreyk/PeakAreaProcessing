# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:28:03 2024

@author: bertve
"""
import numpy as np
import pandas as pd
import datetime as dt

import h5py
try:
    from . import mask_routines as msk_r
except:
    import mask_routines as msk_r

##########################
## IDA analysis objects ##
##########################
class IDA_data(object):
    '''IDA analysis output data object.'''
    def __init__(self, f_hdf5, dict_misc, dict_reaction, dict_dataCollection, tz_info = dt.timezone.utc):
        self.hdf5 = f_hdf5
        
        self.data = None
        self.data_units = None
        
        self.ptr_reaction = None
        self.dataCollection = None
        self.ptr_misc = None
        if not any([label in ('masks','P_inlet') for label in dict_misc.keys()]):
            print('dict_misc not containing all required labels')
            raise ValueError
        
        self.dict_misc = dict_misc
        self.dict_reaction = dict_reaction
        self.dict_dataCollection = dict_dataCollection
        
        self.tz_info = tz_info
    
    def get_PTR_data_for_processing(self):
        if self.data is None:
            self.init_data()
        
        
        df_data = self.data
        data_description = self.data_description
        data_units = self.data_units
        
        if self.ptr_reaction is None:
            self.init_ptr_reaction()
        
        for to_get in ['P_drift', 'T_drift', 'U_drift']:
            id_method = 'exact'
            if 'match' in self.dict_reaction[to_get].keys():
                id_method = self.dict_reaction[to_get]['match']
            
            
            if id_method == 'exact':
                match = self.dict_reaction[to_get]['column']
            elif id_method == 'StartsWith':
                reaction_keys = self.ptr_reaction.keys()
                for key in reaction_keys:
                    if not key.startswith(self.dict_reaction[to_get]['column']):
                        continue
                    match = key
                    break
                    
            if to_get == 'P_drift':
                correction = 1.
                if not self.dict_reaction[to_get]['units'] == 'mbar':
                    print('Correction to P_drift units needed')
                    raise ValueError
                df_P_drift = self.ptr_reaction[match]*correction
            elif to_get == 'U_drift':
                if not self.dict_reaction[to_get]['units'] == 'V':
                    print('Correction to U_drift units needed')
                    raise ValueError
                df_U_drift = self.ptr_reaction[match]
            elif to_get == 'T_drift':
                if not self.dict_reaction[to_get]['units'] == 'degrees C':
                    print('Correction to T_drift units needed')
                    raise ValueError
                
                df_T_drift = self.ptr_reaction[match]
        
        if self.ptr_misc is None:
            self.init_ptr_misc()
            
        # correct units
        correction = 1.
        if self.dict_misc['P_inlet']['units'] == 'hecto Torr':
            correction = 1.e2
            
        if not self.dict_misc['P_inlet']['units'] in ['hecto Torr','Torr']:
            print('Expected P_inlet to be in Torr or a unit covered for conversion to Torr in the PAP')
            raise ValueError
        
        df_P_inlet = self.ptr_misc[self.dict_misc['P_inlet']['column']]*correction
        
        masks = self.get_masks()
        
        sst, sst_units = self.get_single_spec_time()
        
        return df_data, data_description, data_units, sst, sst_units, df_P_drift, df_U_drift, df_T_drift, df_P_inlet, masks
    
    def get_single_spec_time(self):
        if self.dataCollection is None:
            self.init_dataCollection()
        
        to_get = self.dict_dataCollection['single_spec_time']['column']
        units  = self.dict_dataCollection['single_spec_time']['units']
        match  = self.dict_dataCollection['single_spec_time']['match']
        
        if match == 'exact':
            sst = self.dataCollection[to_get]
        else:
            print('Error, single spectrum time match method not set, returning NaN')
            sst = np.NaN
            
        return sst, units
        
        
    def get_dataframe_from_hdf5(self, group):
        '''
        Read groups from IDA output files organised in hdf5 format into pandas dataframe with time as index and the description as columns.
        '''
        f = h5py.File(self.hdf5,'r')
        # Retrieve times
        times = f[group]['time']
        times = np.array(times)
        times = times.flatten()
    
        # Get the description of the columns
        col = f[group]['description']
        col = np.array(col)
        col = col.flatten()
    
        # Get the data
        data = f[group]['TS']
        data = np.array(data)
    
        # Construct dataframe
        df = pd.DataFrame(data, index=times, columns = col)
        
        # Transform index to datetime
        df['time'] = pd.to_datetime(df.index-719529.0, unit='D')
        df.set_index('time',inplace=True)
        df.index = df.index.tz_localize(self.tz_info)
        f.close()
        
        return df

    def get_mz_exact(self,source='file'):
        if source == 'file':
            f = h5py.File(self.hdf5,'r')
            str_mz_exact = f['time_series']['description'][:]
            mz_exact = []
            for mz_string in str_mz_exact:
                mz_exact.append(float(mz_string.split()[1]))
            f.close()
        elif source == 'self':
            if self.data is None:
                self.init_data()
            mz_exact = self.data.columns
            
        return mz_exact
    
    def get_renaming(self, df_clusters, double_peaks='sum'):
        mz_exact = self.get_mz_exact(source='self')
        
        renaming = {}
        for mz_col in mz_exact:
            diff = abs(mz_col-df_clusters.index)
            i = np.argmin(diff)
            cluster_label = df_clusters.index[i]
            cluster_min, cluster_max = df_clusters.iloc[i].cluster_min, df_clusters.iloc[i].cluster_max
            if ((mz_col >= cluster_min) & (mz_col <= cluster_max)):
                renaming[mz_col] = cluster_label
                
        return renaming
    
    def rebase_to_mz_clusters(self,df_clusters,double_peaks='sum'):
        ## Handle double peaks in clusters
        if not double_peaks in ('drop', 'sum'):
            print('WARNING: method to handle double peaks not recognised. Default to drop.')
            double_peaks = 'drop'

        renaming = self.get_renaming(df_clusters,double_peaks)
        
        self.data.drop(columns=[col for col in self.data if col not in renaming.keys()],inplace=True)
        self.data.rename(columns=renaming,inplace=True)
        
        if double_peaks == 'drop':
            self.data = self.data.loc[:, ~self.data.columns.duplicated(),]
                
        elif double_peaks == 'sum':
            # groupby axis = 1 will be depreciated, group the transposed and transpose after
            #self.data = self.data.groupby(by=self.data.columns,axis=1).sum()            
            self.data = self.data.T.groupby(by=self.data.columns).sum().T
            
        else:
            print('Error: method not recognised')
            
        return None
    
    def get_time_axis(self):
        f = h5py.File(self.hdf5,'r')
        # Retrieve times
        times = f['time_series']['time']
        times = np.array(times)
        times = times.flatten()
            
        times = pd.to_datetime(times-719529.0, unit='D').tz_localize(self.tz_info)
        
        f.close()
        
        return times

    def init_data(self):
        self.data = self.get_dataframe_from_hdf5('time_series')
        # Transform the column names to mz/ratios
        self.data.columns = self.get_mz_exact(source='file')
        
        self.data_description = 'Peak Area' # Peak Area
        self.data_units = 'cps' # counts per second
        return None
    
    def init_ptr_misc(self):
        self.ptr_misc = self.get_dataframe_from_hdf5('PTR-Misc')
        return None

    def init_ptr_reaction(self):
        self.ptr_reaction = self.get_dataframe_from_hdf5('PTR-Reaction')
        return None
    
    def init_dataCollection(self):
        self.dataCollection = self.get_dataframe_from_hdf5('DataCollection')
        return None
        
    def get_masks(self):
        if self.ptr_misc is None:
            self.init_ptr_misc()
        
        if not all([x in self.dict_misc['masks'].keys() for x in ['column', 'invalid', 'calib', 'zero']]):
            print('ERROR: expected at least column, invalid, calib, and zero keys in dict_misc')
            raise ValueError
        
        column = self.dict_misc['masks']['column']
        tmp = self.dict_misc['masks'].copy()
        tmp.pop('column')
        masks = {}
        for key, val in tmp.items():
            masks[key] = msk_r.get_mask(self.ptr_misc, column, val)
            
        # If the column is NaN we assume the measurement is invalid
        masks['invalid'] = masks['invalid'] | self.ptr_misc[column].isna()
        
        del [tmp]
        
        return masks
    
    def contains_calib(self,threshold=10):
        if self.ptr_misc is None:
            self.init_ptr_misc()
        
        masks = self.get_masks()

        return masks['calib'].sum() >= threshold
    
    def get_calib_period(self):
        masks = self.get_masks()
        if self.data is None:
            t_ax = self.get_time_axis()
        else:
            t_ax = self.data.index
            
        t_start = t_ax[masks['calib']].min()
        t_stop  = t_ax[masks['calib']].max()
        
        return t_start, t_stop
        
            