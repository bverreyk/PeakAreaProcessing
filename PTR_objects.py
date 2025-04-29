# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:53:41 2024

@author: B. Verreyken
"""
# Date formats
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt

import os
import glob

# Computing
from sklearn import cluster

# Plotting
from matplotlib import pyplot as plt

# Local modules
try:
    from . import mz_routines as mz_r
    from . import calib_routines as calib_r
    from . import IDA_reader as IDA_reader
    from . import mask_routines as msk_r
except:
    import mz_routines as mz_r
    import calib_routines as calib_r
    import IDA_reader as IDA_reader
    import mask_routines as msk_r

__version__ = 'v2.2.8'

######################
## Support routines ##
######################
def check_create_output_dir(path):
    '''
    Check if the directory specified for output exists.
    Should only the last subdirectory in the string not exist, create it.
    '''
    exists = os.path.isdir(path)
    if not exists:
        tmp_path = ''
        split = path.split(os.sep)
        for i, subdir in enumerate(split):
            tmp_path += subdir + os.sep
            if os.path.isdir(tmp_path):
                continue
            
            if i == len(split)-2:
                os.makedirs(tmp_path)
            break
        
    return os.path.isdir(path)

def check_keys(keywords, dict_to_check, dict_name):
    for key in keywords:
        if not key in dict_to_check.keys():
            print('ERROR: Expected {} in {}'.format(key, dict_name))
            return ValueError
        
    return None

def decimal_rounder_floor(number, decimal_places=4):
    str_format = '{:.' + str(decimal_places) + 'f}'
    rounded = float(str_format.format(number))
    if rounded > number:
        rounded = rounded - 10**(-1*decimal_places)
    rounded = float(str_format.format(rounded))
    return rounded

def decimal_rounder_ceil(number, decimal_places=4):
    str_format = '{:.' + str(decimal_places) + 'f}'
    rounded = float(str_format.format(number))
    if rounded < number:
        rounded = rounded + 10**(-1*decimal_places)
    rounded = float(str_format.format(rounded))
    return rounded

def get_timezone(offset):
    if not offset.startswith('UTC'):
        return ValueError
    if len(offset) == 3:
        offset = 0
    else:
        offset = float(offset[3:])
    
    return dt.timezone(dt.timedelta(hours=offset))

######################
## Campaign objects ##
######################
class TOF_campaign(object):
    def __init__(self, name, dir_base, data_input_level, data_output_level, 
                 data_config, processing_config, calibration_config = {'calib_bottle_change':False},
                 year = None, instrument = 'TOF4000',
                 calibrations_analysis = 'integrated',
                 hdf5_files = False,
                 mz_selection = {'method':None}, time_info = {'IDA_output':'UTC','PAP_output':'UTC+1','anc_in':'UTC+1'}):
        '''
        name - string
          Contains the name of the analysis/campaign, the first subdirectory for 
          PAP to look for in the dir_base
        dir_base - string
          Contains the string used as a base for the campaign. The first directory
          PAP will look for in the base is the year
        data_input_level - string
          Level to be used as input for PAP. 
          Can be "archived", use the df_file_list_info provided in the output log directory
          to generate the full processing. Used in case IDA analysis has been redone and we
          need to combine data from different levels to obtain a full analysis.
        data_output_level - string
          Level to be used to store PAP output
        data_config -  dictionary
          contains all the variables used by the IDA reader to construct an IDA data object.
        processing_config - dictionary
          acc_interval - string
            sting to be read by resamplers to accumulate data from the IDA output
          acc_interval_calib
            string to be read by resampler to accumumate data to process calibrations
          origin - string
            origin parameter to be read by resampler, use 'start_day'
          offset - dt.timedelta
            offset parameter to be used by resampler
          processing_interval - dt.timedelta
            interval used to process one slap of data.
            Make sure to have at least 1 interval of zero measurements during this time.
            Make sure that the processing interval is a multiple of output interval.
          output_interval - dt.timedelta
            Period that is covered in 1 output interval. Make sure that the processing
            interval is a multiple of output interval.
          tdelta_buf_zero - dt.timedelta
            buffer used at the end of the zero measurements that is dropped for clculations
          tdelta_avg_zero - dt.timedelta
            Averaging period used to calculate the zero value for the zero measurement
            If tdelta_buf_zero + tdelta_avg_zero is longer than the zero measurement, the
            zero measurement is not taken into account.
          tdelta_min_zero - dt.timedelta
            minimum period for zero measurements to last for being taken into account
          tdelta_buf_calib - dt.timedelta
            buffer used at the end of the calibration measurements that is dropped for clculations
          tdelta_avg_calib - dt.timedelta
            Averaging period used to calculate the calibration value for the zero measurement
            If tdelta_buf_zero + tdelta_avg_zero is longer than the calib measurement, the
            calib measurement is not taken into account.
          tdelta_min_calib - dt.timedelta
            minimum period for calibration measurements to last for being taken into account
          tdelta_stability - dt.timedelta
            interval prior to tdelta_buf_calib to check stability of the signal. To be used
            as a check to see if the calibration is stable or not.
          rate_coeff_col_calib
            column in the anciliry calibration table to be used for calculating the transmission
            information. k_ionicon [1.e-9 cm3 molecule-1 s-1] is used as a standard. Note that
            the units are assumed to be 1.e-9 cm3 molecule-1 s-1            
          k_reac_default
            If no k_reac is known, use this value as a default value.
            If the k_reac is known, put it in the cluster table after clustering and before processing
            the campaign.
          Xr0_default
            The Xr0 that is used as a default. If not default, put it in the cluster table after clustering
            and before processing the campaign.
          tdelta_trim_start - dt.timedelta
            time after any switch from the valve signal where data are discarded
          tdelta_trim_stop - dt.timedelta
            time before any switch fron the valve signal where data are discarded
          precision_calc - string
            Poisson - Assumes raw counts to be poisson distributed and precision is calculated from square root
            Distribution - Only possible with resampling, uses the std during the averaging interval as a measure of uncertainty
        year
          None - Look for all year in the <base_dir>/<name> file
                 CURRENTLY NOT SUPPORTED
          integer - Containing the year
          array - containing multiple years (string!)
                  CURRENTLY NOT SUPPORTED
        instrument - string
          Name of the instrument, used in the construction of the sting where
          PAP will read/write the input/output
        calibrations_analysis 
          "integrated" - calibrations should be found in between the regular
                         data. (Default)
          "dedicated" - The calibrations have been processed with dedicated IDA
                        nearby zero measurements.
        mz_selection - dictionary
          "method",
            None - use exact mz values in files. Not compatible with nesting
                   between different IDA analysis. (Default)
            "cluster" - use cluster dataframe containing the average mz value of
                       the cluster together with upper and lower boundaries.
                       In case cluster is passed, additional inforation is 
                       required to see if we shoud read cluster information or 
                       perform the clustering for the current object.
            "exact" - use a list of exact values and match the closest signals
                      CURRENTLY NOT SUPPORTED
        '''
        #########################
        ## CAMPAIGN ATTRIBUTES ##
        #########################
        ## Name
        self.name = name
        
        ## Instrument name
        self.instrument = instrument

        ## Base path
        if os.path.isdir(dir_base):
            self.dir_base = dir_base
        else:
            print('Base directory not found.')
            raise ValueError
        
        ## Year
        if year is None:
            year = glob.gblo('{}2???',format(self.dir_base))
            
        elif isinstance(year, list):
            year = [str(y) for y in year]
            
        elif isinstance(year, int):
            year = [str(year)]
            
        self.year = year
        
        ## Data output
        self.data_output_level = data_output_level        
        dir_o_log = '{1}Logs{0}{2}_{3}{0}'.format(os.sep,dir_base,name,data_output_level)
        if check_create_output_dir(dir_o_log):
            self.dir_o_log   = dir_o_log
        else:
            raise ValueError
        
        ## Calibration approach: integrated/dedicated
        if not calibrations_analysis in ('integrated', 'dedicated'):
            raise ValueError
        self.calibrations_analysis = calibrations_analysis
        calib_config = calibration_config.copy()
        self.calib_bottle_change = calib_config['calib_bottle_change']
        calib_config.pop('calib_bottle_change')
        self.calibration_config = calib_config

        ## Processing configuration        
        keywords = ['acc_interval',
                    'acc_interval_calib',
                    'origin',
                    'offset',
                    'processing_interval',
                    'output_interval',
                    'tdelta_buf_zero',
                    'tdelta_avg_zero',
                    'tdelta_min_zero',
                    'tdelta_buf_calib',
                    'tdelta_avg_calib',
                    'tdelta_min_calib',
                    'tdelta_stability',
                    'rate_coeff_col_calib',
                    'k_reac_default',
                    'Xr0_default',
                    'tdelta_trim_start',
                    'tdelta_trim_stop',
                    'precision_calc',
                    'cc_t_interp',
                    'tr_t_interp',
                    ]
        
        check_keys(keywords, processing_config, 'processing_config')
        self.processing_config = processing_config
        
        ## Connection m/z between files
        if not 'method' in mz_selection.keys():
            raise ValueError
            
        if not mz_selection['method'] in (None,'cluster','exact'):
            raise ValueError
        
        keywords = []
        if mz_selection['method'] == 'cluster':
            if not mz_selection['archived']:
                if not 'algorithm' in mz_selection.keys():
                    print('Specify an algorithm for clustering.')
                    raise ValueError
                if mz_selection['algorithm'] == 'DBSCAN':
                    keywords.append('eps')
                    keywords.append('min_samples')
                    keywords.append('resolving')
                    keywords.append('add exact')
            keywords.append('tolerance')
            keywords.append('double_peaks')
            
        elif mz_selection['method'] == 'exact':
            keywords.append('mz_exact')
            keywords.append('tolerance')

        check_keys(keywords, mz_selection, 'mz_selection')
        self.mz_selection = mz_selection
        self.data_config = data_config
        
        ## timing information
        keywords = ['PAP_output',
                    'IDA_output'
                    ]
        check_keys(keywords, time_info, 'time_info')
        
        time_info['tz_in'] = get_timezone(time_info['IDA_output'])
        time_info['tz_out'] = get_timezone(time_info['PAP_output'])
        time_info['tz_anc_in'] = get_timezone(time_info['anc_in'])
        
        self.time_info = time_info

        ## List input files
        full_archive = False
        self.data_input_level = data_input_level
        if self.data_input_level == 'Archived':
            full_archive = True
        
        try:
            self.df_file_list_info = self.get_files_dataframe(full_archive,hdf5_files=hdf5_files)
        except:
            print('Error in retrieving available data.')
            print('Creating hdf5_files.csv in output log')
            list_IDA = self.get_list_hdf5_IDA()
            df = pd.DataFrame(data=list_IDA,columns=['file'])
            df.to_csv('{}hdf5_files.csv'.format(self.dir_o_log))
           
        ## create clusters dataframe
        self.df_clusters = self.get_df_clusters()

    def __str__(self):
        return self.name
    
    def get_files_dataframe(self, full_archive = False, hdf5_files=False):
        '''
        Datafiles and start/end times excluding dedicated calibrations.
        '''
        f_info = '{}available_data.csv'.format(self.dir_o_log)
        try:
            df_info = pd.read_csv(f_info,index_col=0,dtype={'file':str,'calib':bool})
            df_info['t_start'] = pd.to_datetime(df_info['t_start'])
            df_info['t_stop'] = pd.to_datetime(df_info['t_stop'])
        except:
            df_info = None
            
        if not full_archive:
            print('Gather data on all files')
            if hdf5_files:
                list_hdf5 = pd.read_csv('{}hdf5_files.csv'.format(self.dir_o_log)).file
            else:
                list_hdf5 = self.get_list_hdf5_IDA()
            files = []
            t_min = []
            t_max = []
            calib = []
            for f_hdf5 in list_hdf5:
                if ((not df_info is None) and 
                    (f_hdf5 in df_info.file.values)):
                    continue
                
                files.append(f_hdf5)
                IDA_object = IDA_reader.IDA_data(f_hdf5, **self.data_config, tz_info=self.time_info['tz_in'])
                times = IDA_object.get_time_axis()
                t_min.append(times.min())
                t_max.append(times.max())
                if self.calibrations_analysis == 'integrated':
                    calib.append(IDA_object.contains_calib())
                else:
                    calib.append(False)
            
            tmp = pd.DataFrame({
                'file':files,
                't_start':t_min,
                't_stop':t_max,
                'calib':calib,
                })
            
            df_info = pd.concat([df_info,tmp],ignore_index=True)
        
        # Check overlap
        df_info.sort_values(by='t_start',inplace=True)
        entries = len(df_info.index)
        index_val = df_info.index.values
        for ind in np.arange(entries):
            if ind == entries-1:
                break
            
            t_end = df_info.loc[index_val[ind],'t_stop']
            t_next_start = df_info.loc[index_val[ind+1],'t_start']
            
            if t_next_start <= t_end:
                print('Error: overlap between files {} and {}.'.format(ind, ind+1))
                # raise ValueError

        if not full_archive:
            df_info.to_csv(f_info)
        
        try: # Try assigning time zone if not in file
            df_info['t_start'] = df_info['t_start'].dt.tz_localize(tz=self.time_info['tz_in'])
            df_info['t_stop'] = df_info['t_stop'].dt.tz_localize(tz=self.time_info['tz_in'])
        except: # If time zone already present, do nothing
            pass
        
        return df_info
    
    def get_list_hdf5_IDA(self):
        list_hdf5_ambient = []
        for year in self.year:
            base_path = '{1}{0}{2}{0}{3}{0}Nominal{0}'.format(os.sep,self.dir_base,year,self.instrument)
            if self.calibrations_analysis == 'dedicated':
                base_path = '{1}Ambient{0}'.format(os.sep,base_path)
            base_path = '{1}{2}{0}'.format(os.sep,base_path,self.data_input_level)
            
            [list_hdf5_ambient.append(f) for f in glob.glob('{1}??{0}IDA_Export_*.h5'.format(os.sep,base_path))]

        list_hdf5_ambient = np.sort(list_hdf5_ambient)
        
        return list(list_hdf5_ambient)
    
    def get_list_hdf5_IDA_calibrations(self):
        list_hdf5_calib = []
        
        if self.calibrations_analysis == 'dedicated':
            for year in self.year:
                base_path = '{1}{0}{2}{0}{3}{0}Nominal{0}'.format(os.sep,self.dir_base,year,self.instrument)
                base_path = '{1}Calibrations{0}'.format(os.sep,base_path)                
                base_path = '{1}{2}{0}'.format(os.sep,base_path,self.data_input_level)
                [list_hdf5_calib.append(f) for f in glob.glob('{}IDA_Export_*.h5'.format(base_path))]
        
        else:
            list_hdf5_calib = self.df_file_list_info[self.df_file_list_info.calib.astype(bool)].file.values
            
        list_hdf5_calib = np.sort(list_hdf5_calib)
        return list(list_hdf5_calib)
    
    def get_list_hdf5_IDA_all(self):
        list_hdf5 = self.get_list_hdf5_IDA()
        list_hdf5_calib = self.get_list_hdf5_IDA_calibrations()
        for calib in list_hdf5_calib:
            if not calib in list_hdf5:
                list_hdf5.append(calib)
        return list_hdf5
    
    def get_clustering_info(self, list_hdf5 = None):
        list_hdf5 = self.df_file_list_info.file.values
        if self.calibrations_analysis == 'dedicated':
            list_hdf5 = np.append(list_hdf5, self.get_list_hdf5_IDA_calibrations())
            
        x = []
        y = []
        
        n = []
        
        index = 0
        for f_hdf5 in list_hdf5:
            index += 1
            IDA_object = IDA_reader.IDA_data(f_hdf5, **self.data_config, tz_info=self.time_info['tz_in'])
            mz = IDA_object.get_mz_exact()
            n.append(len(mz))
            
            for mzv in mz:
                x.append(mzv)
                y.append(index)
        
        x = np.array(x)
        y = np.array(y)
        return x, y, n
    
    def get_clustering_DBSCAN(self, eps, min_samples, x = None, y = None):
        if ((x is None) or (y is None)):
            x, y, n = self.get_clustering_info()
        
        model = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(x.reshape(-1,1))
        labels = model.fit_predict(x.reshape(-1,1))
        
        df = pd.DataFrame({'m/z':x,'file ID':y,'cluster label':labels})
        
        grouped = df.groupby(by='cluster label')
        
        peaks_pdf = grouped.mean()
        limits_min = grouped.min()
        limits_max = grouped.max()
        
        limits_min['m/z'] = [decimal_rounder_floor(value,self.mz_selection['decimal_places']) for value in limits_min['m/z']]
        limits_max['m/z'] = [decimal_rounder_ceil(value,self.mz_selection['decimal_places']) for value in limits_max['m/z']]
        
        # organise peaks and limits in arrays
        peaks = []
        limits = []
        for lbl in peaks_pdf.index:
            # lbl -1 indicates noise (i.e., unclustered peak)
            if lbl == -1:
                continue

            peaks.append(peaks_pdf.loc[lbl,'m/z'])
            limits.append([limits_min.loc[lbl,'m/z'],limits_max.loc[lbl,'m/z']])

        # Sort peaks and limits according to mz values
        zipped = zip(peaks, limits)
        zipped = sorted(zipped)
        peaks, limits = zip(*zipped)
        
        df_clusters = pd.DataFrame(data=np.array(limits),columns=['cluster_min','cluster_max'],index=np.array(peaks).round(self.mz_selection['decimal_places']),dtype=float)
        
        return df_clusters
    
    def write_scan_clustering_DBSCAN(self, eps_range = None, min_samples_range = None, mode = 'show'):
        x, y, n = self.get_clustering_info()
        
        if eps_range is None:
            if self.instrument == 'TOF4000':
                eps_range = np.arange(0.001,0.01,0.001)
        
        if min_samples_range is None:
            min_samples_range = [int(max(y)*perc/100.) for perc in np.arange(10,50,10)]
        
        for min_samples in min_samples_range:
            n_clusters = []
            n_unclustered = []
            n_clusters_double = []
            min_cluster_width = []
            max_cluster_width = []
            avg_cluster_width = []
            for eps in eps_range:
                df_clusters = self.get_clustering_DBSCAN(eps, min_samples, x = x, y = y)
                peaks, limits = mz_r.unpack_df_clusters(df_clusters)
                
                widths = []
                unclustered = np.ones(len(x),dtype=bool)
                n_cluster_double = 0
                for ll, ul in limits:
                    cluster_mask = np.where(x <= ll, False, True) & np.where(x >= ul, False, True)
                    if not len(y[cluster_mask]) == len(np.unique(y[cluster_mask])):
                        n_cluster_double += 1
                    unclustered = unclustered & ~(cluster_mask)
    
                    widths.append(ul-ll)
    
                widths = np.array(widths)
                n_clusters.append(len(peaks))
                n_clusters_double.append(n_cluster_double)
                n_unclustered.append(unclustered.sum())
                min_cluster_width.append(widths.min())
                max_cluster_width.append(widths.max())
                avg_cluster_width.append(widths.mean())
    
            f, ax = plt.subplots(1,2,figsize=(15,5))
            tax = ax[0].twinx()
            tax.grid()
    
            ax[0].plot(eps_range,n_clusters,label='nclusters')
            tax.fill_between(eps_range,0,np.array(n_unclustered)/len(x)*100.,label='noise [% of total peaks]',alpha = 0.25,color='tab:red')
            tax.fill_between(eps_range,0,np.array(n_clusters_double)/len(peaks)*100.,label='double peak in cluster [% of total clusters]',alpha = 0.25,color='tab:orange')
            tax.set_ylabel('%')
            ax[0].legend()
            tax.legend()
    
            ax[1].plot(eps_range,min_cluster_width,label='min')
            ax[1].plot(eps_range,max_cluster_width,label='max')
            ax[1].plot(eps_range,avg_cluster_width,label='avg')
            ax[1].legend()
    
            plt.suptitle('min_samples {}'.format(min_samples))
            if mode == 'show':
                plt.show()
            elif mode == 'write':
                plt.savefig('{}clustering_scan_DBSCAN_minsamples{}.png'.format(self.dir_o_log,min_samples))
            
        return None
    
    def add_anc_df_clusters(self, df_clusters):
        df_clusters['k [1.e-9 cm3 molecule-1 s-1]'] = self.processing_config['k_reac_default']
        df_clusters['Xr0'] = self.processing_config['Xr0_default']
        df_clusters['FY'] = 1. # Fragmentation yield
        df_clusters['IF'] = 1. # Isotopic multiplication factor
        
        return df_clusters
    
    def get_df_clusters(self):
        if self.mz_selection['archived']:
            df_clusters = pd.read_csv('{}clusters.csv'.format(self.dir_o_log),index_col=0)
            
        
        else:
            if self.mz_selection['method'] == 'cluster':
                if self.mz_selection['algorithm'] == 'DBSCAN':
                    eps = self.mz_selection['eps']
                    min_samples = self.mz_selection['min_samples']
                    
                    df_clusters = self.get_clustering_DBSCAN(eps, min_samples, self.get_list_hdf5_IDA_all())
                    
                    resolving = self.mz_selection['resolving']
                    fFWHM = self.mz_selection['fFWHM']
                    df_clusters = mz_r.get_selected_fFWHM(df_clusters,resolving,fFWHM)
                    
                    add_exact = self.mz_selection['add exact']
                    tol = self.mz_selection['tolerance']
                    df_clusters = mz_r.manually_add_clusters(df_clusters,mz_exact=add_exact,how='nearest',tol=tol)
                    
                    self.add_anc_df_clusters(df_clusters)
                    
                    df_clusters.to_csv('{}clusters.csv'.format(self.dir_o_log))
                    self.mz_selection['archived'] = True
                else:
                    print('Only DBSCAN supported for clustering algorithm')
                    raise ValueError
                    
            elif self.mz_selection['method'] == 'exact':
                clusters = np.array(self.calibrations_analysis['exact']).round(self.mz_selection['decimal_places'])
                df_clusters = pd.DataFrame(data=clusters,columns=['cluster_min'],index=clusters,dtype=float)
                df_clusters['cluster_max'] = df_clusters.cluster_min
                df_clusters['stable'] = None
                df_clusters['selected'] = True
                
                self.add_anc_df_clusters(df_clusters)
                
            elif self.mz_selection['method'] is None:
                df_clusters = None            
            
        return df_clusters
    
    def get_PTR_data(self,t_start, t_stop, tz_info = None):
        if t_start.tzinfo is None:
            if tz_info is None:
                tz_info = self.time_info['tz_in']
            t_start = t_start.replace(tzinfo=tz_info)
            t_stop = t_stop.replace(tzinfo=tz_info)
        
        # Only selected clusters are elligable for processing
        df_clusters = self.df_clusters[self.df_clusters['selected']]
        
        tmp_data = []
        tmp_U_drift = []
        tmp_P_drift = []
        tmp_T_drift = []
        tmp_P_inlet = []
        tmp_masks = []
        tmp_sst = []
        for i, f_hdf5 in enumerate(self.df_file_list_info.file):
            if t_start > self.df_file_list_info.iloc[i].t_stop:
                continue
            if t_stop  < self.df_file_list_info.iloc[i].t_start:
                continue

            IDA_object = IDA_reader.IDA_data(f_hdf5, **self.data_config, tz_info=self.time_info['tz_in'])
            IDA_object.rebase_to_mz_clusters(df_clusters,double_peaks=self.mz_selection['double_peaks'])
            df_data, data_description, data_units, sst, sst_units, df_P_drift, df_U_drift, df_T_drift, df_P_inlet, masks = IDA_object.get_PTR_data_for_processing()
            
            t_selection = (df_data.index >= t_start) & (df_data.index <= t_stop)
            
            for key in masks.keys():
                masks[key] = masks[key][t_selection]

            df_data = df_data[t_selection]
            df_U_drift = df_U_drift[t_selection]
            df_T_drift = df_T_drift[t_selection]
            df_P_drift = df_P_drift[t_selection]
            df_P_inlet = df_P_inlet[t_selection]
            sst        = sst[t_selection]
            
            tmp_masks.append(masks)
            tmp_data.append(df_data)
            
            tmp_P_drift.append(df_P_drift)
            tmp_U_drift.append(df_U_drift)
            tmp_T_drift.append(df_T_drift)
            
            tmp_P_inlet.append(df_P_inlet)
            tmp_sst.append(sst)
            
        if len(tmp_data) >= 1:
            df_data = pd.concat(tmp_data)
            
            df_P_drift = pd.concat(tmp_P_drift)
            df_U_drift = pd.concat(tmp_U_drift)
            df_T_drift = pd.concat(tmp_T_drift)
            df_P_inlet = pd.concat(tmp_P_inlet)
            
            sst = pd.concat(tmp_sst)
        
            masks = None
            for tmp in tmp_masks:
                if masks is None:
                    masks = tmp
                    continue
                
                for key in tmp.keys():
                    # WRONG
                    masks[key] = np.concatenate((masks[key],tmp[key]))
            
            PTR_data_object = PTR_data(df_data, data_description, data_units, sst, sst_units, df_P_drift, df_U_drift, df_T_drift, df_P_inlet, masks, df_clusters)
            
        else:
            PTR_data_object = None
        
        return PTR_data_object
    
    def get_dir_o_calib(self):
        dir_o_cal = '{1}Calibrations{0}'.format(os.sep,self.dir_o_log)
        check_create_output_dir(dir_o_cal)
        return dir_o_cal
    
    def get_calibration_ancillary(self, date):
        year = self.year[0]
        print('Calibration ancilary input is expected to be located in the first year of measurements.')
        base_path = '{1}{0}{2}{0}{3}{0}Misc{0}Calibrations{0}input{0}'.format(os.sep,self.dir_base,year,self.instrument)
        if not self.calib_bottle_change:
            calibration_ancillary = pd.read_csv('{}Calibration_Ancillary.csv'.format(base_path))
            
        else:
            # YYYYMMDD_HHMM from when the bottle was put in place
            list_calib_anc = glob.glob('{}Calibration_Ancillary_????????_????.csv'.format(base_path))

            reference = None
            diff = None
            for f in list_calib_anc:
                dt_tmp = dt.datetime.strptime(f[-17:-4],'%Y%m%d_%H%M').replace(tzinfo=self.time_info['tz_in']) # datetime temporary
                td_tmp = date - dt_tmp # timedelta temporary
                if td_tmp < dt.timedelta(0):
                    continue
                    
                if reference is None:
                    reference = f
                    diff = td_tmp
                    continue
                    
                if diff > td_tmp:
                    reference = f
                    diff = td_tmp
            
            print('Calibration ancillary file: ', reference)
            calibration_ancillary = pd.read_csv(reference)
        
        calibration_ancillary.set_index('mz_exact',inplace=True)        
        return calibration_ancillary
    
    def get_archived_calibrations(self, tr_corrected=True, deconstructed=True):
        dir_o_cal = self.get_dir_o_calib()
        
        f_calibrations = '{}calibrations.csv'.format(dir_o_cal)
        f_transmission = '{}transmission.csv'.format(dir_o_cal)
        f_stability    = '{}stability.csv'.format(dir_o_cal)
        f_anc          = '{}anc.csv'.format(dir_o_cal)
        
        if os.path.exists(f_calibrations):
            df_calibrations = pd.read_csv(f_calibrations,index_col=0)
            df_transmission = pd.read_csv(f_transmission,index_col=0)
            df_stability    = pd.read_csv(f_stability,index_col=0)
            df_anc          = pd.read_csv(f_anc,index_col=0)
            
            try: # attribute tz info
                df_calibrations.ctime = pd.to_datetime(df_calibrations.ctime).dt.tz_localize(tz=self.time_info['tz_in'])
                df_transmission.ctime = pd.to_datetime(df_transmission.ctime).dt.tz_localize(tz=self.time_info['tz_in'])
                df_stability.ctime    = pd.to_datetime(df_stability.ctime).dt.tz_localize(tz=self.time_info['tz_in'])
                df_anc.ctime    = pd.to_datetime(df_anc.ctime).dt.tz_localize(tz=self.time_info['tz_in'])
            except: # tz info already provided
                df_calibrations.ctime = pd.to_datetime(df_calibrations.ctime)
                df_transmission.ctime = pd.to_datetime(df_transmission.ctime)
                df_stability.ctime    = pd.to_datetime(df_stability.ctime)
                df_anc.ctime    = pd.to_datetime(df_anc.ctime)
            
            if deconstructed:
                df_calibrations, df_calibrations_rprec, df_calibrations_racc = deconstruct_df_calibrations(df_calibrations)
            else:
                df_calibrations_rprec = None
                df_calibrations_racc  = None
            
            if tr_corrected:
                # Correct calibration factors here to obtain it in tc-ncps ppbv-1
                df_cc_tmp = df_calibrations.copy()
                drop = ['file','ctime']
                for key in df_cc_tmp.columns:
                    if key.startswith('I_cps_'):
                        drop.append(key)
                        
                # df_cc_tmp.drop(['I_cps_H3O1_21', 'I_cps_H5O2_38','file','ctime'],axis=1,inplace=True)
                df_cc_tmp.drop(drop, axis=1,inplace=True)
                df_cc_tmp.columns = [float(mz) for mz in df_cc_tmp.columns]
                
                df_tr_interp = df_transmission.drop(['file','ctime'],axis=1)
                df_tr_interp.columns = [float(mz) for mz in df_tr_interp.columns]
                for col in df_cc_tmp.columns:
                    if not col in df_tr_interp.columns:
                        df_tr_interp[col] = np.nan

                df_tr_interp.sort_index(axis=1, inplace=True)
                df_tr_interp.interpolate(method='values',axis=1,inplace=True)
                df_tr_interp.columns = [str(mz) for mz in df_tr_interp.columns]
                subset = df_tr_interp.columns.intersection(df_calibrations.columns)
                df_calibrations[subset] = df_calibrations[subset]/df_tr_interp[subset]
            
        else:
            df_calibrations       = None
            df_transmission       = None
            df_stability          = None
            df_anc                = None
            df_calibrations_racc  = None
            df_calibrations_rprec = None
        
        return df_calibrations, df_calibrations_rprec, df_calibrations_racc, df_transmission, df_stability, df_anc
    
    def archive_calibrations(self, df_calibrations, df_transmission, df_stability, df_anc):
        dir_o_cal = self.get_dir_o_calib()
        check_create_output_dir(dir_o_cal)
        
        f_calibrations = '{}calibrations.csv'.format(dir_o_cal)
        f_transmission = '{}transmission.csv'.format(dir_o_cal)
        f_stability    = '{}stability.csv'.format(dir_o_cal)
        f_anc          = '{}anc.csv'.format(dir_o_cal)

        df_calibrations.to_csv(f_calibrations)
        df_transmission.to_csv(f_transmission)
        df_stability.to_csv(f_stability)
        df_anc.to_csv(f_anc)
        
        return None
    
    def process_calibrations(self,output_calibrations=False):
        print('start process calibrations, {}'.format(self.calibrations_analysis))
        
        df_calibrations, df_calibrations_rprec, df_calibrations_racc, df_transmission, df_stability, df_anc = self.get_archived_calibrations(tr_corrected=False, deconstructed=False)
        
        list_hdf5_calib = self.get_list_hdf5_IDA_calibrations()
        for f_hdf5 in list_hdf5_calib:
            print('--------')
            print('Process calibration')
            print(f_hdf5)
            if ((not df_calibrations is None) and
                (f_hdf5 in df_calibrations.file.values)
                ):
                print('--already processed--')
                continue
            
            IDA_object = IDA_reader.IDA_data(f_hdf5, **self.data_config, tz_info=self.time_info['tz_in'])
            t_start, t_stop = IDA_object.get_calib_period()
            
            t_start = t_start - dt.timedelta(minutes=35)
            t_stop  = t_stop + dt.timedelta(minutes=15)
            
            if self.calibrations_analysis == 'integrated':
                PTR_data_object = self.get_PTR_data(t_start, t_stop)
            
            elif self.calibrations_analysis == 'dedicated':
                IDA_object.rebase_to_mz_clusters(self.df_clusters,double_peaks=self.mz_selection['double_peaks'])
                df_data, data_description, data_units, sst, sst_units, df_P_drift, df_U_drift, df_T_drift, df_P_inlet, masks = IDA_object.get_PTR_data_for_processing()
                PTR_data_object = PTR_data(df_data, data_description, data_units, sst, sst_units, df_P_drift, df_U_drift, df_T_drift, df_P_inlet, masks, self.df_clusters)
            
            PTR_data_object.resample(self.processing_config['acc_interval_calib'],self.processing_config['origin'],self.processing_config['offset'])
            # If we explicitely ask to calculate precision using the Poisson assumption, reset the df_prec here
            if self.processing_config['precision_calc'] == 'Poisson':
                PTR_data_object.set_precision_poisson()

            kwords = ['tdelta_buf_zero',
                      'tdelta_avg_zero',
                      'tdelta_buf_calib',
                      'tdelta_avg_calib',
                      'tdelta_stability',
                      'rate_coeff_col_calib',
                      'precision_calc',
                     ]
            
            dir_o_cal = self.get_dir_o_calib()
            calibration_ancillary = self.get_calibration_ancillary(t_start+((t_stop-t_start)/2))

            new_trans, new_calib, new_stability, new_anc = PTR_data_object.process_calibration(calibration_ancillary, dir_o_cal, 
                                                                                               **{key: self.processing_config[key] for key in kwords},
                                                                                               **self.calibration_config,
                                                                                               Q_ptrms_corr=self.name)
            
            if ((new_trans is None) or
                (new_calib is None) or
                (new_stability is None) or
                (new_anc is None)):
                continue

            new_trans['file']     = f_hdf5
            new_calib['file']     = f_hdf5
            new_stability['file'] = f_hdf5
            new_anc['file']       = f_hdf5
            
            # To save the calibration and transmission convert the keys to strings in order to append. If not, small fluctuation in floats result in seperate columns
            new_trans     = pd.DataFrame([{str(key): new_trans[key] for key in new_trans.keys()}])
            new_calib     = pd.DataFrame([{str(key): new_calib[key] for key in new_calib.keys()}])
            new_stability = pd.DataFrame([{str(key): new_stability[key] for key in new_stability.keys()}])
            new_anc       = pd.DataFrame([{str(key): new_anc[key] for key in new_anc.keys()}])

            if df_calibrations is None:
                df_calibrations = new_calib
                df_transmission = new_trans
                df_stability    = new_stability
                df_anc          = new_anc
                
            else:
                df_calibrations = pd.concat([df_calibrations,new_calib],ignore_index=True)
                df_transmission = pd.concat([df_transmission,new_trans],ignore_index=True)
                df_stability    = pd.concat([df_stability,new_stability],ignore_index=True)
                df_anc          = pd.concat([df_anc,new_anc],ignore_index=True)
                        
            self.archive_calibrations(df_calibrations, df_transmission, df_stability, df_anc)
            
            if output_calibrations:
                t_ref = PTR_data_object.df_data.index.min()
                tmp_calibrations, tmp_calibrations_rprec, tmp_calibrations_racc, tmp_transmission, tmp_stability, tmp_anc = self.get_archived_calibrations(tr_corrected=True, deconstructed=True)
                tmp_tr_coeff = self.get_transmissionCoefficients('interp', PTR_data_object.df_data.index, tmp_transmission, PTR_data_object.df_clusters,
                                                                t_interp = 'nearest', t_pairing = 'both')
                tmp_cc_coeff, tmp_cc_coeff_rprec, tmp_cc_coeff_racc = self.get_calibrationCoefficients(PTR_data_object.df_data.index, tmp_calibrations, 
                                                                                                    tmp_calibrations_rprec, 
                                                                                                    tmp_calibrations_racc, 
                                                                                                    PTR_data_object.df_clusters,
                                                                                                    t_interp = 'nearest',
                                                                                                    t_pairing = 'both')
                
                
                dir_o = '{1}{2}{0}{3}{0}{4}{0}{5}{0}'.format(os.sep,self.dir_base,t_ref.year,self.instrument,'Nominal',self.data_output_level)
                check_create_output_dir(dir_o)
                dir_o = dir_o+'calibrations'+os.sep
                check_create_output_dir(dir_o)
                dir_o += t_ref.strftime('%m') + os.sep
                check_create_output_dir(dir_o)

                PTR_data_object.save_output(dir_o, self.name, self.instrument, tmp_tr_coeff, tmp_cc_coeff, masks = ['zero','calib'], time_info=self.time_info, processing_config=self.processing_config)
            
            print('--done--')
            del PTR_data_object

        return None

    def get_Xr0_mz(self, df_clusters):
        mask = (df_clusters.Xr0 != self.processing_config['Xr0_default'] )
        dict_Xr0 = df_clusters[mask].Xr0.to_dict()

        return dict_Xr0
    
    def get_k_reac_mz(self, df_clusters):
        mask = (df_clusters['k [1.e-9 cm3 molecule-1 s-1]'] != self.processing_config['k_reac_default'] )
        dict_k_react_mz = df_clusters[mask]['k [1.e-9 cm3 molecule-1 s-1]'].to_dict()
        return dict_k_react_mz
    
    def get_multiplier_mz(self, df_clusters):
        multiplier = df_clusters.IF/df_clusters.FY
        mask = (multiplier != 1.)
        dict_multiplier = multiplier[mask].to_dict()
        return dict_multiplier
    
    def get_transmissionCoefficients(self, mz_function, dt_axis, df_transmission, df_clusters, mz_func = 'interp', t_interp = 'nearest', t_pairing='both'):
        df_tmp = df_transmission.copy()
        df_tmp.set_index('ctime',inplace=True)
        df_tmp.drop('file',axis=1,inplace=True)
        df_tmp.columns = [float(col) for col in df_tmp.columns]
        
        if not t_pairing in ('both','before','after'):
            raise ValueError
        if t_pairing == 'before':
            df_tmp = df_tmp[df_tmp.index <= dt_axis.max()]
        elif t_pairing == 'after':
            df_tmp = df_tmp[df_tmp.index >= dt_axis.min()]
        
        if mz_func == 'interp':
            if t_interp == 'nearest':
                idx = abs(df_tmp.index-dt_axis.mean()).argmin()
                
                dict_tr_clusters = {}
                for mz in df_tmp.columns:
                    mz_col = mz_r.select_mz_cluster_from_exact(mz, df_clusters, tol = 0.01, mute = True)
                    if np.isnan(mz_col):
                        continue
                    dict_tr_clusters[mz_col] = df_tmp.iloc[idx][mz]
                
                for key in df_clusters[df_clusters.selected].index:
                    if key in dict_tr_clusters.keys():
                        continue
                    dict_tr_clusters[key] = np.nan
                    
                df_tr_clusters = pd.DataFrame([dict_tr_clusters])
                df_tr_clusters.interpolate(method='values',axis=1,inplace=True)
                
            elif t_interp == 'interp_to_data_level':
                # 1. Intepollate in time
                ########################
                tstart = dt_axis.index[0]
                tstop  = dt_axis.index[-1]
                df_tmp[tstart]  = np.nan                                                         # Add the start of t_ax to the transmission dataframe
                df_tmp[tstop]   = np.nan                                                         # Add the end of t_ax to the transmission dataframe
                df_tmp = df_tmp.interpolate(method='values',axis=0)                              # Interpolate the calibrated transmissions to the start and end of the dataframe
                df_tmp = df_tmp[(df_tmp.index >= tstart) & (df_tmp.index <= tstop)]              # Restrict the known calibrated transmissions dataframe to the processing period
                tmp = pd.DataFrame(index=dt_axis,columns=df_tmp.columns,data=np.nan)             # Create transmission dataframe for processing and initialize to nan
                tmp.drop(index=[idx for idx in tmp.index if idx in df_tmp.index],inplace=True)   # Drop the entries where transmissions were interpolated/known above
                df_tr_clusters = pd.concat([df_tmp,tmp],axis=0)                                  # Concat the nan dataframe with the known transmission dataframe
                df_tr_clusters.sort_index(axis=0,inplace=True)                                   # Sort the index in time
                df_tr_clusters.interpolate(method='values',axis=0,inplace=True)                  # Perform interpolation in time
                
                df_tr_clusters.drop(index=[idx for idx in df_tr_clusters.index if idx not in dt_axis],inplace=True) # In case a transmission is determined at time t (during a calibration) that is not in the t_ax, drop the value    
                del[df,_tmp,tmp]
                
                # 2. Interpolate transmission curve in mz space
                ###############################################
                tr_nan_col = []
                tr_max_col = []
                tr_min_col = []
                for col in df_tr_clusters[df_clusters.selected].index:
                    if not col in df_tr_clusters.columns:
                        if col > df_tr_clusters.columns.max():
                            tr_max_col.append(col)
                        elif col < df_tr_clusters.columns.min():
                            df_tmp.append(col)
                        else:
                            df_tmp.append(col)

                tr_max = df_tmp[df_tr_clusters.columns.max()]
                
                df_nan = pd.DataFrame(index=df_tr_clusters.index,columns=tr_nan_col,data=np.nan)
                df_min = pd.DataFrame(index=df_tr_clusters.index,columns=tr_min_col,data=1.)
                concat = [df_tr_clusters,df_nan,df_min]
                if not len(tr_max_col) == 0:
                    df_max = pd.DataFrame(index=df_tr_clusters.index,columns=tr_max_col,data=np.array([tr_max.values for i in np.arange(len(tr_max_col))]).transpose())
                    concat.append(df_max)

                df_tr_clusters = pd.concat(concat,axis=1)
                df_tr_clusters.sort_index(axis=1, inplace=True)
                df_tr_clusters.interpolate(method='values',axis=1,inplace=True)
                
                df_tr_clusters.drop(columns=[c for c in df_tr_clusters.columns if c not in df_clusters[df_clusters.selected].index],inplace=True)
            elif t_interp == 'interp_to_PAP_interval':
                print('Error, interp_to_PAP_interval not yet coded for interpolating the transmission coefficients')
                raise ValueError
            else:
                print('Error, this interpolation method is not recognised')
                raise ValueError

        return df_tr_clusters
    
    def get_calibrationCoefficients(self, dt_axis, df_calibrations, df_calibrations_rprec, df_calibrations_racc, df_clusters, t_interp = 'nearest', t_pairing='both'):
        '''
        Get information extracted from calibrations. 
        '''
        df_tmp = df_calibrations.copy()
        df_tmp.set_index('ctime',inplace=True)
        # df_tmp.drop(['file','I_cps_H3O1_21','I_cps_H5O2_38'],axis=1,inplace=True)
        drop = ['file']
        for col in df_tmp.columns:
            if col.startswith('I_cps_'):
                drop.append(col)
        df_tmp.drop(drop, axis=1,inplace=True)
        df_tmp.columns = [float(col) for col in df_tmp.columns]

        df_tmp_rprec = df_calibrations_rprec.copy()
        df_tmp_rprec.set_index('ctime',inplace=True)
        df_tmp_rprec.columns = [float(col) for col in df_tmp_rprec.columns]

        df_tmp_racc = df_calibrations_racc.copy()
        df_tmp_racc.set_index('ctime',inplace=True)
        df_tmp_racc.columns = [float(col) for col in df_tmp_racc.columns]

        if not t_pairing in ('both','before','after'):
            raise ValueError
            
        if t_pairing == 'before':
            df_tmp       = df_tmp[df_tmp.index <= dt_axis.max()]
            df_tmp_rprec = df_tmp_rprec[df_tmp_rprec.index <= dt_axis.max()]
            df_tmp_racc  = df_tmp_racc[df_tmp_racc.index <= dt_axis.max()]
            
        elif t_pairing == 'after':
            df_tmp       = df_tmp[df_tmp.index >= dt_axis.min()]
            df_tmp_rprec = df_tmp_rprec[df_tmp_rprec.index >= dt_axis.min()]
            df_tmp_racc  = df_tmp_racc[df_tmp_racc.index >= dt_axis.min()]
        
        if t_interp == 'nearest':
            dict_cc_clusters       = {}
            dict_cc_clusters_rprec = {}
            dict_cc_clusters_racc  = {}
            
            idx = abs(df_tmp.index-dt_axis.mean()).argmin()
            for mz in df_tmp.columns:
                mz_col = mz_r.select_mz_cluster_from_exact(mz, df_clusters, tol = 0.01, mute = True)
                if np.isnan(mz_col):
                    continue
                dict_cc_clusters[mz_col] = df_tmp.iloc[idx][mz]
                dict_cc_clusters_rprec[mz_col] = df_tmp_rprec.iloc[idx][mz]
                dict_cc_clusters_racc[mz_col] = df_tmp_racc.iloc[idx][mz]
            
            df_cc_clusters       = pd.DataFrame([dict_cc_clusters])
            df_cc_clusters_rprec = pd.DataFrame([dict_cc_clusters_rprec])
            df_cc_clusters_racc  = pd.DataFrame([dict_cc_clusters_racc])
            
        else:
            print('Error, configuration not supported')
            raise ValueError
                
        return df_cc_clusters, df_cc_clusters_rprec, df_cc_clusters_racc
    
    def process(self, ongoing = False, t_start = None, t_stop=None, masks = [], output_threshold = 1, file_format = 'conf0', masses=[33.033],output_full=False):
        '''
        masks: array of strings, contains the masks for which output is asked,
            Usefull for filtering out profile measurements where no EC can be calculated
        output_threshold: int, minimum number of values during output_interval in order for the data to be saved. 
            Usefull to filter HH files with insufficient amount of data.
        '''
        ##################
        ## Calibrations ##
        ##################
        # Check, process and archive calibrations
        #########################################
        self.process_calibrations()
        df_calibrations, df_calibrations_rprec, df_calibrations_racc, df_transmission, df_stability, df_anc = self.get_archived_calibrations(tr_corrected=True, deconstructed=True)
        
        # Anc: Calibration breaks
        #########################
        year = self.year[0]
        base_path = '{1}{0}{2}{0}{3}{0}Misc{0}Calibrations{0}input{0}'.format(os.sep,self.dir_base,year,self.instrument)
        df_calibration_breaks = pd.read_csv('{}calib_info_table.csv'.format(base_path))
        for col in df_calibration_breaks.columns:
            if not ((col.startswith('t_') or (col.startswith('date_')))):
                continue
            
            df_calibration_breaks[col] = pd.to_datetime(df_calibration_breaks[col],origin=-25569,unit='D')
            df_calibration_breaks[col] = df_calibration_breaks[col].dt.tz_localize(tz=self.time_info['tz_anc_in'])
        
        
        ######################
        ## Process the data ##
        ######################
        # File based
        ############
        if ongoing:
            print('Error, not yet supported')
            raise ValueError
        
        # Time based
        ############
        else:
            if t_start is None:
                t_start = self.df_file_list_info.t_start.min()
            
            if t_stop is None:
                t_stop   = self.df_file_list_info.t_stop.max()
            
            if t_start.tzinfo is None:
                t_start = t_start.replace(tzinfo = self.time_info['tz_in'])
            if t_stop.tzinfo is None:
                t_stop = t_stop.replace(tzinfo = self.time_info['tz_in'])
            
            # process days as a whole, start is start of the day and stop is end of the day
            start = dt.datetime(year = t_start.year,
                                month = t_start.month,
                                day = t_start.day,
                                tzinfo = self.time_info['tz_in']
                                )

            stop  = dt.datetime(year = t_stop.year,
                                month = t_stop.month,
                                day = t_stop.day,
                                tzinfo = self.time_info['tz_in']
                                )
            
            stop = stop + dt.timedelta(hours=24)
            
            n_intervals = np.ceil((stop-start)/self.processing_config['processing_interval'])
            
            for i_interval in np.arange(n_intervals):
                print('start processing interval')
                # Get the data interval to process
                ##################################
                stop = start + self.processing_config['processing_interval']
                print('{} -- {}'.format(start.strftime('%Y-%b-%d %H:%M:%S %Z'),stop.strftime('%Y-%b-%d %H:%M:%S %Z')))
                
                PTR_data_object = None
                if (t_start <= start) and (t_stop >= stop):
                    PTR_data_object = self.get_PTR_data(start, stop)
                elif t_start > start:
                    PTR_data_object = self.get_PTR_data(t_start, stop)
                else:
                    PTR_data_object = self.get_PTR_data(start, t_stop)
                    
                
                # Check if data is found
                if PTR_data_object is None:
                    print('No data found between {} and {}.'.format(start.strftime('%Y-%m-%d %H:%M %Z'), stop.strftime('%Y-%m-%d %H:%M %Z')))
                    start = stop
                    continue
                
                # Check if we cross a calibration break
                #######################################
                t_pairing = 'both'
                matches = 0
                for i in np.arange(len(df_calibration_breaks.index)):
                    t_ref_before_start = df_calibration_breaks.iloc[i].t_first_valid_before
                    t_ref_before_end   = df_calibration_breaks.iloc[i].t_last_valid_before

                    t_ref_after_start = df_calibration_breaks.iloc[i].t_first_valid_after
                    t_ref_after_end   = df_calibration_breaks.iloc[i].t_last_valid_after
                    
                    if (t_ref_before_start <= stop) and (t_ref_before_end >= start):
                        t_pairing = 'before'
                        matches += 1
                        
                    if (t_ref_after_start <= stop) and (t_ref_after_end >= start):
                        t_pairing = 'after'
                        matches += 1
                        
                    if matches > 1:
                        print('----')
                        print('Error: data in processing interval should be matches with dedicated before and after calibrations. Adapt the processing interval so that this is no longer the case or split explicitely.')
                        print('Split data between {}, and {}'.format(start.strftime('%Y-%m-%d %H:%M'), stop.strftime('%Y-%m-%d %H:%M')))
                        print('----')
                        raise ValueError
                    
                # Resample the data and trim the data masks
                ###########################################
                print('Trim and resample')
                PTR_data_object.trim_masks(self.processing_config['tdelta_trim_start'], self.processing_config['tdelta_trim_stop'])

                # Note: the resampling routine sets the df_prec based on the distribution assumption
                if not self.processing_config['acc_interval'] is None:
                    PTR_data_object.resample(self.processing_config['acc_interval'], self.processing_config['origin'], self.processing_config['offset'])
                
                # If we don't resample or we explicitely ask to calculate precision using the Poisson assumption, (re)set the df_prec here
                if ((self.processing_config['precision_calc'] == 'Poisson' ) or
                    (self.processing_config['acc_interval'] is None)):
                    PTR_data_object.set_precision_poisson()
                
                # Get the ancilarry dataframes/dictironary to process the data
                ##############################################################
                print('Get ancillary')
                dt_axis = PTR_data_object.df_data.index
                df_tr_coeff = self.get_transmissionCoefficients('interp', dt_axis, df_transmission, PTR_data_object.df_clusters,
                                                                t_interp = self.processing_config['tr_t_interp'], t_pairing = t_pairing)                # Get transmissions
                if np.sum(np.isnan(df_tr_coeff.values)) == len(df_tr_coeff.values):
                    print('----')
                    print('Warning, no valid transmissions found to process data.')
                    print('Check between {} and {}.'.format(start.strftime('%Y-%m-%d %H:%M'), stop.strftime('%Y-%m-%d %H:%M')))
                    print('----')
                    continue
                
                PTR_data_object.Tr_PH1_to_PH2 = df_tr_coeff[PTR_data_object.mz_col_primIon['H5O2_38.033']].mean()**-1.                                  # Obtain Tr_PH1_to_PH2 from the transmission dataframe
                
                dict_Xr0 = self.get_Xr0_mz(PTR_data_object.df_clusters)                                                                                 # Get Xr0 dictionary for non default values
                
                df_cc_coeff, df_cc_coeff_rprec, df_cc_coeff_racc = self.get_calibrationCoefficients(dt_axis, df_calibrations, 
                                                                                                    df_calibrations_rprec, 
                                                                                                    df_calibrations_racc, 
                                                                                                    PTR_data_object.df_clusters,
                                                                                                    t_interp = self.processing_config['cc_t_interp'],
                                                                                                    t_pairing = t_pairing)                              # Get calibration coefficients

                k_reac_mz       = self.get_k_reac_mz(PTR_data_object.df_clusters)                                                                       # Get the reaction rate coefficients for the clusters not using the default value
                k_reac_default  = self.processing_config['k_reac_default']                                                                              # Set the default
                multiplier      = self.get_multiplier_mz(PTR_data_object.df_clusters)                                                                   # Get the multiplier (i.e., accounting for fragmentation, isotopic ratio,...) for the identified clusters that have a not 1 value
                
                # Process the data
                ##################
                print('process data')
                kw_zero = {key: self.processing_config[key] for key in self.processing_config.keys() if 'zero' in key}
                
                PTR_data_object.transform_data_ncr(dict_Xr0,self.processing_config['Xr0_default'])                                                      # Normalise signal
                PTR_data_object.transform_data_trcnrc(df_tr_coeff)                                                                                      # Correct for transmission
                PTR_data_object.transform_data_VMR(df_cc_coeff,df_cc_coeff_rprec,df_cc_coeff_racc,k_reac_mz,k_reac_default,multiplier)                  # Transform to VMR, use transmission corrected calibration coefficients when available, otherwise use first principles.
                PTR_data_object.set_zero(**kw_zero,mute=True)                                                                                           # Set zero dataframe
                PTR_data_object.transform_data_subtract_zero(**kw_zero, mute = True)                                                                    # Subtract zero
                
                # Save the data
                ###############
                if 'EBAS' in file_format:
                    dir_o = '{1}{2}{0}{3}{0}{4}{0}{5}{0}'.format(os.sep,self.dir_base,start.year,self.instrument,'Nominal',self.data_output_level)
                    check_create_output_dir(dir_o)
                    dir_o += start.strftime('%m') + os.sep
                    check_create_output_dir(dir_o)
                    PTR_data_object.save_output_EBAS(dir_o, masses = masses, file_format = file_format)

                else:
                    n_outputs = np.ceil((stop - start)/self.processing_config['output_interval'])
                    for i_output in np.arange(n_outputs):
                        t_start = start+i_output*self.processing_config['output_interval']
                        t_end   = t_start+self.processing_config['output_interval']
                        
                        tmp = PTR_data_object.get_PTR_data_subsample_time(t_start, t_end)
                        dir_o = '{1}{2}{0}{3}{0}{4}{0}{5}{0}'.format(os.sep,self.dir_base,start.year,self.instrument,'Nominal',self.data_output_level)
                        check_create_output_dir(dir_o)
                        dir_o += t_start.strftime('%m') + os.sep
                        check_create_output_dir(dir_o)
                        if ((self.processing_config['output_interval'] is None) or 
                            (self.processing_config['output_interval'] < dt.timedelta(hours=24))):
                            dir_o += t_start.strftime('%d') + os.sep
                            check_create_output_dir(dir_o)
                        
                        tmp.save_output(dir_o, self.name, self.instrument, df_tr_coeff, df_cc_coeff,
                                        masks=masks,threshold=output_threshold,file_format=file_format,
                                        time_info=self.time_info, processing_config=self.processing_config,
                                        output_full=output_full)
                        del tmp
                del PTR_data_object
                
                # Setup the next cycle of processing
                ####################################
                start = stop
        
        return None
    
    
    
######################################
## PTR data object support routines ##
######################################
    
def deconstruct_df_calibrations(df_calibrations):
    # Exctract the uncertainty columns out of the calibrations file
    # tmp_calibrations = df_calibrations.drop(['I_cps_H3O1_21', 'I_cps_H5O2_38','file'],axis=1)
    drop = ['file']
    for col in df_calibrations.columns:
        if col.startswith('I_cps'):
            drop.append(col)
    tmp_calibrations = df_calibrations.drop(drop, axis=1)

    mz_columns = []
    mz_rprec_columns = []
    mz_racc_columns  = []
    ancillary = ['ctime']
    rprec_correction_factor = 1.
    racc_correction_factor = 1.
    for col in tmp_calibrations.columns:
        if col in ancillary:
            continue
        
        if '_rprec' in col:
            mz_rprec_columns.append(col)
            if '[%]' in col:
                rprec_correction_factor = 1.e-2
            
        elif '_racc' in col:
            mz_racc_columns.append(col)
            if '[%]' in col:
                racc_correction_factor = 1.e-2
                
        else:
            mz_columns.append(col)
    
    df_calibrations_runc = tmp_calibrations.drop(mz_columns,axis=1).copy()
    
    df_calibrations_rprec = df_calibrations_runc.drop(mz_racc_columns,axis=1).copy()
    df_calibrations_rprec[mz_rprec_columns] = df_calibrations_rprec[mz_rprec_columns].mul(rprec_correction_factor)
    df_calibrations_rprec.columns = [col.split('_')[0] for col in df_calibrations_rprec.columns]
    
    df_calibrations_racc = df_calibrations_runc.drop(mz_rprec_columns,axis=1).copy()
    df_calibrations_racc[mz_racc_columns] = df_calibrations_racc[mz_racc_columns].mul(racc_correction_factor)
    df_calibrations_racc.columns = [col.split('_')[0] for col in df_calibrations_racc.columns]
    
    df_calibrations.drop(mz_rprec_columns,axis=1,inplace=True)
    df_calibrations.drop(mz_racc_columns,axis=1,inplace=True)

    return df_calibrations, df_calibrations_rprec, df_calibrations_racc

def get_data_corrected(df_I, df_clusters, selected_multiplier, default_multiplier = 1.):
    # Selected multiplier is assumed to be a dataframe of multipliers in a single row with columns mz(_exact/_col)
    # Possibility to set a default multiplier if the data has been corrected before with a wrong factor
    selected_multiplier.columns = [mz_r.select_mz_cluster_from_exact(mz, df_clusters, tol = 0.01, mute = True) for mz in selected_multiplier.columns]
    
    subset = df_I.columns.intersection(selected_multiplier.columns)
    df_I[subset] = (df_I[subset]*selected_multiplier[subset].iloc[0]/default_multiplier).astype(np.float64)

    return df_I

def to_matlabtime(dt_ax):
    dt_ax = dt_ax - dt.datetime(year=2024,month=1,day=1,hour=0,second=0,tzinfo=dt.timezone(dt.timedelta(hours=0)))
    dt_ax = 739252.+(dt_ax/dt.timedelta(hours=24))

    return dt_ax

#####################
## PTR data object ##
#####################
class PTR_data(object):
    '''PTR data object to be transformed to concentrations.'''
    
    def __init__(self, df_data, data_description, data_units, sst, sst_units, df_P_drift, df_U_drift, df_T_drift, df_P_inlet, masks, df_clusters, list_primary_ions=None, df_I_primary_ion=None):
        self.df_data = df_data
        self.data_description = data_description
        self.data_units = data_units
        
        # sst = single spectrum time; used to calculate total counts
        self.sst = sst
        self.sst_units = sst_units
        self.df_absolute_counts = None
        if not self.df_data is None:
            self.df_absolute_counts = self.get_data_totalcounts()


        self.df_zero = None
        self.df_zero_prec = None
        self.zero_description = None
        self.zero_prec_description = None
        self.zero_units = None
        
        self.df_prec = None
        self.prec_description = None
        self.prec_units = None

        self.df_racc = None

        self.df_P_drift = df_P_drift
        self.df_U_drift = df_U_drift
        self.df_T_drift = df_T_drift
        self.df_P_inlet = df_P_inlet
        
        self.masks = masks
        
        self.df_clusters = df_clusters

        self.list_primary_ions = [
#            'H3O1_19.018',
            'H3O1_21.022',
#            'C1O1_29.002',
#            'N2_29.013',
            'N1O1_29.997',
            'N1O1_31.005',
#            'O2_31.989',
#            'O2_32.997',
            'O2_33.997',
#            'H5O2_37.028',
            'H5O2_38.033',
#            'H5O2_39.033',
#            'H7O3_55.039',
#            'H9O4_73.050',
            ]
            
        if not list_primary_ions is None:
            self.list_primary_ions = list_primary_ions
        
        if not 'H3O1_21.022' in self.list_primary_ions:
            print('Unable to do normalisation without "H3O1_21.022" in list_primary_ions')
            raise ValueError
            
        if not 'H5O2_38.033' in self.list_primary_ions:
            print('Unable to do normalisation without "H5O2_38.033" in list_primary_ions')
            raise ValueError

        
        self.mz_col_primIon = {}
        for prim_ion in self.list_primary_ions:
            mass = float(prim_ion.split('_')[-1])
            col = mz_r.select_mz_cluster_from_exact(mass, self.df_clusters, tol = 0.01, mute = True)
            if np.isnan(col):
                continue
            
            self.mz_col_primIon[prim_ion] = col
        
        
#        self.mz_col_21 = mz_r.select_mz_cluster_from_exact(21.022, self.df_clusters, tol = 0.01, mute = True)
#        self.mz_col_38 = mz_r.select_mz_cluster_from_exact(38.033, self.df_clusters, tol = 0.01, mute = True)
        
        self.Tr_PH1_to_PH2 = None
        self.default_CC_kinetic = None

        # SISWEB
        # self.FPH1 = 500
        # self.FPH2 = 624.2

        # SOP ACTRIS
        # self.FPH1 = 487
        # self.FPH2 = 645

        # Python isotopologue
        self.FPH1 = 488
        self.FPH2 = 669
        
        # Chemcalc.org
        # self.FPH1 = 476.19
        # self.FPH2 = 669.23
        
        # IONICON DataBase
        # self.FPH1 = 487.64
        # self.FPH2 = 748.98

        # IONICON multiplier
        # self.FPH1 = 500
        # self.FPH2 = 750
        
        
    def __del__(self):
        del self.df_data
        del self.data_description
        del self.data_units
        
        del self.sst
        del self.sst_units
        del self.df_absolute_counts

        del self.df_zero
        del self.df_zero_prec
        del self.zero_description
        del self.zero_units
        
        del self.df_prec
        del self.prec_description
        del self.prec_units        
        del self.df_racc

        del self.df_P_drift
        del self.df_U_drift
        del self.df_T_drift
        del self.df_P_inlet
        
        del self.masks
        
        del self.df_clusters
        
        del self.list_primary_ions
        del self.mz_col_primIon
        # del self.mz_col_21
        # del self.mz_col_38
        
        del self.Tr_PH1_to_PH2
        del self.default_CC_kinetic

        del self.FPH1
        del self.FPH2
                
    def get_PTR_data_subsample_time(self, t_start, t_end, tz_info = None):
        mask = (self.df_data.index >= t_start) & (self.df_data.index < t_end)
        masks = {}
        for key in self.masks.keys():
            masks[key] = self.masks[key][mask]

        PTR_data_object = PTR_data(None, self.data_description, self.data_units, self.sst[mask], self.sst_units, 
                                   self.df_P_drift[mask], self.df_U_drift[mask], self.df_T_drift, self.df_P_inlet, 
                                   masks, self.df_clusters)
        
        PTR_data_object.df_data = self.df_data[mask]
        PTR_data_object.df_absolute_counts = self.df_absolute_counts[mask]

        if not self.df_zero is None:
            PTR_data_object.df_zero = self.df_zero[mask]
        if not self.df_zero_prec is None:
            PTR_data_object.df_zero_prec = self.df_zero_prec[mask]
        PTR_data_object.zero_description = self.zero_description
        PTR_data_object.zero_prec_description = self.zero_prec_description
        PTR_data_object.zero_units = self.zero_units
        
        if not self.df_prec is None:
            PTR_data_object.df_prec = self.df_prec[mask]
        PTR_data_object.prec_description = self.prec_description
        PTR_data_object.prec_units = self.prec_units
        if not self.df_racc is None:
            PTR_data_object.df_racc = self.df_racc[mask]

        PTR_data_object.Tr_PH1_to_PH2 = self.Tr_PH1_to_PH2
        PTR_data_object.default_CC_kinetic = self.default_CC_kinetic

        PTR_data_object.FPH1 = self.FPH1
        PTR_data_object.FPH2 = self.FPH2
                
        return PTR_data_object
    
    def get_PTR_data_subsample_mz(self, mz_selection):
        # if not self.mz_col_21 in mz_selection:
        #    mz_selection.append(self.mz_col_21)
        #if not self.mz_col_38 in mz_selection:
        #    mz_selection.append(self.mz_col_38)
        # Add all primary ion signals
        for key, mz_col in self.mz_col_primIon.keys():
            if not mz_col in mz_selection:
                mz_selection.append(mz_col)

        PTR_data_object = PTR_data(self.df_data[mz_selection], self.data_description, self.data_units, self.sst, self.sst_units,
                                   self.df_P_drift, self.df_U_drift, self.df_T_drift, self.df_P_inlet, 
                                   self.masks, self.df_clusters)
        
        PTR_data_object.Tr_PH1_to_PH2 = self.Tr_PH1_to_PH2
        
        return PTR_data_object    
    
    def resample(self, acc_interval, origin, offset):
        # Adapt masks as needed
        #######################
        for key in self.masks.keys():
            method = 'all'
            if key == 'invalid':
                method = 'any'
            self.masks[key] = msk_r.get_resampled_mask(self.df_data.index,self.masks[key],acc_interval,origin=origin,offset=offset,method=method)
        
        # Precision and data
        ####################
        # Based on Distribution
        resampled      = self.df_data.resample(acc_interval,origin=origin,offset=offset)
        self.df_prec   = resampled.std()/np.sqrt(resampled.count())
        
        # data resampling and relative precision
        self.df_data   = resampled.mean()
        
        # Correct total counts to allow calcluation according to Poisson later on
        self.df_absolute_counts = self.df_absolute_counts.resample(acc_interval,origin=origin,offset=offset).sum()
        
        # Ancillary data
        ################
        self.df_P_drift = self.df_P_drift.resample(acc_interval,origin=origin,offset=offset).mean()
        self.df_U_drift = self.df_U_drift.resample(acc_interval,origin=origin,offset=offset).mean()
        self.df_T_drift = self.df_T_drift.resample(acc_interval,origin=origin,offset=offset).mean()
        self.df_P_inlet = self.df_P_inlet.resample(acc_interval,origin=origin,offset=offset).mean()
        
        self.sst = self.sst.resample(acc_interval, origin=origin,offset=offset).sum()
                
        return None
    
    def trim_masks(self, tdelta_trim_start, tdelta_trim_stop):
        # Trim masks #
        tmp = {}
        for key in self.masks.keys():
            # The calculation masks have already been trimmed previously
            if 'calc' in key:
                continue
            if 'invalid'in key:
                continue
            if 'trimmed' in key:
                continue
            tmp['{}_trimmed'.format(key)] = msk_r.get_trimmed_mask(self.masks[key],self.df_data.index,tdelta_after_start=tdelta_trim_start,tdelta_before_stop=tdelta_trim_stop)
    
        self.masks.update(tmp)
        
        return None
    
    def set_data_absolute_counts(self):
        if not self.data_units == 'cps':
            print('data not in correct units to compute absolute counts')
            raise ValueError
        
        resolution = np.diff(self.df_data.index).mean()/np.timedelta64(1, 's')
        counts = resolution*self.df_data
        
        return counts
    
    def set_precision_poisson(self):
        self.df_prec = self.get_precision_poisson()
        
        return None
       
    def get_precision_poisson(self):
        df_prec = self.df_absolute_counts.pow(0.5)
        df_rprec = df_prec/abs(self.df_absolute_counts)
        df_prec = abs(self.df_data)*df_rprec
        
        return df_prec
        
    def transform_data_correction(self, selected_multiplier, default_multiplier = 1.):
        self.df_data = get_data_corrected(self.df_data, self.df_clusters, selected_multiplier, default_multiplier = 1.)
        return None

    def get_data_totalcounts(self):
        correction = np.nan
        if self.sst_units == 'ms':
            correction = 1.e3
        else:
            print('spectrum scanning time units not set, return NaN')
        
        if self.data_units != 'cps':
            print('Error, units not correct to make accumulation of signal')
            raise ValueError
            
        df_data_totalcounts = self.df_data.mul(self.sst,axis=0)/correction
        
        return df_data_totalcounts

    def get_zero(self, tdelta_buf_zero = dt.timedelta(minutes=1), tdelta_avg_zero=dt.timedelta(minutes=5), tdelta_min_zero=dt.timedelta(minutes=20), zero='interp', zero_unc='Precision', mute = False):
        if not 'zero_trimmed' in self.masks.keys():
            print('Warning, untrimmed zero mask used to infer zero measurements.')
            mask_calc_zero = msk_r.get_representative_mask_from_multiple_intervals(self.df_data, self.masks['zero'], tdelta_buf_zero, tdelta_avg_zero, tdelta_min_zero, mute=mute)
            
        else:
            mask_calc_zero = msk_r.get_representative_mask_from_multiple_intervals(self.df_data, self.masks['zero_trimmed'], tdelta_buf_zero, tdelta_avg_zero, tdelta_min_zero, mute=mute)
        
        i_start, i_stop = msk_r.get_start_stop_from_mask(mask_calc_zero)
        
        if ((mask_calc_zero is None) or
            (mask_calc_zero.sum() == 0)):
            print('Error: No representative zero measurement available from {} to {}'.format(self.df_data.index.min().strftime(format='%Y-%m-%d %H:%M %Z'),self.df_data.index.max().strftime(format='%Y-%m-%d %H:%M %Z')))
            raise ValueError
        
        df_I_tmp = self.df_data.copy()
        df_I_tmp_prec = self.df_prec.copy()
        
        # zero correction for the primary ions is not applicable
        # df_I_tmp[[self.mz_col_21,self.mz_col_38]] = 0
        # df_I_tmp_prec[[self.mz_col_21,self.mz_col_38]] = 0
        subset = df_I_tmp.columns.intersection(self.mz_col_primIon.values())
        df_I_tmp[subset] = 0
        df_I_tmp_prec[subset] = 0

        if zero == 'constant':
            print('Zero correction constant: precision from std on all zero measuremnts in processed data sample.')
            df_I_zero = df_I_tmp[mask_calc_zero].mean()
            df_I_zero_prec = df_I_tmp[mask_calc_zero].std()
        
        else:
            df_I_zero = df_I_tmp.copy()
            df_I_zero[~mask_calc_zero] = np.nan
            df_I_zero_prec = df_I_tmp_prec.copy()
            df_I_zero_prec[~mask_calc_zero] = np.nan
            
            for i in np.arange(len(i_start)):
                if zero_unc == 'Precision':
                    df_I_zero_prec.iloc[i_start[i]-1:i_stop[i]+1] = df_I_zero_prec.iloc[i_start[i]-1:i_stop[i]+1].mean()
                
                elif zero_unc == 'Distribution':
                    df_I_zero_prec.iloc[i_start[i]-1:i_stop[i]+1] = df_I_zero.iloc[i_start[i]-1:i_stop[i]+1].std()
                
                else:
                    print('Error: uncertainty method for zero ({}) not recognized.'.format(zero_unc))
                    raise ValueError
            
                df_I_zero.iloc[i_start[i]-1:i_stop[i]+1] = df_I_zero.iloc[i_start[i]-1:i_stop[i]+1].mean()
                
            if zero == 'interp':
                df_I_zero.interpolate(method='values',inplace=True)
                df_I_zero.ffill(inplace=True)
                df_I_zero.bfill(inplace=True)
                
                # Interpolate uncertainties, the interpolation is not adding information and thus reducing the error between zero measurements would not make sense
                df_I_zero_prec.interpolate(method='values',inplace=True)
                df_I_zero_prec.ffill(inplace=True)
                df_I_zero_prec.bfill(inplace=True)
            
            elif zero == 'ffill':
                df_I_zero.ffill(inplace=True)
                df_I_zero.bfill(inplace=True)
            
                df_I_zero_prec.ffill(inplace=True)
                df_I_zero_prec.bfill(inplace=True)
                
            elif zero == 'bfill':
                df_I_zero.bfill(inplace=True)
                df_I_zero.ffill(inplace=True)
            
                df_I_zero_prec.bfill(inplace=True)
                df_I_zero_prec.ffill(inplace=True)
                
            else:
                print('Zero method not supported')
                raise ValueError
        
        del [df_I_tmp, df_I_tmp_prec]
        
        return df_I_zero, df_I_zero_prec

    def set_zero(self, tdelta_buf_zero = dt.timedelta(minutes=1), tdelta_avg_zero=dt.timedelta(minutes=5), tdelta_min_zero=dt.timedelta(minutes=20), zero='interp', zero_unc='Distribution', mute = False):
        self.df_zero, self.df_zero_prec = self.get_zero(tdelta_buf_zero = tdelta_buf_zero, tdelta_avg_zero = tdelta_avg_zero, tdelta_min_zero=tdelta_min_zero, zero = zero, zero_unc=zero_unc, mute = mute)
        self.zero_description = '{}, zero values'.format(self.data_description)
        self.zero_prec_description = 'Precision on zero values'
        self.zero_units = self.data_units
        
        return None

    def get_data_zero_corrected(self, tdelta_buf_zero = dt.timedelta(minutes=1), tdelta_avg_zero=dt.timedelta(minutes=5), tdelta_min_zero=dt.timedelta(minutes=20), zero='interp', zero_unc='Distribution', mute = False):
        if self.zero_units != self.data_units:
            self.set_zero(tdelta_buf_zero = tdelta_buf_zero, tdelta_avg_zero = tdelta_avg_zero, tdelta_min_zero = tdelta_min_zero, zero = zero, zero_unc = zero_unc, mute = mute)
        
        df_I_zc = self.df_data - self.df_zero
        df_I_zc_prec = (self.df_prec.pow(2)+self.df_zero_prec.pow(2)).pow(0.5)
        
        return df_I_zc, df_I_zc_prec
        
    def transform_data_subtract_zero(self, tdelta_buf_zero = dt.timedelta(minutes=1), tdelta_avg_zero=dt.timedelta(minutes=5), tdelta_min_zero=dt.timedelta(minutes=20), zero='constant', zero_unc='Distribution', mute = False):
        self.df_data, self.df_prec = self.get_data_zero_corrected(tdelta_buf_zero=tdelta_buf_zero, tdelta_avg_zero=tdelta_avg_zero, tdelta_min_zero=tdelta_min_zero, zero = zero, zero_unc = zero_unc, mute = mute)
        
        return None

    def get_data_ncr(self, dict_Xr0={}, Xr0_default=1.):
        '''
        Convert dataframe with units cps to units of a normalised count rate.
        The normalised count rate of product ion P+ is equal to the count rate of P+ that would correspond with a reactant H3O+ count rate of 1.e6 cps
        '''
        # df: dataframe with absolute count rates observed from the PTR insturment
        # mz_col_21: column name for the H3O+ isotope at 21 (the count rate at 19 is too large to measure faithfully)
        # mz_col_38: column name for the H3O+.H2O isotope at 38
        # Tr_PH1_to_PH2: Ratio of transmission factors from 21 to 38 (i.e., Tr_21/Tr_38)
        # FPH1: Factor multiplication to cxonvert from isotope signal
        # FPH2: Factor multiplication to convert from isotope signal
        # XR0: ratio from reactivity of unionized compound with protonated water and a watercluster of protonated water (i.e., k_37/k_19, theoretical value)
        #      Note that the theoretical value may differ significantly from the experimental one!
        if not self.data_units == 'cps':
            print('Error, expected units of dataframe to be cps to normalise signal')
            raise ValueError
            
        if self.Tr_PH1_to_PH2 is None:
            print('Error, set Tr_PH1_to_PH2 before normalising the ion counts')
            raise ValueError
        
        df_ncr = self.df_data.copy() # copy the original dataframe as to make sure we do not change it
        df_ncr_prec = self.df_prec.copy() # copy the original dataframe as to make sure we do not change it
        df_ncr_rprec = df_ncr_prec/abs(df_ncr)
        
        # I_cr_21, I_cr_38 = df_ncr[self.mz_col_21], df_ncr[self.mz_col_38]
        # I_cr_21_prec, I_cr_38_prec = df_ncr_prec[self.mz_col_21], df_ncr_prec[self.mz_col_38]
        I_cr_21, I_cr_38 = df_ncr[self.mz_col_primIon['H3O1_21.022']], df_ncr[self.mz_col_primIon['H5O2_38.033']]
        I_cr_21_prec, I_cr_38_prec = df_ncr_prec[self.mz_col_primIon['H3O1_21.022']], df_ncr_prec[self.mz_col_primIon['H5O2_38.033']]

        # Correct for Xr0 default
        #########################
        Xr0 = Xr0_default
        numenator = 1.e6
        denominator = (self.FPH1*I_cr_21)+(self.Tr_PH1_to_PH2*Xr0*self.FPH2*I_cr_38)
        norm = numenator/denominator
        # Force this precision as type np.float64 to get sufficient precision 
        norm_rprec = ((self.FPH1*I_cr_21_prec).pow(2)+(self.Tr_PH1_to_PH2*Xr0*self.FPH2*I_cr_38_prec).pow(2))/denominator.pow(2).astype(np.float64)
        
        # Selection of data with default Xr0
        subset = self.df_data.columns.difference(dict_Xr0.keys())
        
        # Normalize data
        df_ncr[subset] = df_ncr[subset].mul(norm,axis=0)
        
        # Update rprec
        df_ncr_rprec[subset] = (df_ncr_rprec[subset].pow(2).add(norm_rprec.pow(2),axis=0)).pow(0.5)
        
        # Correct for custom Xr0
        ########################
        subset = self.df_data.columns.intersection(dict_Xr0.keys())
        for c in subset:
            Xr0 = dict_Xr0[c]
            denominator = (self.FPH1*I_cr_21)+(self.Tr_PH1_to_PH2*Xr0*self.FPH2*I_cr_38)
            norm = numenator/denominator
            norm_rprec = ((self.FPH1*I_cr_21_prec).pow(2)+(self.Tr_PH1_to_PH2*Xr0*self.FPH2*I_cr_38_prec).pow(2))/denominator.pow(2).astype(np.float64)
            
            df_ncr[c] = df_ncr[c].mul(norm,axis=0)
            df_ncr_rprec[c] = (df_ncr_rprec[c].pow(2).add(norm_rprec.pow(2),axis=0)).pow(0.5)
        
        # Correct prec
        ##############
        df_ncr_prec = df_ncr_rprec.mul(abs(df_ncr))
        
        # Reset the primary ion signals
        ###############################
        subset = df_ncr.columns.intersection(self.mz_col_primIon.values())
        df_ncr[subset] = self.df_data[subset]
        df_ncr_prec[subset] = self.df_prec[subset]

        return df_ncr, df_ncr_prec
    
    def transform_data_ncr(self, dict_Xr0={}, Xr0_default=1.):
        self.df_data, self.df_prec = self.get_data_ncr(dict_Xr0, Xr0_default)
        self.data_description = 'Normalised Ion count'
        self.data_units = 'ncps'
        
        self.prec_description = 'Precision on Normalised Ion count'
        self.prec_units = 'ncps'

        return None

    def get_data_trcnrc(self, transmissions):
        if not self.data_units == 'ncps':
            print('Error, normalise signal before correcting for transmission effects')
            raise ValueError
            
        transmissions =  transmissions.drop(columns=[col for col in transmissions if col not in self.df_data.columns]).iloc[0] # Select the column of transmissions in order to apply the correction column-wise
        df_trcncr = self.df_data/transmissions
        df_trcncr_prec = self.df_prec/transmissions
        
        return df_trcncr, df_trcncr_prec

    def transform_data_trcnrc(self, transmissions):
        self.df_data, self.df_prec = self.get_data_trcnrc(transmissions)
        self.data_description = 'Transmission corrected normalised ion count'
        self.data_units = 'tc-ncps'
        
        self.prec_description = 'Precision on Transmission corrected normalised ion count'
        self.prec_units = 'tc-ncps'

        return None
    
    def get_data_VMR(self, df_cc_coeff, df_cc_coeff_rprec, df_cc_coeff_racc, k_reac_mz = {}, k_reac_default = 2, multiplier={}):
        '''
        Transform normalised data to volume mixing ratio, take into account the transmission, direct calibration, and chemical rates.
        There is a possibility to define a multiplicator to take into account isotopic ratio.
          - df_calibration [tc-ncps ppbv-1]
          - k_reac(_mz/_default) [1.e-9 cm3 molecule-1 s-1]
        '''
        if not self.data_units == 'tc-ncps':
            print('Error, to calculate VMRs the signal is assumed to be in transmission corrected normalized counts per second [tc-ncps]')
            raise ValueError
        
        t_reac, N_DT = calib_r.get_ancilarry_PTRT_drift_calib_av(self.df_P_drift.mean(),
                                                                 self.df_T_drift.mean(),
                                                                 self.df_U_drift.mean())
        
        CC_kinetic = k_reac_default*N_DT*t_reac*1.e-3
        CC_kinetic = CC_kinetic*1.e-9 # Correct for the units in k_default
        self.default_CC_kinetic = CC_kinetic

        df_ppbv = self.df_data.copy()
        df_ppbv_prec = self.df_prec.copy()
        
        # Set relative accuracy to 0.5 for all compounds, correct for calirbated later on
        df_ppbv_racc = pd.DataFrame(data=0.5, index=df_ppbv.index, columns=df_ppbv.columns)
        
        ## Default calculation from first principles
        ############################################
        # select uncalibrated compounds specifically
        subset = df_ppbv.columns.difference(df_cc_coeff.keys())
        df_ppbv[subset] = df_ppbv[subset]/CC_kinetic.astype(np.float64)
        
        # Scale precision for uncalibrated and set accuracy to 50% for all compounds
        df_ppbv_prec[subset] = df_ppbv_prec[subset]/CC_kinetic.astype(np.float64)
        
        # Corrections due to deviation of default reaction rate constants
        tmp = pd.DataFrame(k_reac_mz, index=[0])
        subset = tmp.columns.difference(df_cc_coeff.columns)
        corrections = tmp[subset].pow(-1)
        default_multiplier = k_reac_default**-1
        
        # Only uncalibrated compounds were selected to be corrected and thus no correction to calibrated needed here
        df_ppbv = get_data_corrected(df_ppbv, self.df_clusters, 
                                     corrections, default_multiplier = default_multiplier)
        df_ppbv_prec = get_data_corrected(df_ppbv_prec, self.df_clusters, 
                                          corrections, default_multiplier = default_multiplier)

        # Corrections using the multiplicator (generally isotopic ratio combined with fractioning factor)
        multiplier = pd.DataFrame(multiplier, index=[0])
        subset = multiplier.columns.difference(df_cc_coeff.columns)
        corrections = multiplier[subset]
        default_multiplier = 1.
        
        # Only uncalibrated compounds were selected to be corrected and thus no correction to calibrated needed here
        df_ppbv = get_data_corrected(df_ppbv, self.df_clusters, 
                                     corrections, default_multiplier = default_multiplier)
        df_ppbv_prec = get_data_corrected(df_ppbv_prec, self.df_clusters, 
                                          corrections, default_multiplier = default_multiplier)

        # Calculate MR of calibrated compounds and set accuracy and precision
        #####################################################################
        cc_corrections = df_cc_coeff.pow(-1)
        df_ppbv = get_data_corrected(df_ppbv, self.df_clusters,
                                     cc_corrections, default_multiplier=1.)
        df_ppbv_prec = get_data_corrected(df_ppbv_prec, self.df_clusters,
                                     cc_corrections, default_multiplier=1.)
                
        # Set racc for calibrated compounds
        subset = df_ppbv_racc.columns.intersection(df_cc_coeff_racc.columns)
        df_ppbv_racc[subset] = 1.
        df_ppbv_racc[subset] = df_ppbv_racc[subset].mul(df_cc_coeff_racc[subset].iloc[0])

        # Correct prec for calibrated compounds
        df_tmp_rprec = pd.DataFrame(0,index=df_ppbv_prec.index, columns=df_ppbv_prec.columns)
        subset = df_tmp_rprec.columns.intersection(df_cc_coeff_rprec.columns)
        df_tmp_rprec[subset] = 1.
        df_tmp_rprec[subset] = df_tmp_rprec[subset].mul(df_cc_coeff_rprec[subset].iloc[0])
        
        df_ppbv_prec[subset] = df_ppbv[subset].mul(((df_ppbv_prec[subset]/df_ppbv[subset]).pow(2)+df_tmp_rprec[subset].pow(2)).pow(0.5))
        del [df_tmp_rprec]
        
        # The transformation of primary ions is not relevant so revert here
        # for mz_col in [self.mz_col_21, self.mz_col_38]:
        subset = df_ppbv.columns.intersection(self.mz_col_primIon.values())
        df_ppbv[subset] = self.df_data[subset]
        df_ppbv_racc[subset] = 0.
        
        return df_ppbv, df_ppbv_prec, df_ppbv_racc
    
    def transform_data_VMR(self, df_cc_coeff, df_cc_coeff_rprec, df_cc_coeff_racc, k_reac_mz = {}, k_reac_default = 2, multiplier={}):
        self.df_data, self.df_prec, self.df_racc = self.get_data_VMR(df_cc_coeff, df_cc_coeff_rprec, df_cc_coeff_racc, 
                                                                                    k_reac_mz, k_reac_default, multiplier)
        
        self.data_description = 'Mixing Ratio'
        self.data_units = 'ppbv'
        
        self.prec_description = 'Precision on Mixing Ratio'
        self.prec_units = 'ppbv'
        
        return None

    def get_calib_calc_masks(self, mz_exact_infer_masks=59.049,
                             tdelta_buf_zero=dt.timedelta(minutes=1),tdelta_avg_zero=dt.timedelta(minutes=5),tdelta_min_zero=dt.timedelta(minutes=20),
                             tdelta_buf_calib=dt.timedelta(minutes=1),tdelta_avg_calib=dt.timedelta(minutes=5),tdelta_min_calib=dt.timedelta(minutes=70),
                             tdelta_stability=dt.timedelta(minutes=60),
                             ):
        
        calib_start, calib_stop = msk_r.get_start_stop_from_mask(self.masks['calib'])
        zero_start, zero_stop = msk_r.get_start_stop_from_mask(self.masks['zero'])
        
        # In case there is no signal to excplicitely define the measurement type, infer the zero and calibration masks
        if ((len(zero_stop) == 0) or 
            (len(calib_stop) == 0)):
            print('ERROR: no stops of zero/calib measurement types detected,')
            print('Calibration and zero data from {} inferred from mz = {}.'.format(self.df_data.index.mean(), mz_exact_infer_masks))
            mask_zero, mask_calib = msk_r.infer_zero_and_calib_mask(self.df_data,
                                                                    mz_exact_infer_masks,
                                                                    estimator_averaging = '5min',
                                                                    method='max',
                                                                    zero_tolerance=100,
                                                                    calib_tolerance=5,
                                                                    filter_buffer=dt.timedelta(seconds=60),
                                                                    match_switches=dt.timedelta(seconds=600),
                                                                    version=2)
            print('Overwriting the zero and calibration mask values from inferred.')
            self.masks['zero'] = mask_zero
            self.masks['calib'] = mask_calib
            
            # if inferred, trim 10 seconds before and after identified switch
            self.trim_masks(dt.timedelta(seconds=10), dt.timedelta(seconds=10))


        # Zero interval averaging mask
        mask_calc_zero = msk_r.get_representative_mask_from_multiple_intervals(self.df_data, self.masks['zero'], tdelta_buf_zero,  tdelta_avg_zero, tdelta_min_zero)
    
        # Calibration interval averaging mask
        mask_calc_calib = msk_r.get_representative_mask_from_multiple_intervals(self.df_data, self.masks['calib'], tdelta_buf_calib, tdelta_avg_calib, tdelta_min_calib)
    
        # Check stability mask
        mask_check_stability = msk_r.get_representative_mask_from_multiple_intervals(self.df_data, self.masks['calib'], tdelta_buf_calib, tdelta_stability, tdelta_min_calib)
        
        if (((self.masks['invalid'] & mask_calc_zero).sum() >= 1) or 
            ((self.masks['invalid'] & mask_calc_calib).sum() >= 1)):
            print('WARNING: invalid measurements during calibration/zero intervals,')
            print('Calibration data from {} not processed for further analysis.'.format(self.df_data.index.mean()))

        
        return mask_calc_zero, mask_calc_calib, mask_check_stability

    def process_RH_calibration(self, calibration_ancillary, dir_o_cal,
                             tdelta_buf_zero=dt.timedelta(minutes=1),tdelta_avg_zero=dt.timedelta(minutes=5),tdelta_min_zero=dt.timedelta(minutes=20),
                             tdelta_buf_calib=dt.timedelta(minutes=1),tdelta_avg_calib=dt.timedelta(minutes=5),tdelta_min_calib=dt.timedelta(minutes=70),
                             tdelta_stability=dt.timedelta(minutes=60), zero = 'ffill',
                             rate_coeff_col_calib = '', Xr0_default = 1,
                             Q_calib = 10, Q_zero_air = 800, mz_exact_infer_masks = 59.049,
                             Q_ptrms_corr = 'BE-Vie_2022',
                             zero_tolerance = 100, calib_tolerance = 5, infer_method = 'max',
                             estimator_averaging='5min',filter_buffer=dt.timedelta(seconds=60),match_switches=dt.timedelta(seconds=1800),
                             allow_below_zero=True, scan = None):
        
        mz_exact = calibration_ancillary.index.values
        k = calibration_ancillary[rate_coeff_col_calib]
        mixrat_bottle = calibration_ancillary['mixrat_bottle [ppbv]']
        fr = calibration_ancillary['fr']
        
        # Indicate if we use this signal for transmission: i.e., no fragmentation expected
        Tr_selection = calibration_ancillary['Tr_selection']
        Tr_selection = calibration_ancillary[Tr_selection].index.values
        
        # Infer the masks from signal since the valve signal is not correct during the adapted RH setup
        mask_zero, mask_calib = msk_r.infer_zero_and_calib_mask(self.df_data, 
                                                                  mz_exact_infer_masks,
                                                                  estimator_averaging=estimator_averaging,
                                                                  method=infer_method,
                                                                  zero_tolerance=zero_tolerance,
                                                                  calib_tolerance=calib_tolerance,
                                                                  filter_buffer=filter_buffer,
                                                                  match_switches=match_switches,
                                                                  version=2)
        self.masks['zero'] = mask_zero
        self.masks['calib'] = mask_calib

        # Trim the masks with a standard wider buffer as the infering may produce some errors
        self.trim_masks(dt.timedelta(seconds=10), dt.timedelta(seconds=10))
        
        # Get averaging masks
        mask_calc_zero = msk_r.get_representative_mask_from_multiple_intervals(self.df_data,mask_zero,tdelta_buf_zero,tdelta_avg_zero, tdelta_min_zero)
        mask_calc_calib = msk_r.get_representative_mask_from_multiple_intervals(self.df_data,mask_calib,tdelta_buf_calib,tdelta_avg_calib, tdelta_min_calib)

        # Plot the masks as a way to make sure that everything has gone according to expectations
        f, ax = plt.subplots(1,1)
        
        mz_col = mz_r.select_mz_cluster_from_exact(mz_exact_infer_masks, self.df_clusters, tol = 0.01, mute = True)
        if np.isnan(mz_col):
            print('No signal at {} found, default identifying masks to isoprene.'.format(mz_exact_infer_masks))
            mz_col = mz_r.select_mz_cluster_from_exact(69.069, self.df_clusters, tol = 0.01, mute = True)
        
        self.df_data[mz_col].plot(c='slategray',linewidth=0,marker='.',markersize=2,ax=ax,label='mz = {}'.format(mz_exact_infer_masks))
    
        self.df_data[mz_col][mask_zero].plot(c='k',linewidth=0,marker='.',markersize=2,ax=ax,label='inferred zero')
        self.df_data[mz_col][mask_calib].plot(c='g',linewidth=0,marker='.',markersize=2,ax=ax,label='inferred calib')
        
        self.df_data[mz_col][mask_calc_zero | mask_calc_calib].plot(c='goldenrod',linewidth=0,marker='.',markersize=2,ax=ax,label='avg zero/calib')
        ax.set_ylabel('CR mz = {}'.format(mz_exact_infer_masks))
    
        # Get legend handles and labels to merge
        h,l = ax.get_legend_handles_labels()
    
        #Merging two legends
        ax.legend(h, l, title_fontsize='10',bbox_to_anchor=(1.1,1.05),loc='upper left', ncol=1)
        plt.show()
        plt.close()
        
        Q_calib = calib_r.get_Q_calib_corrected(Q_calib)
        
        P_inlet = self.df_P_inlet.mean()

        Q_PTRMS = calib_r.get_Q_PTRMS_corrected(P_inlet,campaign=Q_ptrms_corr)

        # Get the Tr_PH1_to_PH2 factor
        ##############################
        anc = []
        for mz in [33.033, 42.034]:
            mz_anc = {}
            mz_anc['mz']  = mz
            mz_anc['col'] = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)
            mz_anc['mixingRatio_driftTube'] = calib_r.get_mixingRatio_driftTube_Calib(Q_calib,Q_PTRMS,Q_zero_air,mixrat_bottle[mz])
            mz_anc['k'] = k[mz]
            mz_anc['fractioning'] = fr[mz]
            mz_anc['Xr0'] = Xr0_default
        
            anc.append(mz_anc)
        
        t_reac, N_DT = calib_r.get_ancilarry_PTRT_drift_calib_av(self.df_P_drift.mean(),
                                                                 self.df_T_drift.mean(),
                                                                 self.df_U_drift.mean())
        
        if not self.data_units == 'cps':
            print('Error, units not correct to calculate Tr_PH1_to_PH2')
            print(self.data_units)
            raise ValueError
        
        # self.Tr_PH1_to_PH2 = calib_r.get_Tr_PH1_to_PH2(self.df_data, mask_calc_zero, mask_calc_calib, 
        #                                                self.mz_col_21, self.mz_col_38, anc[0], anc[1], N_DT, t_reac,
        #                                                FPH1=self.FPH1, FPH2=self.FPH2,Xr0_default=Xr0_default)
        self.Tr_PH1_to_PH2 = calib_r.get_Tr_PH1_to_PH2(self.df_data, mask_calc_zero, mask_calc_calib, 
                                                       self.mz_col_primIon['H3O1_21.022'], self.mz_col_primIon['H5O2_38.033'], anc[0], anc[1], N_DT, t_reac,
                                                       FPH1=self.FPH1, FPH2=self.FPH2,Xr0_default=Xr0_default)
        
        N = 21
        Xr0 = {}
        for mz in mz_exact:
            if np.isnan(mixrat_bottle[mz]):
                continue
            
            mz_col = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)
            if np.isnan(mz_col):
                continue
            
            tmp_PTR_data = self.get_PTR_data_subsample_mz([mz_col])
            # tmp_PTR_data.transform_data_subtract_zero(tdelta_buf_zero = tdelta_buf_zero, tdelta_avg_zero=tdelta_avg_zero, zero=zero)
            
            if scan is None:
                if allow_below_zero:
                    Xrs = [-20,0,20]
                else:
                    Xrs = [0, 10, 20]
    
                r_stds = []
                for i in np.arange(2*N+1):
                    if i == len(Xrs):
                        A, B = sorted(r_stds)[0:2]
                        iA = np.argmin(abs(np.array(r_stds)-A))
                        iB = np.argmin(abs(np.array(r_stds)-B))
                
                        dist = (Xrs[iA]+Xrs[iB])/2.-Xrs[iA]
                        Xrs.append(Xrs[iA]+dist) # In between the two lowest values
                        Xrs.append(Xrs[iA]-dist) # Symmetric round the lowest value to check if we didn't
                        
                        if ((not allow_below_zero) and 
                            (Xrs[-1] < 0)):
                            Xrs = Xrs[:-1]
                        
                    tmp, tmp_prec = tmp_PTR_data.get_data_ncr({mz_col:Xrs[i]},Xr0_default)[mask_calc_calib]
                    r_std = tmp[mz_col].std()/abs(tmp[mz_col]).mean()
                
                    r_stds.append(r_std)
            else:
                r_stds = []
                Xrs = scan
                for Xr in Xrs:
                    tmp, tmp_prec = tmp_PTR_data.get_data_ncr({mz_col:Xr},Xr0_default)[mask_calc_calib]
                    r_std = tmp[mz_col].std()/abs(tmp[mz_col]).mean()
                    r_stds.append(r_std)
            
            i = np.argmin(np.array(r_stds))
            print('{}: {:.3f} ({:.2f} %)'.format(mz_col, Xrs[i],r_stds[i]*100.))
            
            Xr0[mz] = Xrs[i]
            
            # Show plots to compute Xr0
            f, axs = plt.subplots(1,2,figsize=(7.5,3))
            axs = axs.flatten()
            # self.df_data[[self.mz_col_38,mz_col]].plot(ax=axs[0],linewidth=1)
            self.df_data[[self.mz_col_primIon['H5O2_38.033'],mz_col]].plot(ax=axs[0],linewidth=1)
            axs[0].set_ylabel('PA [cps]')
            axs[0].legend()
            
            for iXr0 in [1,0,Xrs[i]]:
                tmp, tmp_prec = tmp_PTR_data.get_data_ncr({mz_col:iXr0},Xr0_default)[mask_calc_calib]
                axs[1].plot(np.arange(len(tmp[mz_col])),tmp[mz_col],label='Xr,0={:.3f}'.format(iXr0),linewidth=1)
                axs[1].legend()
                axs[1].set_xlabel('sample')
                axs[1].set_ylabel('PA [ncps]')
            
            plt.suptitle('X$_r,_0$ ({}): {:.3f} (Rel. std = {:.2f} %)'.format(mz_col, Xrs[i], r_stds[i]*100.))
            plt.tight_layout()
            
            plt.show()
            plt.close()
            
        return Xr0

    def process_calibration(self, calibration_ancillary, dir_o_cal,
                             tdelta_buf_zero=dt.timedelta(minutes=1),tdelta_avg_zero=dt.timedelta(minutes=5),tdelta_min_zero=dt.timedelta(minutes=20),
                             tdelta_buf_calib=dt.timedelta(minutes=1),tdelta_avg_calib=dt.timedelta(minutes=5),
                             tdelta_stability=dt.timedelta(minutes=60), zero = 'ffill',
                             rate_coeff_col_calib = '', dict_Xr0={}, Xr0_default=1.,
                             Q_zero_air = 800, Q_zero_air_unc = 10, # sccm, 1% of full scale (0-1000 sccm)
                             Q_calib = 10, Q_calib_unc = 0.1,       # unc = 1% stated uncertainty on set point 
                             precision_calc = 'Distribution', Q_ptrms_corr = 'BE-Vie_2022'):
        
        rstd = {}
        cc_coeff = {}
        tr_coeff = {}
        anc_info = {}
        
        mz_exact = calibration_ancillary.index.values
        k = calibration_ancillary[rate_coeff_col_calib]
        mixrat_bottle = calibration_ancillary['mixrat_bottle [ppbv]']
        mixrat_bottle_unc = calibration_ancillary['1_sigm [ppbv]']
        fr = calibration_ancillary['fr']
        
        # Indicate if we use this signal for transmission: i.e., no fragmentation expected
        Tr_selection = calibration_ancillary['Tr_selection']
        Tr_selection = calibration_ancillary[Tr_selection].index.values
        
        # Assume we don't have to trim the masks due to the buffers for the zero and calibration measurements
        # we do apply the trimming here explicitely to supress warnings when calculating the representativity masks
        self.trim_masks(dt.timedelta(seconds=0), dt.timedelta(seconds=0)) 
        mask_calc_zero, mask_calc_calib, mask_check_stability = self.get_calib_calc_masks(tdelta_buf_zero=tdelta_buf_zero,tdelta_avg_zero=tdelta_avg_zero,
                                                                                          tdelta_buf_calib=tdelta_buf_calib,tdelta_avg_calib=tdelta_avg_calib,
                                                                                          tdelta_stability=tdelta_stability)
        
        ctime = self.df_data.index[mask_calc_calib].mean()
        
        if ((mask_calc_zero.sum() == 0) or 
            (mask_calc_calib.sum() == 0)):
                print('Error during Calibration at {}, no representative zero/calibration measurements found.'.format(ctime))
                return None, None, None, None
        
        Q_calib     = calib_r.get_Q_calib_corrected(Q_calib)      # Correction from reference point of flow meter
        Q_calib_unc = calib_r.get_Q_calib_corrected(Q_calib_unc)  # Correction from reference point of flow meter
        
        P_inlet = self.df_P_inlet[mask_calc_calib].mean()
        P_inlet_unc = self.df_P_inlet[mask_calc_calib].std()
        P_inlet_unc = max(P_inlet_unc,P_inlet*0.001) # For corrected axes, there is no variability so make sure there is some error associated with this variable here
        
        Q_PTRMS, Q_PTRMS_unc = calib_r.get_Q_PTRMS_corrected(P_inlet, P_inlet_unc, campaign=Q_ptrms_corr)
        
        anc_info['ctime'] = ctime.round('1s')
        anc_info['P_inlet']        = P_inlet
        anc_info['P_inlet_unc']    = P_inlet_unc
        anc_info['Q_PTRMS']        = Q_PTRMS
        anc_info['Q_PTRMS_unc']    = Q_PTRMS_unc
        anc_info['Q_calib']        = Q_calib
        anc_info['Q_calib_unc']    = Q_calib_unc
        anc_info['Q_zero_air']     = Q_zero_air
        anc_info['Q_zero_air_unc'] = Q_zero_air_unc
        
        # Get the Tr_PH1_to_PH2 factor
        ##############################
        anc = []
        for mz in [33.033, 42.034]:
            mz_anc = {}
            mz_anc['mz']  = mz
            mz_anc['col'] = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)
            mz_anc['mixingRatio_driftTube'] = calib_r.get_mixingRatio_driftTube_Calib(Q_calib,Q_PTRMS,Q_zero_air,mixrat_bottle[mz])
            mz_anc['k'] = k[mz]
            mz_anc['fractioning'] = fr[mz]
            if mz_anc['col'] in dict_Xr0.keys():
                mz_anc['Xr0'] = dict_Xr0[mz_anc['col']]
            else:
                mz_anc['Xr0'] = Xr0_default

            anc.append(mz_anc)
        
        t_reac, t_reac_unc, N_DT, N_DT_unc = calib_r.get_ancilarry_PTRT_drift_calib_av_unc(self.df_P_drift.mean(),self.df_P_drift.std(),
                                                                                           self.df_T_drift.mean(),self.df_T_drift.std(),
                                                                                           self.df_U_drift.mean(),self.df_U_drift.std())
        
        anc_info['t_reac']     = t_reac
        anc_info['N_DT']       = N_DT
        
        anc_info['t_reac_unc'] = t_reac_unc
        anc_info['N_DT_unc']   = N_DT_unc
        
        if not self.data_units == 'cps':
            print('Error, units not correct to calculate Tr_PH1_to_PH2')
            print(self.data_units)
            raise ValueError
        
        # self.Tr_PH1_to_PH2 = calib_r.get_Tr_PH1_to_PH2(self.df_data, mask_calc_zero, mask_calc_calib, 
        #                                                self.mz_col_21, self.mz_col_38, anc[0], anc[1], N_DT, t_reac, 
        #                                                FPH1=self.FPH1, FPH2=self.FPH2,Xr0_default=Xr0_default)

        self.Tr_PH1_to_PH2 = calib_r.get_Tr_PH1_to_PH2(self.df_data, mask_calc_zero, mask_calc_calib, 
                                                       self.mz_col_primIon['H3O1_21.022'], self.mz_col_primIon['H5O2_38.033'], anc[0], anc[1], N_DT, t_reac,
                                                       FPH1=self.FPH1, FPH2=self.FPH2,Xr0_default=Xr0_default)
        
        # Get transmissions and calibration coefficients
        ################################################
        # Zero is not expected to be subtracted for the calibr_r routines!
        self.transform_data_ncr(dict_Xr0=dict_Xr0,Xr0_default=Xr0_default)
        
        # Check stability of signal during calibrations
        tmp = self.df_data[mask_check_stability].std()/self.df_data[mask_check_stability].mean()
        for mz in mz_exact:
            mz_col = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)
            if not mz_col in tmp.keys():
                rstd[mz] = np.nan
                continue
            rstd[mz] = tmp[mz_col]
        
        # Get transmissions
        for mz in Tr_selection:
            if not mz in mixrat_bottle.keys():
                continue
            if np.isnan(mixrat_bottle[mz]):
                continue
            
            mz_col = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)
            if not mz_col in self.df_data.keys():
                print('Transmission not determined for mz {}.'.format(mz))
                tr_coeff[mz] = np.nan
                continue
            
            MR_DT = calib_r.get_mixingRatio_driftTube_Calib(Q_calib, Q_PTRMS, Q_zero_air, mixrat_bottle[mz])
            
            Tr = calib_r.get_transmissionCoefficient(self.df_data, mask_calc_zero, mask_calc_calib,  mz_col,  MR_DT,  k[mz],  fr[mz], N_DT, t_reac)
            
            tr_coeff[mz] = Tr
        
        # Add transmission at 21 (used for normalisation, important to interpolate the Tr_coeff in processing)
        tr_coeff[21.022] = 1.
        
        # Get calibration coefficients --  Here we have to subtract zero before we can calculate the CCs
        self.transform_data_subtract_zero(tdelta_buf_zero = tdelta_buf_zero, tdelta_avg_zero=tdelta_avg_zero, tdelta_min_zero=tdelta_min_zero, zero=zero)
        for mz in mz_exact:
            if not mz in mixrat_bottle[mixrat_bottle.notna()].keys():
                continue
        
            mz_col = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)
            if mz_col in self.df_data.keys():
                signal = self.df_data[mask_calc_calib][mz_col].mean()

                # signal_unc = self.df_prec[mask_calc_calib][mz_col].mean()/np.sqrt(mask_calc_calib.sum())
                signal_unc = self.df_data[mask_calc_calib][mz_col].std()/np.sqrt(mask_calc_calib.sum())

            else:
                print('WARNING NO SIGNAL FOUND FOR MZ {} IN IDA FILE WITH CALIBRATION'.format(mz))
                signal = np.nan
                signal_unc = np.nan
            
            # If the mz value is not in the calibration standard, continue to next loop
            if mixrat_bottle[mz] == 1:
                continue
            
            dir_o_calibration = '{1}{2}{0}'.format(os.sep,dir_o_cal,ctime.strftime('%Y%m%d_%Hh%M'))
            
            check_create_output_dir(dir_o_calibration)
            if not np.isnan(signal):
                self.save_plot_masks(mz,additional_masks = {'zero calc':mask_calc_zero,'mask stability check':mask_check_stability,'calib calc':mask_calc_calib}, dir_o = dir_o_calibration)
            
            MR_DT, MR_DT_unc = calib_r.get_mixingRatio_driftTube_Calib_unc(Q_calib, Q_calib_unc,
                                                                           Q_PTRMS, Q_PTRMS_unc,
                                                                           Q_zero_air, Q_zero_air_unc,
                                                                           mixrat_bottle[mz], mixrat_bottle_unc[mz])
            anc_info['MR_DT_mz_{}'.format(mz)] = MR_DT
            anc_info['MR_DT_acc_mz_{}'.format(mz)] = MR_DT_unc
            
            CC = signal/MR_DT
            CC_racc = MR_DT_unc/MR_DT*100.
            CC_rprec = signal_unc/signal*100.
            
            cc_coeff[mz] = CC
            cc_coeff['{}_racc [%]'.format(mz)] = CC_racc
            cc_coeff['{}_rprec [%]'.format(mz)] = CC_rprec
        
        tr_coeff['ctime'] = ctime.round('1s')
        cc_coeff['ctime'] = ctime.round('1s')
        rstd['ctime']     = ctime.round('1s')
        
        # cc_coeff['I_cps_H3O1_21'] = self.df_data[mask_calc_calib][self.mz_col_21].mean()
        # cc_coeff['I_cps_H5O2_38'] = self.df_data[mask_calc_calib][self.mz_col_38].mean()
        for key, val in self.mz_col_primIon.items():
            try:
                cc_coeff['I_cps_{}'.format(key)] = self.df_data[mask_calc_calib][val].mean()
            except:
                cc_coeff['I_cps_{}'.format(key)] = np.nan
            
        return tr_coeff, cc_coeff, rstd, anc_info
    
    def save_plot_masks(self, mz, additional_masks = {}, dir_o = './'):
        mz_col = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)

        f, ax = plt.subplots(1,1)
        ax = self.plot_masks(mz_col,ax,additional_masks)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title('m/z = {} (cluster at {}), {}'.format(mz, mz_col, self.df_data.index.mean().strftime('%Y-%m-%d')))
        plt.savefig('{}{}.png'.format(dir_o,mz))
        plt.close()
        
        return None
        
    def show_plot_masks(self, mz, additional_masks = {}):
        mz_col = mz_r.select_mz_cluster_from_exact(mz, self.df_clusters, tol = 0.01, mute = True)

        f, ax = plt.subplots(1,1)
        ax = self.plot_masks(mz_col,ax,additional_masks)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.title('m/z = {} (cluster at {}), {}'.format(mz, mz_col, self.df_data.index.mean().strftime('%Y-%m-%d')))
        plt.show()
        plt.close()
        
        return None
    
    def plot_masks(self, mz_col, ax, additional_masks = {}):
        masks = []
        for mask in self.masks.keys():
            if self.masks[mask].sum() >= 1:
                masks.append(mask)
                
        if ((not np.isnan(mz_col)) and 
            (mz_col in self.df_data.columns)):
            color = iter(plt.cm.nipy_spectral(np.linspace(0, 1, len(masks)+len(additional_masks.keys()))))
            
            shown = np.zeros(self.masks['zero'].shape,dtype=bool)
            for mask in masks: 
                c = next(color)
                self.df_data[mz_col][self.masks[mask]].plot(c=c,linewidth=0,marker='.',markersize=2,ax=ax,label=mask)
                shown = (shown | self.masks[mask])
                
            for mask in additional_masks.keys(): 
                c = next(color)
                self.df_data[mz_col][additional_masks[mask]].plot(c=c,linewidth=0,marker='.',markersize=2,ax=ax,label=mask)
                shown = (shown | additional_masks[mask])
                
            if (~shown).sum()>=1:
                self.df_data[mz_col][~shown].plot(c='slategray',linewidth=0,marker='.',markersize=2,ax=ax, label = 'not shown')
        
        ax.set_ylabel('I [{}]'.format(self.data_units))
        
        return ax
    
    def get_output_filename(self, time, key, campaign, instrument, file_format):
        outName = ''
        if file_format == 'conf0':
            outName = '{}_{}_{}_{}.h5'.format(key.split('_')[0],campaign,instrument,time.strftime('%Y%m%d-%H%M%S'))
        elif file_format == 'EC':
            outName = '{}_{}_{}_{}.h5'.format(key.split('_')[0],campaign,instrument,time.strftime('%Y_%m_%d__%H_%M_%S'))
            
        return outName

    def save_output_EBAS(self, dir_o, masses = [], file_format = 'EBAS_crist_pp'):
        if len(masses)==0:
            mz_sel = self.df_data.columns.values
        else:
            tmp = mz_r.select_mz_clusters_from_exact(masses, self.df_clusters)
            mz_sel = self.df_data.columns.intersection(tmp)
            mz_mis = mz_sel.difference(tmp)
    
        df_data = self.df_data[mz_sel]*1.e3
        df_prec = self.df_prec[mz_sel]*1.e3
        df_DL = 3.*self.df_zero_prec[mz_sel]*1.e3
        df_expUnc = 2.*(self.df_prec[mz_sel].pow(2)+(self.df_data[mz_sel]*self.df_racc[mz_sel]).pow(2)).pow(0.5)*1.e3

        df_data[mz_mis] = np.nan
        df_prec[mz_mis] = np.nan
        df_DL[mz_mis] = np.nan
        df_expUnc[mz_mis] = np.nan

        # df_zero = self.df_zero[mz_sel]*1.e3
        # df_zero_prec = self.df_zero_prec[mz_sel]*1.e3
        
        mask = np.zeros(len(self.masks['invalid']))
        setup_flags = np.array(self.masks['invalid']*0.999)
        setup_flags += self.masks['zero']*0.684
        setup_flags += self.masks['calib']*0.683
    
        mask += self.masks['invalid']
        mask += self.masks['zero']
        mask += self.masks['calib']
        # Setup different levels as valid but with local contamination
        for level in np.arange(1,8):
            setup_flags += self.masks['K{}_trimmed'.format(level)]*0.559
            mask += self.masks['K{}_trimmed'.format(level)]
    
        mask += self.masks['K8_trimmed']
        setup_flags += (mask==0)*0.999
        if (mask>1).sum() >= 1:
            print('Error, {} multi-masked points: flags associated points,'.format((mask>1).sum()))
            print(df_data.index[mask>1])
            print('Setting flag to 0.999')
            setup_flags[mask>1] = 0.999
        
        df_flag = pd.DataFrame(1, index=df_data.index, columns=df_data[mz_sel].columns)
        df_flag = df_flag.mul(setup_flags,axis=0)
    
        df_flag[((df_data<df_DL) & 
                 (df_flag != 0.684) &
                 (df_flag != 0.999))] = 0.147
        
        df_flag[mz_mis] = 0.147
    
        df_data.columns = ['{:.4f}'.format(col) for col in df_data.columns]
        df_expUnc.columns = ['{:.4f}_expUnc'.format(col) for col in df_expUnc.columns]
        df_prec.columns = ['{:.4f}_prec'.format(col) for col in df_prec.columns]
        df_flag.columns = ['{:.4f}_flag'.format(col) for col in df_flag.columns]
        
        # df_zero.columns = ['{:.4f}_zero'.format(col) for col in df_zero.columns]
        # df_zero_prec.columns = ['{:.4f}_zeroPrec'.format(col) for col in df_zero_prec.columns]
    
        df_concat = pd.concat([df_data, df_expUnc, df_prec, df_flag],axis=1)
        
        # df_concat = pd.concat([df_concat, df_zero, df_zero_prec],axis=1)
    
        df_concat['t_start'] = df_concat.index
        df_concat['t_stop']  = df_concat.t_start+pd.to_timedelta(self.sst*1.e-3,unit='s')
    
        df_concat.fillna(-999, inplace = True)
    
        if file_format == 'EBAS_crist_pp':
            df_concat['t_start'] = to_matlabtime(df_concat['t_start'])
            df_concat['t_stop'] = to_matlabtime(df_concat['t_stop'])
            
        column_order = ['t_start', 't_stop']
        for mz in mz_sel:
            column_order.append('{:.4f}'.format(mz))
            column_order.append('{:.4f}_expUnc'.format(mz))
            column_order.append('{:.4f}_prec'.format(mz))
            column_order.append('{:.4f}_flag'.format(mz))
            
            # column_order.append('{:.4f}_zero'.format(mz))
            # column_order.append('{:.4f}_zeroPrec'.format(mz))
        
        fname = 'EBAS_{}'.format(df_concat.index.min().strftime('%Y%m%d-%H%M%S'))
        
        df_concat[column_order].to_csv('{}{}{}.csv'.format(dir_o,os.sep,fname))
        np.savetxt('{}{}{}.txt'.format(dir_o,os.sep,fname), df_concat[column_order].values, delimiter='\t')
    
        return None
   
    
    def write_hdf5File(self, f_output, tmask, df_tr_coeff, df_cc_coeff, time_info=None, processing_config={}):
        mz = self.df_data.columns.values
        sel_vmr = self.df_data.columns.difference(self.mz_col_primIon.values())
        sel_primIon = self.df_data.columns.intersection(self.mz_col_primIon.values())
        
        if time_info is None:
            time = self.df_data[tmask].index.values
        else:
            time = self.df_data[tmask].tz_convert(time_info['tz_out']).index

        da_data = xr.DataArray(
            data=self.df_data[tmask][sel_vmr].values,
            dims=["time","mz"],
            coords=dict(
                mz=self.df_data[sel_vmr].columns.values,
                time=time,
            ),
            attrs=dict(
                description=self.data_description,
                units=self.data_units,
            ),
        )
        
        da_prec = xr.DataArray(
            data=self.df_prec[tmask][sel_vmr].values,
            dims=["time","mz"],
            coords=dict(
                mz=self.df_data[sel_vmr].columns.values,
                time=time,
            ),
            attrs=dict(
                description=self.prec_description,
                units=self.prec_units,
            ),
        )
        
        if self.data_units == 'ppbv':
            prim_ion_units = "Transmission corrected counts per second"
        if self.data_units == 'tr-ncps':
            prim_ion_units = "Transmission corrected counts per second"
        if self.data_units == 'ncps':
            prim_ion_units = "cps"
            
        da_primIon = xr.DataArray(
            data=self.df_data[tmask][sel_primIon].values,
            dims=["time","mz"],
            coords=dict(
                mz=self.df_data[sel_primIon].columns.values,
                time=time,
            ),
            attrs=dict(
                description="Primary ion signals",
                units=prim_ion_units,
            ),
        )
        
        if self.prec_units == 'ppbv':
            prim_ion_units = "Transmission corrected counts per second"
        if self.prec_units == 'tr-ncps':
            prim_ion_units = "Transmission corrected counts per second"
        if self.prec_units == 'ncps':
            prim_ion_units = "cps"
        da_primIon_prec = xr.DataArray(
            data=self.df_prec[tmask][sel_primIon].values,
            dims=["time","mz"],
            coords=dict(
                mz=self.df_data[sel_primIon].columns.values,
                time=time,
            ),
            attrs=dict(
                description="Precision on primary ion signals",
                units=prim_ion_units,
            ),
        )
        
        tmp_mask = self.df_clusters.index.isin(mz)
        limits_min = xr.DataArray(
            data=self.df_clusters[tmp_mask].cluster_min.values,
            dims=["mz"],
            coords=dict(
                mz=self.df_clusters[tmp_mask].index.values,
            ),
            attrs=dict(
                description="Cluster min",
                units="atomic mass unit",
            ),
        )
        
        limits_max = xr.DataArray(
            data=self.df_clusters[tmp_mask].cluster_max.values,
            dims=["mz"],
            coords=dict(
                mz=self.df_clusters[tmp_mask].index.values,
            ),
            attrs=dict(
                description="Cluster max",
                units="atomic mass unit",
            ),
        )
        
        Xr0 = xr.DataArray(
            data=self.df_clusters[tmp_mask].Xr0.values,
            dims=["mz"],
            coords=dict(
                mz=self.df_clusters[tmp_mask].index.values,
            ),
            attrs=dict(
                description="Xr0",
                units="",
            ),
        )
        
        # Set the first principles VMR CC parameters to NaN for directly calibrated compounds
        tmp = self.df_clusters[tmp_mask][['k [1.e-9 cm3 molecule-1 s-1]','FY', 'IF']]
        cal_sel = tmp.index.intersection(df_cc_coeff.columns)
        tmp.loc[[idx in cal_sel for idx in tmp.index],:] = np.nan
        tmp.loc[[idx in sel_primIon for idx in tmp.index],:] = np.nan
        k_reac = xr.DataArray(
            data=tmp['k [1.e-9 cm3 molecule-1 s-1]'].values,
            dims=["mz"],
            coords=dict(
                mz=tmp.index.values,
            ),
            attrs=dict(
                description="Ion/Molecule reaction rate constant",
                units="1.e-9 cm3 molecule-1 s-1",
            ),
        )
        
        FY = xr.DataArray(
            data=tmp.FY.values,
            dims=["mz"],
            coords=dict(
                mz=tmp.index.values,
            ),
            attrs=dict(
                description="Fragmentation yield to correct signal due to fragmentation in the drift tube",
                units="",
            ),
        )
        
        IF = xr.DataArray(
            data=tmp.IF.values,
            dims=["mz"],
            coords=dict(
                mz=tmp.index.values,
            ),
            attrs=dict(
                description="Isotopic factor to correct isotopic ratio",
                units="",
            ),
        )
        
        # Should still be adapted to take time interpolated transmissions into account
        transmission = xr.DataArray(
            data=df_tr_coeff.T.values[:,0],
            dims=["mz"],
            coords=dict(
                mz=df_tr_coeff.columns.values,
            ),
            attrs=dict(
                description="Transmission coefficient",
                units="Transmission relative to Tr_21",
            ),
        )
        
        calibration = xr.DataArray(
            data=df_cc_coeff.T.values[:,0],
            dims=["mz"],
            coords=dict(
                mz=df_cc_coeff.columns.values,
            ),
            attrs=dict(
                description="Transmission corrected calibration coefficient",
                units="ppbv tc-ncps-1",
            ),
        )
                    
        da_sst = xr.DataArray(
            data=self.sst[tmask].values,
            dims=["time"],
            coords=dict(
                time=time,
            ),
            attrs=dict(
                description='Measurement time interval',
                units=self.sst_units,
            ),
        )

        data_vars = {'Signal':da_data,'Signal_precision':da_prec,
                     'Signal_primaryIons':da_primIon,'Signal_primaryIons_prec':da_primIon_prec,
                     'cluster_min':limits_min,'cluster_max':limits_max,
                     'Xr0':Xr0,'k_reac':k_reac,'FY':FY,'IF':IF,
                     'transmission':transmission,'calibration':calibration,
                     'MeasurementsTimeInterval':da_sst}
        
        if not (self.df_zero is None):
            da_zero = xr.DataArray(
                data=self.df_zero[tmask][sel_vmr].values,
                dims=["time","mz"],
                coords=dict(
                    mz=self.df_zero[sel_vmr].columns.values,
                    time=time,
                ),
                attrs=dict(
                    description=self.zero_description,
                    units=self.zero_units,
                ),
            )
            data_vars['zero'] = da_zero

            da_zero_prec = xr.DataArray(
                data=self.df_zero_prec[tmask][sel_vmr].values,
                dims=["time","mz"],
                coords=dict(
                    mz=self.df_zero[sel_vmr].columns.values,
                    time=time,
                ),
                attrs=dict(
                    description=self.zero_prec_description,
                    units=self.zero_units,
                ),
            )
            data_vars['zero_precision'] = da_zero_prec
            
        if not (self.df_racc is None):
            da_acc = xr.DataArray(
                data=self.df_racc[tmask].mul(abs(self.df_data[tmask]))[sel_vmr].values,
                dims=["time","mz"],
                coords=dict(
                    mz=self.df_data[sel_vmr].columns.values,
                    time=time,
                ),
                attrs=dict(
                    description=self.data_description,
                    units=self.data_units,
                ),
            )
            data_vars['Signal_accuracy'] = da_acc
    
        attrs = {}
        attrs['Peak Area Analysis version'] = __version__
        attrs['tz_info'] = str(time_info['tz_out'])
        if self.default_CC_kinetic is None:
            attrs['default_CC_kinetic'] = np.nan
        else:
            attrs['default_CC_kinetic'] = self.default_CC_kinetic.round(4)
        attrs['FPH1'] = self.FPH1
        attrs['FPH2'] = self.FPH2
        
        valid_types = (str, int, float, complex, np.number, np.ndarray, list, tuple) # Valid types for serialization of netcdf output
        for key, val in processing_config.items():
            key = 'config_{}'.format(key)
            if isinstance(val, valid_types):
                attrs[key] = val
            else:
                attrs[key] = str(val)
        
        print('Writing: {}'.format(f_output))
        
        ds_PTR = xr.Dataset(data_vars, attrs=attrs)
        ds_PTR.to_netcdf(f_output,engine='h5netcdf')
            
        return None
   
    
    def save_output(self, dir_o, campaign, instrument, df_tr_coeff, df_cc_coeff, masks = [], threshold = 1, file_format = 'conf0', time_info=None, processing_config={},output_full=False):
        # Save the processed data
        #########################
        rest_mask = np.zeros(len(self.masks['invalid']))
        
        for key in self.masks.keys():
            if not 'trimmed' in key:
                continue
            if self.masks[key].sum() < threshold:
                continue
            if 'invalid' in key:
                continue
            if ((len(masks) != 0) and 
                (not key.split('_')[0] in masks)
                ):
                continue
            
            rest_mask += self.masks[key]

            if time_info is None:
                time = self.df_data[self.masks[key]].index.values
            else:
                time = self.df_data[self.masks[key]].tz_convert(time_info['tz_out']).index
                
            out_time = pd.to_datetime(time.min()) # Convert from numpy datetime to dt.datetime                
            file_name = self.get_output_filename(out_time, key, campaign, instrument, file_format)
            f_output = '{}{}'.format(dir_o, file_name)
            
            self.write_hdf5File(f_output, self.masks[key], df_tr_coeff, df_cc_coeff, time_info=time_info,processing_config=processing_config)

        if (rest_mask>1).sum() >= 1:
            print('Error, {} multi-outputed points!'.format((rest_mask>1).sum()))
            raise ValueError
            
        # trimmed and invalid data
        if output_full:
            rest_mask = (rest_mask == 0)
            if time_info is None:
                time = self.df_data[rest_mask].index.values
            else:
                time = self.df_data[rest_mask].tz_convert(time_info['tz_out']).index
                
            out_time = pd.to_datetime(time.min()) # Convert from numpy datetime to dt.datetime
            
            file_name = self.get_output_filename(out_time, 'full', campaign, instrument, file_format)
            f_output = '{}{}'.format(dir_o, file_name)
            self.write_hdf5File(f_output, rest_mask, df_tr_coeff, df_cc_coeff, time_info=time_info,processing_config=processing_config)

        return None
