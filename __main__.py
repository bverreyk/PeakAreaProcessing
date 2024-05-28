# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:52:49 2024

@author: bertve
"""
import numpy as np
import pandas as pd
import datetime as dt

import os

import h5py

import sys
sys.path.append('C:/Users/bertve/Documents/GitLab/')

from PAP import PTR_objects
from PAP import IDA_reader


######################################
## IDA output object configurations ##
######################################
# Obtain dict to identify masks
###############################
data_config = {}

dict_misc = {}
dict_misc['masks']  = {'column':b'AI1_Act []',
                       'invalid': 3.50,
                       'calib':   3.00,
                       'zero':    2.50,
                       'K8':      2.00,
                       'K7':      1.75,
                       'K6':      1.50,
                       'K5':      1.25,
                       'K4':      1.00,
                       'K3':      0.75,
                       'K2':      0.50,
                       'K1':      0.25,
                       }

dict_misc['P_inlet'] = {'column':b'AI2_Act []',
                        'units':'deci Torr',
                        }

dict_reaction = {}
dict_reaction['U_drift'] = {'column':b'Udrift_Act [V]',
                            'units':'V'
                            }

dict_reaction['T_drift'] = {'column':b'T-Drift_Act [\xb0C]',
                            'units':'degrees C'
                            }

dict_reaction['P_drift'] = {'column':b'p-Drift_Act [mbar]',
                            'units':'mbar'
                            }

data_config['dict_misc'] = dict_misc
data_config['dict_reaction'] = dict_reaction

################################
## Data processing parameters ##
################################
processing_config = {}

# Timing information
processing_config['acc_interval'] = None                          # Accumumation interval for processing the data, None is at DAQ resolution
processing_config['origin'] = 'start_day'                         # Origin parameter used by the resample function
processing_config['offset'] = dt.timedelta(hours=0,minutes=0)     # Offset parameters used by the resample fuction

processing_config['processing_interval'] = dt.timedelta(hours=24) # Period where data is processed as 1
processing_config['output_interval'] = dt.timedelta(hours=24)     # Period covered in one output file

# Zero configuration
processing_config['tdelta_buf_zero']  = dt.timedelta(seconds=10)   # buffer between end of zero measurements and end of averaging interval
processing_config['tdelta_avg_zero']  = dt.timedelta(minutes=5)    # period over which to calculate zero
processing_config['tdelta_min_zero']  = dt.timedelta(minutes=25)   # minimum length of zero measurements to consider doing a zero calculation (take 5 minute buffer from standard procedure)
processing_config['zero']   = 'interp'

# Calibration configuration
processing_config['acc_interval_calib'] = '20s'                    # Accumulation interval for determining the calibration factors
processing_config['tdelta_buf_calib'] = dt.timedelta(seconds=10)   # buffer between end of calibration measurements and end of averaging interval
processing_config['tdelta_avg_calib'] = dt.timedelta(minutes=5)    # period over which to take data for calibration processing
processing_config['tdelta_stability'] = dt.timedelta(minutes=60)   # period over which to calculate rstd in order to estimate the stability during calibrations
processing_config['tdelta_min_calib'] = dt.timedelta(minutes=70)   # minimum length of calibration measurement to consider a zero calculation (take 5 minute buffer from standard procedure)

# General trim of data when switching sources
processing_config['tdelta_trim_start'] = dt.timedelta(seconds=10)  # Trim time after switching source in the manifold
processing_config['tdelta_trim_stop']  = dt.timedelta(seconds=10)  # Trim time before switching source in the manifold

# Processing configuration -- calculate concentrations
processing_config['rate_coeff_col_calib'] = 'k_ionicon [1.e-9 cm3 molecule-1 s-1]' # Name of the column used in the ancilary calibration file to compute calibration and transmission coefficients
processing_config['Xr0_default']          = 1.                                     # Default value for Xr0, if you want to adapt these, first construct the df_clusters file and adapt directly here before processing the data
processing_config['k_reac_default']       = 2.                                     # Default reaction rate coefficient, in [1.e-9 cm3 molecule-1 s-1]

###################################
## Campaign object configuration ##
###################################
campaign_config = {}

campaign_config['name'] = 'BE-Vie_2023'
campaign_config['dir_base']  = 'D:Data\RTG\BE-Vie\\'
#campaign_config['dir_base']  = 'C:/Users/bertve/Desktop/Data/RTG/BE-Vie/'
#campaign_config['data_input_level'] = 'L1.1.3'
campaign_config['data_input_level'] = 'Archived'     # If input_level is 'Archived', use te available file in the output level to potentially combine IDA runs between different data levels.
campaign_config['data_output_level'] = 'L1.2.X.4'   # Xr0_default = 1, optimization based on Xr0 from calibration on DD/MM/YYYY, trim = 10s, acc_interval = None, processing_interval = 24h, output_interval = 30 min

campaign_config['instrument'] = 'TOF4000'
campaign_config['year'] = 2023
campaign_config['calibrations_analysis'] = 'integrated' # 'integrated' OR 'dedicated'
campaign_config['mz_selection'] = {
                                   'method':'cluster',
                                   'archived':True,                                                         # Is there a cluster file available? If so, set archived to True -> Don't perform clustering.
                                   'algorithm':'DBSCAN',                                                    # 
                                   'eps':0.004,                                                             # eps parameter for DBSCAN
                                   'min_samples':39,                                                        # min_samples parameter for DBSCAN
                                   'resolving':4000,                                                        # Resolving power of the PTR-MS -- Should implement adaptive on the spectrum
                                   'fFWHM':0.5,                                                             # fraction of FWHM to be the widest allowed cluster width to be considered stable
                                   'add exact':[21.020,31.018,33.033,38.033,47.013,73.065,79.054],          # Exact masses to add even if the clusters were not found to be stable
                                   'tolerance':0.005,                                                       # Tollerance on the matching for exact masses (transmission and calibration) to clusters
                                   'decimal places':4,                                                      # Save the cluster location and edges to 4 decimal places. (Issue: this may omit the edge peaks atm.)
                                   'double_peaks':'sum',                                                    # How to handle two peaks in a cluster in the same file.
                                   }

##########
## Main ##
##########
campaign_config['data_config'] = data_config
campaign_config['processing_config'] = processing_config

Vielsalm_2023 = PTR_objects.TOF_campaign(**campaign_config)
#Vielsalm_2023.write_scan_clustering_DBSCAN()

PTR_data = Vielsalm_2023.process(t_start=dt.datetime.strptime('2023-06-08 00:00', '%Y-%m-%d %H:%M'),
                                 t_stop=dt.datetime.strptime('2023-06-09 00:00', '%Y-%m-%d %H:%M'))
