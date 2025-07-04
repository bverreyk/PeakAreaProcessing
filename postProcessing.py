# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 11:33:56 2025

@author: bertve
"""


from scipy import ndimage
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import os

## Transform datasets
def transform_mz(ds_orig,mz_selection,tolerance=0.05):
    ds = ds_orig.copy()
    new_ax = np.array(ds.mz)
    for mz in mz_selection:
        diff = min(abs(new_ax-mz))
        if diff < tolerance:
            i_diff = np.argmin(abs(new_ax-mz))
            new_ax[i_diff] = mz

    ds['mz'] = new_ax
    return ds

def transform_dataset(ds_orig, multiplier, new_units = None):
    ds = ds_orig.copy()
  
    ds['Signal'] = ds.Signal*multiplier
    ds['Signal'] = ds.Signal.assign_attrs(ds_orig.Signal.attrs)
    
    ds['Signal_precision'] = ds.Signal_precision*multiplier
    ds['Signal_precision'] = ds.Signal_precision.assign_attrs(ds_orig.Signal_precision.attrs)

    ds['Signal_accuracy'] = ds.Signal_accuracy*multiplier
    ds['Signal_accuracy'] = ds.Signal_accuracy.assign_attrs(ds_orig.Signal_accuracy.attrs)

    ds['zero_precision'] = ds.zero_precision*multiplier
    ds['zero_precision'] = ds.zero_precision.assign_attrs(ds_orig.zero_precision.attrs)

    ds['zero'] = ds.zero*multiplier
    ds['zero'] = ds.zero.assign_attrs(ds_orig.zero.attrs)
    
    if not new_units is None:
        ds.Signal.attrs['units'] = new_units
        ds.Signal_precision.attrs['units'] = new_units
        ds.Signal_accuracy.attrs['units'] = new_units
        ds.zero_precision.attrs['units'] = new_units
        ds.zero.attrs['units'] = new_units
        
        
    return ds

def get_ds_sensitivity(ds,include_calib=False):
    sensitivity = xr.DataArray(np.ones(len(ds.mz))*ds.default_CC_kinetic, coords=[ds.mz], dims=["mz"], attrs=dict(description='Sensitivity',units='tc-ncps ppbv-1'))
    sensitivity = np.where(ds.k_reac != ds.config_k_reac_default, ds.k_reac/ds.config_k_reac_default*sensitivity,sensitivity)
    sensitivity = np.where(ds.FY != 1, ds.FY/1*sensitivity,sensitivity)
    sensitivity = np.where(ds.IF != 1, 1/ds.IF*sensitivity,sensitivity)
    if include_calib:
        sensitivity = np.where(~np.isnan(ds.calibration), ds.calibration, sensitivity)
    else:
        sensitivity = np.where(~np.isnan(ds.calibration), np.nan, sensitivity)

    sensitivity = xr.DataArray(sensitivity, coords=[ds.mz], dims=["mz"])
    return sensitivity

def apply_transmission(ds):
    '''
    Used to analyse output from calibrations that are generated in the calibration analysis step.
    '''
    if not ds.Signal.attrs['units'] == 'ncps':
        print('Error, wrong units for apply_transmission')
        raise ValueError

    ds['Signal'] = ds.Signal/ds.transmission
    ds['Signal_precision'] = ds.Signal_precision/ds.transmission
    ds['Signal_primaryIons'] = ds.Signal_primaryIons/ds.transmission

    ds.Signal.attrs['units'] = 'tc-ncps'
    ds.Signal_precision.attrs['units'] = 'tc-ncps'
    ds.Signal_primaryIons.attrs['units'] = 'tc-ncps'
    return ds
    

def to_trncps(ds):
    if not ds.Signal.attrs['units'] == 'ppbv':
        print('Error, wrong units for to_trncsp')
        raise ValueError
        
    sensitivity = get_ds_sensitivity(ds,include_calib=True)
    ds = transform_dataset(ds, sensitivity, new_units='tr-ncps')
    return ds

def transform_sensitivity(ds_orig,new_sensitivity):
    '''Transform remove the IF, k_reac, FY, default_CC_kinetic, data and combine in one sensitivity matrix. This sensitivity can be transformed to a new axis.'''
    if not ds_orig.Signal.attrs['units'] == 'ppbv':
        print('Error, wrong units to transform sensitivity')
        raise ValueError
        
    ds = ds_orig.copy()

    sensitivity = get_ds_sensitivity(ds,include_calib=False)
    # Set NaN values (corresponding with directly calibrated signals) to 1    
    multiplier = np.where(np.isnan(sensitivity),1,sensitivity/new_sensitivity)

    d = ds.attrs
    d.pop('config_k_reac_default')
    d.pop('default_CC_kinetic')
    ds = ds._replace(attrs=d)
    ds = ds.drop_vars(['k_reac','FY','IF'])
    ds['sensitivity'] = new_sensitivity

    ds = transform_dataset(ds, multiplier)
    return ds

def read_data_ToF4000(hdf5_files,period=None,t_offset=dt.timedelta(hours=0),mz_sel=69.069,mz_columns=[],tol=0.0001,mute=False,transform_to_trncps=False):
    ds = None
    for f in hdf5_files:
        if not period is None:
            if not any([p in f for p in period]):
                continue
        if not mute:
            print(f)

        try:
            tmp = xr.open_dataset(f)
        except:
            print('Could not open dataset')
            print('----------------------')
            continue
        
        if transform_to_trncps:
            if tmp.Signal.units == 'ppbv':
                tmp = to_trncps(tmp)
            if tmp.Signal.units == 'ncps':
                tmp = apply_transmission(tmp)

        tmp['time'] = pd.to_datetime(tmp.time)+t_offset

        if len(mz_columns) == 0:
            mz_columns = tmp.mz

        # Drop any data where the mz_sel is NaN (i.e., times where the mask is not active when the cluster is identified in PAP)
        source = f.split('\\')[-1].split('_')[0]
        msk = np.isnan(tmp.Signal.sel(mz=mz_sel,method='nearest').values)

        tmp = tmp.drop_sel(time=tmp.time[msk])
        tmp = tmp.assign(valve_signal=(['time'],[source]*len(tmp.time)))

        if len(mz_columns) != 0:
            tmp = tmp.sel(mz=mz_columns,method='nearest',tolerance=tol)

        #In order to be able to combine ds, include a time variable in their data array
        tmp['transmission'] = tmp.transmission.expand_dims(dim={tmp.time.name: tmp.time.values}, axis=0)
        tmp['calibration']  = tmp.calibration.expand_dims(dim={tmp.time.name: tmp.time.values}, axis=0)

        if ds is None:
            ds = tmp

            tmp.close()
            lst = [tmp]
            del lst
            continue

        if len(tmp.time) == 0:
            continue
        
        ds = xr.merge([tmp,ds])
        tmp.close()
        lst = [tmp]
        del lst
                
    ds = ds.sortby('time')
    ds = ds.sortby('mz')
    return ds

## Transform datalevel to update sensitivity
def transform_PAP_output_Level(files_PAP,df_sensitivity,old_data_level,new_data_level):
    old_sensitivities = {}
    for file in files_PAP:
        # Check if output exists
        new_file = file.replace(old_data_level,new_data_level).replace('\\','/')
        print(new_file)
        
        tmp_path = ''
        split = new_file.split('/')
        for i, subdir in enumerate(split):
            tmp_path += subdir + os.sep
            # reached file level
            if i == len(split)-1:
                break
            if os.path.isdir(tmp_path):
                continue
            else:
                os.makedirs(tmp_path)
    
        exists = os.path.exists(new_file)
        if exists:
            print('-- already processed --')
            continue
        else:
            print('-- start transformation to L2.1 --')
            
        tmp = xr.open_dataset(file)
        # transform mz axis
        mz_selection = df_sensitivity.columns.astype(float)
        tmp = transform_mz(tmp,df_sensitivity.columns.values.astype(float))
    
        # Get sensitivyt in the dataset
        sensitivity = get_ds_sensitivity(tmp,include_calib=False)
        old_sensitivities[file] = sensitivity.to_pandas()
    
        df_sensitivity.columns = df_sensitivity.columns.astype(float)
        for mz in tmp.mz.values:
            if not (mz in mz_selection):
                df_sensitivity[mz] = np.nan
            
        df_sensitivity.sort_index(axis=1,inplace=True)
        new_sensitivity = xr.DataArray(
            data=df_sensitivity.values[0],
            dims=["mz"],
            coords=dict(mz=df_sensitivity.columns)
        )
        
        tmp = transform_sensitivity(tmp, new_sensitivity)
        tmp.attrs['Peak Area Analysis version'] = tmp.attrs['Peak Area Analysis version'] + ', postprocessed for publication.'
    
        # update accuracy
        mz_firstPrinciples = df_sensitivity.dropna(axis=1).columns
        df_accuracy = tmp.Signal_accuracy.to_pandas()
        df_signal = tmp.Signal.to_pandas()
        df_tmp = df_accuracy.copy()
        df_tmp[mz_firstPrinciples] = np.sqrt(0.25**2.+0.50**2.)*df_signal[mz_firstPrinciples]
        tmp['Signal_accuracy'] = xr.DataArray(df_tmp.values,coords=[df_tmp.index, df_tmp.columns], dims=['time','mz'],attrs=tmp.Signal_accuracy.attrs)
        tmp['Signal_accuracy'] = tmp.Signal_accuracy.assign_attrs(description='Accuracy on Mixing Ratio') # To be corrected in the PAP output modules
        # output
        tmp.to_netcdf(new_file)
        tmp.close()
    
    df_old_sensitivities = pd.DataFrame(old_sensitivities)
    
    return df_old_sensitivities

## Formatting correctly
def transform_PAP_output_to_publishable(ds_orig,df_deadtime=None,flag_source_performance=True):
    ds = xr.Dataset()
    ds['concentration'] = ds_orig.Signal
    ds['precision'] = ds_orig.Signal_precision
    ds['accuracy'] = ds_orig.Signal_accuracy

    ## Additional 100% uncertainty for HCHO due to backreaction occuring in the drift tube
    mz_HCHO = ds.mz.values[np.argmin(abs(ds.mz.values-31.018))]
    additional_uncertainty = ds['concentration'].where(ds.mz==mz_HCHO, other = 0)

    ds['expanded_uncertainty'] = 2.*np.sqrt(ds.precision**2.+ds.accuracy**2.+additional_uncertainty**2.) 
    ds['expanded_uncertainty'] = ds.expanded_uncertainty.assign_attrs(description='expanded uncertanty (k=2)',units=ds_orig.Signal_precision.units)

    ds['primary_ion_signal'] = ds_orig.Signal_primaryIons
    
    ds['limit_of_detection'] = 3.*ds_orig.zero_precision
    ds['limit_of_detection'] = ds.limit_of_detection.assign_attrs(description='limit of detection',units=ds_orig.zero_precision.units)
    
    df_masks = ds_orig.valve_signal.to_pandas()
    masks = {}
    for val in df_masks.unique():
        if val == 'full':
            key = 'invalid'
        else:
            key = val
        
        df_masks_tmp = pd.Series(np.zeros(len(df_masks.index)),index=df_masks.index)
        inters = df_masks.index.intersection(df_masks[df_masks == val].index)    
        df_masks_tmp.loc[inters] = 1
        masks[key] = df_masks_tmp.values
    
    # Heights
    heights=np.zeros(len(ds.time))*np.nan
    height = {}
    height['K2'] = 3
    height['K3'] = 11
    height['K4'] = 19
    height['K5'] = 27
    height['K6'] = 35
    height['K7'] = 43
    height['K8'] = 51
    
    # meta data for PAP configuration
    d = ds_orig.attrs.copy()
    d.pop('config_Xr0_default')
    d.pop('config_rate_coeff_col_calib')
    d.pop('config_origin')
    d.pop('config_offset')
    d.pop('config_processing_interval')
    d.pop('config_tdelta_buf_zero')
    d.pop('config_tdelta_avg_zero')
    d.pop('config_tdelta_min_zero')
    d.pop('config_acc_interval_calib')
    d.pop('config_tdelta_buf_calib')
    d.pop('config_tdelta_avg_calib')
    d.pop('config_tdelta_stability')
    d.pop('config_tdelta_min_calib')
    d.pop('config_tdelta_trim_start')
    d.pop('config_tdelta_trim_stop')
    ds = ds._replace(attrs=d)
    
    # Flagging!!
    mask = np.zeros(len(masks['invalid']))
    setup_flags = np.array(masks['invalid']*0.999)
    setup_flags += masks['zero']*0.684
    setup_flags += masks['calib']*0.683

    mask += masks['invalid']
    mask += masks['zero']
    mask += masks['calib']
    
    for level in np.arange(2,9):
        setup_flags += masks['K{}'.format(level)]*0.000
        mask += masks['K{}'.format(level)]
        heights[masks['K{}'.format(level)].astype(bool)] = height['K{}'.format(level)]


    setup_flags += (mask==0)*0.999
    if (mask>1).sum() >= 1:
        print('Error, {} multi-masked points: flags associated points,'.format((mask>1).sum()))
        print(ds.time[mask>1])
        print('Setting flag to 0.999')
        setup_flags[mask>1] = 0.999

    #### SETUP QC FLAG FOR SOURCE PURITY ####
    if flag_source_performance:
        ref   = ds.FPH1*ds.primary_ion_signal.sel(mz=21.0201,method='nearest')+ ds.FPH2*ds.primary_ion_signal.sel(mz=38.0332,method='nearest')
        O2_NO = 1.1*ds.primary_ion_signal.sel(mz=29.9972,method='nearest')    + 245.09*ds.primary_ion_signal.sel(mz=33.9939,method='nearest')
        H5O2  = ds.FPH2*ds.primary_ion_signal.sel(mz=38.0332,method='nearest')
        QA_mask = ((O2_NO/ref > 0.03) | (H5O2/ref > 0.20))
    else:
        QA_mask = np.zeros(len(masks['invalid']))
    
    #### Expand flags to apply to individual m/z signals ####
    da_flag_source = xr.DataArray(setup_flags,coords=[ds.time], dims=['time'],attrs=dict(description='measurement type flag'))
    da_flag_source = da_flag_source.expand_dims(dim={ds.mz.name: ds.mz.values}, axis=1)
    
    da_flag_QAQC   = xr.DataArray(QA_mask,coords=[ds.time], dims=['time'],attrs=dict(description='QA/QC type flag')) # will be multiplied by the correct value later
    da_flag_QAQC = da_flag_QAQC.expand_dims(dim={ds.mz.name: ds.mz.values}, axis=1)

    #### SETUP QC FLAG FOR LOD ####
    da_flag_QAQC.values = da_flag_QAQC.values + (ds.concentration < ds.limit_of_detection)
    da_flag_QAQC.values = (da_flag_QAQC.values >= 1)*.147

    #### Set nan concentrations to invalid ####
    da_flag_source.values = np.where(np.isnan(ds.concentration.values),0.999,da_flag_source.values)
    
    #### Flag dead time ####
    if not df_deadtime is None:
        for mz in ds.mz.values:
            if not mz in df_deadtime.columns:
                df_deadtime[mz] = dt.timedelta(minutes=-999)
        td_buffer = pd.to_timedelta(pd.to_timedelta(ds_orig.attrs['config_tdelta_trim_stop'])) # buffer where values are flagged as invalid
        t_dead = df_deadtime.loc['m/z dead time'][ds.mz.values]+td_buffer+dt.timedelta(minutes=1) # Add 1 minute as the flagging rounds down and I would like to round up...
        t_last_zero = ds.time.min()-t_dead.max()
        t_since_zero = dt.timedelta(days=1)
        for i, time in enumerate(ds.time.values):
            if setup_flags[i] == 0.684:
                t_last_zero = time
                continue

            t_since_zero = time - t_last_zero
            if t_since_zero > max(t_dead):
                continue
            da_flag_source.values[i,:] = np.where((da_flag_source.values[i,:]==0)&
                                                  (t_dead >= t_since_zero),
                                                  0.999, # Invalidated by database co-ordinator
                                                  da_flag_source.values[i,:]
                                                 )            
            
    #### Combine flags ####
    flag_comb = np.where((da_flag_source.values>0),da_flag_source.values+da_flag_QAQC.values*1.e-3,da_flag_QAQC.values)
    da_flag = xr.DataArray(flag_comb,coords=[ds.time,ds.mz], dims=['time','mz'],attrs=dict(description='num_flag'))
    da_flag.values = np.char.mod('%.6f', da_flag.values)

    ds['measurement_height'] = xr.DataArray(heights,coords=[ds.time], dims=['time'],attrs=dict(description='measurement height',units='m a.g.l.'))
    ds['flag_source'] = da_flag_source
    ds['flag_QAQC'] = da_flag_QAQC
    ds['flag'] = da_flag
    
    #### Complete time dimension
    start = pd.Timestamp(ds.time.min().values).floor('D')   # Midnight of start day
    end = pd.Timestamp(ds.time.max().values).ceil('D')      # Midnight of *next* day if not already midnight
    
    full_time = pd.date_range(start=start, end=end, freq='1min')
    ds = ds.reindex(time=full_time)
    ds['time'] = ds.time.assign_attrs(description='time (start of measurement interval)')
    ds['flag_source'] = ds['flag_source'].fillna(0.999)
    ds['flag_QAQC'] = ds['flag_QAQC'].fillna(0.000)
    ds['flag'] = ds['flag'].fillna('0.999000')
    ds['flag'] = ds.flag.astype(str)

    #### Calculate alternative time representation
    ds['year'] = ds.time.dt.year
    time_ns = ds.time.values.astype('datetime64[ns]')  # nanosecond precision
    midnight_ns = time_ns.astype('datetime64[D]')      # rounds down to midnight
    time_since_midnight_ns = (time_ns - midnight_ns).astype('timedelta64[ns]').astype(np.int64)
    ns_per_day = 24 * 60 * 60 * 1.e9                   # Total nanoseconds in a day
    tod = time_since_midnight_ns / ns_per_day          # time of day
    doy = ds.time.dt.dayofyear                         # day of year
    doy = doy + tod - 1                                # day of year (fraction)
    ds['day_of_year'] =  xr.DataArray(doy,coords=[ds.time], dims=['time'],attrs=dict(description='time since new year', units='days'))
    
    return ds

def get_doy_fraction(dt_ax):
    time_ns = dt_ax.values.astype('datetime64[ns]')           # nanosecond precision
    midnight_ns = time_ns.astype('datetime64[D]')      # rounds down to midnight
    time_since_midnight_ns = (time_ns - midnight_ns).astype('timedelta64[ns]').astype(np.int64)
    ns_per_day = 24 * 60 * 60 * 1.e9                   # Total nanoseconds in a day
    tod = time_since_midnight_ns / ns_per_day          # time of day
    try:
        doy = dt_ax.dt.dayofyear                       # day of year
    except:
        doy = dt_ax.dayofyear
    doy = doy + tod - 1                                # day of year (fraction)
    return doy

def transform_publishable_hdf5_to_csv(ds,filter_mz=[21.022,29.997,33.997,38.033]):
    ## get pandas dataframes ##
    df_concentration = ds.concentration.to_pandas()
    df_height = ds.measurement_height.to_pandas()
    df_prec = ds.precision.to_pandas()
    df_acc  = ds.accuracy.to_pandas()
    df_expunc = ds.expanded_uncertainty.to_pandas()
    df_lod = ds.limit_of_detection.to_pandas()

    ## Set flag ##
    df_flag_source = ds.flag_source.to_pandas()
    df_flag_QAQC   = ds.flag_QAQC.to_pandas()
    df_flag_plume  = ds.flag_plume.to_pandas()*0.559

    flag_columns = df_flag_source.columns
    flag_index   = df_flag_source.index
    if ((df_flag_plume.columns != flag_columns).any() or 
        (df_flag_QAQC.columns != flag_columns).any()):
        raise ValueError
    if ((df_flag_plume.index != flag_index).any() or 
        (df_flag_QAQC.index != flag_index).any()):
        raise ValueError

    combined = ((df_flag_source.values>0.0) + (df_flag_plume.values>0.0))
    if combined.max() > 1:
        print('invalid data flagged as plume')
        raise ValueError

    ## Replace zero/calibration flag ##
    df_flag_source = df_flag_source.where(~((df_flag_source == 0.683) | (df_flag_source == 0.684)), 0.980)
    
    ## Combine flags
    flag = df_flag_source.values+df_flag_plume.values
    flag = np.where((flag>0),flag+df_flag_QAQC.values*1.e-3,df_flag_QAQC.values)
    flag = np.char.mod('%.6f', flag)

    df_flag = pd.DataFrame(data=flag,columns=flag_columns,index=flag_index)
    df_flag.columns = ['numflag_{:.3f}'.format(mz) for mz in df_flag.columns]

    ## set NaN ##
    df_concentration.where(df_flag_source==0,other=np.nan,inplace=True)
    df_prec.where(df_flag_source==0,other=np.nan,inplace=True)
    df_acc.where(df_flag_source==0,other=np.nan,inplace=True)
    df_expunc.where(df_flag_source==0,other=np.nan,inplace=True)
    df_lod.where(df_flag_source==0,other=np.nan,inplace=True)

    ## Raname columns of the dataframes
    df_height.name = 'height'
    df_concentration.columns = ['Conc_{:.3f}'.format(mz) for mz in df_concentration.columns]
    df_prec.columns = ['Prec_{:.3f}'.format(mz) for mz in df_prec.columns]
    df_acc.columns = ['Acc_{:.3f}'.format(mz) for mz in df_acc.columns]
    df_expunc.columns = ['ExpUnc_{:.3f}'.format(mz) for mz in df_expunc.columns]
    df_lod.columns = ['LoD_{:.3f}'.format(mz) for mz in df_lod.columns]

    ## Combine ##
    df_concat = pd.concat([df_height,df_concentration,df_prec,df_acc,df_expunc,df_lod,df_flag],axis=1)

    df_concat.index.name = 'TIMESTAMP_START'
    df_concat['TIMESTAMP_END'] = df_concat.index+dt.timedelta(minutes=1)
    df_concat['DOY_START'] = get_doy_fraction(df_concat.index)
    df_concat['DOY_END'] = get_doy_fraction(df_concat.TIMESTAMP_END)

    col_order = ['TIMESTAMP_END','DOY_START','DOY_END','height']
    for mz in ds.mz:
        # Filter some columns
        if mz in filter_mz:
            continue
        col_order.append('Conc_{:.3f}'.format(mz))
        col_order.append('Prec_{:.3f}'.format(mz))
        col_order.append('Acc_{:.3f}'.format(mz))
        col_order.append('ExpUnc_{:.3f}'.format(mz))
        col_order.append('LoD_{:.3f}'.format(mz))
        col_order.append('numflag_{:.3f}'.format(mz))
            
    df_concat = df_concat[col_order]
    
    return df_concat

def flag_plumes(ds, df_flag):
    tmp = df_flag.reindex(ds.time, axis=0, method='bfill', limit=30) # indexed at the end of the intervals
    tmp.fillna(0,inplace=True)
    ds['flag_plume'] = xr.DataArray(tmp,dims=['time','mz'],coords=[tmp.index-dt.timedelta(minutes=60),tmp.columns])
    return ds


## EXTRACT PROFILES
def boolean_vector2ranges(x):
    df1=pd.DataFrame({'location':range(len(x)),
                      'bool':x,
                     })
    df1['group'] = ndimage.label(df1['bool'].astype(int))[0]
    return df1.loc[(df1['group']!=0),:].groupby('group')['location'].agg(['min','max'])

def get_profiles(ds):
    print('Getting profiles')
    # Get mask for middle of the profile measurements to select all the profiles.
    profile_measurements = [valve in ('K4') for valve in ds.valve_signal.values]
    profile_measurements = boolean_vector2ranges(x=profile_measurements)
    
    n       = len(profile_measurements.index)
    heights = [3, 11, 19, 27, 35, 43, 51]
    mz_ax   = ds.mz.values

    profiles   = np.zeros(shape=(n,7,len(mz_ax)))*np.nan
    time_start = np.zeros(n)*np.nan
    time_stop  = np.zeros(n)*np.nan
    
    print('Found {} profile measurements.'.format(n))
    for i in np.arange(n):
        if i%(n/10)<1:
            print('Processing profile {}/{}'.format(i,n))
        t_min, t_max = profile_measurements.iloc[i]['min'], profile_measurements.iloc[i]['max']

        # Select half hour for the profile measurements + 5 minutes before to get K8 concentrations
        t_min = pd.to_datetime(ds.time.values[t_min]).floor(freq='30min')-dt.timedelta(minutes=5)
        t_max = pd.to_datetime(ds.time.values[t_max]).ceil(freq='30min')
        
        tmp = ds.sel(time=slice(t_min, t_max))
        time_start[i] = t_min.to_numpy()
        time_stop[i]  = t_max.to_numpy()
        for h, valve_signal in enumerate(['K{}'.format(k) for k in np.arange(2,9)]):
            mask_source = (tmp.valve_signal == valve_signal)
            for j, mz in enumerate(mz_ax):
                profiles[i,h,j] = tmp.Signal[mask_source].sel(mz=mz,time=slice(t_min,t_max)).median()

    data_vars = {}
    da_profiles = xr.DataArray(data=profiles,
                               dims=['time','height','mz'],
                               coords=dict(time=time_start,height=heights,mz=mz_ax),
                               attrs=dict(description='Mixing Ratio',units='ppbv'))
    data_vars['profiles'] = da_profiles
    
    da_time_start = xr.DataArray(data=time_start,
                                 dims=['time'],
                                 coords=dict(time=time_start),
                                 attrs=dict(description='start time of profile measurement')
                                )
    data_vars['time_start'] = da_time_start
    
    da_time_stop  = xr.DataArray(data=time_stop,
                                 dims=['time'],
                                 coords=dict(time=time_start),
                                 attrs=dict(description='end time of profile measurement')
                                )
    data_vars['time_stop'] = da_time_stop

    ds_profiles = xr.Dataset(data_vars)
    return ds_profiles