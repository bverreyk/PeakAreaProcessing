# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:25:19 2024

@author: bertve
"""

import numpy as np

import pandas as pd
import datetime as dt

try:
    from . import mz_routines as mz_r
except:
    import mz_routines as mz_r
    
def get_mask(df, column, select, eps_l = 0.1, eps_h = 0.1):
    '''
    Get a mask where the value of column in de is located between select-eps_l < value < select+eps_l
    '''
    return (df[column]>(select-eps_l)) & (df[column]<(select+eps_h))

def filter_noisy_maskLimits(index, limits, limit='start', buffer=dt.timedelta(seconds=60)):
    results = []
    if limit == 'start':
        check = -1
    elif limit == 'stop':
        check = +1
    else:
        print('Error, could not make out if I should filter starts or stops.')
        
    for i in np.arange(len(limits)):
        if (i == 0) and (limit == 'start'):
            results.append(limits[i])
            continue
            
        if (i == len(limits)-1) and (limit == 'stop'):
            results.append(limits[i])
            continue
        
        dt_from_check = index[limits[i]] - index[limits[i+check]]
        if abs(dt_from_check).total_seconds() < buffer.total_seconds():
            continue
            
        results.append(limits[i])
        
    return results

def get_harmonized_filtered_start_stop(start,stop,times,buff=dt.timedelta(seconds=60)):
    # Make sure the arrays are sorted
    start = sorted(start)
    stop = sorted(stop)
    
    # initialise arrays to store harmonised start/stop values
    hstart, hstop = [], []
    
    # Get first start and first stop
    i_start = np.argmin(start)
    i_stop = np.argmin(stop)
    
    v_start = start[i_start]
    v_stop = stop[i_stop]    
    # If first stop is before first start: Assume the start is the first measurement
    if v_stop <= v_start:
        v_start = 0

    if times[v_stop]-times[v_start] >= buff :
        hstart.append(v_start)
        hstop.append(v_stop)
    
    
    # Match every start afther the last stop to a next stop
    for i in np.arange(len(start)):
        i_start = np.argmax(start>v_stop)
        
        # i_start only zero if no value above v_stop in the array
        if (i_start == 0):
            break
            
        v_start = start[i_start]
        i_stop = np.argmax(stop>v_start)
        # i_stop only zero is no value above v_start in the array
        if i_stop == 0:
            v_stop = len(times)-1
            if times[v_stop]-times[v_start] >= buff :
                hstart.append(v_start)
                hstop.append(v_stop)
            break
            
        # check for first stop after buffer
        for j in np.arange(i_stop,len(stop)):
            v_stop = stop[j]
            if times[v_stop]-times[v_start] >= buff :
                hstart.append(v_start)
                hstop.append(v_stop)
                break
            
    return hstart, hstop

def get_start_stop_from_mask(mask):
    '''
    Get the indeces where the mask starts/stops flagging data as to be considered.
    '''
    if ((mask is None) or 
        (len(mask) == 0)):
        return np.array([]), np.array([])
    
    mask=np.array(mask)
    diff = np.diff(mask.astype(int))
    start = np.where(diff==1)[0]
    stop  = np.where(diff==-1)[0]
    
    # NOTE: Start index is 1 earlier than first flagged
    for i in np.arange(len(start)):
        if start[i] == 0:
            continue
        start[i] = start[i]-1
    
    # if the first measurement is in the mask, insert it as a start:
    if mask[0] == True:
        start = np.insert(start,0,0)
    
    # if the last measurement is in the mask, instert it as a stop:
    if mask[-1] == True:
        if len(stop) == 0:
            stop = np.insert(stop,0,len(mask)-1)
        else:
            stop = np.insert(stop,-1,len(mask)-1)
        
    return np.sort(start), np.sort(stop)

def get_trimmed_mask(mask,time_axis,tdelta_after_start=dt.timedelta(seconds=10),tdelta_before_stop=dt.timedelta(seconds=5)):
    starts, stops = get_start_stop_from_mask(mask)
    result = mask.copy()
    for start in starts:
        to_cut = (time_axis >= time_axis[start]) & (time_axis <= time_axis[start]+tdelta_after_start)
        result = result & ~to_cut
        
    for stop in stops:
        to_cut = (time_axis <= time_axis[stop]) & (time_axis >= time_axis[stop]-tdelta_before_stop)
        result = result & ~to_cut
        
    return result

def infer_zero_and_calib_mask(df, mz_exact, zero_tolerance = 100, calib_tolerance = 5, method = 'mode', estimator_averaging = '5min', zero_interval = dt.timedelta(minutes = 30), calib_interval = dt.timedelta(minutes=75), interval_warning = dt.timedelta(minutes = 1),filter_buffer=dt.timedelta(seconds=60),match_switches=dt.timedelta(seconds=600),version=2):
    '''
    Infer zero and calibration masks for measurements where no explicit discrimination between measurement types is available.
    Tolerance of variability on zero and calib in [%] of the simple estimates for zero and calibration count rates. 
    The simple estimates are determined by the minimum average value during the calibration (zero) and the mode of the average (default) for the calibration. For the calibration it is also possible to use the maximum average by setting method to 'max'. The averaging interval is set to 5 minutes by default.
    If you are using this to infer information on different consequitive calibrations, be cautious about the different options and always check that you are selecting the correct data. 
    '''
    mz_col = mz_r.select_mz_col_from_exact(df.columns,mz_exact)
    # Take out some of the variability by looking at averages
    resamp = df[mz_col].resample(estimator_averaging).mean()
    
    # Zero should always be below ambient air! Note this could be an issue for some compounds!
    zero = resamp.min()
    
    if method == 'mode':
        # The majority of measurements is the calibration, 
        # i.e., the mode of the average is a good estimator for the calibration signal
        calib = resamp.quantile(0.5)
    elif method == 'max':
        # The calibration mixing ratio is expecter to be higher than any ambient measurement present in the data
        calib = resamp.max()
    else:
        print('Error: method "{}" not recognised, defaulting to "mode"'.format(method))
        calib = resamp.quantile(0.5)
    
    init_mask_zero = get_mask(df,mz_col,zero,eps_l=zero*zero_tolerance/100.,eps_h=zero*zero_tolerance/100.)
    init_zero_start, init_zero_stop  = get_start_stop_from_mask(init_mask_zero)

    # As the inferred masks may be noisy, only consider extrema where there is no other point within a <buffer> time
    init_zero_start = filter_noisy_maskLimits(df.index, init_zero_start, limit='start', buffer=filter_buffer)
    init_zero_stop = filter_noisy_maskLimits(df.index, init_zero_stop, limit='stop',buffer=filter_buffer)

    init_mask_calib = get_mask(df,mz_col,calib,eps_l=calib*calib_tolerance/100.,eps_h=calib*calib_tolerance/100.)
    init_calib_start, init_calib_stop  = get_start_stop_from_mask(init_mask_calib)

    init_calib_start = filter_noisy_maskLimits(df.index, init_calib_start, limit='start', buffer=filter_buffer)
    init_calib_stop = filter_noisy_maskLimits(df.index, init_calib_stop, limit='stop',buffer=filter_buffer)

    infer_mask_zero = np.zeros(len(df.index)).astype(bool)
    infer_mask_calib = np.zeros(len(df.index)).astype(bool)
    
    
    #########################################################################################
    # Version 1: Match between start/end of calibration/zero and create mask based on this. #
    #########################################################################################
    if version == 1:
        # Find where the data switches from zero to calib
        cswitch = []
        for i_cstart in init_calib_start:
            for j_zstop in init_zero_stop:
                # Check if there is a stop of zero and start of calib within 10s of each other
                if abs(df.index[i_cstart]-df.index[j_zstop]) < match_switches:
                    cswitch.append(j_zstop)
                    break

        # Find where the data switches from calib to zero
        zswitch = [init_zero_start[0]]
        for i_zstart in init_zero_start:
            for j_cstop in init_calib_stop:
                # Check if there is a stop of zero and start of calib within 10s of each other
                if abs(df.index[i_zstart]-df.index[j_cstop]) < match_switches:
                    zswitch.append(j_cstop)
                    break
        zswitch.append(init_zero_stop[-1])

        #Assume that the following:
        # 1. Zero measruement always comes frist
        # 2. Assume only switches between zero and calibrations during the file
        # 3. Continue switching between zero/calib until the final calibration measurement
        for i in np.arange(len(zswitch)-2):
            zero_start = df.index[zswitch[i]]
            zero_stop = df.index[cswitch[i]-1]
            calib_start = df.index[cswitch[i]]
            calib_stop = df.index[zswitch[i+1]]

            infer_mask_zero = (infer_mask_zero) | ((df.index >= zero_start) & (df.index <= zero_stop))
            infer_mask_calib = (infer_mask_calib) | ((df.index >= calib_start) & (df.index <= calib_stop))

            if abs(zero_stop - zero_start - zero_interval) >= interval_warning:
                print('WARNING: duration zero measurement ({} min) deviates from expected ({} min)'.format((zero_stop - zero_start).seconds/60.,zero_interval.seconds/60.))

            if abs(calib_stop - calib_start - calib_interval) >= interval_warning:
                print('WARNING: duration calibration measurement ({} min) deviates from expected ({} min)'.format((calib_stop - calib_start).seconds/60.,calib_interval.seconds/60.))
    
    ################################################################################
    # Don't do any matching but just apply the filtered start/stop to obtain masks #
    ################################################################################
    elif version == 2:
        init_zero_start, init_zero_stop = get_harmonized_filtered_start_stop(init_zero_start, init_zero_stop, df.index, buff=match_switches)
        for idx_start, idx_stop in zip(init_zero_start, init_zero_stop):
            start, stop = df.index[idx_start], df.index[idx_stop]
            infer_mask_zero = (infer_mask_zero) | ((df.index >= start) & (df.index <= stop))
            if abs(stop - start - zero_interval) >= interval_warning:
                print('WARNING: duration zero measurement ({:.2f} min) deviates from expected ({:.2f} min)'.format((stop - start).seconds/60.,zero_interval.seconds/60.))
                
        init_calib_start, init_calib_stop = get_harmonized_filtered_start_stop(init_calib_start, init_calib_stop, df.index,buff=match_switches)
        for idx_start, idx_stop in zip(init_calib_start, init_calib_stop):
            start, stop = df.index[idx_start], df.index[idx_stop]
            infer_mask_calib = (infer_mask_calib) | ((df.index >= start) & (df.index <= stop))
            if abs(stop - start - calib_interval) >= interval_warning:
                print('WARNING: duration calibration measurement ({:.2f} min) deviates from expected ({:.2f} min)'.format((stop - start).seconds/60.,calib_interval.seconds/60.))
                
    return infer_mask_zero, infer_mask_calib

def get_representative_mask(df, dt_begin, dt_end, tdelta_buf, tdelta_avg, mute = False):
    '''
    Get the masks used to calculate zero/calibration (normalised) count rates.
    This mask is calculated using the datetime of the end of the relevant measurements, together with timedeltas consistent with buffering and averaging intervals.
    Pass a pandas dataframe containing the time information of measurements in the index.
    '''
    dt_stop  = dt_end - tdelta_buf
    dt_start = dt_stop - tdelta_avg
    mask = ((df.index >= dt_start) & (df.index < dt_stop))

    # If the start of the averaging interval if before the begining, warn the user and return a mask that does not select any measurements
    if dt_start < dt_begin:
        if not mute:
            print('Error, the start of your averaging interval is located before the beginning of your measurement type')
        mask = mask*False
    
    return mask

def get_representative_mask_from_multiple_intervals(df,mask,tdelta_buf,tdelta_avg,tdelta_min,mute = False):
    starts, stops = get_start_stop_from_mask(mask)
    
    mask_calc = None
    for i in np.arange(len(starts)):
        dt_begin = df.index[starts[i]]
        dt_end = df.index[stops[i]]
        if dt_end - dt_begin < tdelta_min: # interval not sufficiently long to look for representative interval
            continue
        
        tmp = get_representative_mask(df, dt_begin, dt_end, tdelta_buf, tdelta_avg, mute = mute)
        if mask_calc is None:
            mask_calc = tmp
        
        mask_calc = (mask_calc) | tmp

    if mask_calc is None:
        print('No representative intervals identified in the dataframe.')
        mask_calc = np.zeros(len(df.index),dtype=bool)

    return mask_calc

def get_resampled_mask(time_axis,mask,interval,origin='start_day',offset=dt.timedelta(hours=0,minutes=0),method='all'):
    resampled_mask = None
    resamp = pd.Series(index=time_axis,data=mask).resample(interval,origin=origin,offset=offset)
    if method == 'all':
        resampled_mask = np.where(resamp.min()==False,False,True)
    elif method == 'any':
        resampled_mask = np.where(resamp.max()==True,True,False)
    else:
        print('WARNING: resampling method for mask not recognised. Use all/any.')
        
    return resampled_mask
