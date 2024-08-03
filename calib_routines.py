# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:16:35 2024

@author: bertve
"""
import numpy as np

def get_ancilarry_PTRT_drift_calib_av(P_drift_calib_av,T_drift_calib_av,U_drift_calib_av,drift_tube_L=9.2):
    '''
    Calculate the reaction time and the number density in the drift tube given the pressure [mbar], temperature [centigrade] and electrical potential [V] in/over the drift tube. Drift tube length is set to 9.2 cm as a default.
    '''
    mu0 = 2.8 # [cm2/Vs]
        
    L_DT = drift_tube_L # [cm]
    T_DT = 273.15 + T_drift_calib_av # [K]
    p_DT = P_drift_calib_av # [mbar]
    U=U_drift_calib_av # [V] 
    
    mu= (T_DT/283)*(1013/p_DT)*mu0
    t_reac=L_DT**2./(mu*U) # [s]
    N_DT=(p_DT*100/(1.38e-23*T_DT)/1e6) # [molecules/cm3]
    
    return t_reac, N_DT

def get_Q_PTRMS_corrected(P_inlet,campaign='Vie_2022'):
    Q_PTRMS = np.nan
    
    p_offset = 20 # Pressure drop from Catalitic converter, estimate by Niels and to be optimized
    # Emperical relation between P_inlet en Q_PTRMS from fit Niels Schoon (2022)
    if campaign == 'BE-Vie_2022':
        A = 0.25833 # sccm Torr-1
        B = -92.48601 # sccm
    
    elif campaign == 'BE-Vie_2023':
        A = 0.26419
        B = -94.48497
    else:
        # Default to Q_PTRMS = 80
        print('Warning, campaign not parameterised to calculate Q_PTRMS, default to = 80.')
        return 80.
        
    Q_PTRMS = A*(P_inlet-p_offset)-B


    return Q_PTRMS

def get_Q_calib_corrected(Q_calib,campaign='RTG_BE-Vie_2022'):
    if campaign == 'RTG_BE-Vie_2022':
        # Correction as the flow meters are using different reference points (0 deg centigrade vs 20)
        Q_calib = Q_calib*(273.16+20)/273.16
    return Q_calib

def get_mixingRatio_driftTube_Calib(Q_calib,Q_PTRMS,Q_zero_air,MR_bottle):
    '''
    Calculate the expected mixing ratio [ppbv] of the calibration gass in the drift tube.
    '''
    # Q_calib: Flow of calibration gass [slm]
    # Q_zero_air: flow from catalytic converter to dilute the calibration mixture
    # Q_PTRMS: flow entering the PTR instrument
    # MR_bottle: Mixing ratio of the compound in the calibration bottle        
    return (Q_calib/(Q_PTRMS+Q_zero_air))*MR_bottle

def get_normalisedCountRate_df(df, mz_col_21, mz_col_38, Tr_PH1_to_PH2, FPH1 = 500, FPH2 = 624.2, XR0={},Xr0_default=1.):
    '''
    Convert dataframe with units cps to units of a normalised count rate.
    The normalised count rate of product ion P+ is equal to the count rate of P+ that would correspond with a reactant H3O+ count rate of 1.e6 cps
    '''
    # df: dataframe with absolute count rates observed from the PTR insturment
    # mz_col_21: column name for the H3O+ isotope at 21 (the count rate at 19 is too large to measure faithfully)
    # mz_col_38: column name for the H3O+.H2O isotope at 38
    # Tr_PH1_to_PH2: Ratio of transmission factors from 21 to 38 (i.e., Tr_21/Tr_38)
    # FPH1: Factor multiplication to convert from isotope signal
    # FPH2: Factor multiplication to convert from isotope signal
    # XR0: ratio from reactivity of unionized compound with protonated water and a watercluster of protonated water (i.e., k_37/k_19, theoretical value)
    #      Note that the theoretical value may differ significantly from the experimental one!
    
    ncr_df = df.copy() # copy the original dataframe as to make sure we do not change it
    
    I_nrc_21, I_nrc_38 = ncr_df[mz_col_21], ncr_df[mz_col_38]
    
    Xr0 = Xr0_default
    norm_uncorrected = 1.e6/((FPH1*I_nrc_21)+(FPH2*Tr_PH1_to_PH2*I_nrc_38*Xr0))    
    for c in ncr_df.columns:
        if not c in XR0.keys():
            ncr_df[c] = ncr_df[c]*norm_uncorrected
        else:
            Xr0 = XR0[c]        
            norm = 1.e6/((FPH1*I_nrc_21)+(FPH2*Tr_PH1_to_PH2*I_nrc_38*Xr0))
            ncr_df[c] = ncr_df[c]*norm
    
    return ncr_df

def get_transmissionCoefficient(ncr_df, mask_calc_zero, mask_calc_calib, mz_col, MR_calib, k, fr, N, t_react):
    '''
    Get Transmission coefficient from calibration measurements and the normalised count rate dataframe. 
    The formula used calculates transmissions relative to Tr_21.
    '''
    # nrc_df: dataframe with normalised count rates for the calibration measurement
    # mask_calc_zero: mask for selecting zero measurements used in this dataframe
    # mask_calc_calib: mask for selecting calibration measurements used in this dataframe
    # mz_col: column name for the desired mz value of which the transmisison will be calculated
    # MR_calib: Mixing ratio of the calibration gas in the drift tube
    # k: expected kinetic rate constant relevant for reaction between the expected compound at mass mz and H3O+
    # fr: fraction pattern coefficient in the PTR instrument for the expected compound at mass mz
    # N: Number density in the dirft tube
    # t_react: Reation time in the drift tube
    CC_R = 1.e-3*t_react*k*1.e-9*N/fr
    zero = ncr_df[mask_calc_zero][mz_col].mean()
    
    I_ncr = (ncr_df[mask_calc_calib][mz_col]-zero).mean()
    
    Tr = I_ncr/MR_calib/CC_R
    
    return Tr

def get_Tr_PH1_to_PH2(df, mask_calc_zero, mask_calc_calib, mz_col_21, mz_col_38, mz_1_anc, mz_2_anc, N, t_reac, Tr_PH1_to_PH2_ini = 1, nsteps = 10, eps = 0.01, verbose = False):
    '''
    Get Tr_PH1_to_PH2 used for normalisation of the count rates based on an itterative proccess.
    Two other calibrated masses close to mz = 38 are used to interpolate the expected value of Tr_PH1_to_PH2 and compare with the initial guess.
    Dictionaries mz_1_anc and mz_2_anc have to contain:
    1. The column name of the calibrated compound as it is saved in the dataframe with raw count rates (df) ['col']
    2. The expected mixing ratio of the compound in the drift tube
    3. The expected kinetic rate constant (in 1.e-9 cm3 molecule-1 s-1)
    4. How much it is expected to fraction in the drift tube
    '''
    # Don't calculate ncr for all columns in the different loops to save time
    selected = [mz_col_21,mz_col_38,mz_1_anc['col'],mz_2_anc['col']]
    XR0={}
    if (('Xr0' in mz_1_anc.keys()) and
        ('Xr0' in mz_2_anc.keys())):
        XR0[mz_1_anc['col']] = mz_1_anc['Xr0']
        XR0[mz_2_anc['col']] = mz_2_anc['Xr0']
        print('--- Taking into account Xr0 to compute Tr_PH1_to_PH2 ---')
        
    tdf = df.drop(columns=[col for col in df if col not in selected], inplace=False)

    success = False
    Tr_PH1_to_PH2 = Tr_PH1_to_PH2_ini
    for i in np.arange(nsteps):
        ncr_tr_df = get_normalisedCountRate_df(tdf, mz_col_21, mz_col_38, Tr_PH1_to_PH2_ini,XR0=XR0)

        Tr_1 = get_transmissionCoefficient(ncr_tr_df, mask_calc_zero, mask_calc_calib,  
                                           mz_1_anc['col'],  mz_1_anc['mixingRatio_driftTube'],  
                                           mz_1_anc['k'],  mz_1_anc['fractioning'], N, t_reac)

        Tr_2 = get_transmissionCoefficient(ncr_tr_df, mask_calc_zero, mask_calc_calib,  
                                           mz_2_anc['col'],  mz_2_anc['mixingRatio_driftTube'],  
                                           mz_2_anc['k'],  mz_2_anc['fractioning'], N, t_reac)        
        
        interp = np.interp(38.033,[mz_1_anc['mz'],mz_2_anc['mz']],[Tr_1,Tr_2])
        
        Tr_PH1_to_PH2 = 1./interp

        if verbose:
            print('Initial value: {:.2e}, Tr_1: {:.5e}, Tr_2: {:.5e}, new value: {:.2e} (inverse of {})'.format(Tr_PH1_to_PH2_ini,Tr_1, Tr_2, Tr_PH1_to_PH2,interp))
            
        if abs(Tr_PH1_to_PH2_ini - Tr_PH1_to_PH2)/Tr_PH1_to_PH2 < eps:
            success = True
            break

        Tr_PH1_to_PH2_ini = Tr_PH1_to_PH2

    if not success:
        print('Tr_PH1_to_PH2 not converged')
        Tr_PH1_to_PH2 = np.nan
        
    if verbose:
        if success:
            print('Succes: Tr_PH1_to_PH2 converged at {:.2e} in {} steps'.format(Tr_PH1_to_PH2, i))
        else:
            print('Error: Tr_PH1_to_PH2 not converged in {} steps.'.format(nsteps))
    
    return Tr_PH1_to_PH2