# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:49:13 2024

@author: bertve
"""

import pandas as pd
import numpy as np

k = {} # Reaction rate constant [1.e-9 cm3 molecule-1 s-1]
k_ionicon = {}
k_Holzinger = {}
k_calculated = {}
fr = {}
mixrat_bottle = {} # [ppb]
unc_mixrat_bottle = {} # [ppbv]
Tr_selection = {} # boolean

# H3O+ isotope
mz = 21.022
mixrat_bottle[mz] = np.nan
unc_mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False


# Protonated Methanol
mz = 33.033
mixrat_bottle[mz] = 1068
unc_mixrat_bottle[mz] = 70/2 # coverage factor 2 and stated uncertainty 48
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.20, 2.20, 2.20
k_calculated[mz] = 2.196
fr[mz] = 1.014
Tr_selection[mz] = True

# O2.H+ isotoop (impurity ion)
mz = 33.989
mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

# H3O+.H2O isotope
mz = 38.033
mixrat_bottle[mz] = np.nan
unc_mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

# H3O+.H2O isotope
mz = 39.033
mixrat_bottle[mz] = np.nan
unc_mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

# Protonated Isoprene fragment
mz = 41.039
mixrat_bottle[mz] = 508
unc_mixrat_bottle[mz] = 15/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 1.85, 1.85, 1.85
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

# Protonated Acetonitrile
mz = 42.034
mixrat_bottle[mz] = 490
unc_mixrat_bottle[mz] = 23/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 3.10, 4.40, 3.10
k_calculated[mz] = 3.782
fr[mz] = 1.026
Tr_selection[mz] = True

# Protonated Acetaldehyde
mz = 45.033
mixrat_bottle[mz] = 1006
unc_mixrat_bottle[mz] = 36/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 3.03, 3.03, 3.03
k_calculated[mz] = 2.962
fr[mz] = 1.025
Tr_selection[mz] = True

# Protonated Ethanol
mz = 47.049
mixrat_bottle[mz] = 493
unc_mixrat_bottle[mz] = 20/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 3.00, 3.00, 3.00
k_calculated[mz] = np.nan
fr[mz] = 1
Tr_selection[mz] = False

# Protonated Acetone
mz = 59.049
mixrat_bottle[mz] = 1026
unc_mixrat_bottle[mz] = 33/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 3.25, 3.70, 3.25
k_calculated[mz] = 3.095
fr[mz] = 1.036
Tr_selection[mz] = True

# Protonated Isoprene
mz = 69.070
mixrat_bottle[mz] = 508
unc_mixrat_bottle[mz] = 15/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 1.85, 1.85, 1.85
k_calculated[mz] = np.nan
fr[mz] = 1.057
Tr_selection[mz] = False

# Protonated Isoprene oxidation product (50/50 MACR/MVK)
mz = 71.049
mixrat_bottle[mz] = 470+501
unc_mixrat_bottle[mz] = ((14/2)**2+(15/2)**2)**.5
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.72, 2.72, 2.72
k_calculated[mz] = (3.287+3.506)/2.
fr[mz] = 1.048
Tr_selection[mz] = False

# Protonated MEK
mz = 73.065
mixrat_bottle[mz] = 502
unc_mixrat_bottle[mz] = 19/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 3.25, 3.25, 3.25
k_calculated[mz] = 3.052
fr[mz] = 1.048
Tr_selection[mz] = False

# (C6H5).H+, estimate impurity contribution from oxygen using protonated benzene fragementation
mz = 78.046
mixrat_bottle[mz] = np.nan
unc_mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

# Protonated Benzene
mz = 79.054
mixrat_bottle[mz] = 506
unc_mixrat_bottle[mz] = 15/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 1.97, 2.00, 1.97
k_calculated[mz] = 1.932
fr[mz] = 1.068
Tr_selection[mz] = True

# Protonated Benzene isotope
mz = 80.058
mixrat_bottle[mz] = 506
unc_mixrat_bottle[mz] = 15/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 1.97, 2.00, 1.97
k_calculated[mz] = 1.932
fr[mz] = 16.6
Tr_selection[mz] = False

# Protonated monoterpene fragment (100% sabinene)
mz = 81.070
mixrat_bottle[mz] = 982
unc_mixrat_bottle[mz] = 30/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.04, 2.04, 2.04
k_calculated[mz] = np.nan
fr[mz] = 1.068
Tr_selection[mz] = False

# Protonated dehydrated cis-3-hexenol
mz = 83.086
mixrat_bottle[mz] = 986
unc_mixrat_bottle[mz] = 31/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2, 2, 2
k_calculated[mz] = np.nan
fr[mz] = 1.069
Tr_selection[mz] = False

# Protonated Toluene
mz = 93.070
mixrat_bottle[mz] = 491
unc_mixrat_bottle[mz] = 15/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.15, 2.12, 2.12
k_calculated[mz] = 2.062
fr[mz] = 1.080
Tr_selection[mz] = True

# Protonated cis-3-hexenol
# mz = 101.096
# mixrat_bottle[mz] = 986
# unc_mixrat_bottle[mz] = 31/2
# k[mz], k_ionicon[mz], k_Holzinger[mz] = 2, 2, 2
# k_calculated[mz] = np.nan
# fr[mz] = 1.069
# Tr_selection[mz] = False


# Protonated m-xylene
mz = 107.086
mixrat_bottle[mz] = 487
unc_mixrat_bottle[mz] = 15/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.31, 2.26, 2.31
k_calculated[mz] = 2.203
fr[mz] = 1.092
Tr_selection[mz] = True

# Protonated 1,2,4-trimethylbenzene
mz = 121.101
mixrat_bottle[mz] = 482
unc_mixrat_bottle[mz] = 17/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.40, 2.20, 2.40
k_calculated[mz] = 2.404
fr[mz] = 1.105
Tr_selection[mz] = False

# Protonated 1,2,3-trifluorobenzene
mz = 133.026
mixrat_bottle[mz] = 531
unc_mixrat_bottle[mz] = 22/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.46, 2.46, 2.46
k_calculated[mz] = np.nan
fr[mz] = 1.067
Tr_selection[mz] = False

# Protonated monoterpene (100% sabinene)
mz = 137.132
mixrat_bottle[mz] = 982
unc_mixrat_bottle[mz] = 31/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.04, 2.04, 2.04
k_calculated[mz] = np.nan
fr[mz] = 1.117
Tr_selection[mz] = False

# 
mz = 179.929
mixrat_bottle[mz] = np.nan
unc_mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

# Protonated 1,2,4-trichlorobenzene
mz = 180.937
mixrat_bottle[mz] = 485
unc_mixrat_bottle[mz] = 15/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = 2.40, 2.20, 2.40
k_calculated[mz] = 2.183
fr[mz] = 2.4777
Tr_selection[mz] = True

# Iodobenzene
mz = 203.9445
mixrat_bottle[mz] = np.nan
unc_mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

# Protonated D4-Siloxane
mz = 297.082
mixrat_bottle[mz] = 989
unc_mixrat_bottle[mz] = 32/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, 2.00, 2.99
k_calculated[mz] = np.nan
fr[mz] = 1.5
Tr_selection[mz] = False

# Protonated D5-Siloxane
mz = 371.101
mixrat_bottle[mz] = 1008
unc_mixrat_bottle[mz] = 31/2
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, 2.00, 3.39
k_calculated[mz] = np.nan
fr[mz] = 1.7
Tr_selection[mz] = False

# Diiodobenzene
mz = 330.8475
mixrat_bottle[mz] = np.nan
unc_mixrat_bottle[mz] = np.nan
k[mz], k_ionicon[mz], k_Holzinger[mz] = np.nan, np.nan, np.nan
k_calculated[mz] = np.nan
fr[mz] = np.nan
Tr_selection[mz] = False

calibration_ancillary = pd.DataFrame([mixrat_bottle, unc_mixrat_bottle, k, k_ionicon, k_Holzinger, k_calculated, fr, Tr_selection],index=['mixrat_bottle [ppbv]','1_sigm [ppbv]','k [1.e-9 cm3 molecule-1 s-1]','k_ionicon [1.e-9 cm3 molecule-1 s-1]','k_Holzinger [1.e-9 cm3 molecule-1 s-1]','k_calculated [1.e-9 cm3 molecule-1 s-1]','fr','Tr_selection']).transpose()
calibration_ancillary.index.name = 'mz_exact'

calibration_ancillary.to_csv('./Calibration_Ancillary_2024.csv'.format())