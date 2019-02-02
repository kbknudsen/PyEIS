#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:23:40 2018

This script contains tools for extracting impedance data from data files. Currently following data files are supported
    - Bio-Logic '.mpt' files
    - Gamry's '.DTA' files

@author: Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
"""
#Python dependencies
from __future__ import division
import pandas as pd
import numpy as np
from scipy.constants import codata

#### Extracting .mpt files with PEIS or GEIS data
def correct_text_EIS(text_header):
    '''Corrects the text of '*.mpt' and '*.dta' files into readable parameters without spaces, ., or /
    
    <E_we> = averaged Wew value for each frequency
    <I> = Averaged I values for each frequency
    |E_we| = module of Ewe
    |I_we| = module of Iwe
    Cs/F = Capacitance caluculated using an R+C (series) equivalent circuit
    Cp/F = Capacitance caluculated using an R-C (parallel) equivalent circuit
    Ref.:
        - EC-Lab User's Manual
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    if text_header == 'freq/Hz' or text_header == '  Freq(Hz)':
        return 'f'
    elif text_header == 'Re(Z)/Ohm' or text_header == "Z'(a)":
        return 're'
    elif text_header == '-Im(Z)/Ohm' or text_header == "Z''(b)":
        return 'im'
#    elif text_header == "Z''(b)":
#        return 'im_neg'
    elif text_header == '|Z|/Ohm':
        return 'Z_mag'
    elif text_header == 'Phase(Z)/deg':
        return 'Z_phase'
    elif text_header == 'time/s' or text_header == 'Time(Sec)':
        return 'times'
    elif text_header == '<Ewe>/V' or text_header == 'Bias':
        return 'E_avg'
    elif text_header == '<I>/mA':
        return 'I_avg'
    elif text_header == 'Cs/F':
        return 'Cs' ####
    elif text_header == 'Cp/F':
        return 'Cp'
    elif text_header == 'cycle number':
        return 'cycle_number'
    elif text_header == 'Re(Y)/Ohm-1':
        return 'Y_re'
    elif text_header == 'Im(Y)/Ohm-1':
        return 'Y_im'
    elif text_header == '|Y|/Ohm-1':
        return 'Y_mag'
    elif text_header == 'Phase(Y)/deg':
        return 'Y_phase'
    elif text_header == 'Time':
        return 'times'
    elif text_header == 'Freq':
        return 'f'
    elif text_header == 'Zreal':
        return 're'
    elif text_header == 'Zimag':
        return 'im'
    elif text_header == 'Zmod':
        return 'Z_mag'
    elif text_header == 'Vdc':
        return 'E_avg'
    elif text_header == 'Idc':
        return 'I_avg'
    elif text_header == 'I/mA':
        return 'ImA'
    elif text_header == 'Ewe/V':
        return 'EweV'
    elif text_header == 'half cycle':
        return 'half_cycle'
    elif text_header == 'Ns changes':
        return 'Ns_changes'
    else:
        return text_header
    
def extract_mpt(path, EIS_name):
    '''
    Extracting PEIS and GEIS data files from EC-lab '.mpt' format, coloums are renames following correct_text_EIS()
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    EIS_init = pd.read_csv(path+EIS_name, sep='\t', nrows=1,header=0,names=['err'], encoding='latin1') #findes line that states skiplines
#    EIS_test_header_names = pd.read_csv(path+EIS_name, sep='\t', skiprows=int(EIS_init.err[0][18:20])-1, encoding='latin1') #locates number of skiplines
    EIS_test_header_names = pd.read_csv(path+EIS_name, sep='\t', skiprows=int(EIS_init.err[0][18:-1])-1, encoding='latin1') #locates number of skiplines
    names_EIS = []
    for j in range(len(EIS_test_header_names.columns)):
        names_EIS.append(correct_text_EIS(EIS_test_header_names.columns[j])) #reads coloumn text
#    return pd.read_csv(path+EIS_name, sep='\t', skiprows=int(EIS_init.err[0][18:20]), names=names_EIS, encoding='latin1')
    return pd.read_csv(path+EIS_name, sep='\t', skiprows=int(EIS_init.err[0][18:-1]), names=names_EIS, encoding='latin1')

def extract_dta(path, EIS_name):
    '''
    Extracting data files from Gamry '.DTA' format, coloums are renames following correct_text_EIS()
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    dummy_col = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J','K','L','M','N','O','P']
    init = pd.read_csv(path+EIS_name, encoding='latin1', sep='\t', names=dummy_col)
    ZC = pd.Index(init.A)
    header_loc = ZC.get_loc('ZCURVE')+1  ##ZC.get_loc('ZCURVE')+3
    
    header_names_raw = pd.read_csv(path+EIS_name, sep='\t', skiprows=header_loc, encoding='latin1') #locates number of skiplines
    header_names = []
    for j in range(len(header_names_raw.columns)):
        header_names.append(correct_text_EIS(header_names_raw.columns[j])) #reads coloumn text
    data = pd.read_csv(path+EIS_name, sep='\t', skiprows=ZC.get_loc('ZCURVE')+3, names=header_names, encoding='latin1')
    data.update({'im': np.abs(data.im)})
    data = data.assign(cycle_number = 1.0)
    return data

def extract_solar(path, EIS_name):
    '''
    Extracting data files from Solartron's '.z' format, coloums are renames following correct_text_EIS()
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    dummy_col = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J','K','L','M','N','O','P']
    init = pd.read_csv(path+EIS_name, encoding='latin1', sep='\t', names=dummy_col)
    ZC = pd.Index(init.A)
    header_loc = ZC.get_loc('  Freq(Hz)')
    
    header_names_raw = pd.read_csv(path+EIS_name, sep='\t', skiprows=header_loc, encoding='latin1') #locates number of skiplines
    header_names = []
    for j in range(len(header_names_raw.columns)):
        header_names.append(correct_text_EIS(header_names_raw.columns[j])) #reads coloumn text
    data = pd.read_csv(path+EIS_name, sep='\t', skiprows=header_loc+2, names=header_names, encoding='latin1')
    data.update({'im': -data.im})
    data = data.assign(cycle_number = 1.0)
    return data

#
#print()
#print('---> Data Extraction Script Loaded (v. 0.0.2 - 06/27/18)')