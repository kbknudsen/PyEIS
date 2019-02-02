#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:13:33 2018

@author: Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
"""
#Python dependencies
from __future__ import division
import pandas as pd
import numpy as np
from scipy.constants import codata
from pylab import *
from scipy.optimize import curve_fit
import mpmath as mp
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
#from scipy.optimize import leastsq
pd.options.mode.chained_assignment = None

#Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns
import matplotlib.ticker as mtick
mpl.rc('mathtext', fontset='stixsans', default='regular')
mpl.rcParams.update({'axes.labelsize':22})
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16)
mpl.rc('legend',fontsize=14)

from scipy.constants import codata
F = codata.physical_constants['Faraday constant'][0]
Rg = codata.physical_constants['molar gas constant'][0]

### Importing PyEIS add-ons
from .PyEIS_Data_extraction import *
from .PyEIS_Lin_KK import *
from .PyEIS_Advanced_tools import *

### Frequency generator
##
#
def freq_gen(f_start, f_stop, pts_decade=7):
    '''
    Frequency Generator with logspaced freqencies
    
    Inputs
    ----------
    f_start = frequency start [Hz]
    f_stop = frequency stop [Hz]
    pts_decade = Points/decade, default 7 [-]
    
    Output
    ----------
    [0] = frequency range [Hz]
    [1] = Angular frequency range [1/s]
    '''
    f_decades = np.log10(f_start) - np.log10(f_stop)
    f_range = np.logspace(np.log10(f_start), np.log10(f_stop), num=np.around(pts_decade*f_decades), endpoint=True)
    w_range = 2 * np.pi * f_range
    return f_range, w_range

### Simulation Element Functions
##
#
def elem_L(w, L):
    '''
    Simulation Function: -L-
    Returns the impedance of an inductor
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    L = Inductance [ohm * s]
    '''
    return 1j*w*L

def elem_C(w,C):
    '''
    Simulation Function: -C-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    C = Capacitance [F]    
    '''
    return 1/(C*(w*1j))

def elem_Q(w,Q,n):
    '''
    Simulation Function: -Q-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    '''
    return 1/(Q*(w*1j)**n)

### Simulation Curciuts Functions
##
#

def cir_RsC(w, Rs, C):
    '''
    Simulation Function: -Rs-C-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series resistance [Ohm]
    C = Capacitance [F]    
    '''
    return Rs + 1/(C*(w*1j))

def cir_RsQ(w, Rs, Q, n):
    '''
    Simulation Function: -Rs-Q-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    '''
    return Rs + 1/(Q*(w*1j)**n)

def cir_RQ(w, R='none', Q='none', n='none', fs='none'):
    '''
    Simulation Function: -RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RQ_fit()
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    R = Resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    fs = Summit frequency of RQ circuit [Hz]
    '''
    if R == 'none':
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q == 'none':
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n == 'none':
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    return (R/(1+R*Q*(w*1j)**n))

def cir_RsRQ(w, Rs='none', R='none', Q='none', n='none', fs='none'):
    '''
    Simulation Function: -Rs-RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RQ_fit()
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series resistance [Ohm]
    R = Resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase elelment exponent [-]
    fs = Summit frequency of RQ circuit [Hz]
    '''
    if R == 'none':
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q == 'none':
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n == 'none':
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    return Rs + (R/(1+R*Q*(w*1j)**n))

def cir_RC(w, C='none', R='none', fs='none'):
    '''
    Simulation Function: -RC-
    Returns the impedance of an RC circuit, using RQ definations where n=1. see cir_RQ() for details
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    R = Resistance [Ohm]
    C = Capacitance [F]
    fs = Summit frequency of RC circuit [Hz]
    '''
    return cir_RQ(w, R=R, Q=C, n=1, fs=fs)

def cir_RsRQRQ(w, Rs, R='none', Q='none', n='none', fs='none', R2='none', Q2='none', n2='none', fs2='none'):
    '''
    Simulation Function: -Rs-RQ-RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RQ_fit()
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [Ohm]
    
    R = Resistance [Ohm]
    Q = Constant phase element [s^n/ohm]
    n = Constant phase element exponent [-]
    fs = Summit frequency of RQ circuit [Hz]

    R2 = Resistance [Ohm]
    Q2 = Constant phase element [s^n/ohm]
    n2 = Constant phase element exponent [-]
    fs2 = Summit frequency of RQ circuit [Hz]
    '''
    if R == 'none':
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q == 'none':
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n == 'none':
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))

    if R2 == 'none':
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    elif Q2 == 'none':
        Q2 = (1/(R2*(2*np.pi*fs2)**n2))
    elif n2 == 'none':
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))
        
    return Rs + (R/(1+R*Q*(w*1j)**n)) + (R2/(1+R2*Q2*(w*1j)**n2))

def cir_RsRQQ(w, Rs, Q, n, R1='none', Q1='none', n1='none', fs1='none'):
    '''
    Simulation Function: -Rs-RQ-Q-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]
    
    R1 = Resistance in (RQ) circuit [ohm]
    Q1 = Constant phase element in (RQ) circuit [s^n/ohm]
    n1 = Constant phase elelment exponent in (RQ) circuit [-]
    fs1 = Summit frequency of RQ circuit [Hz]

    Q = Constant phase element of series Q [s^n/ohm]
    n = Constant phase elelment exponent of series Q [-]
    '''
    return Rs + cir_RQ(w, R=R1, Q=Q1, n=n1, fs=fs1) + elem_Q(w,Q,n)

def cir_RsRQC(w, Rs, C, R1='none', Q1='none', n1='none', fs1='none'):
    '''
    Simulation Function: -Rs-RQ-C-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]
    
    R1 = Resistance in (RQ) circuit [ohm]
    Q1 = Constant phase element in (RQ) circuit [s^n/ohm]
    n1 = Constant phase elelment exponent in (RQ) circuit [-]
    fs1 = summit frequency of RQ circuit [Hz]

    C = Constant phase element of series Q [s^n/ohm]
    '''
    return Rs + cir_RQ(w, R=R1, Q=Q1, n=n1, fs=fs1) + elem_C(w, C=C)

def cir_RsRCC(w, Rs, R1, C1, C):
    '''
    Simulation Function: -Rs-RC-C-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]
    
    R1 = Resistance in (RQ) circuit [ohm]
    C1 = Constant phase element in (RQ) circuit [s^n/ohm]

    C = Capacitance of series C [s^n/ohm]
    '''
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_C(w, C=C)

def cir_RsRCQ(w, Rs, R1, C1, Q, n):
    '''
    Simulation Function: -Rs-RC-Q-
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Rs = Series Resistance [ohm]
    
    R1 = Resistance in (RQ) circuit [ohm]
    C1 = Constant phase element in (RQ) circuit [s^n/ohm]

    Q = Constant phase element of series Q [s^n/ohm]
    n = Constant phase elelment exponent of series Q [-]
    '''
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_Q(w,Q,n)
    
def Randles_coeff(w, n_electron, A, E='none', E0='none', D_red='none', D_ox='none', C_red='none', C_ox='none', Rg=Rg, F=F, T=298.15):
    '''
    Returns the Randles coefficient sigma [ohm/s^1/2]. 
    Two cases: a) ox and red are both present in solution here both Cred and Dred are defined, b) In the particular case where initially
    only Ox species are present in the solution with bulk concentration C*_ox, the surface concentrations may be calculated as function
    of the electrode potential following Nernst equation. Here C_red and D_red == 'none'
    
    Ref.:
        - Lasia, A.L., ISBN: 978-1-4614-8932-0, "Electrochemical Impedance Spectroscopy and its Applications" 
        - Bard A.J., ISBN: 0-471-04372-9, Faulkner L. R. (2001) "Electrochemical methods: Fundamentals and applications". New York: Wiley.

    Kristian B. Knudsen (kknu@berkeley.edu // kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    n_electron = number of e- [-]
    A = geometrical surface area [cm2]
    D_ox = Diffusion coefficent of oxidized specie [cm2/s]
    D_red = Diffusion coefficent of reduced specie [cm2/s]
    C_ox = Bulk concetration of oxidized specie [mol/cm3]
    C_red = Bulk concetration of reduced specie [mol/cm3]
    T = Temperature [K]
    Rg = Gas constant [J/molK]
    F = Faradays consntat [C/mol]
    E = Potential [V]
        if reduced specie is absent == 'none'
    E0 = formal potential [V]
        if reduced specie is absent == 'none'
    
    Returns
    ----------
    Randles coefficient [ohm/s^1/2]
    '''
    if C_red != 'none' and D_red != 'none':
        sigma = ((Rg*T) / ((n_electron**2) * A * (F**2) * (2**(1/2)))) * ((1/(D_ox**(1/2) * C_ox)) + (1/(D_red**(1/2) * C_red)))
    elif C_red == 'none' and D_red == 'none' and E!='none' and E0!= 'none':
        f = F/(Rg*T)
        x = (n_electron*f*(E-E0))/2
        func_cosh2 = (np.cosh(2*x)+1)/2
        sigma = ((4*Rg*T) / ((n_electron**2) * A * (F**2) * C_ox * ((2*D_ox)**(1/2)) )) * func_cosh2
    else:
        print('define E and E0')
    Z_Aw = sigma*(w**(-0.5))-1j*sigma*(w**(-0.5))
    return Z_Aw

def cir_Randles(w, n_electron, D_red, D_ox, C_red, C_ox, Rs, Rct, n, E, A, Q='none', fs='none', E0=0, F=F, Rg=Rg, T=298.15):
    '''
    Simulation Function: Randles -Rs-(Q-(RW)-)-
    Return the impedance of a Randles circuit with full complity of the warbug constant
    NOTE: This Randles circuit is only meant for semi-infinate linear diffusion

    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    n_electron = number of e- [-]
    A = geometrical surface area [cm2]
    D_ox = Diffusion coefficent of oxidized specie [cm2/s]
    D_red = Diffusion coefficent of reduced specie [cm2/s]
    C_ox = Concetration of oxidized specie [mol/cm3]
    C_red = Concetration of reduced specie [mol/cm3]
    T = Temperature [K]
    Rg = Gas constant [J/molK]
    F = Faradays consntat [C/mol]
    E = Potential [V]
        if reduced specie is absent == 'none'
    E0 = Formal potential [V]
        if reduced specie is absent == 'none'

    Rs = Series resistance [ohm]
    Rct = charge-transfer resistance [ohm]

    Q = Constant phase element used to model the double-layer capacitance [F]
    n = expononent of the CPE [-]
    
    Returns
    ----------
    The real and imaginary impedance of a Randles circuit [ohm]
    '''
    Z_Rct = Rct
    Z_Q = elem_Q(w,Q,n)
    Z_w = Randles_coeff(w, n_electron=n_electron, E=E, E0=E0, D_red=D_red, D_ox=D_ox, C_red=C_red, C_ox=C_ox, A=A, T=T, Rg=Rg, F=F)
    return Rs + 1/(1/Z_Q + 1/(Z_Rct+Z_w))


def cir_Randles_simplified(w, Rs, R, n, sigma, Q='none', fs='none'):
    '''
    Simulation Function: Randles -Rs-(Q-(RW)-)-
    Return the impedance of a Randles circuit with a simplified
    
    NOTE: This Randles circuit is only meant for semi-infinate linear diffusion
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    if R == 'none':
        R = (1/(Q*(2*np.pi*fs)**n))
    elif Q == 'none':
        Q = (1/(R*(2*np.pi*fs)**n))
    elif n == 'none':
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    
    Z_Q = 1/(Q*(w*1j)**n)
    Z_R = R
    Z_w = sigma*(w**(-0.5))-1j*sigma*(w**(-0.5))
    
    return Rs + 1/(1/Z_Q + 1/(Z_R+Z_w))

# Polymer electrolytes

def cir_C_RC_C(w, Ce, Cb='none', Rb='none', fsb='none'):
    '''
    Simulation Function: -C-(RC)-C-
    
    This circuit is often used for modeling blocking electrodes with a polymeric electrolyte, which exhibts a immobile ionic species in bulk that gives a capacitance contribution
    to the otherwise resistive electrolyte
    
    Ref:
    - MacCallum J.R., and Vincent, C.A. "Polymer Electrolyte Reviews - 1" Elsevier Applied Science Publishers LTD, London, Bruce, P. "Electrical Measurements on Polymer Electrolytes" 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Ce = Interfacial capacitance [F]
    Rb = Bulk/series resistance [Ohm]
    Cb = Bulk capacitance [F]
    fsb = summit frequency of bulk (RC) circuit [Hz]
    '''
    Z_C = elem_C(w,C=Ce)
    Z_RC = cir_RC(w, C=Cb, R=Rb, fs=fsb)
    return Z_C + Z_RC

def cir_Q_RQ_Q(w, Qe, ne, Qb='none', Rb='none', fsb='none', nb='none'):
    '''
    Simulation Function: -Q-(RQ)-Q-
    
    Modified cir_C_RC_C() circuits that can be used if electrodes and bulk are not behaving like ideal capacitors
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    Qe = Interfacial capacitance modeled with a CPE [F]
    ne = Interfacial constant phase element exponent [-]
    
    Rb = Bulk/series resistance [Ohm]
    Qb = Bulk capacitance modeled with a CPE [s^n/ohm]
    nb = Bulk constant phase element exponent [-]
    fsb = summit frequency of bulk (RQ) circuit [Hz]
    '''
    Z_Q = elem_Q(w,Q=Qe,n=ne)
    Z_RQ = cir_RQ(w, Q=Qb, R=Rb, fs=fsb, n=nb)
    return Z_Q + Z_RQ

def tanh(x):
    '''
    As numpy gives errors when tanh becomes very large, above 10^250, this functions is used for np.tanh
    '''
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def cir_RCRCZD(w, L, D_s, u1, u2, Cb='none', Rb='none', fsb='none', Ce='none', Re='none', fse='none'):
    '''
    Simulation Function: -RC_b-RC_e-Z_D
    
    This circuit has been used to study non-blocking electrodes with an ioniocally conducting electrolyte with a mobile and immobile ionic specie in bulk, this is mixed with a
    ionically conducting salt. This behavior yields in a impedance response, that consists of the interfacial impendaces -(RC_e)-, the ionically conducitng polymer -(RC_e)-,
    and the diffusional impedance from the dissolved salt.
    
    Refs.:
        - Sørensen, P.R. and Jacobsen T., Electrochimica Acta, 27, 1671-1675, 1982, "Conductivity, Charge Transfer and Transport number - An AC-Investigation
        of the Polymer Electrolyte LiSCN-Poly(ethyleneoxide)"
        - MacCallum J.R., and Vincent, C.A. "Polymer Electrolyte Reviews - 1" Elsevier Applied Science Publishers LTD, London
        Bruce, P. "Electrical Measurements on Polymer Electrolytes" 

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    w = Angular frequency [1/s]
    L = Thickness of electrode [cm]
    D_s = Diffusion coefficient of dissolved salt [cm2/s]
    u1 = Mobility of the ion reacting at the electrode interface
    u2 = Mobility of other ion

    Re = Interfacial resistance [Ohm]
    Ce = Interfacial  capacitance [F]
    fse = Summit frequency of the interfacial (RC) circuit [Hz]

    Rb = Bulk/series resistance [Ohm]
    Cb = Bulk capacitance [F]
    fsb = Summit frequency of the bulk (RC) circuit [Hz]
    '''
    Z_RCb = cir_RC(w, C=Cb, R=Rb, fs=fsb)
    Z_RCe = cir_RC(w, C=Ce, R=Re, fs=fse)
    alpha = ((w*1j*L**2)/D_s)**(1/2)
    Z_D = Rb * (u2/u1) * (tanh(x=alpha)/alpha)
    return Z_RCb + Z_RCe + Z_D
    
# Transmission lines

def cir_RsTLsQ(w, Rs, L, Ri, Q='none', n='none'):
    '''
    Simulation Function: -Rs-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance (Q)
    
    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance). 
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    
    Inputs
    -----------
    Rs = Series resistance [ohm]
        
    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    Q = Interfacial capacitance of non-faradaic interface [F/cm]
    n = exponent for the interfacial capacitance [-]
    '''    
    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri # ohm/cm
    Lam = (Phi/X1)**(1/2) #np.sqrt(Phi/X1)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)  #Handles coth with x having very large or very small numbers

    Z_TLsQ = Lam * X1 * coth_mp        

    return Rs + Z_TLsQ

def cir_RsRQTLsQ(w, Rs, R1, fs1, n1, L, Ri, Q, n, Q1='none'):
    '''
    Simulation Function: -Rs-RQ-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance(Q)
    
    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance). 
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    
    Inputs
    -----------
    Rs = Series resistance [ohm]
    
    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = Exponent for RQ circuit [-]
    Q1 = Constant phase element of RQ circuit [s^n/ohm]
    
    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    Q = Interfacial capacitance of non-faradaic interface [F/cm]
    n = Exponent for the interfacial capacitance [-]
    
    Output
    -----------
    Impdance of Rs-(RQ)1-TLsQ
    '''    
    Z_RQ = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)
    
    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)
    
    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_TLsQ = Lam * X1 * coth_mp
    
    return Rs + Z_RQ + Z_TLsQ

def cir_RsTLs(w, Rs, L, Ri, R='none', Q='none', n='none', fs='none'):
    '''
    Simulation Function: -Rs-TLs-
    TLs = Simplified Transmission Line, with a faradaic interfacial impedance (RQ)
    
    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance). 
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    
    Inputs
    -----------
    Rs = Series resistance [ohm]
    
    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    R = Interfacial Charge transfer resistance [ohm*cm]
    fs = Summit frequency of interfacial RQ circuit [Hz]
    n = Exponent for interfacial RQ circuit [-]
    Q = Constant phase element of interfacial capacitance [s^n/Ohm] 

    Output
    -----------
    Impedance of Rs-TLs(RQ)
    '''
    Phi = cir_RQ(w, R, Q, n, fs)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)    

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
    
    Z_TLs = Lam * X1 * coth_mp
    
    return Rs + Z_TLs

def cir_RsRQTLs(w, Rs, L, Ri, R1, n1, fs1, R2, n2, fs2, Q1='none', Q2='none'):
    '''
    Simulation Function: -Rs-RQ-TLs-
    TLs = Simplified Transmission Line, with a faradaic interfacial impedance (RQ)
    
    The simplified transmission line assumes that Ri is much greater than Rel (electrode resistance). 
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    
    Inputs
    -----------
    Rs = Series resistance [ohm]
      
    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = Exponent for RQ circuit [-]
    Q1 = Constant phase element of RQ circuit [s^n/(ohm * cm)]
    
    L = Length/Thickness of porous electrode [cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    R2 = Interfacial Charge transfer resistance [ohm*cm]
    fs2 = Summit frequency of interfacial RQ circuit [Hz]
    n2 = Exponent for interfacial RQ circuit [-]
    Q2 = Constant phase element of interfacial capacitance [s^n/Ohm] 
    
    Output
    -----------
    Impedance of Rs-(RQ)1-TLs(RQ)2
    '''
    Z_RQ = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)
    
    Phi = cir_RQ(w=w, R=R2, Q=Q2, n=n2, fs=fs2)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)    

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
    
    Z_TLs = Lam * X1 * coth_mp
    
    return Rs + Z_RQ + Z_TLs

### Support function

def sinh(x):
    '''
    As numpy gives errors when sinh becomes very large, above 10^250, this functions is used instead of np/mp.sinh()
    '''
    return (1 - np.exp(-2*x))/(2*np.exp(-x))

def coth(x):
    '''
    As numpy gives errors when coth becomes very large, above 10^250, this functions is used instead of np/mp.coth()
    '''
    return (1 + np.exp(-2*x))/(1 - np.exp(-2*x))

###

def cir_RsTLQ(w, L, Rs, Q, n, Rel, Ri):
    '''
    Simulation Function: -R-TLQ- (interfacial non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
                
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ------------------
    Rs = Series resistance [ohm]

    Q = Constant phase element for the interfacial capacitance [s^n/ohm]        
    n = exponenet for interfacial RQ element [-]
    
    Rel = electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = thickness of porous electrode [cm]
    
    Output
    --------------
    Impedance of Rs-TL
    '''
    #The impedance of the series resistance
    Z_Rs = Rs
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    
    return Z_Rs + Z_TL

def cir_RsRQTLQ(w, L, Rs, Q, n, Rel, Ri, R1, n1, fs1, Q1='none'):
    '''
    Simulation Function: -R-RQ-TLQ- (interfacial non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
                
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ------------------
    Rs = Series resistance [ohm]
    
    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = exponent for RQ circuit [-]
    Q1 = constant phase element of RQ circuit [s^n/(ohm * cm)]

    Q = Constant phase element for the interfacial capacitance [s^n/ohm]        
    n = exponenet for interfacial RQ element [-]
    
    Rel = electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = thickness of porous electrode [cm]
    
    Output
    --------------
    Impedance of Rs-TL
    '''
    #The impedance of the series resistance
    Z_Rs = Rs
    
    #The (RQ) circuit in series with the transmission line
    Z_RQ1 = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    
    return Z_Rs + Z_RQ1 + Z_TL

def cir_RsTL(w, L, Rs, R, fs, n, Rel, Ri, Q='none'):
    '''
    Simulation Function: -R-TL- (interfacial reacting, i.e. non-blocking)
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
                
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ------------------
    Rs = Series resistance [ohm]
        
    R = Interfacial charge transfer resistance [ohm * cm]
    fs = Summit frequency for the interfacial RQ element [Hz]
    n = Exponenet for interfacial RQ element [-]
    Q = Constant phase element for the interfacial capacitance [s^n/ohm]
    
    Rel = Electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = Thickness of porous electrode [cm]
    
    Output
    --------------
    Impedance of Rs-TL
    '''
    #The impedance of the series resistance
    Z_Rs = Rs
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = cir_RQ(w, R=R, Q=Q, n=n, fs=fs)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL

def cir_RsRQTL(w, L, Rs, R1, fs1, n1, R2, fs2, n2, Rel, Ri, Q1='none', Q2='none'):
    '''
    Simulation Function: -R-RQ-TL- (interfacial reacting, i.e. non-blocking)
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    Ref.:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
                
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ------------------
    Rs = Series resistance [ohm]
        
    R1 = Charge transfer resistance of RQ circuit [ohm]
    fs1 = Summit frequency for RQ circuit [Hz]
    n1 = exponent for RQ circuit [-]
    Q1 = constant phase element of RQ circuit [s^n/(ohm * cm)]
    
    R2 = interfacial charge transfer resistance [ohm * cm]
    fs2 = Summit frequency for the interfacial RQ element [Hz]
    n2 = exponenet for interfacial RQ element [-]
    Q2 = Constant phase element for the interfacial capacitance [s^n/ohm]
    
    Rel = electronic resistance of electrode [ohm/cm]
    Ri = Ionic resistance inside of flodded pores [ohm/cm]
    L = thickness of porous electrode [cm]
    
    Output
    --------------
    Impedance of Rs-TL
    '''
    #The impedance of the series resistance
    Z_Rs = Rs
    
    #The (RQ) circuit in series with the transmission line
    Z_RQ1 = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = cir_RQ(w, R=R2, Q=Q2, n=n2, fs=fs2)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)
    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    return Z_Rs + Z_RQ1 + Z_TL

# Transmission lines with solid-state transport

def cir_RsTL_1Dsolid(w, L, D, radius, Rs, R, Q, n, R_w, n_w, Rel, Ri):
    '''
    Simulation Function: -R-TL(Q(RW))-
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    Warburg element is specific for 1D solid-state diffusion
    
    Refs:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Illig, J., Physically based Impedance Modelling of Lithium-ion Cells, KIT Scientific Publishing (2014)
        - Scipioni, et al., ECS Transactions, 69 (18) 71-80 (2015) 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ------------------
    Rs = Series resistance [ohm]
        
    R = particle charge transfer resistance [ohm*cm^2]
    Q = Summit frequency peak of RQ element in the modified randles element of a particle [Hz]
    n = exponenet for internal RQ element in the modified randles element of a particle [-]
    
    Rel = electronic resistance of electrode [ohm/cm]
    Ri = ionic resistance of solution in flooded pores of electrode [ohm/cm]
    R_w = polarization resistance of finite diffusion Warburg element [ohm]
    n_w = exponent for Warburg element [-]
    
    L = thickness of porous electrode [cm]
    D = solid-state diffusion coefficient [cm^2/s]
    radius = average particle radius [cm]
    
    Output
    --------------
    Impedance of Rs-TL(Q(RW))
    '''
    #The impedance of the series resistance
    Z_Rs = Rs
    
    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D
    
    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x
    
    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R
    Z_Q = elem_Q(w,Q=Q,n=n)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)        
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)
    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    return Z_Rs + Z_TL

def cir_RsRQTL_1Dsolid(w, L, D, radius, Rs, R1, fs1, n1, R2, Q2, n2, R_w, n_w, Rel, Ri, Q1='none'):
    '''
    Simulation Function: -R-RQ-TL(Q(RW))-
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    Warburg element is specific for 1D solid-state diffusion
    
    Refs:
        - De Levie R., and Delahay P., Advances in Electrochemistry and Electrochemical Engineering, p. 329, Wiley-Interscience, New York (1973)
        - Bisquert J. Electrochemistry Communications 1, 1999, 429-435, "Anamalous transport effects in the impedance of porous film electrodes"
        - Bisquert J. J. Phys. Chem. B., 2000, 104, 2287-2298, "Doubling exponent models for the analysis of porous film electrodes by impedance.
        Relaxation of TiO2 nanoporous in aqueous solution"
        - Illig, J., Physically based Impedance Modelling of Lithium-ion Cells, KIT Scientific Publishing (2014)
        - Scipioni, et al., ECS Transactions, 69 (18) 71-80 (2015) 
    
    David Brown (demoryb@berkeley.edu)
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ------------------
    Rs = Series resistance [ohm]
    
    R1 = charge transfer resistance of the interfacial RQ element [ohm*cm^2]
    fs1 = max frequency peak of the interfacial RQ element[Hz]
    n1 = exponenet for interfacial RQ element
    
    R2 = particle charge transfer resistance [ohm*cm^2]
    Q2 = Summit frequency peak of RQ element in the modified randles element of a particle [Hz]
    n2 = exponenet for internal RQ element in the modified randles element of a particle [-]
    
    Rel = electronic resistance of electrode [ohm/cm]
    Ri = ionic resistance of solution in flooded pores of electrode [ohm/cm]
    R_w = polarization resistance of finite diffusion Warburg element [ohm]
    n_w = exponent for Warburg element [-]
    
    L = thickness of porous electrode [cm]
    D = solid-state diffusion coefficient [cm^2/s]
    radius = average particle radius [cm]
    
    Output
    ------------------
    Impedance of R-RQ-TL(Q(RW))
    '''
    #The impedance of the series resistance
    Z_Rs = Rs
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    Z_RQ = cir_RQ(w=w, R=R1, Q=Q1, n=n1, fs=fs1)
    
    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D
    
    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x
    
    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R2
    Z_Q = elem_Q(w,Q=Q2,n=n2)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)
#        
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    
    return Z_Rs + Z_RQ + Z_TL

### Fitting Circuit Functions
##
#

def elem_C_fit(params, w):
    '''
    Fit Function: -C-
    '''
    C = params['C']
    return 1/(C*(w*1j))

def elem_Q_fit(params, w):
    '''
    Fit Function: -Q-
    
    Constant Phase Element for Fitting
    '''
    Q = params['Q']
    n = params['n']
    return 1/(Q*(w*1j)**n)

def cir_RsC_fit(params, w):
    '''
    Fit Function: -Rs-C-
    '''
    Rs = params['Rs']
    C = params['C']
    return Rs + 1/(C*(w*1j))

def cir_RsQ_fit(params, w):
    '''
    Fit Function: -Rs-Q-
    '''
    Rs = params['Rs']
    Q = params['Q']
    n = params['n']
    return Rs + 1/(Q*(w*1j)**n)

def cir_RC_fit(params, w):
    '''
    Fit Function: -RC-
    Returns the impedance of an RC circuit, using RQ definations where n=1
    '''
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['C']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("C") == -1: #elif Q == 'none':
        R = params['R']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['C']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['C']
    return cir_RQ(w, R=R, Q=C, n=1, fs=fs)


def cir_RQ_fit(params, w):
    '''
    Fit Function: -RQ-
    Return the impedance of an RQ circuit:
    Z(w) = R / (1+ R*Q * (2w)^n)
    
    See Explanation of equations under cir_RQ()
    
    The params.keys()[10:] finds the names of the user defined parameters that should be interated over if X == -1, if the paramter is not given, it becomes equal to 'none'
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        n = params['n']
        Q = params['Q']
    return R/(1+R*Q*(w*1j)**n)

def cir_RsRQ_fit(params, w):
    '''
    Fit Function: -Rs-RQ-
    Return the impedance of an Rs-RQ circuit. See details for RQ under cir_RsRQ_fit()
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['Q']
        n = params['n']
        
    Rs = params['Rs']
    return Rs + (R/(1+R*Q*(w*1j)**n))

def cir_RsRQRQ_fit(params, w):
    '''
    Fit Function: -Rs-RQ-RQ-
    Return the impedance of an Rs-RQ circuit. See details under cir_RsRQRQ()
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    if str(params.keys())[10:].find("'R'") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'Q'") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'n'") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("'fs'") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['Q']
        n = params['n']

    if str(params.keys())[10:].find("'R2'") == -1: #if R == 'none':
        Q2 = params['Q2']
        n2 = params['n2']
        fs2 = params['fs2']
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    if str(params.keys())[10:].find("'Q2'") == -1: #elif Q == 'none':
        R2 = params['R2']
        n2 = params['n2']
        fs2 = params['fs2']
        Q2 = (1/(R2*(2*np.pi*fs2)**n2))
    if str(params.keys())[10:].find("'n2'") == -1: #elif n == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        fs2 = params['fs2']
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))
    if str(params.keys())[10:].find("'fs2'") == -1: #elif fs == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        n2 = params['n2']

    Rs = params['Rs']
    return Rs + (R/(1+R*Q*(w*1j)**n)) + (R2/(1+R2*Q2*(w*1j)**n2))

def cir_Randles_simplified_Fit(params, w):
    '''
    Fit Function: Randles simplified -Rs-(Q-(RW)-)-
    Return the impedance of a Randles circuit. See more under cir_Randles_simplified()

    NOTE: This Randles circuit is only meant for semi-infinate linear diffusion
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    if str(params.keys())[10:].find("'R'") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'Q'") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("'n'") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("'fs'") == -1: #elif fs == 'none':
        R = params['R']
        Q = params['Q']
        n = params['n']
    
    Rs = params['Rs']
    sigma = params['sigma']
    
    Z_Q = 1/(Q*(w*1j)**n)
    Z_R = R
    Z_w = sigma*(w**(-0.5))-1j*sigma*(w**(-0.5))
    
    return Rs + 1/(1/Z_Q + 1/(Z_R+Z_w))

def cir_RsRQQ_fit(params, w):
    '''
    Fit Function: -Rs-RQ-Q-
    
    See cir_RsRQQ() for details
    '''
    Rs = params['Rs']
    Q = params['Q']
    n = params['n']
    Z_Q = 1/(Q*(w*1j)**n)

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))
    
    return Rs + Z_RQ + Z_Q

def cir_RsRQC_fit(params, w):
    '''
    Fit Function: -Rs-RQ-C-
    
    See cir_RsRQC() for details
    '''
    Rs = params['Rs']
    C = params['C']
    Z_C = 1/(C*(w*1j))

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))
    
    return Rs + Z_RQ + Z_C

def cir_RsRCC_fit(params, w):
    '''
    Fit Function: -Rs-RC-C-
    
    See cir_RsRCC() for details
    '''
    Rs = params['Rs']
    R1 = params['R1']
    C1 = params['C1']
    C = params['C']
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_C(w, C=C)

def cir_RsRCQ_fit(params, w):
    '''
    Fit Function: -Rs-RC-Q-
    
    See cir_RsRCQ() for details
    '''
    Rs = params['Rs']
    R1 = params['R1']
    C1 = params['C1']
    Q = params['Q']
    n = params['n']
    return Rs + cir_RC(w, C=C1, R=R1, fs='none') + elem_Q(w,Q,n)

# Polymer electrolytes
    
def cir_C_RC_C_fit(params, w):
    '''
    Fit Function: -C-(RC)-C-
    
    See cir_C_RC_C() for details
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    # Interfacial impedance
    Ce = params['Ce']
    Z_C = 1/(Ce*(w*1j))
    
    # Bulk impendance
    if str(params.keys())[10:].find("Rb") == -1: #if R == 'none':
        Cb = params['Cb']
        fsb = params['fsb']
        Rb = (1/(Cb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("Cb") == -1: #elif Q == 'none':
        Rb = params['Rb']
        fsb = params['fsb']
        Cb = (1/(Rb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("fsb") == -1: #elif fs == 'none':
        Rb = params['Rb']
        Cb = params['Cb']
    Z_RC = (Rb/(1+Rb*Cb*(w*1j)))
    

    return Z_C + Z_RC

def cir_Q_RQ_Q_Fit(params, w):
    '''
    Fit Function: -Q-(RQ)-Q-
    
    See cir_Q_RQ_Q() for details
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    # Interfacial impedance
    Qe = params['Qe']
    ne = params['ne']
    Z_Q = 1/(Qe*(w*1j)**ne)
    
    # Bulk impedance
    if str(params.keys())[10:].find("Rb") == -1: #if R == 'none':
        Qb = params['Qb']
        nb = params['nb']
        fsb = params['fsb']
        Rb = (1/(Qb*(2*np.pi*fsb)**nb))
    if str(params.keys())[10:].find("Qb") == -1: #elif Q == 'none':
        Rb = params['Rb']
        nb = params['nb']
        fsb = params['fsb']
        Qb = (1/(Rb*(2*np.pi*fsb)**nb))
    if str(params.keys())[10:].find("nb") == -1: #elif n == 'none':
        Rb = params['Rb']
        Qb = params['Qb']
        fsb = params['fsb']
        nb = np.log(Qb*Rb)/np.log(1/(2*np.pi*fsb))
    if str(params.keys())[10:].find("fsb") == -1: #elif fs == 'none':
        Rb = params['Rb']
        nb = params['nb']
        Qb = params['Qb']
    Z_RQ =  Rb/(1+Rb*Qb*(w*1j)**nb)

    return Z_Q + Z_RQ

def cir_RCRCZD_fit(params, w):
    '''
    Fit Function: -RC_b-RC_e-Z_D
    
    See cir_RCRCZD() for details

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    # Interfacial impendace
    if str(params.keys())[10:].find("Re") == -1: #if R == 'none':
        Ce = params['Ce']
        fse = params['fse']
        Re = (1/(Ce*(2*np.pi*fse)))
    if str(params.keys())[10:].find("Ce") == -1: #elif Q == 'none':
        Re = params['Rb']
        fse = params['fsb']
        Ce = (1/(Re*(2*np.pi*fse)))
    if str(params.keys())[10:].find("fse") == -1: #elif fs == 'none':
        Re = params['Re']
        Ce = params['Ce']
    Z_RCe = (Re/(1+Re*Ce*(w*1j)))

    # Bulk impendance
    if str(params.keys())[10:].find("Rb") == -1: #if R == 'none':
        Cb = params['Cb']
        fsb = params['fsb']
        Rb = (1/(Cb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("Cb") == -1: #elif Q == 'none':
        Rb = params['Rb']
        fsb = params['fsb']
        Cb = (1/(Rb*(2*np.pi*fsb)))
    if str(params.keys())[10:].find("fsb") == -1: #elif fs == 'none':
        Rb = params['Rb']
        Cb = params['Cb']
    Z_RCb = (Rb/(1+Rb*Cb*(w*1j)))
    
    # Mass transport impendance
    L = params['L']
    D_s = params['D_s']
    u1 = params['u1']
    u2 = params['u2']
    
    alpha = ((w*1j*L**2)/D_s)**(1/2)
    Z_D = Rb * (u2/u1) * (tanh(alpha)/alpha)
    return Z_RCb + Z_RCe + Z_D

# Transmission lines

def cir_RsTLsQ_fit(params, w):
    '''
    Fit Function: -Rs-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance (Q)
    See more under cir_RsTLsQ()
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Q = params['Q']
    n = params['n']    
    
    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri # ohm/cm
    Lam = (Phi/X1)**(1/2) #np.sqrt(Phi/X1)

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)  #Handles coth with x having very large or very small numbers
#    
#    Z_TLsQ = Lam * X1 * coth_mp    
    Z_TLsQ = Lam * X1 * coth(x)

    return Rs + Z_TLsQ

def cir_RsRQTLsQ_Fit(params, w):
    '''
    Fit Function: -Rs-RQ-TLsQ-
    TLs = Simplified Transmission Line, with a non-faradaic interfacial impedance (Q)
    See more under cir_RsRQTLsQ
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Q = params['Q']
    n = params['n']

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))
    

    Phi = 1/(Q*(w*1j)**n)
    X1 = Ri
    Lam = (Phi/X1)**(1/2)

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers

    Z_TLsQ = Lam * X1 * coth_mp
    
    return Rs + Z_RQ + Z_TLsQ

def cir_RsTLs_Fit(params, w):
    '''
    Fit Function: -Rs-RQ-TLs-
    TLs = Simplified Transmission Line, with a faradaic interfacial impedance (RQ)
    See mor under cir_RsTLs()
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        n = params['n']
        Q = params['Q']
    Phi = R/(1+R*Q*(w*1j)**n)

    X1 = Ri
    Lam = (Phi/X1)**(1/2)    
    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers

    Z_TLs = Lam * X1 * coth_mp
    
    return Rs + Z_TLs

def cir_RsRQTLs_Fit(params, w):
    '''
    Fit Function: -Rs-RQ-TLs-
    TLs = Simplified Transmission Line with a faradaic interfacial impedance (RQ)
    See more under cir_RsRQTLs()
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']

    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ = (R1/(1+R1*Q1*(w*1j)**n1))

    if str(params.keys())[10:].find("R2") == -1: #if R == 'none':
        Q2 = params['Q2']
        n2 = params['n2']
        fs2 = params['fs2']
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    if str(params.keys())[10:].find("Q2") == -1: #elif Q == 'none':
        R2 = params['R2']
        n2 = params['n2']
        fs2 = params['fs2']
        Q2 = (1/(R2*(2*np.pi*fs2)**n1))
    if str(params.keys())[10:].find("n2") == -1: #elif n == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        fs2 = params['fs2']
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))
    if str(params.keys())[10:].find("fs2") == -1: #elif fs == 'none':
        R2 = params['R2']
        n2 = params['n2']
        Q2 = params['Q2']    
    Phi = (R2/(1+R2*Q2*(w*1j)**n2))
    X1 = Ri
    Lam = (Phi/X1)**(1/2)    

    x = L/Lam
    x_mp = mp.matrix(x) #x in mp.math format
    coth_mp = []
    for i in range(len(Lam)):
        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
    
    Z_TLs = Lam * X1 * coth_mp
    
    return Rs + Z_RQ + Z_TLs

def cir_RsTLQ_fit(params, w):
    '''
    Fit Function: -R-TLQ- (interface non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']
    Q = params['Q']
    n = params['n']
    
    #The impedance of the series resistance
    Z_Rs = Rs
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL

def cir_RsRQTLQ_fit(params, w):
    '''
    Fit Function: -R-RQ-TLQ- (interface non-reacting, i.e. blocking electrode)
    Transmission line w/ full complexity, which both includes Ri and Rel
                
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']
    Q = params['Q']
    n = params['n']
    
    #The impedance of the series resistance
    Z_Rs = Rs
    
    #The (RQ) circuit in series with the transmission line
    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    if str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    if str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ1 = (R1/(1+R1*Q1*(w*1j)**n1))
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    Phi = elem_Q(w, Q=Q, n=n)
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_RQ1 + Z_TL

def cir_RsTL_Fit(params, w):
    '''
    Fit Function: -R-TLQ- (interface reacting, i.e. non-blocking)
    Transmission line w/ full complexity, which both includes Ri and Rel
    
    See cir_RsTL() for details
            
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']

    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R") == -1: #if R == 'none':
        Q = params['Q']
        n = params['n']
        fs = params['fs']
        R = (1/(Q*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("Q") == -1: #elif Q == 'none':
        R = params['R']
        n = params['n']
        fs = params['fs']
        Q = (1/(R*(2*np.pi*fs)**n))
    if str(params.keys())[10:].find("n") == -1: #elif n == 'none':
        R = params['R']
        Q = params['Q']
        fs = params['fs']
        n = np.log(Q*R)/np.log(1/(2*np.pi*fs))
    if str(params.keys())[10:].find("fs") == -1: #elif fs == 'none':
        R = params['R']
        n = params['n']
        Q = params['Q']

    Phi = (R/(1+R*Q*(w*1j)**n))
    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float(mp.sinh(x_mp[i]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_TL

def cir_RsRQTL_fit(params, w):
    '''
    Fit Function: -R-RQ-TL- (interface reacting, i.e. non-blocking)
    Transmission line w/ full complexity including both includes Ri and Rel
                
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    Rel = params['Rel']

    #The impedance of the series resistance
    Z_Rs = Rs

    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    elif str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ1 = (R1/(1+R1*Q1*(w*1j)**n1))
#    
#    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R2") == -1: #if R == 'none':
        Q2 = params['Q2']
        n2 = params['n2']
        fs2 = params['fs2']
        R2 = (1/(Q2*(2*np.pi*fs2)**n2))
    elif str(params.keys())[10:].find("Q2") == -1: #elif Q == 'none':
        R2 = params['R2']
        n2 = params['n2']
        fs2 = params['fs2']
        Q2 = (1/(R2*(2*np.pi*fs2)**n1))
    elif str(params.keys())[10:].find("n2") == -1: #elif n == 'none':
        R2 = params['R2']
        Q2 = params['Q2']
        fs2 = params['fs2']
        n2 = np.log(Q2*R2)/np.log(1/(2*np.pi*fs2))
    elif str(params.keys())[10:].find("fs2") == -1: #elif fs == 'none':
        R2 = params['R2']
        n2 = params['n2']
        Q2 = params['Q2']
    Phi = (R2/(1+R2*Q2*(w*1j)**n2))

    X1 = Ri
    X2 = Rel
    Lam = (Phi/(X1+X2))**(1/2)    

    x = L/Lam
#    x_mp = mp.matrix(x) #x in mp.math format
#    coth_mp = []
#    sinh_mp = []
#    for i in range(len(Lam)):
#        coth_mp.append(float(mp.coth(x_mp[i]).real)+float((mp.coth(x_mp[i]).imag))*1j) #Handles coth with x having very large or very small numbers
#        sinh_mp.append(float(((1-mp.exp(-2*x_mp[i]))/(2*mp.exp(-x_mp[i]))).real) + float(((1-mp.exp(-2*x_mp[i]))/(2*mp.exp(-x_mp[i]))).real)*1j)
#        sinh_mp.append(float(mp.sinh(x_mp[i]).real)+float((mp.sinh(x_mp[i]).imag))*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/np.array(sinh_mp))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*Lam)/sinh(x))) + Lam * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)

    return Z_Rs + Z_RQ1 + Z_TL

def cir_RsTL_1Dsolid_fit(params, w):
    '''
    Fit Function: -R-TL(Q(RW))-
    Transmission line w/ full complexity
    
    See cir_RsTL_1Dsolid() for details
    
    David Brown (demoryb@berkeley.edu)
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    radius = params['radius']
    D = params['D']
    R = params['R']
    Q = params['Q']
    n = params['n']
    R_w = params['R_w']
    n_w = params['n_w']
    Rel = params['Rel']
    Ri = params['Ri']

    #The impedance of the series resistance
    Z_Rs = Rs
    
    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D
    
    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x
    
    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R
    Z_Q = elem_Q(w=w, Q=Q, n=n)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)
    
    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    
    return Z_Rs + Z_TL

def cir_RsRQTL_1Dsolid_fit(params, w):
    '''
    Fit Function: -R-RQ-TL(Q(RW))-
    Transmission line w/ full complexity, which both includes Ri and Rel. The Warburg element is specific for 1D solid-state diffusion
    
    See cir_RsRQTL_1Dsolid() for details
    
    David Brown (demoryb@berkeley.edu)
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    L = params['L']
    Ri = params['Ri']
    radius = params['radius']
    D = params['D']
    R2 = params['R2']
    Q2 = params['Q2']
    n2 = params['n2']
    R_w = params['R_w']
    n_w = params['n_w']
    Rel = params['Rel']
    Ri = params['Ri']
    #The impedance of the series resistance
    Z_Rs = Rs
    
    # The Interfacial impedance is given by an -(RQ)- circuit
    if str(params.keys())[10:].find("R1") == -1: #if R == 'none':
        Q1 = params['Q1']
        n1 = params['n1']
        fs1 = params['fs1']
        R1 = (1/(Q1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("Q1") == -1: #elif Q == 'none':
        R1 = params['R1']
        n1 = params['n1']
        fs1 = params['fs1']
        Q1 = (1/(R1*(2*np.pi*fs1)**n1))
    elif str(params.keys())[10:].find("n1") == -1: #elif n == 'none':
        R1 = params['R1']
        Q1 = params['Q1']
        fs1 = params['fs1']
        n1 = np.log(Q1*R1)/np.log(1/(2*np.pi*fs1))
    elif str(params.keys())[10:].find("fs1") == -1: #elif fs == 'none':
        R1 = params['R1']
        n1 = params['n1']
        Q1 = params['Q1']
    Z_RQ1 = (R1/(1+R1*Q1*(w*1j)**n1))

    #The impedance of a 1D Warburg Element
    time_const = (radius**2)/D
    
    x = (time_const*w*1j)**n_w
    x_mp = mp.matrix(x)
    warburg_coth_mp = []
    for i in range(len(w)):
        warburg_coth_mp.append(float(mp.coth(x_mp[i]).real)+float(mp.coth(x_mp[i]).imag)*1j)

    Z_w = R_w * np.array(warburg_coth_mp)/x
    
    # The Interfacial impedance is given by a Randles Equivalent circuit with the finite space warburg element in series with R2
    Z_Rct = R2
    Z_Q = elem_Q(w,Q=Q2,n=n2)
    Z_Randles = 1/(1/Z_Q + 1/(Z_Rct+Z_w)) #Ohm

    # The Impedance of the Transmission Line
    lamb = (Z_Randles/(Rel+Ri))**(1/2)
    x = L/lamb
#    lamb_mp = mp.matrix(x)
#    sinh_mp = []
#    coth_mp = []
#    for j in range(len(lamb_mp)):
#        sinh_mp.append(float(mp.sinh(lamb_mp[j]).real)+float((mp.sinh(lamb_mp[j]).imag))*1j)
#        coth_mp.append(float(mp.coth(lamb_mp[j]).real)+float(mp.coth(lamb_mp[j]).imag)*1j)
#        
#    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/np.array(sinh_mp))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * np.array(coth_mp)

    Z_TL = ((Rel*Ri)/(Rel+Ri)) * (L+((2*lamb)/sinh(x))) + lamb * ((Rel**2 + Ri**2)/(Rel+Ri)) * coth(x)
    
    return Z_Rs + Z_RQ1 + Z_TL

### Least-Squares error function
def leastsq_errorfunc(params, w, re, im, circuit, weight_func):
    '''
    Sum of squares error function for the complex non-linear least-squares fitting procedure (CNLS). The fitting function (lmfit) will use this function to iterate over
    until the total sum of errors is minimized.
    
    During the minimization the fit is weighed, and currently three different weigh options are avaliable:
        - modulus
        - unity
        - proportional
    
    Modulus is generially recommended as random errors and a bias can exist in the experimental data.
        
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------
    - params: parameters needed for CNLS
    - re: real impedance
    - im: Imaginary impedance
    - circuit:
      The avaliable circuits are shown below, and this this parameter needs it as a string.
        - C
        - Q
        - R-C
        - R-Q
        - RC
        - RQ
        - R-RQ
        - R-RQ-RQ
        - R-RQ-Q
        - R-(Q(RW))
        - R-(Q(RM))
        - R-RC-C
        - R-RC-Q
        - R-RQ-Q
        - R-RQ-C
        - RC-RC-ZD
        - R-TLsQ
        - R-RQ-TLsQ
        - R-TLs
        - R-RQ-TLs
        - R-TLQ
        - R-RQ-TLQ
        - R-TL
        - R-RQ-TL
        - R-TL1Dsolid (reactive interface with 1D solid-state diffusion)
        - R-RQ-TL1Dsolid

    - weight_func
      Weight function
        - modulus
        - unity
        - proportional
    '''
    if circuit == 'C':
        re_fit = elem_C_fit(params, w).real
        im_fit = -elem_C_fit(params, w).imag
    elif circuit == 'Q':
        re_fit = elem_Q_fit(params, w).real
        im_fit = -elem_Q_fit(params, w).imag
    elif circuit == 'R-C':
        re_fit = cir_RsC_fit(params, w).real
        im_fit = -cir_RsC_fit(params, w).imag
    elif circuit == 'R-Q':
        re_fit = cir_RsQ_fit(params, w).real
        im_fit = -cir_RsQ_fit(params, w).imag
    elif circuit == 'RC':
        re_fit = cir_RC_fit(params, w).real
        im_fit = -cir_RC_fit(params, w).imag
    elif circuit == 'RQ':
        re_fit = cir_RQ_fit(params, w).real
        im_fit = -cir_RQ_fit(params, w).imag
    elif circuit == 'R-RQ':
        re_fit = cir_RsRQ_fit(params, w).real
        im_fit = -cir_RsRQ_fit(params, w).imag
    elif circuit == 'R-RQ-RQ':
        re_fit = cir_RsRQRQ_fit(params, w).real
        im_fit = -cir_RsRQRQ_fit(params, w).imag
    elif circuit == 'R-RC-C':
        re_fit = cir_RsRCC_fit(params, w).real
        im_fit = -cir_RsRCC_fit(params, w).imag
    elif circuit == 'R-RC-Q':
        re_fit = cir_RsRCQ_fit(params, w).real
        im_fit = -cir_RsRCQ_fit(params, w).imag
    elif circuit == 'R-RQ-Q':
        re_fit = cir_RsRQQ_fit(params, w).real
        im_fit = -cir_RsRQQ_fit(params, w).imag
    elif circuit == 'R-RQ-C':
        re_fit = cir_RsRQC_fit(params, w).real
        im_fit = -cir_RsRQC_fit(params, w).imag
    elif circuit == 'R-(Q(RW))':
        re_fit = cir_Randles_simplified_Fit(params, w).real
        im_fit = -cir_Randles_simplified_Fit(params, w).imag
    elif circuit == 'R-(Q(RM))':
        re_fit = cir_Randles_uelectrode_fit(params, w).real
        im_fit = -cir_Randles_uelectrode_fit(params, w).imag
    elif circuit == 'C-RC-C':
        re_fit = cir_C_RC_C_fit(params, w).real
        im_fit = -cir_C_RC_C_fit(params, w).imag
    elif circuit == 'Q-RQ-Q':
        re_fit = cir_Q_RQ_Q_Fit(params, w).real
        im_fit = -cir_Q_RQ_Q_Fit(params, w).imag
    elif circuit == 'RC-RC-ZD':
        re_fit = cir_RCRCZD_fit(params, w).real
        im_fit = -cir_RCRCZD_fit(params, w).imag
    elif circuit == 'R-TLsQ':
        re_fit = cir_RsTLsQ_fit(params, w).real
        im_fit = -cir_RsTLsQ_fit(params, w).imag
    elif circuit == 'R-RQ-TLsQ':
        re_fit = cir_RsRQTLsQ_Fit(params, w).real
        im_fit = -cir_RsRQTLsQ_Fit(params, w).imag
    elif circuit == 'R-TLs':
        re_fit = cir_RsTLs_Fit(params, w).real
        im_fit = -cir_RsTLs_Fit(params, w).imag
    elif circuit == 'R-RQ-TLs':
        re_fit = cir_RsRQTLs_Fit(params, w).real
        im_fit = -cir_RsRQTLs_Fit(params, w).imag
    elif circuit == 'R-TLQ':
        re_fit = cir_RsTLQ_fit(params, w).real
        im_fit = -cir_RsTLQ_fit(params, w).imag
    elif circuit == 'R-RQ-TLQ':
        re_fit = cir_RsRQTLQ_fit(params, w).real
        im_fit = -cir_RsRQTLQ_fit(params, w).imag
    elif circuit == 'R-TL':
        re_fit = cir_RsTL_Fit(params, w).real
        im_fit = -cir_RsTL_Fit(params, w).imag
    elif circuit == 'R-RQ-TL':
        re_fit = cir_RsRQTL_fit(params, w).real
        im_fit = -cir_RsRQTL_fit(params, w).imag
    elif circuit == 'R-TL1Dsolid':
        re_fit = cir_RsTL_1Dsolid_fit(params, w).real
        im_fit = -cir_RsTL_1Dsolid_fit(params, w).imag
    elif circuit == 'R-RQ-TL1Dsolid':
        re_fit = cir_RsRQTL_1Dsolid_fit(params, w).real
        im_fit = -cir_RsRQTL_1Dsolid_fit(params, w).imag
    else:
        print('Circuit is not defined in leastsq_errorfunc()')
        
    error = [(re-re_fit)**2, (im-im_fit)**2] #sum of squares
    
    #Different Weighing options, see Lasia
    if weight_func == 'modulus':
        weight = [1/((re_fit**2 + im_fit**2)**(1/2)), 1/((re_fit**2 + im_fit**2)**(1/2))]
    elif weight_func == 'proportional':
        weight = [1/(re_fit**2), 1/(im_fit**2)]
    elif weight_func == 'unity':
        unity_1s = []
        for k in range(len(re)):
            unity_1s.append(1) #makes an array of [1]'s, so that the weighing is == 1 * sum of squres.
        weight = [unity_1s, unity_1s]
    else:
        print('weight not defined in leastsq_errorfunc()')
        
    S = np.array(weight) * error #weighted sum of squares 
    return S

### Fitting Class
class EIS_exp:
    '''
    This class is used to plot and/or analyze experimental impedance data. The class has three major functions:
        - EIS_plot()
        - Lin_KK()
        - EIS_fit()

    - EIS_plot() is used to plot experimental data with or without fit
    - Lin_KK() performs a linear Kramers-Kronig analysis of the experimental data set.
    - EIS_fit() performs complex non-linear least-squares fitting of the experimental data to an equivalent circuit
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    -----------
        - path: path of datafile(s) as a string
        - data: datafile(s) including extension, e.g. ['EIS_data1', 'EIS_data2']
        - cycle: Specific cycle numbers can be extracted using the cycle function. Default is 'none', which includes all cycle numbers.
        Specific cycles can be extracted using this parameter, insert cycle numbers in brackets, e.g. cycle number 1,4, and 6 are wanted. cycle=[1,4,6]
        - mask: ['high frequency' , 'low frequency'], if only a high- or low-frequency is desired use 'none' for the other, e.g. maks=[10**4,'none']
    '''
    def __init__(self, path, data, cycle='off', mask=['none','none']):
        self.df_raw0 = []
        self.cycleno = []
        for j in range(len(data)):
            if data[j].find(".mpt") != -1: #file is a .mpt file
                self.df_raw0.append(extract_mpt(path=path, EIS_name=data[j])) #reads all datafiles
            elif data[j].find(".DTA") != -1: #file is a .dta file
                self.df_raw0.append(extract_dta(path=path, EIS_name=data[j])) #reads all datafiles
            elif data[j].find(".z") != -1: #file is a .z file
                self.df_raw0.append(extract_solar(path=path, EIS_name=data[j])) #reads all datafiles
            else:
                print('Data file(s) could not be identified')

            self.cycleno.append(self.df_raw0[j].cycle_number)
            if np.min(self.cycleno[j]) <= np.max(self.cycleno[j-1]):
                if j > 0: #corrects cycle_number except for the first data file
                    self.df_raw0[j].update({'cycle_number': self.cycleno[j]+np.max(self.cycleno[j-1])}) #corrects cycle number
#            else:
#                print('__init__ Error (#1)')

        #currently need to append a cycle_number coloumn to gamry files

        # adds individual dataframes into one
        if len(self.df_raw0) == 1:
            self.df_raw = self.df_raw0[0]
        elif len(self.df_raw0) == 2:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1]], axis=0)
        elif len(self.df_raw0) == 3:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2]], axis=0)
        elif len(self.df_raw0) == 4:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3]], axis=0)
        elif len(self.df_raw0) == 5:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4]], axis=0)
        elif len(self.df_raw0) == 6:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5]], axis=0)
        elif len(self.df_raw0) == 7:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6]], axis=0)
        elif len(self.df_raw0) == 8:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7]], axis=0)
        elif len(self.df_raw0) == 9:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7], self.df_raw0[8]], axis=0)
        elif len(self.df_raw0) == 10:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7], self.df_raw0[8], self.df_raw0[9]], axis=0)
        elif len(self.df_raw0) == 11:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7], self.df_raw0[8], self.df_raw0[9], self.df_raw0[10]], axis=0)
        elif len(self.df_raw0) == 12:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7], self.df_raw0[8], self.df_raw0[9], self.df_raw0[10], self.df_raw0[11]], axis=0)
        elif len(self.df_raw0) == 13:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7], self.df_raw0[8], self.df_raw0[9], self.df_raw0[10], self.df_raw0[11], self.df_raw0[12]], axis=0)
        elif len(self.df_raw0) == 14:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7], self.df_raw0[8], self.df_raw0[9], self.df_raw0[10], self.df_raw0[11]], self.df_raw0[12], self.df_raw0[13], axis=0)
        elif len(self.df_raw0) == 15:
            self.df_raw = pd.concat([self.df_raw0[0], self.df_raw0[1], self.df_raw0[2], self.df_raw0[3], self.df_raw0[4], self.df_raw0[5], self.df_raw0[6], self.df_raw0[7], self.df_raw0[8], self.df_raw0[9], self.df_raw0[10], self.df_raw0[11]], self.df_raw0[12], self.df_raw0[13], self.df_raw0[14], axis=0)
        else:
            print("Too many data files || 15 allowed")
        self.df_raw = self.df_raw.assign(w = 2*np.pi*self.df_raw.f) #creats a new coloumn with the angular frequency

        #Masking data to each cycle
        self.df_pre = []
        self.df_limited = []
        self.df_limited2 = []
        self.df = []
        if mask == ['none','none'] and cycle == 'off':
            for i in range(len(self.df_raw.cycle_number.unique())): #includes all data
                self.df.append(self.df_raw[self.df_raw.cycle_number == self.df_raw.cycle_number.unique()[i]])                
        elif mask == ['none','none'] and cycle != 'off':
            for i in range(len(cycle)):
                self.df.append(self.df_raw[self.df_raw.cycle_number == cycle[i]]) #extracting dataframe for each cycle                                
        elif mask[0] != 'none' and mask[1] == 'none' and cycle == 'off':
            self.df_pre = self.df_raw.mask(self.df_raw.f > mask[0])
            self.df_pre.dropna(how='all', inplace=True)
            for i in range(len(self.df_pre.cycle_number.unique())): #Appending data based on cycle number
                self.df.append(self.df_pre[self.df_pre.cycle_number == self.df_pre.cycle_number.unique()[i]])
        elif mask[0] != 'none' and mask[1] == 'none' and cycle != 'off': # or [i for i, e in enumerate(mask) if e == 'none'] == [0]
            self.df_limited = self.df_raw.mask(self.df_raw.f > mask[0])
            for i in range(len(cycle)):
                self.df.append(self.df_limited[self.df_limited.cycle_number == cycle[i]])
        elif mask[0] == 'none' and mask[1] != 'none' and cycle == 'off':
            self.df_pre = self.df_raw.mask(self.df_raw.f < mask[1])
            self.df_pre.dropna(how='all', inplace=True)
            for i in range(len(self.df_raw.cycle_number.unique())): #includes all data
                self.df.append(self.df_pre[self.df_pre.cycle_number == self.df_pre.cycle_number.unique()[i]])
        elif mask[0] == 'none' and mask[1] != 'none' and cycle != 'off': 
            self.df_limited = self.df_raw.mask(self.df_raw.f < mask[1])
            for i in range(len(cycle)):
                self.df.append(self.df_limited[self.df_limited.cycle_number == cycle[i]])
        elif mask[0] != 'none' and mask[1] != 'none' and cycle != 'off':
            self.df_limited = self.df_raw.mask(self.df_raw.f < mask[1])
            self.df_limited2 = self.df_limited.mask(self.df_raw.f > mask[0])
            for i in range(len(cycle)):
                self.df.append(self.df_limited[self.df_limited2.cycle_number == cycle[i]])
        elif mask[0] != 'none' and mask[1] != 'none' and cycle == 'off':
            self.df_limited = self.df_raw.mask(self.df_raw.f < mask[1])
            self.df_limited2 = self.df_limited.mask(self.df_raw.f > mask[0])
            for i in range(len(self.df_raw.cycle_number.unique())):
                self.df.append(self.df_limited[self.df_limited2.cycle_number == self.df_raw.cycle_number.unique()[i]])
        else:
            print('__init__ error (#2)')


    def Lin_KK(self, num_RC='auto', legend='on', plot='residuals', bode='off', nyq_xlim='none', nyq_ylim='none', weight_func='Boukamp', savefig='none'):
        '''
        Plots the Linear Kramers-Kronig (KK) Validity Test
        The script is based on Boukamp and Schōnleber et al.'s papers for fitting the resistances of multiple -(RC)- circuits
        to the data. A data quality analysis can hereby be made on the basis of the relative residuals

        Ref.:
            - Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
            - Boukamp, B.A. J. Electrochem. Soc., 142, 6, 1885-1894 
        
        The function performs the KK analysis and as default the relative residuals in each subplot        
    
        Note, that weigh_func should be equal to 'Boukamp'.
        
        Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
        
        Optional Inputs
        -----------------
        - num_RC:
            - 'auto' applies an automatic algorithm developed by Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
            that ensures no under- or over-fitting occurs
            - can be hardwired by inserting any number (RC-elements/decade)

        - plot: 
            - 'residuals' = plots the relative residuals in subplots correspoding to the cycle numbers picked
            - 'w_data' = plots the relative residuals with the experimental data, in Nyquist and bode plot if desired, see 'bode =' in description
        
        - nyq_xlim/nyq_xlim: Change the x/y-axis limits on nyquist plot, if not equal to 'none' state [min,max] value
        
        - legend:
            - 'on' = displays cycle number
            - 'potential' = displays average potential which the spectra was measured at
            - 'off' = off

        bode = Plots Bode Plot - options:
            'on' = re, im vs. log(freq)
            'log' = log(re, im) vs. log(freq)
            
            're' = re vs. log(freq)
            'log_re' = log(re) vs. log(freq)
            
            'im' = im vs. log(freq)
            'log_im' = log(im) vs. log(freq)
        '''
        if num_RC == 'auto':
            print('cycle || No. RC-elements ||   u')
            self.decade = []
            self.Rparam = []
            self.t_const = []
            self.Lin_KK_Fit = []
            self.R_names = []
            self.KK_R0 = []
            self.KK_R = []
            self.number_RC = []
            self.number_RC_sort = []
    
            self.KK_u = []
            self.KK_Rgreater = []
            self.KK_Rminor = []
            M = 2
            for i in range(len(self.df)):
                self.decade.append(np.log10(np.max(self.df[i].f))-np.log10(np.min(self.df[i].f))) #determine the number of RC circuits based on the number of decades measured and num_RC
                self.number_RC.append(M)
                self.number_RC_sort.append(M) #needed for self.KK_R
                self.Rparam.append(KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC[i]))[0]) #Creates intial guesses for R's
                self.t_const.append(KK_timeconst(w=self.df[i].w, num_RC=int(self.number_RC[i]))) #Creates time constants values for self.number_RC -(RC)- circuits
                
                self.Lin_KK_Fit.append(minimize(KK_errorfunc, self.Rparam[i], method='leastsq', args=(self.df[i].w.values, self.df[i].re.values, self.df[i].im.values, self.number_RC[i], weight_func, self.t_const[i]) )) #maxfev=99
                self.R_names.append(KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC[i]))[1]) #creates R names
                for j in range(len(self.R_names[i])):
                    self.KK_R0.append(self.Lin_KK_Fit[i].params.get(self.R_names[i][j]).value)
            self.number_RC_sort.insert(0,0) #needed for self.KK_R
            for i in range(len(self.df)):
                self.KK_R.append(self.KK_R0[int(np.cumsum(self.number_RC_sort)[i]):int(np.cumsum(self.number_RC_sort)[i+1])]) #assigns resistances from each spectra to their respective df
                self.KK_Rgreater.append(np.where(np.array(self.KK_R)[i] >= 0, np.array(self.KK_R)[i], 0) )
                self.KK_Rminor.append(np.where(np.array(self.KK_R)[i] < 0, np.array(self.KK_R)[i], 0) )
                self.KK_u.append(1-(np.abs(np.sum(self.KK_Rminor[i]))/np.abs(np.sum(self.KK_Rgreater[i]))))
            
            for i in range(len(self.df)):
                while self.KK_u[i] <= 0.75 or self.KK_u[i] >= 0.88:
                    self.number_RC_sort0 = []
                    self.KK_R_lim = []
                    self.number_RC[i] = self.number_RC[i] + 1
                    self.number_RC_sort0.append(self.number_RC)
                    self.number_RC_sort = np.insert(self.number_RC_sort0, 0,0)
                    self.Rparam[i] = KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC[i]))[0] #Creates intial guesses for R's
                    self.t_const[i] = KK_timeconst(w=self.df[i].w, num_RC=int(self.number_RC[i])) #Creates time constants values for self.number_RC -(RC)- circuits
                    self.Lin_KK_Fit[i] = minimize(KK_errorfunc, self.Rparam[i], method='leastsq', args=(self.df[i].w.values, self.df[i].re.values, self.df[i].im.values, self.number_RC[i], weight_func, self.t_const[i]) ) #maxfev=99
                    self.R_names[i] = KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC[i]))[1] #creates R names
                    self.KK_R0 = np.delete(np.array(self.KK_R0), np.s_[0:len(self.KK_R0)])
                    self.KK_R0 = []
                    for q in range(len(self.df)):
                        for j in range(len(self.R_names[q])):
                            self.KK_R0.append(self.Lin_KK_Fit[q].params.get(self.R_names[q][j]).value)
                    self.KK_R_lim = np.cumsum(self.number_RC_sort) #used for KK_R[i]
    
                    self.KK_R[i] = self.KK_R0[self.KK_R_lim[i]:self.KK_R_lim[i+1]] #assigns resistances from each spectra to their respective df
                    self.KK_Rgreater[i] = np.where(np.array(self.KK_R[i]) >= 0, np.array(self.KK_R[i]), 0)
                    self.KK_Rminor[i] = np.where(np.array(self.KK_R[i]) < 0, np.array(self.KK_R[i]), 0)
                    self.KK_u[i] = 1-(np.abs(np.sum(self.KK_Rminor[i]))/np.abs(np.sum(self.KK_Rgreater[i])))
                else:
                    print('['+str(i+1)+']'+'            '+str(self.number_RC[i]),'           '+str(np.round(self.KK_u[i],2)))

        elif num_RC != 'auto': #hardwired number of RC-elements/decade
            print('cycle ||   u')
            self.decade = []
            self.number_RC0 = []
            self.number_RC = []
            self.Rparam = []
            self.t_const = []
            self.Lin_KK_Fit = []
            self.R_names = []
            self.KK_R0 = []
            self.KK_R = []
            for i in range(len(self.df)):
                self.decade.append(np.log10(np.max(self.df[i].f))-np.log10(np.min(self.df[i].f))) #determine the number of RC circuits based on the number of decades measured and num_RC
                self.number_RC0.append(np.round(num_RC * self.decade[i]))
                self.number_RC.append(np.round(num_RC * self.decade[i])) #Creats the the number of -(RC)- circuits
                self.Rparam.append(KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC0[i]))[0]) #Creates intial guesses for R's
                self.t_const.append(KK_timeconst(w=self.df[i].w, num_RC=int(self.number_RC0[i]))) #Creates time constants values for self.number_RC -(RC)- circuits
                self.Lin_KK_Fit.append(minimize(KK_errorfunc, self.Rparam[i], method='leastsq', args=(self.df[i].w.values, self.df[i].re.values, self.df[i].im.values, self.number_RC0[i], weight_func, self.t_const[i]) )) #maxfev=99
                self.R_names.append(KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC0[i]))[1]) #creates R names            
                for j in range(len(self.R_names[i])):
                    self.KK_R0.append(self.Lin_KK_Fit[i].params.get(self.R_names[i][j]).value)
            self.number_RC0.insert(0,0)
    
    #        print(report_fit(self.Lin_KK_Fit[i])) # prints fitting report
    
            self.KK_circuit_fit = []
            self.KK_rr_re = []
            self.KK_rr_im = []
            self.KK_Rgreater = []
            self.KK_Rminor = []
            self.KK_u = []
            for i in range(len(self.df)):
                self.KK_R.append(self.KK_R0[int(np.cumsum(self.number_RC0)[i]):int(np.cumsum(self.number_RC0)[i+1])]) #assigns resistances from each spectra to their respective df
                self.KK_Rx = np.array(self.KK_R)
                self.KK_Rgreater.append(np.where(self.KK_Rx[i] >= 0, self.KK_Rx[i], 0) )
                self.KK_Rminor.append(np.where(self.KK_Rx[i] < 0, self.KK_Rx[i], 0) )
                self.KK_u.append(1-(np.abs(np.sum(self.KK_Rminor[i]))/np.abs(np.sum(self.KK_Rgreater[i])))) #currently gives incorrect values
                print('['+str(i+1)+']'+'       '+str(np.round(self.KK_u[i],2)))
        else:
            print('num_RC incorrectly defined')

        self.KK_circuit_fit = []
        self.KK_rr_re = []
        self.KK_rr_im = []
        for i in range(len(self.df)):
            if int(self.number_RC[i]) == 2:
                self.KK_circuit_fit.append(KK_RC2(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 3:
                self.KK_circuit_fit.append(KK_RC3(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 4:
                self.KK_circuit_fit.append(KK_RC4(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 5:
                self.KK_circuit_fit.append(KK_RC5(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 6:
                self.KK_circuit_fit.append(KK_RC6(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 7:
                self.KK_circuit_fit.append(KK_RC7(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 8:
                self.KK_circuit_fit.append(KK_RC8(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 9:
                self.KK_circuit_fit.append(KK_RC9(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 10:
                self.KK_circuit_fit.append(KK_RC10(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 11:
                self.KK_circuit_fit.append(KK_RC11(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 12:
                self.KK_circuit_fit.append(KK_RC12(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 13:
                self.KK_circuit_fit.append(KK_RC13(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 14:
                self.KK_circuit_fit.append(KK_RC14(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 15:
                self.KK_circuit_fit.append(KK_RC15(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 16:
                self.KK_circuit_fit.append(KK_RC16(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 17:
                self.KK_circuit_fit.append(KK_RC17(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 18:
                self.KK_circuit_fit.append(KK_RC18(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 19:
                self.KK_circuit_fit.append(KK_RC19(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 20:
                self.KK_circuit_fit.append(KK_RC20(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 21:
                self.KK_circuit_fit.append(KK_RC21(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 22:
                self.KK_circuit_fit.append(KK_RC22(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 23:
                self.KK_circuit_fit.append(KK_RC23(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 24:
                self.KK_circuit_fit.append(KK_RC24(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 25:
                self.KK_circuit_fit.append(KK_RC25(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 26:
                self.KK_circuit_fit.append(KK_RC26(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 27:
                self.KK_circuit_fit.append(KK_RC27(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 28:
                self.KK_circuit_fit.append(KK_RC28(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 29:
                self.KK_circuit_fit.append(KK_RC29(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 30:
                self.KK_circuit_fit.append(KK_RC30(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 31:
                self.KK_circuit_fit.append(KK_RC31(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 32:
                self.KK_circuit_fit.append(KK_RC32(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 33:
                self.KK_circuit_fit.append(KK_RC33(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 34:
                self.KK_circuit_fit.append(KK_RC34(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 35:
                self.KK_circuit_fit.append(KK_RC35(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 36:
                self.KK_circuit_fit.append(KK_RC36(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 37:
                self.KK_circuit_fit.append(KK_RC37(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 38:
                self.KK_circuit_fit.append(KK_RC38(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 39:
                self.KK_circuit_fit.append(KK_RC39(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 40:
                self.KK_circuit_fit.append(KK_RC40(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 41:
                self.KK_circuit_fit.append(KK_RC41(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 42:
                self.KK_circuit_fit.append(KK_RC42(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 43:
                self.KK_circuit_fit.append(KK_RC43(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 44:
                self.KK_circuit_fit.append(KK_RC44(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 45:
                self.KK_circuit_fit.append(KK_RC45(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 46:
                self.KK_circuit_fit.append(KK_RC46(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 47:
                self.KK_circuit_fit.append(KK_RC47(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 48:
                self.KK_circuit_fit.append(KK_RC48(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 49:
                self.KK_circuit_fit.append(KK_RC49(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 50:
                self.KK_circuit_fit.append(KK_RC50(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 51:
                self.KK_circuit_fit.append(KK_RC51(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 52:
                self.KK_circuit_fit.append(KK_RC52(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 53:
                self.KK_circuit_fit.append(KK_RC53(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 54:
                self.KK_circuit_fit.append(KK_RC54(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 55:
                self.KK_circuit_fit.append(KK_RC55(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 56:
                self.KK_circuit_fit.append(KK_RC56(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 57:
                self.KK_circuit_fit.append(KK_RC57(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 58:
                self.KK_circuit_fit.append(KK_RC58(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 59:
                self.KK_circuit_fit.append(KK_RC59(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 60:
                self.KK_circuit_fit.append(KK_RC60(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 61:
                self.KK_circuit_fit.append(KK_RC61(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 62:
                self.KK_circuit_fit.append(KK_RC62(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 63:
                self.KK_circuit_fit.append(KK_RC63(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 64:
                self.KK_circuit_fit.append(KK_RC64(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 65:
                self.KK_circuit_fit.append(KK_RC65(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 66:
                self.KK_circuit_fit.append(KK_RC66(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 67:
                self.KK_circuit_fit.append(KK_RC67(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 68:
                self.KK_circuit_fit.append(KK_RC68(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 69:
                self.KK_circuit_fit.append(KK_RC69(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 70:
                self.KK_circuit_fit.append(KK_RC70(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 71:
                self.KK_circuit_fit.append(KK_RC71(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 72:
                self.KK_circuit_fit.append(KK_RC72(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 73:
                self.KK_circuit_fit.append(KK_RC73(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 74:
                self.KK_circuit_fit.append(KK_RC74(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 75:
                self.KK_circuit_fit.append(KK_RC75(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 76:
                self.KK_circuit_fit.append(KK_RC76(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 77:
                self.KK_circuit_fit.append(KK_RC77(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 78:
                self.KK_circuit_fit.append(KK_RC78(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 79:
                self.KK_circuit_fit.append(KK_RC79(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            elif int(self.number_RC[i]) == 80:
                self.KK_circuit_fit.append(KK_RC80(w=self.df[i].w, Rs=self.Lin_KK_Fit[i].params.get('Rs').value, R_values=self.KK_R[i], t_values=self.t_const[i]))
            else:
                print('RC simulation circuit not defined')
                print('   Number of RC = ', self.number_RC)
            self.KK_rr_re.append(residual_real(re=self.df[i].re, fit_re=self.KK_circuit_fit[i].real, fit_im=-self.KK_circuit_fit[i].imag)) #relative residuals for the real part
            self.KK_rr_im.append(residual_imag(im=self.df[i].im, fit_re=self.KK_circuit_fit[i].real, fit_im=-self.KK_circuit_fit[i].imag)) #relative residuals for the imag part

        ### Plotting Linear_kk results
        ##
        #
        ### Label functions
        self.label_re_1 = []
        self.label_im_1 = []
        self.label_cycleno = []
        if legend == 'on':
            for i in range(len(self.df)):
                self.label_re_1.append("Z' (#"+str(i+1)+")")
                self.label_im_1.append("Z'' (#"+str(i+1)+")")
                self.label_cycleno.append('#'+str(i+1))
        elif legend == 'potential':
            for i in range(len(self.df)):
                self.label_re_1.append("Z' ("+str(np.round(np.average(self.df[i].E_avg), 2))+' V)')
                self.label_im_1.append("Z'' ("+str(np.round(np.average(self.df[i].E_avg), 2))+' V)')
                self.label_cycleno.append(str(np.round(np.average(self.df[i].E_avg), 2))+' V')


        if plot == 'w_data':
            fig = figure(figsize=(6, 8), dpi=120, facecolor='w', edgecolor='k')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(311, aspect='equal')
            ax1 = fig.add_subplot(312)
            ax2 = fig.add_subplot(313)
    
            colors = sns.color_palette("colorblind", n_colors=len(self.df))
            colors_real = sns.color_palette("Blues", n_colors=len(self.df)+2)
            colors_imag = sns.color_palette("Oranges", n_colors=len(self.df)+2)
    
            ### Nyquist Plot
            for i in range(len(self.df)):
                ax.plot(self.df[i].re, self.df[i].im, marker='o', ms=4, lw=2, color=colors[i], ls='-', alpha=.7, label=self.label_cycleno[i])
    
            ### Bode Plot
            if bode == 'on':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), self.df[i].re, color=colors_real[i+1], marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_re_1[i])
                    ax1.plot(np.log10(self.df[i].f), self.df[i].im, color=colors_imag[i+1], marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_im_1[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("Z', -Z'' [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best', fontsize=10, frameon=False)

            elif bode == 're':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), self.df[i].re, color=colors_real[i+1], marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("Z' [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best', fontsize=10, frameon=False)

            elif bode == 'log_re':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].re), color=colors_real[i+1], marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("log(Z') [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best', fontsize=10, frameon=False)

            elif bode == 'im':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), self.df[i].im, color=colors_imag[i+1], marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("-Z'' [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best', fontsize=10, frameon=False)

            elif bode == 'log_im':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].im), color=colors_imag[i+1], marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("log(-Z'') [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best', fontsize=10, frameon=False)      

            elif bode == 'log':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].re), color=colors_real[i+1], marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_re_1[i])
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].im), color=colors_imag[i+1], marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_im_1[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("log(Z', -Z'') [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best', fontsize=10, frameon=False)

            ### Kramers-Kronig Relative Residuals    
            for i in range(len(self.df)):
                ax2.plot(np.log10(self.df[i].f), self.KK_rr_re[i]*100, color=colors_real[i+1], marker='D', ls='--', ms=6, alpha=.7, label=self.label_re_1[i])
                ax2.plot(np.log10(self.df[i].f), self.KK_rr_im[i]*100, color=colors_imag[i+1], marker='s', ls='--', ms=6, alpha=.7, label=self.label_im_1[i])
                ax2.set_xlabel("log(f) [Hz]")
                ax2.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]")
                if legend == 'on' or legend == 'potential': 
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
            ax2.axhline(0, ls='--', c='k', alpha=.5)
            
            ### Setting ylims and write 'KK-Test' on RR subplot
            self.KK_rr_im_min = []
            self.KK_rr_im_max = []
            self.KK_rr_re_min = []
            self.KK_rr_re_max = []
            for i in range(len(self.df)):
                self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))    
            if np.min(self.KK_rr_im_min) > np.min(self.KK_rr_re_min):
                ax2.set_ylim(np.min(self.KK_rr_re_min)*100*1.5, np.max(np.abs(self.KK_rr_re_min))*100*1.5)
                ax2.annotate('Lin-KK', xy=[np.min(np.log10(self.df[0].f)), np.max(self.KK_rr_re_max)*100*.9], color='k', fontweight='bold')
            elif np.min(self.KK_rr_im_min) < np.min(self.KK_rr_re_min):
                ax2.set_ylim(np.min(self.KK_rr_im_min)*100*1.5, np.max(self.KK_rr_im_max)*100*1.5)
                ax2.annotate('Lin-KK', xy=[np.min(np.log10(self.df[0].f)), np.max(self.KK_rr_im_max)*100*.9], color='k', fontweight='bold')
                
            ### Figure specifics
            if legend == 'on' or legend == 'potential':
                ax.legend(loc='best', fontsize=10, frameon=False)
            ax.set_xlabel("Z' [$\Omega$]")
            ax.set_ylabel("-Z'' [$\Omega$]")
            if nyq_xlim != 'none':
                ax.set_xlim(nyq_xlim[0], nyq_xlim[1])
            if nyq_ylim != 'none':
                ax.set_ylim(nyq_ylim[0], nyq_ylim[1])
            #Save Figure
            if savefig != 'none':
                fig.savefig(savefig)

        ### Illustrating residuals only
    
        elif plot == 'residuals':
            colors = sns.color_palette("colorblind", n_colors=9)
            colors_real = sns.color_palette("Blues", n_colors=9)
            colors_imag = sns.color_palette("Oranges", n_colors=9)

            ### 1 Cycle
            if len(self.df) == 1:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax = fig.add_subplot(231)
                ax.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax.set_xlabel("log(f) [Hz]")
                ax.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]")
                if legend == 'on' or legend == 'potential':
                    ax.legend(loc='best', fontsize=10, frameon=False)        
                ax.axhline(0, ls='--', c='k', alpha=.5)
                
                ### Setting ylims and write 'KK-Test' on RR subplot
                self.KK_rr_im_min = np.min(self.KK_rr_im)
                self.KK_rr_im_max = np.max(self.KK_rr_im)
                self.KK_rr_re_min = np.min(self.KK_rr_re)
                self.KK_rr_re_max = np.max(self.KK_rr_re)
                if self.KK_rr_re_max > self.KK_rr_im_max:
                    self.KK_ymax = self.KK_rr_re_max
                else:
                    self.KK_ymax = self.KK_rr_im_max
                if self.KK_rr_re_min < self.KK_rr_im_min:
                    self.KK_ymin = self.KK_rr_re_min
                else:
                    self.KK_ymin = self.KK_rr_im_min
                if np.abs(self.KK_ymin) > self.KK_ymax:
                    ax.set_ylim(self.KK_ymin*100*1.5, np.abs(self.KK_ymin)*100*1.5)
                    if legend == 'on':
                        ax.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin)*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin)*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin) < self.KK_ymax:
                    ax.set_ylim(np.negative(self.KK_ymax)*100*1.5, np.abs(self.KK_ymax)*100*1.5)
                    if legend == 'on':
                        ax.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax*100*1.3], color='k', fontweight='bold')

                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 2 Cycles
            elif len(self.df) == 2:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=18)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                #cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax2.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])

                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.3], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 3 Cycles
            elif len(self.df) == 3:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=18)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax2.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best', fontsize=10, frameon=False)        
                ax3.axhline(0, ls='--', c='k', alpha=.5)
                
                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.3], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.3], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 4 Cycles
            elif len(self.df) == 4:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=18)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax2.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                ax3.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=18)
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best', fontsize=10, frameon=False)        
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best', fontsize=10, frameon=False)        
                ax4.axhline(0, ls='--', c='k', alpha=.5)
                
                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')

                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 5 Cycles
            elif len(self.df) == 5:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=18)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best', fontsize=10, frameon=False)        
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=18)
                ax4.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best', fontsize=10, frameon=False)        
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax5.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best', fontsize=10, frameon=False)        
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 6 Cycles
            elif len(self.df) == 6:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best', fontsize=10, frameon=False)        
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_xlabel("log(f) [Hz]")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best', fontsize=10, frameon=False)        
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax5.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best', fontsize=10, frameon=False)        
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax6.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax6.legend(loc='best', fontsize=10, frameon=False)        
                ax6.axhline(0, ls='--', c='k', alpha=.5)
                
                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)
                          
            ### 7 Cycles
            elif len(self.df) == 7:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(331)
                ax2 = fig.add_subplot(332)
                ax3 = fig.add_subplot(333)
                ax4 = fig.add_subplot(334)
                ax5 = fig.add_subplot(335)
                ax6 = fig.add_subplot(336)
                ax7 = fig.add_subplot(337)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best', fontsize=10, frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best', fontsize=10, frameon=False)
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax5.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best', fontsize=10, frameon=False)
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax6.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax6.legend(loc='best', fontsize=10, frameon=False)
                ax6.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 7
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_re[6]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_im[6]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax7.set_xlabel("log(f) [Hz]")
                ax7.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax7.legend(loc='best', fontsize=10, frameon=False)
                ax7.axhline(0, ls='--', c='k', alpha=.5)
                
                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[6]) > self.KK_ymax[6]:
                    ax7.set_ylim(self.KK_ymin[6]*100*1.5, np.abs(self.KK_ymin[6])*100*1.5)
                    if legend == 'on': 
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[6]) < self.KK_ymax[6]:
                    ax7.set_ylim(np.negative(self.KK_ymax[6])*100*1.5, np.abs(self.KK_ymax[6])*100*1.5)
                    if legend == 'on': 
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymax[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK, ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), self.KK_ymax[6]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)      
                           
            ### 8 Cycles
            elif len(self.df) == 8:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(331)
                ax2 = fig.add_subplot(332)
                ax3 = fig.add_subplot(333)
                ax4 = fig.add_subplot(334)
                ax5 = fig.add_subplot(335)
                ax6 = fig.add_subplot(336)
                ax7 = fig.add_subplot(337)
                ax8 = fig.add_subplot(338)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=14)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best', fontsize=10, frameon=False)        
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=14)
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best', fontsize=10, frameon=False)        
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best', fontsize=10, frameon=False)        
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax6.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax6.legend(loc='best', fontsize=10, frameon=False)        
                ax6.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 7
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_re[6]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_im[6]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax7.set_xlabel("log(f) [Hz]")
                ax7.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=14)                
                if legend == 'on' or legend == 'potential':
                    ax7.legend(loc='best', fontsize=10, frameon=False)        
                ax7.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 8
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_re[7]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_im[7]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax8.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax8.legend(loc='best', fontsize=10, frameon=False)        
                ax8.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[6]) > self.KK_ymax[6]:
                    ax7.set_ylim(self.KK_ymin[6]*100*1.5, np.abs(self.KK_ymin[6])*100*1.5)
                    if legend == 'on': 
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[6]) < self.KK_ymax[6]:
                    ax7.set_ylim(np.negative(self.KK_ymax[6])*100*1.5, np.abs(self.KK_ymax[6])*100*1.5)
                    if legend == 'on': 
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymax[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK, ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), self.KK_ymax[6]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[7]) > self.KK_ymax[7]:
                    ax8.set_ylim(self.KK_ymin[7]*100*1.5, np.abs(self.KK_ymin[7])*100*1.5)
                    if legend == 'on': 
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[7]) < self.KK_ymax[7]:
                    ax8.set_ylim(np.negative(self.KK_ymax[7])*100*1.5, np.abs(self.KK_ymax[7])*100*1.5)
                    if legend == 'on': 
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymax[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK, ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), self.KK_ymax[7]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 9 Cycles
            elif len(self.df) == 9:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(331)
                ax2 = fig.add_subplot(332)
                ax3 = fig.add_subplot(333)
                ax4 = fig.add_subplot(334)
                ax5 = fig.add_subplot(335)
                ax6 = fig.add_subplot(336)
                ax7 = fig.add_subplot(337)
                ax8 = fig.add_subplot(338)
                ax9 = fig.add_subplot(339)
                
                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on': 
                    ax1.legend(loc='best', fontsize=10, frameon=False)        
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on': 
                    ax2.legend(loc='best', fontsize=10, frameon=False)        
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on': 
                    ax3.legend(loc='best', fontsize=10, frameon=False)        
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on': 
                    ax4.legend(loc='best', fontsize=10, frameon=False)        
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on': 
                    ax5.legend(loc='best', fontsize=10, frameon=False)        
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on': 
                    ax6.legend(loc='best', fontsize=10, frameon=False)        
                ax6.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 7
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_re[6]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_im[6]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax7.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                ax7.set_xlabel("log(f) [Hz]")
                if legend == 'on':
                    ax7.legend(loc='best', fontsize=10, frameon=False)
                ax7.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 8
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_re[7]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_im[7]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax8.set_xlabel("log(f) [Hz]")
                if legend == 'on':
                    ax8.legend(loc='best', fontsize=10, frameon=False)
                ax8.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 9
                ax9.plot(np.log10(self.df[8].f), self.KK_rr_re[8]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax9.plot(np.log10(self.df[8].f), self.KK_rr_im[8]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax9.set_xlabel("log(f) [Hz]")
                if legend == 'on':
                    ax9.legend(loc='best', fontsize=10, frameon=False)
                ax9.axhline(0, ls='--', c='k', alpha=.5)
                
                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on': 
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on': 
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on': 
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on': 
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on': 
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on': 
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[6]) > self.KK_ymax[6]:
                    ax7.set_ylim(self.KK_ymin[6]*100*1.5, np.abs(self.KK_ymin[6])*100*1.5)
                    if legend == 'on': 
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[6]) < self.KK_ymax[6]:
                    ax7.set_ylim(np.negative(self.KK_ymax[6])*100*1.5, np.abs(self.KK_ymax[6])*100*1.5)
                    if legend == 'on': 
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymax[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK, ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), self.KK_ymax[6]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[7]) > self.KK_ymax[7]:
                    ax8.set_ylim(self.KK_ymin[7]*100*1.5, np.abs(self.KK_ymin[7])*100*1.5)
                    if legend == 'on': 
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[7]) < self.KK_ymax[7]:
                    ax8.set_ylim(np.negative(self.KK_ymax[7])*100*1.5, np.abs(self.KK_ymax[7])*100*1.5)
                    if legend == 'on': 
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymax[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK, ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), self.KK_ymax[7]*100*1.2], color='k', fontweight='bold')

                if np.abs(self.KK_ymin[8]) > self.KK_ymax[8]:
                    ax9.set_ylim(self.KK_ymin[8]*100*1.5, np.abs(self.KK_ymin[8])*100*1.5)
                    if legend == 'on': 
                        ax9.annotate('Lin-KK, #9', xy=[np.min(np.log10(self.df[8].f)), np.abs(self.KK_ymin[8])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax9.annotate('Lin-KK ('+str(np.round(np.average(self.df[8].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[8].f)), np.abs(self.KK_ymin[8])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[8]) < self.KK_ymax[8]:
                    ax9.set_ylim(np.negative(self.KK_ymax[8])*100*1.5, np.abs(self.KK_ymax[8])*100*1.5)
                    if legend == 'on': 
                        ax9.annotate('Lin-KK, #9', xy=[np.min(np.log10(self.df[8].f)), np.abs(self.KK_ymax[8])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax9.annotate('Lin-KK, ('+str(np.round(np.average(self.df[8].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[8].f)), self.KK_ymax[8]*100*1.2], color='k', fontweight='bold')  
                        
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)
            else:
                print('Too many spectras, cannot plot all. Maximum spectras allowed = 9')

    def EIS_fit(self, params, circuit, weight_func='modulus', nan_policy='raise'):
        '''
        EIS_fit() fits experimental data to an equivalent circuit model using complex non-linear least-squares (CNLS) fitting procedure and allows for batch fitting.
        
        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
        
        Inputs
        ------------
        - circuit:
          Choose an equivalent circuits and defined circuit as a string. The following circuits are avaliable.
            - RC
            - RQ
            - R-RQ
            - R-RQ-RQ
            - R-Q
            - R-RQ-Q
            - R-(Q(RW))
            - C-RC-C
            - Q-RQ-Q
            - RC-RC-ZD
            - R-TLsQ
            - R-RQ-TLsQ
            - R-TLs
            - R-RQ-TLs
            - R-TLQ
            - R-RQ-TLQ
            - R-TL
            - R-RQ-TL
            - R-TL1Dsolid (reactive interface with 1D solid-state diffusion)
            - R-RQ-TL1Dsolid

        - weight_func
          The weight function to which the CNLS fitting is performed
            - modulus (default)
            - unity
            - proportional
        
        - nan_policy
        How to handle Nan or missing values in dataset
            - ‘raise’ = raise a value error (default)
            - ‘propagate’ = do nothing
            - ‘omit’ = drops missing data
        
        Returns
        ------------
        Returns the fitted impedance spectra(s) but also the fitted parameters that were used in the initial guesses. To call these use e.g. self.fit_Rs
        '''
        self.Fit = []
        self.circuit_fit = []
        self.fit_E = []
        for i in range(len(self.df)):
            self.Fit.append(minimize(leastsq_errorfunc, params, method='leastsq', args=(self.df[i].w.values, self.df[i].re.values, self.df[i].im.values, circuit, weight_func), nan_policy=nan_policy, maxfev=9999990))
            print(report_fit(self.Fit[i]))
            
            self.fit_E.append(np.average(self.df[i].E_avg))
            
        if circuit == 'C':
            self.fit_C = []
            for i in range(len(self.df)):
                self.circuit_fit.append(elem_C(w=self.df[i].w, C=self.Fit[i].params.get('C').value))
                self.fit_C.append(self.Fit[i].params.get('C').value)
        elif circuit == 'Q':
            self.fit_Q = []
            self.fit_n = []
            for i in range(len(self.df)):
                self.circuit_fit.append(elem_Q(w=self.df[i].w, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value))
                self.fit_Q.append(self.Fit[i].params.get('Q').value)
                self.fit_n.append(self.Fit[i].params.get('n').value)
        elif circuit == 'R-C':
            self.fit_Rs = []
            self.fit_C = []
            for i in range(len(self.df)):
                self.circuit_fit.append(cir_RsC(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, C=self.Fit[i].params.get('C').value))
                self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                self.fit_C.append(self.Fit[i].params.get('C').value)
        elif circuit == 'R-Q':
            self.fit_Rs = []
            self.fit_Q = []
            self.fit_n = []
            for i in range(len(self.df)):
                self.circuit_fit.append(cir_RsQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value))
                self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                self.fit_Q.append(self.Fit[i].params.get('Q').value)
                self.fit_n.append(self.Fit[i].params.get('n').value)
        elif circuit == 'RC':
            self.fit_R = []
            self.fit_C = []
            self.fit_fs = []
            for i in range(len(self.df)):
                if "'C'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RC(w=self.df[i].w, C=self.Fit[i].params.get('C').value, R=self.Fit[i].params.get('R').value, fs='none'))
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_C.append(self.Fit[i].params.get('C').value)
                elif "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RC(w=self.df[i].w, C='none', R=self.Fit[i].params.get('R').value, fs=self.Fit[i].params.get('fs').value))
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_fs.append(self.Fit[i].params.get('R').value)
        elif circuit == 'RQ':
            self.fit_R = []
            self.fit_n = []
            self.fit_fs = []
            self.fit_Q = []
            for i in range(len(self.df)):
                if "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RQ(w=self.df[i].w, R=self.Fit[i].params.get('R').value, Q='none', n=self.Fit[i].params.get('n').value, fs=self.Fit[i].params.get('fs').value))
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)
                elif "'Q'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RQ(w=self.df[i].w, R=self.Fit[i].params.get('R').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, fs='none'))
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
        elif circuit == 'R-RQ':
            self.fit_Rs = []
            self.fit_R = []
            self.fit_n = []
            self.fit_fs = []
            self.fit_Q = []
            for i in range(len(self.df)):
                if "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q='none', n=self.Fit[i].params.get('n').value, fs=self.Fit[i].params.get('fs').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)
                elif "'Q'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, fs='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
        elif circuit == 'R-RQ-RQ':
            self.fit_Rs = []
            self.fit_R = []
            self.fit_n = []
            self.fit_R2 = []
            self.fit_n2 = []
            self.fit_fs = []
            self.fit_fs2 = []
            self.fit_Q = []
            self.fit_Q2 = []
            for i in range(len(self.df)):
                if "'fs'" in str(self.Fit[i].params.keys()) and "'fs2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQRQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q='none', n=self.Fit[i].params.get('n').value, fs=self.Fit[i].params.get('fs').value, R2=self.Fit[i].params.get('R2').value, Q2='none', n2=self.Fit[i].params.get('n2').value, fs2=self.Fit[i].params.get('fs2').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_fs2.append(self.Fit[i].params.get('fs2').value)
                elif "'Q'" in str(self.Fit[i].params.keys()) and "'fs2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQRQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, fs='none', R2=self.Fit[i].params.get('R2').value, Q2='none', n2=self.Fit[i].params.get('n2').value, fs2=self.Fit[i].params.get('fs2').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_fs2.append(self.Fit[i].params.get('fs2').value)
                elif "'fs'" in str(self.Fit[i].params.keys()) and "'Q2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQRQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q='none', n=self.Fit[i].params.get('n').value, fs=self.Fit[i].params.get('fs').value, R2=self.Fit[i].params.get('R2').value, Q2=self.Fit[i].params.get('Q2').value, n2=self.Fit[i].params.get('n2').value, fs2='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)
                elif "'Q'" in str(self.Fit[i].params.keys()) and "'Q2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQRQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, fs='none', R2=self.Fit[i].params.get('R2').value, Q2=self.Fit[i].params.get('Q2').value, n2=self.Fit[i].params.get('n2').value, fs2='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)
        elif circuit == 'R-RC-C':
            self.fit_Rs = []
            self.fit_R1 = []
            self.fit_C1 = []
            self.fit_C = []
            for i in range(len(self.df)):
                self.circuit_fit.append(cir_RsRCC(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, C1=self.Fit[i].params.get('C1').value, C=self.Fit[i].params.get('C').value))
                self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                self.fit_R1.append(self.Fit[i].params.get('R1').value)
                self.fit_C1.append(self.Fit[i].params.get('C1').value)
                self.fit_C.append(self.Fit[i].params.get('C').value)
        elif circuit == 'R-RC-Q':
            self.fit_Rs = []
            self.fit_R1 = []
            self.fit_C1 = []
            self.fit_Q = []
            self.fit_n = []
            for i in range(len(self.df)):
                self.circuit_fit.append(cir_RsRCQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, C1=self.Fit[i].params.get('C1').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value))
                self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                self.fit_R1.append(self.Fit[i].params.get('R1').value)
                self.fit_C1.append(self.Fit[i].params.get('C1').value)
                self.fit_Q.append(self.Fit[i].params.get('Q').value)
                self.fit_n.append(self.Fit[i].params.get('n').value)
        elif circuit == 'R-RQ-Q':
            self.fit_Rs = []
            self.fit_n = []
            self.fit_R1 = []
            self.fit_n1 = []
            self.fit_Q = []
            self.fit_fs1 = []
            self.fit_Q1 = []
            for i in range(len(self.df)):
                if "'fs1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, R1=self.Fit[i].params.get('R1').value, Q1='none', n1=self.Fit[i].params.get('n1').value, fs1=self.Fit[i].params.get('fs1').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, R1=self.Fit[i].params.get('R1').value, Q1=self.Fit[i].params.get('Q1').value, n1=self.Fit[i].params.get('n1').value, fs1='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
        elif circuit == 'R-RQ-C':
            self.fit_Rs = []
            self.fit_C = []
            self.fit_R1 = []
            self.fit_n1 = []
            self.fit_Q1 = []
            self.fit_fs1 = []
            for i in range(len(self.df)):
                if "'fs1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQC(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, C=self.Fit[i].params.get('C').value, R1=self.Fit[i].params.get('R1').value, Q1='none', n1=self.Fit[i].params.get('n1').value, fs1=self.Fit[i].params.get('fs1').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_C.append(self.Fit[i].params.get('C').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQC(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, C=self.Fit[i].params.get('C').value, R1=self.Fit[i].params.get('R1').value, Q1=self.Fit[i].params.get('Q1').value, n1=self.Fit[i].params.get('n1').value, fs1='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_C.append(self.Fit[i].params.get('C').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
        elif circuit == 'R-(Q(RW))':
            self.fit_Rs = []
            self.fit_R = []
            self.fit_n = []
            self.fit_sigma = []
            self.fit_fs = []
            self.fit_Q = []
            for i in range(len(self.df)):
                if "'Q'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Randles_simplified(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q=self.Fit[i].params.get('Q').value, fs='none', n=self.Fit[i].params.get('n').value, sigma=self.Fit[i].params.get('sigma').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_sigma.append(self.Fit[i].params.get('sigma').value)
                elif "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Randles_simplified(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q='none', fs=self.Fit[i].params.get('fs').value, n=self.Fit[i].params.get('n').value, sigma=self.Fit[i].params.get('sigma').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_sigma.append(self.Fit[i].params.get('sigma').value)
        elif circuit == 'R-TLsQ':
            self.fit_Rs = []
            self.fit_Q = []
            self.fit_n = []
            self.fit_Ri = []
            self.fit_L = []
            for i in range(len(self.df)):
                self.circuit_fit.append(cir_RsTLsQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value))
                self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                self.fit_Q.append(self.Fit[i].params.get('Q').value)
                self.fit_n.append(self.Fit[i].params.get('n').value)
                self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                self.fit_L.append(self.Fit[i].params.get('L').value)
        elif circuit == 'R-RQ-TLsQ':
            self.fit_Rs = []
            self.fit_R1 = []
            self.fit_n1 = []
            self.fit_Q = []
            self.fit_n = []
            self.fit_Ri = []
            self.fit_L = []
            self.fit_fs1 = []
            self.fit_Q1 = []
            for i in range(len(self.df)):
                if "'fs1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLsQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, fs1=self.Fit[i].params.get('fs1').value, n1=self.Fit[i].params.get('n1').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, Q1='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLsQ(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, fs1='none', n1=self.Fit[i].params.get('n1').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, Q1=self.Fit[i].params.get('Q1').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
        elif circuit == 'R-TLs':
            self.fit_Rs = []
            self.fit_R = []
            self.fit_n = []
            self.fit_Ri = []
            self.fit_L = []
            self.fit_fs = []
            self.fit_Q = []
            for i in range(len(self.df)):
                if "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsTLs(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, R=self.Fit[i].params.get('R').value, Q='none', n=self.Fit[i].params.get('n').value, fs=self.Fit[i].params.get('fs').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                elif "'Q'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsTLs(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, R=self.Fit[i].params.get('R').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, fs='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R.append(self.Fit[i].params.get('R').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
        elif circuit == 'R-RQ-TLs':
            self.fit_Rs = []
            self.fit_R1 = []
            self.fit_n1 = []
            self.fit_R2 = []
            self.fit_n2 = []
            self.fit_Ri = []
            self.fit_L = []
            self.fit_fs1 = []
            self.fit_fs2 = []
            self.fit_Q1 = []
            self.fit_Q2 = []
            for i in range(len(self.df)):
                if "'fs1'" in str(self.Fit[i].params.keys()) and "'fs2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLs(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, R1=self.Fit[i].params.get('R1').value, n1=self.Fit[i].params.get('n1').value, fs1=self.Fit[i].params.get('fs1').value, R2=self.Fit[i].params.get('R2').value, n2=self.Fit[i].params.get('n2').value, fs2=self.Fit[i].params.get('fs2').value, Q1='none', Q2='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_fs2.append(self.Fit[i].params.get('fs2').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()) and "'fs2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLs(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, R1=self.Fit[i].params.get('R1').value, n1=self.Fit[i].params.get('n1').value, fs1='none', R2=self.Fit[i].params.get('R2').value, n2=self.Fit[i].params.get('n2').value, fs2=self.Fit[i].params.get('fs2').value, Q1=self.Fit[i].params.get('Q1').value, Q2='none'))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_fs2.append(self.Fit[i].params.get('fs2').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                elif "'fs1'" in str(self.Fit[i].params.keys()) and "'Q2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLs(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, R1=self.Fit[i].params.get('R1').value, n1=self.Fit[i].params.get('n1').value, fs1=self.Fit[i].params.get('fs1').value, R2=self.Fit[i].params.get('R2').value, n2=self.Fit[i].params.get('n2').value, fs2='none', Q1='none', Q2=self.Fit[i].params.get('Q2').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()) and "'Q2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLs(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, L=self.Fit[i].params.get('L').value, Ri=self.Fit[i].params.get('Ri').value, R1=self.Fit[i].params.get('R1').value, n1=self.Fit[i].params.get('n1').value, fs1='none', R2=self.Fit[i].params.get('R2').value, n2=self.Fit[i].params.get('n2').value, fs2='none', Q1=self.Fit[i].params.get('Q1').value, Q2=self.Fit[i].params.get('Q2').value))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_L.append(self.Fit[i].params.get('L').value)
        elif circuit == 'R-TLQ':
            self.fit_L = []
            self.fit_Rs = []
            self.fit_Q = []
            self.fit_n = []
            self.fit_Rel = []
            self.fit_Ri = []
            for i in range(len(self.df)):
                self.circuit_fit.append(cir_RsTLQ(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                self.fit_L.append(self.Fit[i].params.get('L').value)            
                self.fit_Rs.append(self.Fit[i].params.get('Rs').value)            
                self.fit_Q.append(self.Fit[i].params.get('Q').value)            
                self.fit_n.append(self.Fit[i].params.get('n').value)            
                self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
        elif circuit == 'R-RQ-TLQ':
            self.fit_Rs = []
            self.fit_L = []
            self.fit_Q = []
            self.fit_n = []
            self.fit_Rel = []
            self.fit_Ri = []
            self.fit_R1 = []
            self.fit_n1 = []
            self.fit_fs1 = []
            self.fit_Q1 = []
            for i in range(len(self.df)):
                if "'fs1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLQ(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value, R1=self.Fit[i].params.get('R1').value, n1=self.Fit[i].params.get('n1').value, fs1=self.Fit[i].params.get('fs1').value, Q1='none'))
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)                    
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTLQ(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value, R1=self.Fit[i].params.get('R1').value, n1=self.Fit[i].params.get('n1').value, fs1='none', Q1=self.Fit[i].params.get('Q1').value))
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n.append(self.Fit[i].params.get('n').value)
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)                    
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
        elif circuit == 'R-TL':
            self.fit_L = []
            self.fit_Rs = []
            self.fit_R = []
            self.fit_fs = []
            self.fit_n = []
            self.fit_Rel = []
            self.fit_Ri = []
            for i in range(len(self.df)):
                if "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsTL(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, fs=self.Fit[i].params.get('fs').value, n=self.Fit[i].params.get('n').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value, Q='none'))                
                    self.fit_L.append(self.Fit[i].params.get('L').value)                
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)                
                    self.fit_R.append(self.Fit[i].params.get('R').value)                
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)                
                    self.fit_n.append(self.Fit[i].params.get('n').value)                
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
        elif circuit == 'R-RQ-TL':
            self.fit_L = []
            self.fit_Rs = []
            self.fit_R1 = []
            self.fit_n1 = []
            self.fit_R2 = []
            self.fit_n2 = []
            self.fit_Rel = []
            self.fit_Ri = []
            self.fit_Q1 = []
            self.fit_Q2 = []
            self.fit_fs1 = []
            self.fit_fs2 = []
            for i in range(len(self.df)):
                if "'Q1'" in str(self.Fit[i].params.keys()) and "'Q2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTL(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, fs1='none', Q1=self.Fit[i].params.get('Q1').value, n1=self.Fit[i].params.get('n1').value, R2=self.Fit[i].params.get('R2').value, fs2='none', Q2=self.Fit[i].params.get('Q2').value, n2=self.Fit[i].params.get('n2').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                    self.fit_L.append(self.Fit[i].params.get('L').value)                    
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)                    
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)                    
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)                    
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)                    
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)                    
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)                    
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)                    
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                elif "'fs1'" in str(self.Fit[i].params.keys()) and "'fs2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTL(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, fs1=self.Fit[i].params.get('fs1').value, Q1='none', n1=self.Fit[i].params.get('n1').value, R2=self.Fit[i].params.get('R2').value, fs2=self.Fit[i].params.get('fs2').value, Q2='none', n2=self.Fit[i].params.get('n2').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                    self.fit_L.append(self.Fit[i].params.get('L').value)                
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)                    
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)                    
                    self.fit_fs2.append(self.Fit[i].params.get('fs2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()) and "'fs2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTL(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, fs1='none', Q1=self.Fit[i].params.get('Q1').value, n1=self.Fit[i].params.get('n1').value, R2=self.Fit[i].params.get('R2').value, fs2=self.Fit[i].params.get('fs2').value, Q2='none', n2=self.Fit[i].params.get('n2').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                    self.fit_L.append(self.Fit[i].params.get('L').value)                
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)                    
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)                    
                    self.fit_fs2.append(self.Fit[i].params.get('fs2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                elif "'fs1'" in str(self.Fit[i].params.keys()) and "'Q2'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTL(w=self.df[i].w, L=self.Fit[i].params.get('L').value, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, fs1=self.Fit[i].params.get('fs1').value, Q1='none', n1=self.Fit[i].params.get('n1').value, R2=self.Fit[i].params.get('R2').value, fs2='none', Q2=self.Fit[i].params.get('Q2').value, n2=self.Fit[i].params.get('n2').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                    self.fit_L.append(self.Fit[i].params.get('L').value)                
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)                    
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)                    
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
        elif circuit == 'R-TL1Dsolid':
            self.fit_L = []
            self.fit_radius = []
            self.fit_D = []
            self.fit_Rs = []
            self.fit_R = []
            self.fit_Q = []
            self.fit_n = []
            self.fit_R_w = []
            self.fit_n_w = []
            self.fit_Rel = []
            self.fit_Ri = []
            for i in range(len(self.df)):
                self.circuit_fit.append(cir_RsTL_1Dsolid(w=self.df[i].w, L=self.Fit[i].params.get('L').value, D=self.Fit[i].params.get('D').value, radius=self.Fit[i].params.get('radius').value, Rs=self.Fit[i].params.get('Rs').value, R=self.Fit[i].params.get('R').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, R_w=self.Fit[i].params.get('R_w').value, n_w=self.Fit[i].params.get('n_w').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                self.fit_L.append(self.Fit[i].params.get('L').value)
                self.fit_radius.append(self.Fit[i].params.get('radius').value)
                self.fit_D.append(self.Fit[i].params.get('D').value)            
                self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                self.fit_R.append(self.Fit[i].params.get('R').value)
                self.fit_Q.append(self.Fit[i].params.get('Q').value)
                self.fit_n.append(self.Fit[i].params.get('n').value)
                self.fit_R_w.append(self.Fit[i].params.get('R_w').value)
                self.fit_n_w.append(self.Fit[i].params.get('n_w').value)
                self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
        elif circuit == 'R-RQ-TL1Dsolid':
            self.fit_L = []
            self.fit_radius = []
            self.fit_D = []
            self.fit_Rs = []
            self.fit_R1 = []
            self.fit_n1 = []
            self.fit_R2 = []
            self.fit_Q2 = []
            self.fit_n2 = []
            self.fit_R_w = []
            self.fit_n_w = []
            self.fit_Rel = []
            self.fit_Ri = []
            self.fit_fs1 = []
            self.fit_Q1 = []
            for i in range(len(self.df)):
                if "'fs1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTL_1Dsolid(w=self.df[i].w, L=self.Fit[i].params.get('L').value, D=self.Fit[i].params.get('D').value, radius=self.Fit[i].params.get('radius').value, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, Q1='none', fs1=self.Fit[i].params.get('fs1').value, n1=self.Fit[i].params.get('n1').value, R2=self.Fit[i].params.get('R2').value, Q2=self.Fit[i].params.get('Q2').value, n2=self.Fit[i].params.get('n2').value, R_w=self.Fit[i].params.get('R_w').value, n_w=self.Fit[i].params.get('n_w').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                    self.fit_L.append(self.Fit[i].params.get('L').value)                    
                    self.fit_radius.append(self.Fit[i].params.get('radius').value)                    
                    self.fit_D.append(self.Fit[i].params.get('D').value)                                
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)                    
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)                    
                    self.fit_fs1.append(self.Fit[i].params.get('fs1').value)                    
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)                    
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)                    
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)                    
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)                    
                    self.fit_R_w.append(self.Fit[i].params.get('R_w').value)                    
                    self.fit_n_w.append(self.Fit[i].params.get('n_w').value)                    
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
                elif "'Q1'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_RsRQTL_1Dsolid(w=self.df[i].w, L=self.Fit[i].params.get('L').value, D=self.Fit[i].params.get('D').value, radius=self.Fit[i].params.get('radius').value, Rs=self.Fit[i].params.get('Rs').value, R1=self.Fit[i].params.get('R1').value, Q1=self.Fit[i].params.get('Q1').value, fs1='none', n1=self.Fit[i].params.get('n1').value, R2=self.Fit[i].params.get('R2').value, Q2=self.Fit[i].params.get('Q2').value, n2=self.Fit[i].params.get('n2').value, R_w=self.Fit[i].params.get('R_w').value, n_w=self.Fit[i].params.get('n_w').value, Rel=self.Fit[i].params.get('Rel').value, Ri=self.Fit[i].params.get('Ri').value))
                    self.fit_L.append(self.Fit[i].params.get('L').value)
                    self.fit_radius.append(self.Fit[i].params.get('radius').value)
                    self.fit_D.append(self.Fit[i].params.get('D').value)            
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_R1.append(self.Fit[i].params.get('R1').value)
                    self.fit_Q1.append(self.Fit[i].params.get('Q1').value)
                    self.fit_n1.append(self.Fit[i].params.get('n1').value)
                    self.fit_R2.append(self.Fit[i].params.get('R2').value)
                    self.fit_Q2.append(self.Fit[i].params.get('Q2').value)
                    self.fit_n2.append(self.Fit[i].params.get('n2').value)
                    self.fit_R_w.append(self.Fit[i].params.get('R_w').value)
                    self.fit_n_w.append(self.Fit[i].params.get('n_w').value)
                    self.fit_Rel.append(self.Fit[i].params.get('Rel').value)
                    self.fit_Ri.append(self.Fit[i].params.get('Ri').value)
        elif circuit == 'C-RC-C':
            self.fit_Ce = []
            self.fit_Rb = []
            self.fit_fsb = []
            self.fit_Cb = []
            for i in range(len(self.df)):
                if "'fsb'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_C_RC_C(w=self.df[i].w, Ce=self.Fit[i].params.get('Ce').value, Cb='none', Rb=self.Fit[i].params.get('Rb').value, fsb=self.Fit[i].params.get('fsb').value))                    
                    self.fit_Ce.append(self.Fit[i].params.get('Ce').value)                    
                    self.fit_Rb.append(self.Fit[i].params.get('Rb').value)
                    self.fit_fsb.append(self.Fit[i].params.get('fsb').value)
                elif "'Cb'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_C_RC_C(w=self.df[i].w, Ce=self.Fit[i].params.get('Ce').value, Cb=self.Fit[i].params.get('Cb').value, Rb=self.Fit[i].params.get('Rb').value, fsb='none'))
                    self.fit_Ce.append(self.Fit[i].params.get('Ce').value)
                    self.fit_Rb.append(self.Fit[i].params.get('Rb').value)
                    self.fit_Cb.append(self.Fit[i].params.get('Cb').value)
        elif circuit == 'Q-RQ-Q':
            self.fit_Qe = []
            self.fit_ne = []
            self.fit_Rb = []
            self.fit_nb = []
            self.fit_fsb = []
            self.fit_Qb = []
            for i in range(len(self.df)):
                if "'fsb'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Q_RQ_Q(w=self.df[i].w, Qe=self.Fit[i].params.get('Qe').value, ne=self.Fit[i].params.get('ne').value, Qb='none', Rb=self.Fit[i].params.get('Rb').value, fsb=self.Fit[i].params.get('fsb').value, nb=self.Fit[i].params.get('nb').value))
                    self.fit_Qe.append(self.Fit[i].params.get('Qe').value)                    
                    self.fit_ne.append(self.Fit[i].params.get('ne').value)                    
                    self.fit_Rb.append(self.Fit[i].params.get('Rb').value)                    
                    self.fit_fsb.append(self.Fit[i].params.get('fsb').value)
                    self.fit_nb.append(self.Fit[i].params.get('nb').value)
                elif "'Qb'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Q_RQ_Q(w=self.df[i].w, Qe=self.Fit[i].params.get('Qe').value, ne=self.Fit[i].params.get('ne').value, Qb=self.Fit[i].params.get('Qb').value, Rb=self.Fit[i].params.get('Rb').value, fsb='none', nb=self.Fit[i].params.get('nb').value))
                    self.fit_Qe.append(self.Fit[i].params.get('Qe').value)
                    self.fit_ne.append(self.Fit[i].params.get('ne').value)
                    self.fit_Rb.append(self.Fit[i].params.get('Rb').value)                    
                    self.fit_Qb.append(self.Fit[i].params.get('Qb').value)
                    self.fit_nb.append(self.Fit[i].params.get('nb').value)
        else:
            print('Circuit was not properly defined, see details described in definition')

    def EIS_plot(self, bode='off', fitting='off', rr='off', nyq_xlim='none', nyq_ylim='none', legend='on', savefig='none'):
        '''
        Plots Experimental and fitted impedance data in three subplots:
            a) Nyquist, b) Bode, c) relative residuals between experimental and fit
        
        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
        
        Optional Inputs
        -----------------
        - bode
          Plots the Bode Plot with the following possibilities
            - 'on' = re, im vs. log(freq)
            - 'log' = log(re, im) vs. log(freq)
            
            - 're' = re vs. log(freq)
            - 'log_re' = log(re) vs. log(freq)
            
            - 'im' = im vs. log(freq)
            - 'log_im' = log(im) vs. log(freq)

        - legend:
          Legend options
            - 'on' = illustrates the cycle number
            - 'off' = off
            - 'potential' = illustrates the potential
        
        - fitting: 
          If EIS_fit() has been called. To plot experimental- and fitted data turn fitting on
            - 'on'
            - 'off' (default)

        - rr: 
         The relative residuals between fit and experimental data
         - 'on' = opens a new subplot
         - 'off' (default)
        
        - nyq_xlim/nyq_xlim: 
          x/y-axis on nyquist plot, if not equal to 'none' state [min,max] value
        '''
        if bode=='off':
            fig = figure(dpi=120, facecolor='w', edgecolor='w')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(111, aspect='equal')

        elif bode=='on' and rr=='off' or bode=='log' and rr=='off' or bode=='re' and rr=='off' or bode=='log_re' and rr=='off' or bode=='im' and rr=='off' or bode=='log_im' and rr=='off' or bode=='log' and rr=='off':
            fig = figure(figsize=(6, 5), dpi=120, facecolor='w', edgecolor='w')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(211, aspect='equal')
            ax1 = fig.add_subplot(212)

        elif bode=='on' and rr=='on' or bode=='log' and rr=='on' or bode=='re' and rr=='on' or bode=='log_re' and rr=='on' or bode=='im' and rr=='on' or bode=='log_im' and rr=='on' or bode=='log' and rr=='on':
            fig = figure(figsize=(6, 8), dpi=120, facecolor='w', edgecolor='k')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(311, aspect='equal')
            ax1 = fig.add_subplot(312)
            ax2 = fig.add_subplot(313)

        ### Colors
        colors = sns.color_palette("colorblind", n_colors=len(self.df))
        colors_real = sns.color_palette("Blues", n_colors=len(self.df)+2)
        colors_imag = sns.color_palette("Oranges", n_colors=len(self.df)+2)

        ### Label functions
        self.label_re_1 = []
        self.label_im_1 = []
        self.label_cycleno = []
        if legend == 'on':
            for i in range(len(self.df)):
                self.label_re_1.append("Z' (#"+str(i+1)+")")
                self.label_im_1.append("Z'' (#"+str(i+1)+")")
                self.label_cycleno.append('#'+str(i+1))
        elif legend == 'potential':
            for i in range(len(self.df)):
                self.label_re_1.append("Z' ("+str(np.round(np.average(self.df[i].E_avg), 2))+' V)')
                self.label_im_1.append("Z'' ("+str(np.round(np.average(self.df[i].E_avg), 2))+' V)')
                self.label_cycleno.append(str(np.round(np.average(self.df[i].E_avg), 2))+' V')



        ### Nyquist Plot
        for i in range(len(self.df)):
            ax.plot(self.df[i].re, self.df[i].im, marker='o', ms=4, lw=2, color=colors[i], ls='-', label=self.label_cycleno[i])
            if fitting == 'on':
                ax.plot(self.circuit_fit[i].real, -self.circuit_fit[i].imag, lw=0, marker='o', ms=8, mec='r', mew=1, mfc='none', label='')

        ### Bode Plot
        if bode=='on':
            for i in range(len(self.df)):
                ax1.plot(np.log10(self.df[i].f), self.df[i].re, color=colors_real[i], marker='D', ms=3, lw=2.25, ls='-', label=self.label_re_1[i])
                ax1.plot(np.log10(self.df[i].f), self.df[i].im, color=colors_imag[i], marker='s', ms=3, lw=2.25, ls='-', label=self.label_im_1[i])
                if fitting == 'on':
                    ax1.plot(np.log10(self.df[i].f), self.circuit_fit[i].real, lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='')
                    ax1.plot(np.log10(self.df[i].f), -self.circuit_fit[i].imag, lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none')
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("Z', -Z'' [$\Omega$]")
                if legend == 'on' or legend == 'potential': 
                    ax1.legend(loc='best', fontsize=10, frameon=False)
            
        elif bode == 're':
            for i in range(len(self.df)):
                ax1.plot(np.log10(self.df[i].f), self.df[i].re, color=colors_real[i], marker='D', ms=3, lw=2.25, ls='-', label=self.label_cycleno[i])
                if fitting == 'on':
                    ax1.plot(np.log10(self.df[i].f), self.circuit_fit[i].real, lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='')
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("Z' [$\Omega$]")
                if legend == 'on' or legend =='potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode == 'log_re':
            for i in range(len(self.df)):
                ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].re), color=colors_real[i], marker='D', ms=3, lw=2.25, ls='-', label=self.label_cycleno[i])
                if fitting == 'on':
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.circuit_fit[i].real), lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='')
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("log(Z') [$\Omega$]")
                if legend == 'on' or legend == 'potential': 
                    ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode=='im':
            for i in range(len(self.df)):
                ax1.plot(np.log10(self.df[i].f), self.df[i].im, color=colors_imag[i], marker='s', ms=3, lw=2.25, ls='-', label=self.label_cycleno[i])
                if fitting == 'on':
                    ax1.plot(np.log10(self.df[i].f), -self.circuit_fit[i].imag, lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none', label='')
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("-Z'' [$\Omega$]")
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode=='log_im':
            for i in range(len(self.df)):
                ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].im), color=colors_imag[i], marker='s', ms=3, lw=2.25, ls='-', label=self.label_cycleno[i])
                if fitting == 'on':
                    ax1.plot(np.log10(self.df[i].f), np.log10(-self.circuit_fit[i].imag), lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none', label='')
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("log(-Z'') [$\Omega$]")
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode == 'log':
            for i in range(len(self.df)):
                ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].re), color=colors_real[i], marker='D', ms=3, lw=2.25,  ls='-', label=self.label_re_1[i])
                ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].im), color=colors_imag[i], marker='s', ms=3, lw=2.25,  ls='-', label=self.label_im_1[i])
                if fitting == 'on':
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.circuit_fit[i].real), lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='')
                    ax1.plot(np.log10(self.df[i].f), np.log10(-self.circuit_fit[i].imag), lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none')
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("log(Z', -Z'') [$\Omega$]")
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best', fontsize=10, frameon=False)

        ### Relative Residuals on Fit
        if rr=='on':
            if fitting == 'off':
                print('Fitting has not been performed, thus the relative residuals cannot be determined')
            elif fitting == 'on':
                self.rr_real = []
                self.rr_imag = []
                for i in range(len(self.df)):
                    self.rr_real.append(residual_real(re=self.df[i].re.values, fit_re=self.circuit_fit[i].real, fit_im=-self.circuit_fit[i].imag))
                    self.rr_imag.append(residual_imag(im=self.df[i].im.values, fit_re=self.circuit_fit[i].real, fit_im=-self.circuit_fit[i].imag))
                    if legend == 'on':
                        ax2.plot(np.log10(self.df[i].f), self.rr_real[i]*100, color=colors_real[i], marker='D', ms=6, lw=1, ls='--', label='#'+str(i+1))
                        ax2.plot(np.log10(self.df[i].f), self.rr_imag[i]*100, color=colors_imag[i], marker='s', ms=6, lw=1, ls='--',label='')
                    elif legend == 'potential':
                        ax2.plot(np.log10(self.df[i].f), self.rr_real[i]*100, color=colors_real[i], marker='D', ms=6, lw=1, ls='--', label=str(np.round(np.average(self.df[i].E_avg.values),2))+' V')
                        ax2.plot(np.log10(self.df[i].f), self.rr_imag[i]*100, color=colors_imag[i], marker='s', ms=6, lw=1, ls='--',label='')

                    ax2.axhline(0, ls='--', c='k', alpha=.5)
                    ax2.set_xlabel("log(f) [Hz]")
                    ax2.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]")

                #Automatic y-limits limits
                self.rr_im_min = []
                self.rr_im_max = []
                self.rr_re_min = []
                for i in range(len(self.df)): # needs to be within a loop if cycles have different number of data points     
                    self.rr_im_min = np.min(self.rr_imag[i])
                    self.rr_im_max = np.max(self.rr_imag[i])
                    self.rr_re_min = np.min(self.rr_real[i])
                    self.rr_re_max = np.max(self.rr_real[i])
                if self.rr_re_max > self.rr_im_max:
                    self.rr_ymax = self.rr_re_max
                else:
                    self.rr_ymax = self.rr_im_max
                if self.rr_re_min < self.rr_im_min:
                    self.rr_ymin = self.rr_re_min
                else:
                    self.rr_ymin  = self.rr_im_min
                if np.abs(self.rr_ymin) > np.abs(self.rr_ymax):
                    ax2.set_ylim(self.rr_ymin *100*1.5, np.abs(self.rr_ymin)*100*1.5)
                    ax2.annotate("$\Delta$Z'", xy=(np.log10(np.min(self.df[0].f)), np.abs(self.rr_ymin )*100*1.2), color=colors_real[-1], fontsize=12)
                    ax2.annotate("$\Delta$-Z''", xy=(np.log10(np.min(self.df[0].f)), np.abs(self.rr_ymin )*100*0.9), color=colors_imag[-1], fontsize=12)
                elif np.abs(self.rr_ymin) < np.abs(self.rr_ymax):
                    ax2.set_ylim(np.negative(self.rr_ymax)*100*1.5, np.abs(self.rr_ymax)*100*1.5)                    
                    ax2.annotate("$\Delta$Z'", xy=(np.log10(np.min(self.df[0].f)), np.abs(self.rr_ymax)*100*1.2), color=colors_real[-1], fontsize=12)
                    ax2.annotate("$\Delta$-Z''", xy=(np.log10(np.min(self.df[0].f)), np.abs(self.rr_ymax)*100*0.9), color=colors_imag[-1], fontsize=12)
    
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best', fontsize=10, frameon=False)

        ### Figure specifics
        if legend == 'on' or legend == 'potential':
            ax.legend(loc='best', fontsize=10, frameon=False)
        ax.set_xlabel("Z' [$\Omega$]")
        ax.set_ylabel("-Z'' [$\Omega$]")
        if nyq_xlim != 'none':
            ax.set_xlim(nyq_xlim[0], nyq_xlim[1])
        if nyq_ylim != 'none':
            ax.set_ylim(nyq_ylim[0], nyq_ylim[1])

        #Save Figure
        if savefig != 'none':
            fig.savefig(savefig) #saves figure if fix text is given

    def Fit_uelectrode(self, params, circuit, D_ox, r, theta_real_red, theta_imag_red, n, T, F, R, Q='none', weight_func='modulus', nan_policy='raise'):
        '''
        Fit the reductive microdisk electrode impedance repsonse following either BV or MHC infinite kientics
    
        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
        '''
        self.Fit = []
        self.circuit_fit = []
        
        for i in range(len(self.df)):
            self.Fit.append(minimize(leastsq_errorfunc_uelectrode, params, method='leastsq', args=(self.df[i].w, self.df[i].re, self.df[i].im, circuit, weight_func, np.average(self.df[i].E_avg), D_ox, r, theta_real_red, theta_imag_red, n, T, F, R), nan_policy=nan_policy, maxfev=9999990))
            print(report_fit(self.Fit[i]))
        
            if circuit == 'R-(Q(RM)),BV_red':
                if "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Rs_QRM_BV_red(w=self.df[i].w, E=np.average(self.df[i].E_avg), E0=self.Fit[i].params.get('E0').value, Rs=self.Fit[i].params.get('Rs').value, fs=self.Fit[i].params.get('fs').value, n_Q=self.Fit[i].params.get('n_Q').value, Q='none', Rct=self.Fit[i].params.get('Rct').value, alpha=self.Fit[i].params.get('alpha').value, C_ox=self.Fit[i].params.get('C_ox').value, D_ox=D_ox, r=r, theta_real_red=theta_real_red, theta_imag_red=theta_imag_red, n=n, T=T, F=F, R=R))
                elif "'Q'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Rs_QRM_BV_red(w=self.df[i].w, E=np.average(self.df[i].E_avg), E0=self.Fit[i].params.get('E0').value, Rs=self.Fit[i].params.get('Rs').value, fs='none', n_Q=self.Fit[i].params.get('n_Q').value, Q=self.Fit[i].params.get('Q').value, Rct=self.Fit[i].params.get('Rct').value, alpha=self.Fit[i].params.get('alpha').value, C_ox=self.Fit[i].params.get('C_ox').value, D_ox=D_ox, r=r, theta_real_red=theta_real_red, theta_imag_red=theta_imag_red, n=n, T=T, F=F, R=R))


    def uelectrode(self, params, circuit, E, alpha, n, C_ox, D_red, D_ox, r, theta_real_red, theta_real_ox, theta_imag_red, theta_imag_ox, F, R, T, weight_func='modulus', nan_policy='raise'):
        '''        
        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
        
        Inputs
        ------------
        - Rs = Series resistance [ohm]
        - Q = constant phase element [s/ohm]
        - n_Q = exponent of Q
        - Rct = Charge transfer resistance [ohm]
        - C_ox = concentration of oxidized specie [mol/cm3]

        - circuit:
            - 'R-(Q(RM))'
            - 'R-RQ-(Q(RM))'

        - weight_func = Weight function, Three options:
            - modulus (default)
            - unity
            - proportional
        
        - nan_policy = if issues occur with this fitting due to nan values 'propagate' should be used. otherwise, 'raise' is default
        
        Returns
        ------------
        Returns the fitted impedance spectra(s) but also the fitted parameters that were used in the initial guesses. To call these use e.g. self.fit_Rs
        '''
        self.Fit = []
        self.circuit_fit = []        
        self.fit_Rs = []
        self.fit_Q = []
        self.fit_fs = []
        self.fit_n_Q = []
        self.fit_Rct = []
        self.fit_E0 = []
        self.fit_Cred = []

        for i in range(len(self.df)):
            self.Fit.append(minimize(leastsq_errorfunc_uelectrode, params, method='leastsq', args=(self.df[i].w, self.df[i].re, self.df[i].im, circuit, weight_func, E, alpha, n, C_ox, D_red, D_ox, r, theta_real_red, theta_real_ox, theta_imag_red, theta_imag_ox, F, R, T), nan_policy=nan_policy, maxfev=9999990))
            print(report_fit(self.Fit[i]))
            if circuit == 'R-(Q(RM))':
                if "'fs'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Rs_QRM(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, fs=self.Fit[i].params.get('fs').value, Q='none', n_Q=self.Fit[i].params.get('n_Q').value, Rct=self.Fit[i].params.get('Rct').value, E=E, E0=self.Fit[i].params.get('E0').value, alpha=alpha, n=n, C_red=self.Fit[i].params.get('C_red').value, C_ox=C_ox, D_red=D_red, D_ox=D_ox, r=r, theta_real_red=theta_real_red, theta_real_ox=theta_real_ox, theta_imag_red=theta_imag_red, theta_imag_ox=theta_imag_ox, T=T, F=F, R=R))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_fs.append(self.Fit[i].params.get('fs').value)
                    self.fit_n_Q.append(self.Fit[i].params.get('n_Q').value)
                    self.fit_Rct.append(self.Fit[i].params.get('Rct').value)
                    self.fit_E0.append(self.Fit[i].params.get('E0').value)
                    self.fit_Cred.append(self.Fit[i].params.get('C_red').value)
                elif "'Q'" in str(self.Fit[i].params.keys()):
                    self.circuit_fit.append(cir_Rs_QRM(w=self.df[i].w, Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, fs='none', n_Q=self.Fit[i].params.get('n_Q').value, Rct=self.Fit[i].params.get('Rct').value, E=E, E0=self.Fit[i].params.get('E0').value, alpha=alpha, n=n, C_red=self.Fit[i].params.get('C_red').value, C_ox=C_ox, D_red=D_red, D_ox=D_ox, r=r, theta_real_red=theta_real_red, theta_real_ox=theta_real_ox, theta_imag_red=theta_imag_red, theta_imag_ox=theta_imag_ox, T=T, F=F, R=R))
                    self.fit_Rs.append(self.Fit[i].params.get('Rs').value)
                    self.fit_Q.append(self.Fit[i].params.get('Q').value)
                    self.fit_n_Q.append(self.Fit[i].params.get('n_Q').value)
                    self.fit_Rct.append(self.Fit[i].params.get('Rct').value)
                    self.fit_E0.append(self.Fit[i].params.get('E0').value)
                    self.fit_Cred.append(self.Fit[i].params.get('C_red').value)

    def uelectrode_sim_fit(self, params, circuit, E, alpha, n, C_ox, D_red, D_ox, r, theta_real_red, theta_real_ox, theta_imag_red, theta_imag_ox, F, R, T, weight_func='modulus', nan_policy='raise'):
        '''
        In development..
        
        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
        
        Inputs
        ------------
        - weight_func = Weight function, Three options:
            - modulus (default)
            - unity
            - proportional
        
        - nan_policy = if issues occur with this fitting due to nan values 'propagate' should be used. otherwise, 'raise' is default
        
        - nyq_xlim/nyq_xlim: x/y-axis on nyquist plot, if not equal to 'none' state [min,max] value
        
        - legend: Display legend
            Turn 'on', 'off'
    
        - bode = Plots Bode Plot - options:
            'on' = re, im vs. log(freq)
            'log' = log(re, im) vs. log(freq)
            
            're' = re vs. log(freq)
            'log_re' = log(re) vs. log(freq)
            
            'im' = im vs. log(freq)
            'log_im' = log(im) vs. log(freq)
        
        - fitting: if EIS_exp_fit() has been called. Plotting exp and fits by = 'on'
            Turn 'on', 'off'
    
        - rr: relative residuals. Gives relative residuals of fit from experimental data. 
            Turn 'on', 'off'
    
        Returns
        ------------
        The fitted impedance spectra(s) but also the fitted parameters that were used in the initial guesses. To call these use e.g. self.fit_Rs
        '''
        self.Fit = minimize(leastsq_errorfunc_uelectrode, params, method='leastsq', args=(self.w, self.re, self.im, circuit, weight_func, E, alpha, n, C_ox, D_red, D_ox, r, theta_real_red, theta_real_ox, theta_imag_red, theta_imag_ox, F, R, T), nan_policy=nan_policy, maxfev=9999990)
        print(report_fit(self.Fit))
    

    def plot_Cdl_E(self, interface, BET_Area, m_electrode):
        '''
        Normalizing Q to C_eff or Cdl using either norm_nonFara_Q_C() or norm_Fara_Q_C()
        
        Refs:
            - G. J.Brug, A.L.G. vandenEeden, M.Sluyters-Rehbach, and J.H.Sluyters, J.Elec-
            troanal. Chem. Interfacial Electrochem., 176, 275 (1984)
            - B. Hirschorn, ElectrochimicaActa, 55, 6218 (2010)
        
        Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
        
        Inputs
        ---------
        interface = faradaic / nonfaradaic
        BET_Area = BET surface area of electrode material [cm]
        m_electrode = mass of electrode [cm2/mg]
        
        Inputs
        ---------
        C_eff/C_dl = Normalized Double-layer capacitance measured from impedance [uF/cm2] (normalized by norm_nonFara_Q_C() or norm_Fara_Q_C())
        '''
        fig = figure(dpi=120, facecolor='w', edgecolor='w')
        fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
        ax = fig.add_subplot(111)

        self.Q_norm = []
        self.E = []
        if interface == 'nonfaradaic':
            self.Q_norm = []
            for i in range(len(self.df)):
                #self.Q_norm.append(norm_nonFara_Q_C(Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value, L=self.Fit[i].params.get('L').value) )
                self.Q_norm.append(norm_nonFara_Q_C(Rs=self.Fit[i].params.get('Rs').value, Q=self.Fit[i].params.get('Q').value, n=self.Fit[i].params.get('n').value) )
                self.E.append(np.average(self.df[i].E_avg))
        
        elif interface == 'faradaic':
            self.Q_norm = []
            for j in range(len(self.df)):
                self.Q_norm.append(norm_Fara_Q_C(Rs=self.Fit[j].params.get('Rs').value, Rct=self.Fit[j].params.get('R').value, n=self.Fit[j].params.get('n').value, fs=self.Fit[j].params.get('fs').value, L=self.Fit[j].params.get('L').value))
                self.E.append(np.average(self.df[j].E_avg))

        self.C_norm = (np.array(self.Q_norm)/(m_electrode*BET_Area))*10**6 #'uF/cm2'
        ax.plot(self.E, self.C_norm, 'o--', label='C$_{dl}$')
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('C$_{dl}$ [$\mu$F/cm$^2$]')


class EIS_sim:
    '''
    Simulates and plots Electrochemical Impedance Spectroscopy based-on build-in equivalent cirucit models
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)    

    Following circuits are implemented:
        - RC
        - RQ
        - R-RQ
        - R-RQ-RQ
        - R-Q
        - R-RQ-Q
        - R-(Q(RW))
        - C-RC-C
        - Q-RQ-Q
        - RC-RC-ZD
        - R-TLsQ
        - R-RQ-TLsQ
        - R-TLs
        - R-RQ-TLs
        - R-TLQ
        - R-RQ-TLQ
        - R-TL
        - R-RQ-TL
        - R-TL1Dsolid (reactive interface with 1D solid-state diffusion)
        - R-RQ-TL1Dsolid
    
    Inputs
    --------
    - nyq_xlim/nyq_xlim: 
        x/y-axis on nyquist plot, if not equal to 'none' state [min,max] value
        
    - bode: Plots following Bode plots
        - 'off'
        - 'on' = re, im vs. log(freq)
        - 'log' = log(re, im) vs. log(freq)
        
        - 're' = re vs. log(freq)
        - 'log_re' = log(re) vs. log(freq)
        
        - 'im' = im vs. log(freq)
        - 'log_im' = log(im) vs. log(freq)
    '''
    def __init__(self, circuit, frange, bode='off', nyq_xlim='none', nyq_ylim='none', legend='on', savefig='none'):
        self.f = frange
        self.w = 2*np.pi*frange
        self.re = circuit.real
        self.im = -circuit.imag

        if bode=='off':
            fig = figure(dpi=120, facecolor='w', edgecolor='w')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(111, aspect='equal')

        elif bode=='on' or bode=='log' or bode=='re' or bode=='log_re' or bode=='im' or bode=='log_im' or bode=='log':
            fig = figure(figsize=(6, 4.5), dpi=120, facecolor='w', edgecolor='w')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(211, aspect='equal')
            ax1 = fig.add_subplot(212)

        colors = sns.color_palette("colorblind", n_colors=1)
        colors_real = sns.color_palette("Blues", n_colors=1)
        colors_imag = sns.color_palette("Oranges", n_colors=1)

        ### Nyquist Plot
        ax.plot(self.re, self.im, color=colors[0], marker='o', ms=4, lw=2, ls='-', label='Sim')

        ### Bode Plot
        if bode=='on':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z', -Z'' [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best', fontsize=10, frameon=False)
            
        elif bode == 're':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z' [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode == 'log_re':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z') [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode=='im':
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("-Z'' [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode=='log_im':
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(-Z'') [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode == 'log':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25,  ls='-', label="Z''")
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25,  ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z', -Z'') [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)
        
        ### Figure specifics
        if legend == 'on': 
            ax.legend(loc='best', fontsize=10, frameon=False)
        ax.set_xlabel("Z' [$\Omega$]")
        ax.set_ylabel("-Z'' [$\Omega$]")
        if nyq_xlim != 'none':
            ax.set_xlim(nyq_xlim[0], nyq_xlim[1])
        if nyq_ylim != 'none':
            ax.set_ylim(nyq_ylim[0], nyq_ylim[1])

        #Save Figure
        if savefig != 'none':
            fig.savefig(savefig) #saves figure if fix text is given

        
    def EIS_sim_fit(self, params, circuit, weight_func='modulus', nan_policy='raise', bode='on', nyq_xlim='none', nyq_ylim='none', legend='on', savefig='none'):
        '''
        This function fits simulations with a selected circuit. This function is mainly used to test fitting functions prior to being used on experimental data
        
        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
        
        Inputs
        ------------
        - Circuit: Equivlaent circuit models
            - RC
            - RQ
            - R-RQ
            - R-RQ-RQ
            - R-Q
            - R-RQ-Q
            - R-(Q(RW))
            - C-RC-C
            - Q-RQ-Q
            - RC-RC-ZD
            - R-TLsQ
            - R-RQ-TLsQ
            - R-TLs
            - R-RQ-TLs
            - R-TLQ
            - R-RQ-TLQ
            - R-TL
            - R-RQ-TL
            - R-TL1Dsolid (reactive interface with 1D solid-state diffusion)
            - R-RQ-TL1Dsolid

        - weight_func = Weight function, Three options:
            - modulus (default)
            - unity
            - proportional
                
        - nyq_xlim/nyq_xlim: x/y-axis on nyquist plot, if not equal to 'none' state [min,max] value
        
        - legend: Display legend
            Turn 'on', 'off'

        - bode = Plots Bode Plot - options:
            'on' = re, im vs. log(freq)
            'log' = log(re, im) vs. log(freq)
            
            're' = re vs. log(freq)
            'log_re' = log(re) vs. log(freq)
            
            'im' = im vs. log(freq)
            'log_im' = log(im) vs. log(freq)
        
        Returns
        ------------
        The fitted impedance spectra(s) but also the fitted parameters that were used in the initial guesses. To call these use e.g. self.fit_Rs
        '''
        self.Fit = minimize(leastsq_errorfunc, params, method='leastsq', args=(self.w, self.re, self.im, circuit, weight_func), maxfev=9999990, nan_policy=nan_policy)
        print(report_fit(self.Fit))

        if circuit == 'C':
            self.circuit_fit = elem_C(w=self.w, C=self.Fit.params.get('C').value)
            self.fit_C = []
            self.fit_C.append(self.Fit.params.get('C').value)
        elif circuit == 'Q':
            self.circuit_fit = elem_Q(w=self.w, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value)
            self.fit_Q = []
            self.fit_Q.append(self.Fit.params.get('Q').value)
            self.fit_n = []
            self.fit_n.append(self.Fit.params.get('n').value)
        elif circuit == 'R-C':
            self.circuit_fit = cir_RsC(w=self.w, Rs=self.Fit.params.get('Rs').value, C=self.Fit.params.get('C').value)
            self.fit_Rs = []
            self.fit_Rs.append(self.Fit.params.get('Rs').value)
            self.fit_C = []
            self.fit_C.append(self.Fit.params.get('C').value)
        elif circuit == 'R-Q':
            self.circuit_fit = cir_RsQ(w=self.w, Rs=self.Fit.params.get('Rs').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value)
            self.fit_Rs = []
            self.fit_Rs.append(self.Fit.params.get('Rs').value)
            self.fit_Q = []
            self.fit_Q.append(self.Fit.params.get('Q').value)
            self.fit_n = []
            self.fit_n.append(self.Fit.params.get('n').value)
        elif circuit == 'RC':
            if "'C'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RC(w=self.w, C=self.Fit.params.get('C').value, R=self.Fit.params.get('R').value, fs='none')
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_C = []
                self.fit_C.append(self.Fit.params.get('C').value)
            elif "'fs'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RC(w=self.w, C='none', R=self.Fit.params.get('R').value, fs=self.Fit.params.get('fs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('R').value)
        elif circuit == 'RQ':
            if "'fs'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RQ(w=self.w, R=self.Fit.params.get('R').value, Q='none', n=self.Fit.params.get('n').value, fs=self.Fit.params.get('fs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('fs').value)
            elif "'Q'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RQ(w=self.w, R=self.Fit.params.get('R').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, fs='none')
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
        elif circuit == 'R-RQ':
            if "'fs'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q='none', n=self.Fit.params.get('n').value, fs=self.Fit.params.get('fs').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('fs').value)
            elif "'Q'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, fs='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
        elif circuit == 'R-RQ-RQ':
            if "'fs'" in str(self.Fit.params.keys()) and "'fs2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQRQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q='none', n=self.Fit.params.get('n').value, fs=self.Fit.params.get('fs').value, R2=self.Fit.params.get('R2').value, Q2='none', n2=self.Fit.params.get('n2').value, fs2=self.Fit.params.get('fs2').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('fs').value)
                self.fit_R2 =[]
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_fs2 = []
                self.fit_fs2.append(self.Fit.params.get('fs2').value)
            elif "'Q'" in str(self.Fit.params.keys()) and "'fs2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQRQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, fs='none', R2=self.Fit.params.get('R2').value, Q2='none', n2=self.Fit.params.get('n2').value, fs2=self.Fit.params.get('fs2').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_fs2 = []
                self.fit_fs2.append(self.Fit.params.get('fs2').value)
            elif "'fs'" in str(self.Fit.params.keys()) and "'Q2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQRQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q='none', n=self.Fit.params.get('n').value, fs=self.Fit.params.get('fs').value, R2=self.Fit.params.get('R2').value, Q2=self.Fit.params.get('Q2').value, n2=self.Fit.params.get('n2').value, fs2='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('fs').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
            elif "'Q'" in str(self.Fit.params.keys()) and "'Q2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQRQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, fs='none', R2=self.Fit.params.get('R2').value, Q2=self.Fit.params.get('Q2').value, n2=self.Fit.params.get('n2').value, fs2='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
        elif circuit == 'R-RC-C':
            self.circuit_fit = cir_RsRCC(w=self.df[i].w, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, C1=self.Fit.params.get('C1').value, C=self.Fit.params.get('C').value)
            self.fit_Rs = []
            self.fit_Rs.append(self.Fit.params.get('Rs').value)
            self.fit_R1 = []
            self.fit_R1.append(self.Fit.params.get('R1').value)
            self.fit_C1 = []
            self.fit_C1.append(self.Fit.params.get('C1').value)
            self.fit_C = []
            self.fit_C.append(self.Fit.params.get('C').value)
        elif circuit == 'R-RC-Q':
            self.circuit_fit = cir_RsRCQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, C1=self.Fit.params.get('C1').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value)
            self.fit_Rs = []
            self.fit_Rs.append(self.Fit.params.get('Rs').value)
            self.fit_R1 =[]
            self.fit_R1.append(self.Fit.params.get('R1').value)
            self.fit_C1 =[]
            self.fit_C1.append(self.Fit.params.get('C1').value)
            self.fit_Q = []
            self.fit_Q.append(self.Fit.params.get('Q').value)
            self.fit_n = []
            self.fit_n.append(self.Fit.params.get('n').value)
        elif circuit == 'R-RQ-Q':
            if "'fs1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQQ(w=self.w, Rs=self.Fit.params.get('Rs').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, R1=self.Fit.params.get('R1').value, Q1='none', n1=self.Fit.params.get('n1').value, fs1=self.Fit.params.get('fs1').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
            if "'Q1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQQ(w=self.w, Rs=self.Fit.params.get('Rs').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, R1=self.Fit.params.get('R1').value, Q1=self.Fit.params.get('Q1').value, n1=self.Fit.params.get('n1').value, fs1='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_fQ = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
        elif circuit == 'R-RQ-C':
            if "'fs1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQC(w=self.w, Rs=self.Fit.params.get('Rs').value, C=self.Fit.params.get('C').value, R1=self.Fit.params.get('R1').value, Q1='none', n1=self.Fit.params.get('n1').value, fs1=self.Fit.params.get('fs1').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_C = []
                self.fit_C.append(self.Fit.params.get('C').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
            elif "'Q1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQC(w=self.df.w, Rs=self.Fit.params.get('Rs').value, C=self.Fit.params.get('C').value, R1=self.Fit.params.get('R1').value, Q1=self.Fit.params.get('Q1').value, n1=self.Fi.params.get('n1').value, fs1='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_C = []
                self.fit_C.append(self.Fit.params.get('C').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
        elif circuit == 'R-(Q(RW))':
            if "'Q'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_Randles_simplified(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q=self.Fit.params.get('Q').value, fs='none', n=self.Fit.params.get('n').value, sigma=self.Fit.params.get('sigma').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_sigma = []
                self.fit_sigma.append(self.Fit.params.get('sigma').value)
            elif "'fs'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_Randles_simplified(w=self.w, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q='none', fs=self.Fit.params.get('fs').value, n=self.Fit.params.get('n').value, sigma=self.Fit.params.get('sigma').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('fs').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_sigma = []
                self.fit_sigma.append(self.Fit.params.get('sigma').value)
        elif circuit == 'R-TLsQ':
            self.circuit_fit = cir_RsTLsQ(w=self.w, Rs=self.Fit.params.get('Rs').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value)
            self.fit_Rs = []
            self.fit_Rs.append(self.Fit.params.get('Rs').value)
            self.fit_Q = []
            self.fit_Q.append(self.Fit.params.get('Q').value)
            self.fit_n = []
            self.fit_n.append(self.Fit.params.get('n').value)
            self.fit_Ri = []
            self.fit_Ri.append(self.Fit.params.get('Ri').value)
            self.fit_L = []
            self.fit_L.append(self.Fit.params.get('L').value)
        elif circuit == 'R-RQ-TLsQ':
            if "'fs1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLsQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, fs1=self.Fit.params.get('fs1').value, n1=self.Fit.params.get('n1').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, Q1='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
            elif "'Q1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLsQ(w=self.w, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, fs1='none', n1=self.Fit.params.get('n1').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, Q1=self.Fit.params.get('Q1').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
        elif circuit == 'R-TLs':
            if "'fs'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsTLs(w=self.w, Rs=self.Fit.params.get('Rs').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, R=self.Fit.params.get('R').value, Q='none', n=self.Fit.params.get('n').value, fs=self.Fit.params.get('fs').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('fs').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
            elif "'Q'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsTLs(w=self.w, Rs=self.Fit.params.get('Rs').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, R=self.Fit.params.get('R').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, fs='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
        elif circuit == 'R-RQ-TLs':
            if "'fs1'" in str(self.Fit.params.keys()) and "'fs2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLs(w=self.w, Rs=self.Fit.params.get('Rs').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, R1=self.Fit.params.get('R1').value, n1=self.Fit.params.get('n1').value, fs1=self.Fit.params.get('fs1').value, R2=self.Fit.params.get('R2').value, n2=self.Fit.params.get('n2').value, fs2=self.Fit.params.get('fs2').value, Q1='none', Q2='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_fs2 = []
                self.fit_fs2.append(self.Fit.params.get('fs2').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
            elif "'Q1'" in str(self.Fit.params.keys()) and "'fs2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLs(w=self.w, Rs=self.Fit.params.get('Rs').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, R1=self.Fit.params.get('R1').value, n1=self.Fit.params.get('n1').value, fs1='none', R2=self.Fit.params.get('R2').value, n2=self.Fit.params.get('n2').value, fs2=self.Fit.params.get('fs2').value, Q1=self.Fit.params.get('Q1').value, Q2='none')
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_fs2 = []
                self.fit_fs2.append(self.Fit.params.get('fs2').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
            elif "'fs1'" in str(self.Fit.params.keys()) and "'Q2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLs(w=self.w, Rs=self.Fit.params.get('Rs').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, R1=self.Fit.params.get('R1').value, n1=self.Fit.params.get('n1').value, fs1=self.Fit.params.get('fs1').value, R2=self.Fit.params.get('R2').value, n2=self.Fit.params.get('n2').value, fs2='none', Q1='none', Q2=self.Fit.params.get('Q2').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
            elif "'Q1'" in str(self.Fit.params.keys()) and "'Q2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLs(w=self.w, Rs=self.Fit.params.get('Rs').value, L=self.Fit.params.get('L').value, Ri=self.Fit.params.get('Ri').value, R1=self.Fit.params.get('R1').value, n1=self.Fit.params.get('n1').value, fs1='none', R2=self.Fit.params.get('R2').value, n2=self.Fit.params.get('n2').value, fs2='none', Q1=self.Fit.params.get('Q1').value, Q2=self.Fit.params.get('Q2').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
        elif circuit == 'R-TLQ':
            self.circuit_fit = cir_RsTLQ(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
            self.fit_L = []
            self.fit_L.append(self.Fit.params.get('L').value)
            self.fit_Rs = []
            self.fit_Rs.append(self.Fit.params.get('Rs').value)
            self.fit_Q = []
            self.fit_Q.append(self.Fit.params.get('Q').value)
            self.fit_n = []
            self.fit_n.append(self.Fit.params.get('n').value)
            self.fit_Rel = []
            self.fit_Rel.append(self.Fit.params.get('Rel').value)
            self.fit_Ri = []
            self.fit_Ri.append(self.Fit.params.get('Ri').value)
        elif circuit == 'R-RQ-TLQ':
            if "'fs1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLQ(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value, R1=self.Fit.params.get('R1').value, n1=self.Fit.params.get('n1').value, fs1=self.Fit.params.get('fs1').value, Q1='none')
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
            elif "'Q1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTLQ(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value, R1=self.Fit.params.get('R1').value, n1=self.Fit.params.get('n1').value, fs1='none', Q1=self.Fit.params.get('Q1').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
        elif circuit == 'R-TL':
            if "'fs'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsTL(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, fs=self.Fit.params.get('fs').value, n=self.Fit.params.get('n').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value, Q='none')
                self.fit_L = []                
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []                
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_fs = []
                self.fit_fs.append(self.Fit.params.get('fs').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
            elif "'Q'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsTL(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, fs='none', n=self.Fit.params.get('n').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value, Q=self.Fit.params.get('Q').value)
                self.fit_L = []                
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []                
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
        elif circuit == 'R-RQ-TL':
            if "'Q1'" in str(self.Fit.params.keys()) and "'Q2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTL(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, fs1='none', Q1=self.Fit.params.get('Q1').value, n1=self.Fit.params.get('n1').value, R2=self.Fit.params.get('R2').value, fs2='none', Q2=self.Fit.params.get('Q2').value, n2=self.Fit.params.get('n2').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
                self.fit_L = []                
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []                
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
            elif "'fs1'" in str(self.Fit.params.keys()) and "'fs2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTL(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, fs1=self.Fit.params.get('fs1').value, Q1='none', n1=self.Fit.params.get('n1').value, R2=self.Fit.params.get('R2').value, fs2=self.Fit.params.get('fs2').value, Q2='none', n2=self.Fit.params.get('n2').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
                self.fit_L = []                
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []                
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_fs2 = []
                self.fit_fs2.append(self.Fit.params.get('fs2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
            elif "'Q1'" in str(self.Fit.params.keys()) and "'fs2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTL(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, fs1='none', Q1=self.Fit.params.get('Q1').value, n1=self.Fit.params.get('n1').value, R2=self.Fit.params.get('R2').value, fs2=self.Fit.params.get('fs2').value, Q2='none', n2=self.Fit.params.get('n2').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
                self.fit_L = []                
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []                
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_fs2 = []
                self.fit_fs2.append(self.Fit.params.get('fs2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
            elif "'fs1'" in str(self.Fit.params.keys()) and "'Q2'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTL(w=self.w, L=self.Fit.params.get('L').value, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, fs1=self.Fit.params.get('fs1').value, Q1='none', n1=self.Fit.params.get('n1').value, R2=self.Fit.params.get('R2').value, fs2='none', Q2=self.Fit.params.get('Q2').value, n2=self.Fit.params.get('n2').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
                self.fit_L = []                
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_Rs = []                
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
        elif circuit == 'R-TL1Dsolid':
                self.circuit_fit = cir_RsTL_1Dsolid(w=self.w, L=self.Fit.params.get('L').value, D=self.Fit.params.get('D').value, radius=self.Fit.params.get('radius').value, Rs=self.Fit.params.get('Rs').value, R=self.Fit.params.get('R').value, Q=self.Fit.params.get('Q').value, n=self.Fit.params.get('n').value, R_w=self.Fit.params.get('R_w').value, n_w=self.Fit.params.get('n_w').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_radius = []
                self.fit_radius.append(self.Fit.params.get('radius').value)
                self.fit_D = []
                self.fit_D.append(self.Fit.params.get('D').value)            
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R = []
                self.fit_R.append(self.Fit.params.get('R').value)
                self.fit_Q = []
                self.fit_Q.append(self.Fit.params.get('Q').value)
                self.fit_n = []
                self.fit_n.append(self.Fit.params.get('n').value)
                self.fit_R_w = []
                self.fit_R_w.append(self.Fit.params.get('R_w').value)
                self.fit_n_w = []
                self.fit_n_w.append(self.Fit.params.get('n_w').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
        elif circuit == 'R-RQ-TL1Dsolid':
            if "'fs1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTL_1Dsolid(w=self.w, L=self.Fit.params.get('L').value, D=self.Fit.params.get('D').value, radius=self.Fit.params.get('radius').value, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, Q1='none', fs1=self.Fit.params.get('fs1').value, n1=self.Fit.params.get('n1').value, R2=self.Fit.params.get('R2').value, Q2=self.Fit.params.get('Q2').value, n2=self.Fit.params.get('n2').value, R_w=self.Fit.params.get('R_w').value, n_w=self.Fit.params.get('n_w').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_radius = []
                self.fit_radius.append(self.Fit.params.get('radius').value)
                self.fit_D = []
                self.fit_D.append(self.Fit.params.get('D').value)            
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_fs1 = []
                self.fit_fs1.append(self.Fit.params.get('fs1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_R_w = []
                self.fit_R_w.append(self.Fit.params.get('R_w').value)
                self.fit_n_w = []
                self.fit_n_w.append(self.Fit.params.get('n_w').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
            elif "'Q1'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RsRQTL_1Dsolid(w=self.w, L=self.Fit.params.get('L').value, D=self.Fit.params.get('D').value, radius=self.Fit.params.get('radius').value, Rs=self.Fit.params.get('Rs').value, R1=self.Fit.params.get('R1').value, Q1=self.Fit.params.get('Q1').value, fs1='none', n1=self.Fit.params.get('n1').value, R2=self.Fit.params.get('R2').value, Q2=self.Fit.params.get('Q2').value, n2=self.Fit.params.get('n2').value, R_w=self.Fit.params.get('R_w').value, n_w=self.Fit.params.get('n_w').value, Rel=self.Fit.params.get('Rel').value, Ri=self.Fit.params.get('Ri').value)
                self.fit_L = []
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_radius = []
                self.fit_radius.append(self.Fit.params.get('radius').value)
                self.fit_D = []
                self.fit_D.append(self.Fit.params.get('D').value)            
                self.fit_Rs = []
                self.fit_Rs.append(self.Fit.params.get('Rs').value)
                self.fit_R1 = []
                self.fit_R1.append(self.Fit.params.get('R1').value)
                self.fit_Q1 = []
                self.fit_Q1.append(self.Fit.params.get('Q1').value)
                self.fit_n1 = []
                self.fit_n1.append(self.Fit.params.get('n1').value)
                self.fit_R2 = []
                self.fit_R2.append(self.Fit.params.get('R2').value)
                self.fit_Q2 = []
                self.fit_Q2.append(self.Fit.params.get('Q2').value)
                self.fit_n2 = []
                self.fit_n2.append(self.Fit.params.get('n2').value)
                self.fit_R_w = []
                self.fit_R_w.append(self.Fit.params.get('R_w').value)
                self.fit_n_w = []
                self.fit_n_w.append(self.Fit.params.get('n_w').value)
                self.fit_Rel = []
                self.fit_Rel.append(self.Fit.params.get('Rel').value)
                self.fit_Ri = []
                self.fit_Ri.append(self.Fit.params.get('Ri').value)
        elif circuit == 'C-RC-C':
            if "'fsb'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_C_RC_C(w=self.w, Ce=self.Fit.params.get('Ce').value, Cb='none', Rb=self.Fit.params.get('Rb').value, fsb=self.Fit.params.get('fsb').value)
                self.fit_Ce = []
                self.fit_Ce.append(self.Fit.params.get('Ce').value)
                self.fit_Rb = []
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_fsb = []
                self.fit_fsb.append(self.Fit.params.get('fsb').value)
            elif "'Cb'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_C_RC_C(w=self.w, Ce=self.Fit.params.get('Ce').value, Cb=self.Fit.params.get('Cb').value, Rb=self.Fit.params.get('Rb').value, fsb='none')
                self.fit_Ce = []
                self.fit_Ce.append(self.Fit.params.get('Ce').value)
                self.fit_Rb = []
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_Cb = []
                self.fit_Cb.append(self.Fit.params.get('Cb').value)
        elif circuit == 'Q-RQ-Q':
            if "'fsb'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_Q_RQ_Q(w=self.w, Qe=self.Fit.params.get('Qe').value, ne=self.Fit.params.get('ne').value, Qb='none', Rb=self.Fit.params.get('Rb').value, fsb=self.Fit.params.get('fsb').value, nb=self.Fit.params.get('nb').value)
                self.fit_Qe = []
                self.fit_Qe.append(self.Fit.params.get('Qe').value)
                self.fit_ne = []
                self.fit_ne.append(self.Fit.params.get('ne').value)
                self.fit_Rb = []
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_fsb = []
                self.fit_fsb.append(self.Fit.params.get('fsb').value)
                self.fit_nb = []
                self.fit_nb.append(self.Fit.params.get('nb').value)
            elif "'Qb'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_Q_RQ_Q(w=self.w, Qe=self.Fit.params.get('Qe').value, ne=self.Fit.params.get('ne').value, Qb=self.Fit.params.get('Qb').value, Rb=self.Fit.params.get('Rb').value, fsb='none', nb=self.Fit.params.get('nb').value)
                self.fit_Qe = []
                self.fit_Qe.append(self.Fit.params.get('Qe').value)
                self.fit_ne = []
                self.fit_ne.append(self.Fit.params.get('ne').value)
                self.fit_Rb = []
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_Qb = []
                self.fit_Qb.append(self.Fit.params.get('Qb').value)
                self.fit_nb = []
                self.fit_nb.append(self.Fit.params.get('nb').value)
        elif circuit == 'RC-RC-ZD':
            self.fit_L = []
            self.fit_D_s = []
            self.fit_u1 = []
            self.fit_u2 = []
            self.fit_Cb = []
            self.fit_Rb = []
            self.fit_fsb = []
            self.fit_Ce = []
            self.fit_Re = []
            self.fit_fse = []
            if "'fsb'" in str(self.Fit.params.keys()) and "'fse'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RCRCZD(w=self.w, L=self.Fit.params.get('L').value, D_s=self.Fit.params.get('D_s').value, u1=self.Fit.params.get('u1').value, u2=self.Fit.params.get('u2').value, Cb='none', Rb=self.Fit.params.get('Rb').value, fsb=self.Fit.params.get('fsb').value, Ce='none', Re=self.Fit.params.get('Re').value, fse=self.Fit.params.get('fse').value)
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_D_s.append(self.Fit.params.get('D_s').value)
                self.fit_u1.append(self.Fit.params.get('u1').value)
                self.fit_u2.append(self.Fit.params.get('u2').value)
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_Re.append(self.Fit.params.get('Re').value)
                self.fit_fsb.append(self.Fit.params.get('fsb').value)
                self.fit_fse.append(self.Fit.params.get('fse').value)
            elif "'Cb'" in str(self.Fit.params.keys()) and "'Ce'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RCRCZD(w=self.w, L=self.Fit.params.get('L').value, D_s=self.Fit.params.get('D_s').value, u1=self.Fit.params.get('u1').value, u2=self.Fit.params.get('u2').value, Cb=self.Fit.params.get('Cb').value, Rb=self.Fit.params.get('Rb').value, fsb='none', Ce=self.Fit.params.get('Ce').value, Re=self.Fit.params.get('Re').value, fse='none')
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_D_s.append(self.Fit.params.get('D_s').value)
                self.fit_u1.append(self.Fit.params.get('u1').value)
                self.fit_u2.append(self.Fit.params.get('u2').value)
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_Re.append(self.Fit.params.get('Re').value)
                self.fit_Cb.append(self.Fit.params.get('Cb').value)
                self.fit_Ce.append(self.Fit.params.get('Ce').value)                
            elif "'Cb'" in str(self.Fit.params.keys()) and "'fse'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RCRCZD(w=self.w, L=self.Fit.params.get('L').value, D_s=self.Fit.params.get('D_s').value, u1=self.Fit.params.get('u1').value, u2=self.Fit.params.get('u2').value, Cb=self.Fit.params.get('Cb').value, Rb=self.Fit.params.get('Rb').value, fsb='none', Ce='none', Re=self.Fit.params.get('Re').value, fse=self.Fit.params.get('fse').value)
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_D_s.append(self.Fit.params.get('D_s').value)
                self.fit_u1.append(self.Fit.params.get('u1').value)
                self.fit_u2.append(self.Fit.params.get('u2').value)
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_Re.append(self.Fit.params.get('Re').value)
                self.fit_Cb.append(self.Fit.params.get('Cb').value)
                self.fit_fse.append(self.Fit.params.get('fse').value)
            elif "'fsb'" in str(self.Fit.params.keys()) and "'Ce'" in str(self.Fit.params.keys()):
                self.circuit_fit = cir_RCRCZD(w=self.w, L=self.Fit.params.get('L').value, D_s=self.Fit.params.get('D_s').value, u1=self.Fit.params.get('u1').value, u2=self.Fit.params.get('u2').value, Cb=self.Fit.params.get('Cb').value, Rb='none', fsb=self.Fit.params.get('fsb').value, Ce=self.Fit.params.get('Ce').value, Re=self.Fit.params.get('Re').value, fse='none')
                self.fit_L.append(self.Fit.params.get('L').value)
                self.fit_D_s.append(self.Fit.params.get('D_s').value)
                self.fit_u1.append(self.Fit.params.get('u1').value)
                self.fit_u2.append(self.Fit.params.get('u2').value)
                self.fit_Rb.append(self.Fit.params.get('Rb').value)
                self.fit_Re.append(self.Fit.params.get('Re').value)
                self.fit_fsb.append(self.Fit.params.get('fsb').value)
                self.fit_Ce.append(self.Fit.params.get('Ce').value)  
        else:
            print('Circuit is not properly defined, see details described in definition')

        fig = figure(figsize=(6, 4.5), dpi=120, facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
        ax = fig.add_subplot(211, aspect='equal')
        ax1 = fig.add_subplot(212)

        colors = sns.color_palette("colorblind", n_colors=1)
        colors_real = sns.color_palette("Blues", n_colors=1)
        colors_imag = sns.color_palette("Oranges", n_colors=1)

        ### Nyquist Plot
        ax.plot(self.re, self.im, color=colors[0], marker='o', ms=4, lw=2, ls='-', label='Sim')
        ax.plot(self.circuit_fit.real, -self.circuit_fit.imag, lw=0, marker='o', ms=8, mec='r', mew=1, mfc='none', label='Fit')

        ### Bode Plot
        if bode=='on':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), self.circuit_fit.real, lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.plot(np.log10(self.f), -self.circuit_fit.imag, lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z', -Z'' [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)
            
        elif bode == 're':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.plot(np.log10(self.f), self.circuit_fit.real, lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z' [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode == 'log_re':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z''")
            ax1.plot(np.log10(self.f), np.log10(self.circuit_fit.real), lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z') [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode=='im':
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), -self.circuit_fit.imag, lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("-Z'' [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode=='log_im':
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), np.log10(-self.circuit_fit.imag), lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(-Z'') [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)

        elif bode == 'log':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25,  ls='-', label="Z''")
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25,  ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), np.log10(self.circuit_fit.real), lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.plot(np.log10(self.f), np.log10(-self.circuit_fit.imag), lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z', -Z'') [$\Omega$]")
            if legend == 'on': 
                ax1.legend(loc='best', fontsize=10, frameon=False)
        
        ### Figure specifics
        if legend == 'on': 
            ax.legend(loc='best', fontsize=10, frameon=False)
        ax.set_xlabel("Z' [$\Omega$]")
        ax.set_ylabel("-Z'' [$\Omega$]")

        if nyq_xlim != 'none':
            ax.set_xlim(nyq_xlim[0], nyq_xlim[1])
        if nyq_ylim != 'none':
            ax.set_ylim(nyq_ylim[0], nyq_ylim[1])

        #Save Figure
        if savefig != 'none':
            fig.savefig(savefig) #saves figure if fix text is given

#print()
#print('---> PyEIS Core Loaded (v. 0.5.7 - 02/01/19)')
#print()