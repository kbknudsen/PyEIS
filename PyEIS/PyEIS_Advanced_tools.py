#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:18:41 2018

This script contains tools for advanced EIS analysis. These scripts are meant for post treatment of resistances and capacitors

@author: Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
"""

# Constants
import numpy as np

from scipy.constants import codata
F = codata.physical_constants['Faraday constant'][0]
qe = codata.physical_constants['elementary charge'][0]
R = codata.physical_constants['molar gas constant'][0]
kB = codata.physical_constants['Boltzmann constant'][0]
kB_eV = codata.physical_constants['Boltzmann constant in eV/K'][0]
N_A = codata.physical_constants['Avogadro constant'][0]
F, R, qe, kB, kB_eV, N_A

### Normalization of constant phase elements
##
#
def norm_nonFara_Q_C(Rs, Q, n, L='none'):
    '''
    Normalziation of a non-faradaic interfacial capacitance (Blocking Electrode)
    
    Following Brug and Hirschorn's normalization of distribtuion of relaxation times
    Ref.:
        - G.J.Brug et al., J.Electroanal. Chem. Interfacial Electrochem., 176, 275 (1984)
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ------------------
    Q = Constant phase element [s^n/ohm]
    n = Exponent of CPE [-]
    Rs = Series Resistance [ohm]
    
    Optional Inputs
    ------------------
    L = Thickness/length of electrode, used in Porous Electrode Theory [cm]
    
    Returns
    ------------------
    C_eff = normalized capacitance for a non-faradaic electrode [s/ohm = F]
    '''
    if L == 'none':
        C_eff = (Q * Rs**(1-n))**(1/n)
    else:
        C_eff = ((Q*L) * Rs**(1-n))**(1/n)
    return C_eff

def norm_Fara_Q_C(Rs, Rct, n, Q='none', fs='none', L='none'):
    '''
    Normalziation of a faradaic interfacial capacitance (Blocking Electrode)
    
    Contains option to use summit frequency (fs) instead of CPE (Q) - valueable for outputs of fits
    
    Following Brug and Hirschorn's normalization of distribtuion of relaxation times
    Ref.:
        - G.J.Brug et al., J.Electroanal. Chem. Interfacial Electrochem., 176, 275 (1984)
        - B.Hirschorn, et al., ElectrochimicaActa, 55, 6218 (2010)
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    
    Inputs
    ----------
    n = Exponent of CPE [-]
    Rs = Series Resistance [ohm]
    Rct = Charge Transfer Resistance [ohm]
    
    Optional Inputs
    ------------------
    Q = Constant phase element [s^n/ohm]
    fs = summit frequencey of fitted spectra [Hz]
    
    Returns
    ----------
    C_eff = normalized capacitance for a faradaic electrode [s/ohm = F]
    '''
    if Q == 'none':
        if L == 'none':
            Q = (1/(Rct*(2*np.pi*fs)**n))
            C_eff = Q**(1/n) * ((Rs*Rct)/(Rs*Rct))**((1-n)/n)
        if L != 'none':
            Rct_norm = Rct / L
            Q = (1/(Rct_norm*(2*np.pi*fs)**n))
            C_eff = Q**(1/n) * ((Rs*Rct_norm)/(Rs*Rct_norm))**((1-n)/n)
    if fs == 'none':
        C_eff = Q**(1/n) * ((Rs*Rct)/(Rs*Rct))**((1-n)/n)
    return C_eff

def Theta(E, E0, n, T=298.15, F=F, R=R):
    '''
    See explantion in C_redox_Estep_semiinfinite()
    Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    '''
    return np.exp(((n*F)/(R*T))*(E-E0))

def Varsigma(D_ox, D_red):
    '''
    See explantion in C_redox_Estep_semiinfinite()
    Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    '''
    return (D_ox/D_red)**(1/2)

def C_redox_Estep_semiinfinite(E, E0, n, C_ox, D_ox, D_red, T=298.15, R=R, F=F):
    '''
    The concentration at the electrode surface (x=0) as a function of potential following Nernst eq. 
    during semi-infinite linear diffusion (Macro disk electrode)
    
    O + ne- --> R
    
    Ref: Bard A.J., Faulkner L. R., ISBN: 0-471-04372-9 (2001) "Electrochemical methods: Fundamentals and applications". New York: Wiley.

    Author: Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    
    returns
    ----------
    [0] = C_red at x=0
    [1] = C_ox at x=0
    '''
    C_red0 = C_ox * (Varsigma(D_ox=D_ox, D_red=D_red)/(1+(Varsigma(D_ox=D_ox, D_red=D_red)*Theta(E=E, E0=E0, n=n, T=T, F=F, R=R))))
    C_ox0 = C_ox * ((Varsigma(D_ox=D_ox, D_red=D_red)*Theta(E=E, E0=E0, n=n, T=T, F=F, R=R))/(1+(Varsigma(D_ox=D_ox, D_red=D_red)*Theta(E=E, E0=E0, n=n, T=T, F=F, R=R))))
    return C_red0, C_ox0

def C_redox_Estep_semihemisperhical(E, E0, n, C_ox, D_ox, D_red, T=298.15, R=R, F=F):
    '''
    The concentration at the electrode surface (x=0) as a function of potential following Nernst eq. 
    during semi-infinite hemispherical diffusion (Micro disk electrode)

    O + ne- --> R
    
    Note: This equation applies only for a reversible system with rapid kinetics
    
    Ref: Bard A.J., Faulkner L. R., ISBN: 0-471-04372-9 (2001) "Electrochemical methods: Fundamentals and applications". New York: Wiley.

    Author: Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    
    returns
    ----------
    [0] = C_red at x=0
    [1] = C_ox at x=0
    '''
    C_red0 = C_ox * ((Varsigma(D_ox, D_red)**2)/(1+Varsigma(D_ox, D_red)**2 * Theta(E, E0, n, T, F, R)))
    C_ox0 = C_ox * (1- (1/(1+Varsigma(D_ox, D_red)**2 * Theta(E, E0, n, T, F, R))))
    return C_red0, C_ox0
#
#print()
#print('---> EIS Advanced Tools Loaded (v. 0.0.2 - 06/15/18)')