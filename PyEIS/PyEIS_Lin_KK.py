#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 4 18:23:35 2018

This script contains the core for the linear Kramer-Kronig analysis

@author: Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
"""
import numpy as np
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit

### Simulation Functions
##
#

def KK_RC2(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1]))

def KK_RC3(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2]))

def KK_RC4(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3]))

def KK_RC5(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4]))

def KK_RC6(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5]))

def KK_RC7(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6]))

def KK_RC8(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7]))

def KK_RC9(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8]))

def KK_RC10(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9]))

def KK_RC11(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10]))

def KK_RC12(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11]))

def KK_RC13(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12]))

def KK_RC14(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13]))

def KK_RC15(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14]))

def KK_RC16(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15]))

def KK_RC17(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16]))

def KK_RC18(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17]))

def KK_RC19(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18]))

def KK_RC20(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19]))

def KK_RC21(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20]))

def KK_RC22(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21]))

def KK_RC23(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22]))

def KK_RC24(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23]))

def KK_RC25(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24]))

def KK_RC26(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25]))

def KK_RC27(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26]))

def KK_RC28(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27]))

def KK_RC29(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28]))

def KK_RC30(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29]))

def KK_RC31(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30]))

def KK_RC32(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31]))

def KK_RC33(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32]))

def KK_RC34(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33]))

def KK_RC35(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))

def KK_RC36(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35]))

def KK_RC37(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36]))

def KK_RC38(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37]))

def KK_RC39(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38]))

def KK_RC40(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39]))

def KK_RC41(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40]))

def KK_RC42(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41]))

def KK_RC43(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42]))

def KK_RC44(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43]))

def KK_RC45(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44]))

def KK_RC46(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45]))

def KK_RC47(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46]))

def KK_RC48(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47]))

def KK_RC49(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48]))

def KK_RC50(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49]))

def KK_RC51(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50]))

def KK_RC52(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51]))

def KK_RC53(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52]))

def KK_RC54(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53]))

def KK_RC55(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54]))

def KK_RC56(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55]))

def KK_RC57(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56]))

def KK_RC58(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57]))


def KK_RC59(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58]))

def KK_RC60(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59]))

def KK_RC61(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60]))

def KK_RC62(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61]))

def KK_RC63(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62]))

def KK_RC64(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63]))

def KK_RC65(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64]))

def KK_RC66(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65]))

def KK_RC67(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66]))

def KK_RC68(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67]))

def KK_RC69(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68]))

def KK_RC70(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69]))

def KK_RC71(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70]))

def KK_RC72(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71]))

def KK_RC73(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72]))

def KK_RC74(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72])) + (R_values[73] /(1+w*1j*t_values[73]))

def KK_RC75(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72])) + (R_values[73] /(1+w*1j*t_values[73])) + (R_values[74] /(1+w*1j*t_values[74]))

def KK_RC76(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72])) + (R_values[73] /(1+w*1j*t_values[73])) + (R_values[74] /(1+w*1j*t_values[74])) + (R_values[75] /(1+w*1j*t_values[75]))

def KK_RC77(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72])) + (R_values[73] /(1+w*1j*t_values[73])) + (R_values[74] /(1+w*1j*t_values[74])) + (R_values[75] /(1+w*1j*t_values[75])) + (R_values[76] /(1+w*1j*t_values[76]))

def KK_RC78(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72])) + (R_values[73] /(1+w*1j*t_values[73])) + (R_values[74] /(1+w*1j*t_values[74])) + (R_values[75] /(1+w*1j*t_values[75])) + (R_values[76] /(1+w*1j*t_values[76])) + (R_values[77] /(1+w*1j*t_values[77]))

def KK_RC79(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72])) + (R_values[73] /(1+w*1j*t_values[73])) + (R_values[74] /(1+w*1j*t_values[74])) + (R_values[75] /(1+w*1j*t_values[75])) + (R_values[76] /(1+w*1j*t_values[76])) + (R_values[77] /(1+w*1j*t_values[77])) + (R_values[78] /(1+w*1j*t_values[78]))

def KK_RC80(w, Rs, R_values, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    return Rs + (R_values[0]/(1+w*1j*t_values[0])) + (R_values[1] /(1+w*1j*t_values[1])) + (R_values[2] /(1+w*1j*t_values[2])) + (R_values[3] /(1+w*1j*t_values[3])) + (R_values[4] /(1+w*1j*t_values[4])) + (R_values[5] /(1+w*1j*t_values[5])) + (R_values[6] /(1+w*1j*t_values[6])) + (R_values[7] /(1+w*1j*t_values[7])) + (R_values[8] /(1+w*1j*t_values[8])) + (R_values[9] /(1+w*1j*t_values[9])) + (R_values[10] /(1+w*1j*t_values[10])) + (R_values[11] /(1+w*1j*t_values[11])) + (R_values[12] /(1+w*1j*t_values[12])) + (R_values[13] /(1+w*1j*t_values[13])) + (R_values[14] /(1+w*1j*t_values[14])) + (R_values[15] /(1+w*1j*t_values[15])) + (R_values[16] /(1+w*1j*t_values[16])) + (R_values[17] /(1+w*1j*t_values[17])) + (R_values[18] /(1+w*1j*t_values[18])) + (R_values[19] /(1+w*1j*t_values[19])) + (R_values[20] /(1+w*1j*t_values[20])) + (R_values[21] /(1+w*1j*t_values[21])) + (R_values[22] /(1+w*1j*t_values[22])) + (R_values[23] /(1+w*1j*t_values[23])) + (R_values[24] /(1+w*1j*t_values[24])) + (R_values[25] /(1+w*1j*t_values[25])) + (R_values[26] /(1+w*1j*t_values[26])) + (R_values[27] /(1+w*1j*t_values[27])) + (R_values[28] /(1+w*1j*t_values[28])) + (R_values[29] /(1+w*1j*t_values[29])) + (R_values[30] /(1+w*1j*t_values[30])) + (R_values[31] /(1+w*1j*t_values[31])) + (R_values[32] /(1+w*1j*t_values[32])) + (R_values[33] /(1+w*1j*t_values[33])) + (R_values[34] /(1+w*1j*t_values[34]))  + (R_values[35] /(1+w*1j*t_values[35])) + (R_values[36] /(1+w*1j*t_values[36])) + (R_values[37] /(1+w*1j*t_values[37])) + (R_values[38] /(1+w*1j*t_values[38])) + (R_values[39] /(1+w*1j*t_values[39])) + (R_values[40] /(1+w*1j*t_values[40])) + (R_values[41] /(1+w*1j*t_values[41])) + (R_values[42] /(1+w*1j*t_values[42])) + (R_values[43] /(1+w*1j*t_values[43])) + (R_values[44] /(1+w*1j*t_values[44])) + (R_values[45] /(1+w*1j*t_values[45])) + (R_values[46] /(1+w*1j*t_values[46])) + (R_values[47] /(1+w*1j*t_values[47])) + (R_values[48] /(1+w*1j*t_values[48])) + (R_values[49] /(1+w*1j*t_values[49])) + (R_values[50] /(1+w*1j*t_values[50])) + (R_values[51] /(1+w*1j*t_values[51])) + (R_values[52] /(1+w*1j*t_values[52])) + (R_values[53] /(1+w*1j*t_values[53])) + (R_values[54] /(1+w*1j*t_values[54])) + (R_values[55] /(1+w*1j*t_values[55])) + (R_values[56] /(1+w*1j*t_values[56])) + (R_values[57] /(1+w*1j*t_values[57])) + (R_values[58] /(1+w*1j*t_values[58])) + (R_values[59] /(1+w*1j*t_values[59])) + (R_values[60] /(1+w*1j*t_values[60])) + (R_values[61] /(1+w*1j*t_values[61])) + (R_values[62] /(1+w*1j*t_values[62])) + (R_values[63] /(1+w*1j*t_values[63])) + (R_values[64] /(1+w*1j*t_values[64])) + (R_values[65] /(1+w*1j*t_values[65])) + (R_values[66] /(1+w*1j*t_values[66])) + (R_values[67] /(1+w*1j*t_values[67])) + (R_values[68] /(1+w*1j*t_values[68])) + (R_values[69] /(1+w*1j*t_values[69])) + (R_values[70] /(1+w*1j*t_values[70])) + (R_values[71] /(1+w*1j*t_values[71])) + (R_values[72] /(1+w*1j*t_values[72])) + (R_values[73] /(1+w*1j*t_values[73])) + (R_values[74] /(1+w*1j*t_values[74])) + (R_values[75] /(1+w*1j*t_values[75])) + (R_values[76] /(1+w*1j*t_values[76])) + (R_values[77] /(1+w*1j*t_values[77])) + (R_values[78] /(1+w*1j*t_values[78])) + (R_values[79] /(1+w*1j*t_values[79]))

### Fitting Functions
##
#

def KK_RC2_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) 

def KK_RC3_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2]))

def KK_RC4_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3]))

def KK_RC5_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4]))

def KK_RC6_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5]))

def KK_RC7_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6]))

def KK_RC8_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7]))

def KK_RC9_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8]))

def KK_RC10_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9]))

def KK_RC11_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10]))

def KK_RC12_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11]))

def KK_RC13_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12]))

def KK_RC14_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13]))

def KK_RC15_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13]))

def KK_RC15_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14]))

def KK_RC16_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15]))

def KK_RC17_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16]))

def KK_RC18_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17]))

def KK_RC19_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18]))

def KK_RC20_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19]))

def KK_RC21_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20]))

def KK_RC22_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21]))

def KK_RC23_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22]))

def KK_RC24_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23]))

def KK_RC25_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24]))

def KK_RC26_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25]))

def KK_RC27_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26]))

def KK_RC28_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27]))

def KK_RC29_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28]))

def KK_RC30_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29]))

def KK_RC31_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30]))

def KK_RC32_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31]))

def KK_RC33_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32]))

def KK_RC34_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33]))

def KK_RC35_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))

def KK_RC36_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35]))

def KK_RC37_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36]))

def KK_RC38_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37]))

def KK_RC39_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38]))

def KK_RC40_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39]))

def KK_RC41_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40]))

def KK_RC42_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41]))

def KK_RC43_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42]))

def KK_RC44_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43]))

def KK_RC45_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44]))

def KK_RC46_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45]))

def KK_RC47_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46]))

def KK_RC48_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47]))

def KK_RC49_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48]))

def KK_RC50_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49]))

def KK_RC51_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50]))

def KK_RC52_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51]))

def KK_RC53_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52]))

def KK_RC54_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53]))

def KK_RC55_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54]))

def KK_RC56_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55]))

def KK_RC57_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56]))

def KK_RC58_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57]))

def KK_RC59_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58]))

def KK_RC60_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59]))

def KK_RC61_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60]))

def KK_RC62_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61]))

def KK_RC63_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62]))    

def KK_RC64_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63]))    

def KK_RC65_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64]))

def KK_RC66_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65]))

def KK_RC67_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66]))    

def KK_RC68_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67]))

def KK_RC69_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68]))

def KK_RC70_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69]))    

def KK_RC71_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70]))    

def KK_RC72_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71]))

def KK_RC73_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72]))

def KK_RC74_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    R74 = params['R74']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72])) + (R74 /(1+w*1j*t_values[73]))

def KK_RC75_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    R74 = params['R74']
    R75 = params['R75']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72])) + (R74 /(1+w*1j*t_values[73])) + (R75 /(1+w*1j*t_values[74]))

def KK_RC76_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    R74 = params['R74']
    R75 = params['R75']
    R76 = params['R76']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72])) + (R74 /(1+w*1j*t_values[73])) + (R75 /(1+w*1j*t_values[74])) + (R76 /(1+w*1j*t_values[75]))

def KK_RC77_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    R74 = params['R74']
    R75 = params['R75']
    R76 = params['R76']
    R77 = params['R77']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72])) + (R74 /(1+w*1j*t_values[73])) + (R75 /(1+w*1j*t_values[74])) + (R76 /(1+w*1j*t_values[75])) + (R77 /(1+w*1j*t_values[76]))

def KK_RC78_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    R74 = params['R74']
    R75 = params['R75']
    R76 = params['R76']
    R77 = params['R77']
    R78 = params['R78']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72])) + (R74 /(1+w*1j*t_values[73])) + (R75 /(1+w*1j*t_values[74])) + (R76 /(1+w*1j*t_values[75])) + (R77 /(1+w*1j*t_values[76])) + (R78 /(1+w*1j*t_values[77]))

def KK_RC79_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    R74 = params['R74']
    R75 = params['R75']
    R76 = params['R76']
    R77 = params['R77']
    R78 = params['R78']
    R79 = params['R79']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72])) + (R74 /(1+w*1j*t_values[73])) + (R75 /(1+w*1j*t_values[74])) + (R76 /(1+w*1j*t_values[75])) + (R77 /(1+w*1j*t_values[76])) + (R78 /(1+w*1j*t_values[77])) + (R79 /(1+w*1j*t_values[78]))

def KK_RC80_fit(params, w, t_values):
    '''
    Kramers-Kronig Function: -RC-
    
    Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
    '''
    Rs = params['Rs']
    R1 = params['R1']
    R2 = params['R2']
    R3 = params['R3']
    R4 = params['R4']
    R5 = params['R5']
    R6 = params['R6']
    R7 = params['R7']
    R8 = params['R8']
    R9 = params['R9']
    R10 = params['R10']
    R11 = params['R11']
    R12 = params['R12']
    R13 = params['R13']
    R14 = params['R14']
    R15 = params['R15']
    R16 = params['R16']
    R17 = params['R17']
    R18 = params['R18']
    R19 = params['R19']
    R20 = params['R20']
    R21 = params['R21']
    R22 = params['R22']
    R23 = params['R23']
    R24 = params['R24']
    R25 = params['R25']
    R26 = params['R26']
    R27 = params['R27']
    R28 = params['R28']
    R29 = params['R29']
    R30 = params['R30']
    R31 = params['R31']
    R32 = params['R32']
    R33 = params['R33']
    R34 = params['R34']
    R35 = params['R35']
    R36 = params['R36']
    R37 = params['R37']
    R38 = params['R38']
    R39 = params['R39']
    R40 = params['R40']
    R41 = params['R41']
    R42 = params['R42']
    R43 = params['R43']
    R44 = params['R44']
    R45 = params['R45']
    R46 = params['R46']
    R47 = params['R47']
    R48 = params['R48']
    R49 = params['R49']
    R50 = params['R50']
    R51 = params['R51']
    R52 = params['R52']
    R53 = params['R53']
    R54 = params['R54']
    R55 = params['R55']
    R56 = params['R56']
    R57 = params['R57']
    R58 = params['R58']
    R59 = params['R59']
    R60 = params['R60']
    R61 = params['R61']
    R62 = params['R62']
    R63 = params['R63']
    R64 = params['R64']
    R65 = params['R65']
    R66 = params['R66']
    R67 = params['R67']
    R68 = params['R68']
    R69 = params['R69']
    R70 = params['R70']
    R71 = params['R71']
    R72 = params['R72']
    R73 = params['R73']
    R74 = params['R74']
    R75 = params['R75']
    R76 = params['R76']
    R77 = params['R77']
    R78 = params['R78']
    R79 = params['R79']
    R80 = params['R80']
    return Rs + (R1/(1+w*1j*t_values[0])) + (R2 /(1+w*1j*t_values[1])) + (R3 /(1+w*1j*t_values[2])) + (R4 /(1+w*1j*t_values[3])) + (R5 /(1+w*1j*t_values[4])) + (R6 /(1+w*1j*t_values[5])) + (R7 /(1+w*1j*t_values[6])) + (R8 /(1+w*1j*t_values[7])) + (R9 /(1+w*1j*t_values[8])) + (R10 /(1+w*1j*t_values[9])) + (R11 /(1+w*1j*t_values[10])) + (R12 /(1+w*1j*t_values[11])) + (R13 /(1+w*1j*t_values[12])) + (R14 /(1+w*1j*t_values[13])) + (R15 /(1+w*1j*t_values[14])) + (R16 /(1+w*1j*t_values[15])) + (R17 /(1+w*1j*t_values[16])) + (R18 /(1+w*1j*t_values[17])) + (R19 /(1+w*1j*t_values[18])) + (R20 /(1+w*1j*t_values[19])) + (R21 /(1+w*1j*t_values[20])) + (R22 /(1+w*1j*t_values[21])) + (R23 /(1+w*1j*t_values[22])) + (R24 /(1+w*1j*t_values[23])) + (R25 /(1+w*1j*t_values[24])) + (R26 /(1+w*1j*t_values[25])) + (R27 /(1+w*1j*t_values[26])) + (R28 /(1+w*1j*t_values[27])) + (R29 /(1+w*1j*t_values[28])) + (R30 /(1+w*1j*t_values[29])) + (R31 /(1+w*1j*t_values[30])) + (R32 /(1+w*1j*t_values[31])) + (R33 /(1+w*1j*t_values[32])) + (R34 /(1+w*1j*t_values[33])) + (R35 /(1+w*1j*t_values[34]))  + (R36 /(1+w*1j*t_values[35])) + (R37 /(1+w*1j*t_values[36])) + (R38 /(1+w*1j*t_values[37])) + (R39 /(1+w*1j*t_values[38])) + (R40 /(1+w*1j*t_values[39])) + (R41 /(1+w*1j*t_values[40])) + (R42 /(1+w*1j*t_values[41])) + (R43 /(1+w*1j*t_values[42])) + (R44 /(1+w*1j*t_values[43])) + (R45 /(1+w*1j*t_values[44])) + (R46 /(1+w*1j*t_values[45])) + (R47 /(1+w*1j*t_values[46])) + (R48 /(1+w*1j*t_values[47])) + (R49 /(1+w*1j*t_values[48])) + (R50 /(1+w*1j*t_values[49])) + (R51 /(1+w*1j*t_values[50])) + (R52 /(1+w*1j*t_values[51])) + (R53 /(1+w*1j*t_values[52])) + (R54 /(1+w*1j*t_values[53])) + (R55 /(1+w*1j*t_values[54])) + (R56 /(1+w*1j*t_values[55])) + (R57 /(1+w*1j*t_values[56])) + (R58 /(1+w*1j*t_values[57])) + (R59 /(1+w*1j*t_values[58])) + (R60 /(1+w*1j*t_values[59])) + (R61 /(1+w*1j*t_values[60])) + (R62 /(1+w*1j*t_values[61])) + (R63 /(1+w*1j*t_values[62])) + (R64 /(1+w*1j*t_values[63])) + (R65 /(1+w*1j*t_values[64])) + (R66 /(1+w*1j*t_values[65])) + (R67 /(1+w*1j*t_values[66])) + (R68 /(1+w*1j*t_values[67])) + (R69 /(1+w*1j*t_values[68])) + (R70 /(1+w*1j*t_values[69])) + (R71 /(1+w*1j*t_values[70])) + (R72 /(1+w*1j*t_values[71])) + (R73 /(1+w*1j*t_values[72])) + (R74 /(1+w*1j*t_values[73])) + (R75 /(1+w*1j*t_values[74])) + (R76 /(1+w*1j*t_values[75])) + (R77 /(1+w*1j*t_values[76])) + (R78 /(1+w*1j*t_values[77])) + (R79 /(1+w*1j*t_values[78])) + (R80 /(1+w*1j*t_values[79]))

### Least-squres function and related functions
##
#
def KK_Rnam_val(re, re_start, num_RC):
    '''
    This function determines the name and initial guesses for resistances for the Linear KK test
    
    Ref.:
        - Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
        - Boukamp, B.A. J. Electrochem. Soc., 142, 6, 1885-1894 
        
    Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    
    Inputs
    -----------
    w = angular frequency
    num_RC = number of -(RC)- circuits
    
    Outputs
    -----------
    [0] = parameters for LMfit
    [1] = R_names
    [2] = number of R in each fit
    '''
    num_RC = np.arange(1,num_RC+1,1)

    R_name = []
    R_initial = []
    for j in range(len(num_RC)):
        R_name.append('R'+str(num_RC[j]))
        R_initial.append(1) #initial guess for Resistances

    params = Parameters()
    for j in range(len(num_RC)):
        params.add(R_name[j], value=R_initial[j])

    params.add('Rs', value=re[re_start], min=-10**5, max=10**5)
    return params, R_name, num_RC

def KK_timeconst(w, num_RC):
    '''
    This function determines the initial guesses for time constants for the Linear KK test
    
    Ref.:
        - Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
        
    Kristian B. Knudsen (kknu@berkeley.edu || Kristianbknudsen@gmail.com)
    '''
    num_RC = np.arange(1,num_RC+1,1)

    t_max = 1/min(w)
    t_min = 1/max(w)
    t_name = []
    t_initial = []
    for j in range(len(num_RC)):
        t_name.append('t'+str(num_RC[j]))
        t_initial.append(10**((np.log10(t_min)) + (j-1)/(len(num_RC)-1) * np.log10(t_max/t_min)) ) #initial guess parameter parameter tau for each -RC- circuit
    return t_initial


#from scipy.optimize import curve_fit

def KK_errorfunc_2(w, re, im, num_RC, t_values):
    '''
    test
    '''
    if num_RC == 51:
        re_fit = curve_fit(lambda w, Rs, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48, R49, R50, R51: KK_RC51_fit(w, Rs, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48, R49, R50, R51, t_values), w, re)[0]
#        im_fit = curve_fit(lambda w, Rs, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48, R49, R50, R51: KK_RC51_fit(w, Rs, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48, R49, R50, R51, t_values)[1], w, im)[0]
    else:
        print('error in this crap')
    return re_fit#, im_fit


def KK_errorfunc(params, w, re, im, num_RC, weight_func, t_values):
    '''
    Sum of squares error function for linear least-squares fitting for the Kramers-Kronig Relations. 
    The fitting function will use this function to iterate over until the return the sum of errors is minimized
    
    The data should be minimized using the weight_func = 'Boukamp'
    
    Ref.: Boukamp, B.A. J. Electrochem. Soc., 142, 6, 1885-1894 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    ------------
        - w = angular frequency
        - re = real impedance
        - im = imaginary impedance
        - num_RC = number of RC-circuits
        - t_values = time constants
        
        weight_func = Weight function, Three options:
            - modulus
            - unity
            - proportional
            - Boukamp
    '''
    if num_RC == 2:
        re_fit = KK_RC2_fit(params, w, t_values).real
        im_fit = -KK_RC2_fit(params, w, t_values).imag
    elif num_RC == 3:
        re_fit = KK_RC3_fit(params, w, t_values).real
        im_fit = -KK_RC3_fit(params, w, t_values).imag
    elif num_RC == 4:
        re_fit = KK_RC4_fit(params, w, t_values).real
        im_fit = -KK_RC4_fit(params, w, t_values).imag
    elif num_RC == 5:
        re_fit = KK_RC5_fit(params, w, t_values).real
        im_fit = -KK_RC5_fit(params, w, t_values).imag
    elif num_RC == 6:
        re_fit = KK_RC6_fit(params, w, t_values).real
        im_fit = -KK_RC6_fit(params, w, t_values).imag
    elif num_RC == 7:
        re_fit = KK_RC7_fit(params, w, t_values).real
        im_fit = -KK_RC7_fit(params, w, t_values).imag
    elif num_RC == 8:
        re_fit = KK_RC8_fit(params, w, t_values).real
        im_fit = -KK_RC8_fit(params, w, t_values).imag
    elif num_RC == 9:
        re_fit = KK_RC9_fit(params, w, t_values).real
        im_fit = -KK_RC9_fit(params, w, t_values).imag
    elif num_RC == 10:
        re_fit = KK_RC10_fit(params, w, t_values).real
        im_fit = -KK_RC10_fit(params, w, t_values).imag
    elif num_RC == 11:
        re_fit = KK_RC11_fit(params, w, t_values).real
        im_fit = -KK_RC11_fit(params, w, t_values).imag
    elif num_RC == 12:
        re_fit = KK_RC12_fit(params, w, t_values).real
        im_fit = -KK_RC12_fit(params, w, t_values).imag
    elif num_RC == 13:
        re_fit = KK_RC13_fit(params, w, t_values).real
        im_fit = -KK_RC13_fit(params, w, t_values).imag
    elif num_RC == 14:
        re_fit = KK_RC14_fit(params, w, t_values).real
        im_fit = -KK_RC14_fit(params, w, t_values).imag
    elif num_RC == 15:
        re_fit = KK_RC15_fit(params, w, t_values).real
        im_fit = -KK_RC15_fit(params, w, t_values).imag
    elif num_RC == 16:
        re_fit = KK_RC16_fit(params, w, t_values).real
        im_fit = -KK_RC16_fit(params, w, t_values).imag
    elif num_RC == 17:
        re_fit = KK_RC17_fit(params, w, t_values).real
        im_fit = -KK_RC17_fit(params, w, t_values).imag
    elif num_RC == 18:
        re_fit = KK_RC18_fit(params, w, t_values).real
        im_fit = -KK_RC18_fit(params, w, t_values).imag
    elif num_RC == 19:
        re_fit = KK_RC19_fit(params, w, t_values).real
        im_fit = -KK_RC19_fit(params, w, t_values).imag
    elif num_RC == 20:
        re_fit = KK_RC20_fit(params, w, t_values).real
        im_fit = -KK_RC20_fit(params, w, t_values).imag
    elif num_RC == 21:
        re_fit = KK_RC21_fit(params, w, t_values).real
        im_fit = -KK_RC21_fit(params, w, t_values).imag
    elif num_RC == 22:
        re_fit = KK_RC22_fit(params, w, t_values).real
        im_fit = -KK_RC22_fit(params, w, t_values).imag
    elif num_RC == 23:
        re_fit = KK_RC23_fit(params, w, t_values).real
        im_fit = -KK_RC23_fit(params, w, t_values).imag
    elif num_RC == 24:
        re_fit = KK_RC24_fit(params, w, t_values).real
        im_fit = -KK_RC24_fit(params, w, t_values).imag
    elif num_RC == 25:
        re_fit = KK_RC25_fit(params, w, t_values).real
        im_fit = -KK_RC25_fit(params, w, t_values).imag
    elif num_RC == 26:
        re_fit = KK_RC26_fit(params, w, t_values).real
        im_fit = -KK_RC26_fit(params, w, t_values).imag
    elif num_RC == 27:
        re_fit = KK_RC27_fit(params, w, t_values).real
        im_fit = -KK_RC27_fit(params, w, t_values).imag
    elif num_RC == 28:
        re_fit = KK_RC28_fit(params, w, t_values).real
        im_fit = -KK_RC28_fit(params, w, t_values).imag
    elif num_RC == 29:
        re_fit = KK_RC29_fit(params, w, t_values).real
        im_fit = -KK_RC29_fit(params, w, t_values).imag
    elif num_RC == 30:
        re_fit = KK_RC30_fit(params, w, t_values).real
        im_fit = -KK_RC30_fit(params, w, t_values).imag
    elif num_RC == 31:
        re_fit = KK_RC31_fit(params, w, t_values).real
        im_fit = -KK_RC31_fit(params, w, t_values).imag
    elif num_RC == 32:
        re_fit = KK_RC32_fit(params, w, t_values).real
        im_fit = -KK_RC32_fit(params, w, t_values).imag
    elif num_RC == 33:
        re_fit = KK_RC33_fit(params, w, t_values).real
        im_fit = -KK_RC33_fit(params, w, t_values).imag
    elif num_RC == 34:
        re_fit = KK_RC34_fit(params, w, t_values).real
        im_fit = -KK_RC34_fit(params, w, t_values).imag
    elif num_RC == 35:
        re_fit = KK_RC35_fit(params, w, t_values).real
        im_fit = -KK_RC35_fit(params, w, t_values).imag
    elif num_RC == 36:
        re_fit = KK_RC36_fit(params, w, t_values).real
        im_fit = -KK_RC36_fit(params, w, t_values).imag
    elif num_RC == 37:
        re_fit = KK_RC37_fit(params, w, t_values).real
        im_fit = -KK_RC37_fit(params, w, t_values).imag
    elif num_RC == 38:
        re_fit = KK_RC38_fit(params, w, t_values).real
        im_fit = -KK_RC38_fit(params, w, t_values).imag
    elif num_RC == 39:
        re_fit = KK_RC39_fit(params, w, t_values).real
        im_fit = -KK_RC39_fit(params, w, t_values).imag
    elif num_RC == 40:
        re_fit = KK_RC40_fit(params, w, t_values).real
        im_fit = -KK_RC40_fit(params, w, t_values).imag
    elif num_RC == 41:
        re_fit = KK_RC41_fit(params, w, t_values).real
        im_fit = -KK_RC41_fit(params, w, t_values).imag
    elif num_RC == 42:
        re_fit = KK_RC42_fit(params, w, t_values).real
        im_fit = -KK_RC42_fit(params, w, t_values).imag
    elif num_RC == 43:
        re_fit = KK_RC43_fit(params, w, t_values).real
        im_fit = -KK_RC43_fit(params, w, t_values).imag
    elif num_RC == 44:
        re_fit = KK_RC44_fit(params, w, t_values).real
        im_fit = -KK_RC44_fit(params, w, t_values).imag
    elif num_RC == 45:
        re_fit = KK_RC45_fit(params, w, t_values).real
        im_fit = -KK_RC45_fit(params, w, t_values).imag
    elif num_RC == 46:
        re_fit = KK_RC46_fit(params, w, t_values).real
        im_fit = -KK_RC46_fit(params, w, t_values).imag
    elif num_RC == 47:
        re_fit = KK_RC47_fit(params, w, t_values).real
        im_fit = -KK_RC47_fit(params, w, t_values).imag
    elif num_RC == 48:
        re_fit = KK_RC48_fit(params, w, t_values).real
        im_fit = -KK_RC48_fit(params, w, t_values).imag
    elif num_RC == 49:
        re_fit = KK_RC49_fit(params, w, t_values).real
        im_fit = -KK_RC49_fit(params, w, t_values).imag
    elif num_RC == 50:
        re_fit = KK_RC50_fit(params, w, t_values).real
        im_fit = -KK_RC50_fit(params, w, t_values).imag
    elif num_RC == 51:
        re_fit = KK_RC51_fit(params, w, t_values).real
        im_fit = -KK_RC51_fit(params, w, t_values).imag
    elif num_RC == 52:
        re_fit = KK_RC52_fit(params, w, t_values).real
        im_fit = -KK_RC52_fit(params, w, t_values).imag
    elif num_RC == 53:
        re_fit = KK_RC53_fit(params, w, t_values).real
        im_fit = -KK_RC53_fit(params, w, t_values).imag
    elif num_RC == 54:
        re_fit = KK_RC54_fit(params, w, t_values).real
        im_fit = -KK_RC54_fit(params, w, t_values).imag
    elif num_RC == 55:
        re_fit = KK_RC55_fit(params, w, t_values).real
        im_fit = -KK_RC55_fit(params, w, t_values).imag
    elif num_RC == 56:
        re_fit = KK_RC56_fit(params, w, t_values).real
        im_fit = -KK_RC56_fit(params, w, t_values).imag
    elif num_RC == 57:
        re_fit = KK_RC57_fit(params, w, t_values).real
        im_fit = -KK_RC57_fit(params, w, t_values).imag
    elif num_RC == 58:
        re_fit = KK_RC58_fit(params, w, t_values).real
        im_fit = -KK_RC58_fit(params, w, t_values).imag
    elif num_RC == 59:
        re_fit = KK_RC59_fit(params, w, t_values).real
        im_fit = -KK_RC59_fit(params, w, t_values).imag
    elif num_RC == 60:
        re_fit = KK_RC60_fit(params, w, t_values).real
        im_fit = -KK_RC60_fit(params, w, t_values).imag
    elif num_RC == 61:
        re_fit = KK_RC61_fit(params, w, t_values).real
        im_fit = -KK_RC61_fit(params, w, t_values).imag
    elif num_RC == 62:
        re_fit = KK_RC62_fit(params, w, t_values).real
        im_fit = -KK_RC62_fit(params, w, t_values).imag
    elif num_RC == 63:
        re_fit = KK_RC63_fit(params, w, t_values).real
        im_fit = -KK_RC63_fit(params, w, t_values).imag
    elif num_RC == 64:
        re_fit = KK_RC64_fit(params, w, t_values).real
        im_fit = -KK_RC64_fit(params, w, t_values).imag
    elif num_RC == 65:
        re_fit = KK_RC65_fit(params, w, t_values).real
        im_fit = -KK_RC65_fit(params, w, t_values).imag
    elif num_RC == 66:
        re_fit = KK_RC66_fit(params, w, t_values).real
        im_fit = -KK_RC66_fit(params, w, t_values).imag
    elif num_RC == 67:
        re_fit = KK_RC67_fit(params, w, t_values).real
        im_fit = -KK_RC67_fit(params, w, t_values).imag
    elif num_RC == 68:
        re_fit = KK_RC68_fit(params, w, t_values).real
        im_fit = -KK_RC68_fit(params, w, t_values).imag
    elif num_RC == 69:
        re_fit = KK_RC69_fit(params, w, t_values).real
        im_fit = -KK_RC69_fit(params, w, t_values).imag
    elif num_RC == 70:
        re_fit = KK_RC70_fit(params, w, t_values).real
        im_fit = -KK_RC70_fit(params, w, t_values).imag
    elif num_RC == 71:
        re_fit = KK_RC71_fit(params, w, t_values).real
        im_fit = -KK_RC71_fit(params, w, t_values).imag
    elif num_RC == 72:
        re_fit = KK_RC72_fit(params, w, t_values).real
        im_fit = -KK_RC72_fit(params, w, t_values).imag
    elif num_RC == 73:
        re_fit = KK_RC73_fit(params, w, t_values).real
        im_fit = -KK_RC73_fit(params, w, t_values).imag
    elif num_RC == 74:
        re_fit = KK_RC74_fit(params, w, t_values).real
        im_fit = -KK_RC74_fit(params, w, t_values).imag
    elif num_RC == 75:
        re_fit = KK_RC75_fit(params, w, t_values).real
        im_fit = -KK_RC75_fit(params, w, t_values).imag
    elif num_RC == 76:
        re_fit = KK_RC76_fit(params, w, t_values).real
        im_fit = -KK_RC76_fit(params, w, t_values).imag
    elif num_RC == 77:
        re_fit = KK_RC77_fit(params, w, t_values).real
        im_fit = -KK_RC77_fit(params, w, t_values).imag
    elif num_RC == 78:
        re_fit = KK_RC78_fit(params, w, t_values).real
        im_fit = -KK_RC78_fit(params, w, t_values).imag
    elif num_RC == 79:
        re_fit = KK_RC79_fit(params, w, t_values).real
        im_fit = -KK_RC79_fit(params, w, t_values).imag
    elif num_RC == 80:
        re_fit = KK_RC80_fit(params, w, t_values).real
        im_fit = -KK_RC80_fit(params, w, t_values).imag
    else:
        print('In KK_errorfunc() - Define num_RC')
    error = [(re-re_fit)**2, (im-im_fit)**2] #sum of squares
    
    if weight_func == 'modulus':
        weight = [1/((re_fit**2 + im_fit**2)**(1/2)), 1/((re_fit**2 + im_fit**2)**(1/2))]
    elif weight_func == 'proportional':
        weight = [1/(re_fit**2), 1/(im_fit**2)]
    elif weight_func == 'unity':
        unity_1s = []
        for k in range(len(re)):
            unity_1s.append(1) #makes an array of [1]'s, so that the weighing is == 1 * sum of squares.
        weight = [unity_1s, unity_1s]
    elif weight_func == 'Boukamp':
        weight = [1/(re**2), 1/(im**2)]
    elif weight_func == 'ignore':
        print('weight ignored')
    S = np.array(weight) * error #weighted sum of squares 
    return S


### Functions for Evaluating Fits
##
#
def residual_real(re, fit_re, fit_im):
    '''
    Relative Residuals as based on Boukamp's definition

    Ref.:
        - Boukamp, B.A. J. Electrochem. SoC., 142, 6, 1885-1894 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    modulus_fit = (fit_re**2 + fit_im**2)**(1/2)
    return (re - fit_re) / modulus_fit

def residual_imag(im, fit_re, fit_im):
    '''
    Relative Residuals as based on Boukamp's  definition

    Ref.:
        - Boukamp, B.A. J. Electrochem. SoC., 142, 6, 1885-1894 
    
    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)
    '''
    modulus_fit = (fit_re**2 + fit_im**2)**(1/2)
    return (im - fit_im) / modulus_fit

#print()
#print('---> Linear Kramers-Kronig Script Loaded (v. 0.0.9 - 11/11/18)')