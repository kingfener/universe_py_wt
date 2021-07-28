#coding=utf-8
'''

this code id for text related processing.

'''

import sys,os,time
import re,codecs,shutil

def symbol_filter(textStrin):
    ''' 
    delete unnormal symbol
    '''
    fmtSym = re.compile(r'[~-%]')

    textStrin = fmtSym.sub(' ',textStrin)
    
    return textStrin




def chi_eng_sym_check():
    '''  use for check input strs class '''

    pass

















