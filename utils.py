# coding=utf-8
'''
    工具类程序
    by @wangtao
'''

import os,re,sys,codecs,shutil

import subprocess
import copy
def py_grep(srcTxt, tarStr):
    strcmd = 'grep ' + tarStr + ' ' + srcTxt
    # aa = subprocess.check_output(strcmd,shell=True)
    try:
        aa = subprocess.check_output(strcmd, shell=True)
    # print('aa= '+ str(aa))
    except Exception:
        aa = ''
    return aa


def py_wc(srcTxt):
    strcmd = 'wc -l ' + srcTxt
    aa = subprocess.check_output(strcmd, shell=True)
    num, name = aa.split()
    # print('aa= '+ str(aa))
    return int(num)











