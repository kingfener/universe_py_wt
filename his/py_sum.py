# -*- coding: utf-8 -*-
# wt 20200310
# USAGE: 
# 获得　ｗａｖ时长  :464.00477944444447
#     tail dictV6_p25_add13_25v2_1911_200126/*-cmb/time_info.txt | grep "wav file time len insec" | cut -d ' ' -f 12
# 获得　有效时长：　　suｍV =266.6076999853115
#     tail dictV6_p25_add13_25v2_1911_200126/*-cmb/time_info.txt | grep "total useful time insec" | cut -d ' ' -f 9
# 获得　ｓｅｇ NUm：suｍV =362855
#     tail dictV6_p25_add13_25v2_1911_200126/*-cmb/time_info.txt | grep "useful seg or line num" | cut -d ' ' -f 8
# 

import sys
import os
import shutil
import codecs
# give file's full path name ,chek the fold exist,if not,make it.
def foldPrepare(fullfilename0):
    fullfilename = fullfilename0
    # if fullfilename0 have '.' ，then treat it a file.
    if fullfilename.find('.')>0:
        fullfilename,f = os.path.split(fullfilename)
    # then fullfilename is dir
    try: 
        os.makedirs(fullfilename)
    except:
        pass #print('mkdir failed.')

def strrepWT(string,orIs,tarS):
    strinfo = re.compile(orIs)
    strDst = strinfo.sub(tarS,string)
    return strDst

def traveseFileFmt2(file_dir,frmStr,fileList):     
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == frmStr:  
                fileList.append(os.path.join(root, file))  
    fileList.sort()
    return fileList 
def log(ID, val):
    print('-----> ' + str(ID) + ' =|' + str(val)+"|\n")
    # print('-----> '+str(ID)+'shape = '+str(val.shape) )


'''
获得　ｗａｖ时长  :464.00477944444447
    tail dictV6_p25_add13_25v2_1911_200126/*-cmb/time_info.txt | grep "wav file time len insec" | cut -d ' ' -f 12
获得　有效时长：　　suｍV =266.6076999853115
    tail dictV6_p25_add13_25v2_1911_200126/*-cmb/time_info.txt | grep "total useful time insec" | cut -d ' ' -f 9
获得　ｓｅｇ NUm：suｍV =362855
    tail dictV6_p25_add13_25v2_1911_200126/*-cmb/time_info.txt | grep "useful seg or line num" | cut -d ' ' -f 8
'''

if __name__=="__main__":

    # strIN="36.61763500000001  141.85822527777788  106.25620027777781  35.432879722222225  15.02388444444444  17.305102777777773  39.584869722222216  6.070021666666666  22.047835555555555  5.56378  3.108850277777778  31.572714722222226  3.5627800000000005"
    strIN='''10.663107689395694
87.40514296940364
65.73081743872099
17.055867908324387
7.9665868581988875
7.440939894807459
30.019730788807692
1.9279588942348158
15.42798326141269
3.948044569220173
2.540696094287579
15.34996622401857
1.1308573944789166'''

    arr = strIN.split('\n')
    suｍV=0
    i=1
    for ss in arr:
        log(i,ss)
        if len(ss.strip())>0:
            suｍV = suｍV + float(ss)
            i+=1
    print("suｍV ="+str(suｍV) )
#  python3 python3 clu_txt_im.py /home/user01/ASR/4-预处理-分析程序/Data-asr/py-score-V4/txt_prepare/chengYu-ok 3.txt












