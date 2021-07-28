# -*- coding: utf-8 -*-
'''
20181022: used for caldulate time ; ok
20190917: for data in aishell format , get teco2dur file from sgement sile( not use librosa to read wav).
        eg:: 122cc4bd-4038-4e56-ad7d-d32c0eb5ecf1-customerA_0005224-0005882 6.56
        note:   " " between utt and timeLen is "space".
    

'''
import sys
import os
import shutil
import codecs
import re
import librosa
import numpy as np
import subprocess
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
# wrong -2018016-not ok ;
def traveseFileFmt3(srcDir,frmStr,fileList): # srcDir = /wav   
    #print('start  traveseFileFmt ********** ')  
    for file1 in os.listdir(srcDir):           # file1 = wav/dev,test,train
        srcDir = os.path.join(srcDir, file1)  
        if os.path.isdir(srcDir):  
            fileList = traveseFileFmt2(srcDir,frmStr,fileList)  
            #listdir(srcDir,frmStr,fileList)
        else:
            if  srcDir[-4:]==frmStr:
                fileList.append(srcDir) 
                print(srcDir+'\n') 
    #rint('end  traveseFileFmt ********** ')   
    fileList.sort()                         
    return fileList  
    
'''def traveseFileFmt2(srcDir,frmStr): # srcDir = /wav
    print('start   traveseFileFmt ********** ')     
    fileList = []
    print('srcDir= ')
    print(srcDir)
    temS= srcDir[-1]
    if ~(temS=='/'):
        srcDir = srcDir+'/'
    for file1 in os.listdir(srcDir):           # file1 = wav/dev,test,train
        srcDir2 = srcDir+file1  # srcdir2 = wav/dev,test,train
        temS= srcDir2[-1]
        #print(srcDir4)
        if srcDir2[-4:]==frmStr:
            fileList.append(srcDir2)
    print('end  traveseFileFmt ********** ')                            
    return fileList   
''' 
     
def traveseFileFmt2(file_dir,frmStr,fileList):     
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == frmStr:  
                fileList.append(os.path.join(root, file))  
    fileList.sort()
    return fileList 
def removeALL(file_dir):     
    for root, dirs, files in os.walk(file_dir):  
        for file0 in files:  
            os.remove(os.path.join(root,file0))  
    os.rmdir(file_dir)
# 从spk+id的 文件名中分割除 spk 和 utt-ID
def spkIDsplit(instr):  
    liSTR=re.findall(r'\d+',instr)
    ID = liSTR[-1]
    num = len(ID)
    spkIDstr = instr[:-num]
    return spkIDstr,ID 
    
    
def getTimeLenSec(file0):
    
    y,sr = librosa.load(file0,sr=None)
    timeLen = librosa.get_duration(y,sr)

    return timeLen

    
if __name__=="__main__":
    # s1: check paras;
    print(sys.argv[0])
    if len(sys.argv) != 3:
        print(len(sys.argv))
        print('usage: '+ sys.argv[0] +' segmentsin reco2durWAVSCP')
        sys.exit()
    segmentsin = sys.argv[1]
    reco2durWAVSCP = sys.argv[2]
    foldPrepare(reco2durWAVSCP)
    
    strcmd = 'cat '+segmentsin+" | cut -d ' ' -f 2,4 > "+reco2durWAVSCP
    print(strcmd)
    aa = subprocess.check_output(strcmd, shell=True)
    print(aa)
    if aa ==b'':
        print('end ok.')
    else:
        print(' err end.')
    tarName=os.path.join(os.path.split(reco2durWAVSCP)[0],"reco2dur")
    shutil.copyfile(reco2durWAVSCP,tarName)
    sys.exit()
    #foldPrepare(dstFile)
    # s2: travese file through dir; get file list ; make parent dir and file list;
    #fileList = traveseFileFmt(srcDir,'.txt')
    
    f = codecs.open(wavSCPin,'r+','utf-8')
    fOut = codecs.open(reco2durWAVSCP,'w+','utf-8')
    
    line = f.readline()
    '14A3862_20170831131035 /c/data/0-ASR-corpus/cor-open-16k/2-open/2-open/wav/train/14A3862/14A3862_20170831131035.wav'
    cnt = 0
    while True:
        if not line:
            break
        else:
            res = line.split(' ')
            name = res[0]
            fullPath  = res[1][:-1]
            timeLen = getTimeLenSec(fullPath)
            cur_str = name + ' ' + str(timeLen)+'\n'
            fOut.write(cur_str)
            line = f.readline()
            cnt = cnt + 1
            if np.mod(cnt,100) == 0:
                print('cur line ='+str(cnt))
    f.close()
    fOut.close()
    print(" end ok .")









