# -*- coding: utf-8 -*-
import sys
import os
import shutil
import codecs
import re
import wave
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
    
def traveseFileSpec(file_dir,fileName,fileList):     
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.split(file)[1] == fileName:  
                fileList.append(os.path.join(root, file))  
    fileList.sort()
    return fileList  
    
  
if __name__=="__main__":
    # s1: check paras;
    print(sys.argv[0])
    if len(sys.argv) != 4:
        print(len(sys.argv))
        print('usage:'+sys.argv[0]+' srcFile Fs wavFile')
        sys.exit()
    srcFile = sys.argv[1]
    Fs = int(sys.argv[2])
    wavFile = sys.argv[3]
    foldPrepare(wavFile)

    f0 = open(srcFile,'rb')
    str_data = f0.read()
    f0.close()

    wave_out = wave.open(wavFile,'wb')
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(Fs)
    wave_out.writeframes(str_data)
    wave_out.close()


    print(' ok end')


