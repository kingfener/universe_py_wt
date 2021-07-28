# -*- coding: utf-8 -*-
'''
# used for audio re-SPK-name
20181017:
# 20190705: used for data rename.

20190925:
    1, back text-seg to content of ori-text .  because um is split one by one.




'''
import sys
import os
import shutil
import codecs
import numpy as np
import re
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
# 注意： ’.‘号的替换需要转义符号
def strrepWT(string,orIs,tarS):
    strinfo = re.compile(orIs)
    strDst = strinfo.sub(tarS,string)
    return strDst
# wrong -2018016-not ok ;
def traveseFileFmt33(srcDir,frmStr,fileList): # srcDir = /wav   
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
def traveseFileFmt2(file_dir,frmStr,fileList):     
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == frmStr:  
                fileList.append(os.path.join(root, file))  
    fileList.sort()
    return fileList 
def traveseFileFmt3(file_dir,frmStr,fileList):     
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == frmStr:  
                fileList.append(os.path.join(root, file))  
    #fileList.sort()
    return fileList     
def removeALL(file_dir):
    for root, dirs, files in os.walk(file_dir):  
        for file0 in files:  
            os.remove(os.path.join(root,file0))
    for root, dirs, files in os.walk(file_dir):         
        for file1 in dirs:  
            str1 = os.path.join(root,file1)
            #print("str1 ="+str1)
            os.rmdir(str1)
                #pause()
    os.rmdir(file_dir)
    print("over ")
# 获得固定 长度的字符串。    
def getSetLenStr(tarLen,datain,padStr):
    if len(padStr) != 1:
        print('error occured in func:getSetLenStr(), padStrLen != 1')
        sys.exit() 
    if len(str(datain)) == tarLen:
        return str(datain)
    elif len(str(datain)) > tarLen:
        print('error occured in func:getSetLenStr(), srcLen > tarLen')
        sys.exit() 
    else:
        dataOutStr = str(datain)
        for i in np.arange(0,tarLen-len(str(datain))):
            dataOutStr = padStr+dataOutStr
        return  dataOutStr        
# not use
def writeUTTID():
    #f1 = codecs.open(os.getcwd()+r"/P6-SRC/spk-numID.txt",'w+','utf-8')
    f1 = codecs.open(os.getcwd()+r"/P6-SRC/utt-ID.txt",'w+','utf-8')
    for i in np.arange(1,10000):
        SS = getSetLenStr(4,i,"0")
        f1.write(SS)
        f1.write('\n')
    f1.close()
    return 0
    #sys.exit()
# not use
def writeSPKnumID():
    #f1 = codecs.open(os.getcwd()+r"/P6-SRC/spk-numID.txt",'w+','utf-8')
    f1 = codecs.open(os.getcwd()+r"/P6-SRC/spk-numID.txt",'w+','utf-8')
    for i in np.arange(1,1000):
        SS = getSetLenStr(3,i,"0")
        f1.write(SS)
        f1.write('\n')
    f1.close()
    return 0
# 用来生成  spk-numID.txt
def writeSpkName():
    #f1 = codecs.open(os.getcwd()+r"/P6-SRC/spk-numID.txt",'w+','utf-8')
    f1 = codecs.open(os.getcwd()+r"/P6-SRC/spk-Name.txt",'w+','utf-8')
    headName = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    print(headName)
    SSall = ''
    for hN in headName:
        for i in np.arange(1,10000):
            SS = getSetLenStr(4,i,"0")
            SS = hN+SS
            SSall = SSall+SS+'\n'
            #f1.write(SS)
            #f1.write('\n')
    SSall = SSall[:-1]        
    f1.write(SSall)
    f1.close()
    return 0    

    
    #sys.exit()
def spkIDsplit(instr):  
    liSTR=re.findall(r'\d+',instr)
    ID = liSTR[-1]
    num = len(ID)
    spkIDstr = instr[:-num]
    return spkIDstr,ID
'''
liSTR=re.findall(r'\\n+',spkList)

'''   
# 文件读为字典：
def readTxtAsDict(fullfilename):
    #fullfilename = "/home/user01/ASR/11-data/engCorpus/voxforge-data-Part/P6-SRC/spk-Exi-Dict.txt"
    dict_data = {}
    spkEXIList = []
    spkUttNumList = []
    with open(fullfilename, 'r') as df:
        for kv in [d.strip().split(' ') for d in df]:
            #print(kv)
            dict_data[kv[0]] = kv[1]
    #  以下遍历是 乱序的。
    for k in dict_data:
        spkEXIList.append(str(k))
        spkUttNumList.append(str(dict_data[k]))
    '''print("spkEXIList ="+str(spkEXIList))    
    print("spkUttNumList ="+str(spkUttNumList))    
    print("dict_data ="+str(dict_data))    
    '''   
    return spkEXIList,spkUttNumList,dict_data 
    
# 读取源txt ，重写内容编号
def reSPKcopy(srcFil2,TxtnewFullName1,filname,fiN):
# reSPKcopy(srcTEX,dstTex,ORIcONT,REPLACEcONT):
# 20180205: 除spkID外，其余转换为小写;标点符号替换为空格。
    f00 = codecs.open(srcFil2,'r+','utf-8')
    f01 = codecs.open(TxtnewFullName1,'w+','utf-8')
    
    spkListSTR = f00.read()
    
    strOK = strrepWT(spkListSTR,filname,fiN+" ")
    # 除spkID外，其余转换为小写。
    strOK11 = strOK.lower()
    strOK = strrepWT(strOK11,fiN.lower(),fiN.upper())
    # 标点符号替换
    strOK = strrepWT(strOK,","," ")
    # 注意： ’.‘号的替换需要转义符号
    strOK = strrepWT(strOK,"\.",' ')
    strOK = strrepWT(strOK,"\"",' ')
  
    f01.write(strOK)
    f00.close()
    f01.close()
    return strOK
# for voip corpus
def spkIDsplitPar(instr):  
    liSTR=re.findall(r'\d+',instr)
    ID = liSTR[-1]
    num = len(ID)
    #spkIDstr = instr[:-num]
    spkIDstr = instr[:len("jurcic-002-")]
    return spkIDstr,ID 
# for cv corpus
def spkIDsplitCV(instr):  
    liSTR=re.findall(r'\d+',instr)
    ID = liSTR[-1]
    num = len(ID)
    #spkIDstr = instr[:-num]
    spkIDstr = instr[:-4]
    return spkIDstr,ID 
    

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


if __name__=="__main__":
    print(sys.argv[0])
    if len(sys.argv) != 3:
        print(len(sys.argv))
        print('usage: '+ sys.argv[0] +' textIn nj')
        sys.exit()
        
    textIn = sys.argv[1]
    nj = int(sys.argv[2])

    # get parent directory
    path,fileName = os.path.split(textIn)
    NUm = py_wc(textIn)
    for ii in range(1,nj+1):
        foldPrepare(os.path.join(path,'split'+str(nj)+'/'+str(ii)))

    pserNum = int(NUm/nj)
    numList = []
    for ii in range(0,nj):
        numList.append(ii*pserNum)
    
    numList.append(NUm)
    
    #print(numList)
    for jj in range(1,nj+1):
        ii = jj-1
        resDir = os.path.join(path,'split'+str(nj)+'/'+str(jj))
        resTxt = os.path.join(resDir,fileName[:-4]+'-'+str(jj)+'.txt')
        
        strcmd = "sed -n "+str(numList[ii]+1)+','+str(numList[ii+1])+"p "+textIn+ ' > '+resTxt
        aa = subprocess.check_output(strcmd, shell=True)

    
    print(' end ok ')
# python3 p6_reNum.py /home/user01/ASR/11-data/engCorpus/voxWAV-TXT_P4 spkList_p5.txt /home/user01/ASR/11-data/engCorpus/voxWAV-TXT_p6_ok










 
