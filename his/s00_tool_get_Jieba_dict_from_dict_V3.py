# -*- coding: utf-8 -*-
'''
# used for audio re-SPK-name
20181017:
# 20190705: used for data rename.
20200108:
    1, 用来将 dict 生成 jieba 分词用的工具。
    2，添加 参看 频率数值的 文件输入。
'''
import sys
import os
import shutil
import codecs
#import numpy as np
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
  
def py_grep_eng(srcTxt, tarStr):
    strcmd = "grep '" + tarStr+ "' " + srcTxt
    # log(999,strcmd)
    aa = ''
    # sys.exit()
    # aa = subprocess.check_output(strcmd,shell=True)
    try:
        res = subprocess.check_output(strcmd, shell=True).decode('utf-8').strip()
        sss = res.split('\n')
        # log(777,res)
        # log(888,sss)
        nn = len(sss)
        if nn >=1:
            for ii in range(0,nn):
                curRes = sss[ii]
                # log(101,curRes)
                # log(102,curRes.split(' ')[0])
                # log(103,tarStr[1:])
                if curRes.split('\t')[0]==tarStr[1:]:
                    aa = curRes
                    # log(100,aa)
                    # log(100,"*****************")
                    # sys.exit()
                    break
    except Exception:
        pass
    # log(2000,aa+'\n')
    return aa  

def log(ID,val):
    print('-----> '+str(ID)+' = |'+str(val)+"|\n")    

def get_txt_line_Dict2(srcTxt,spStr=' '):
    # 备注：　此函数　只能保留第一次出现的 dict
    f = codecs.open(srcTxt,'r+','utf-8')
    line = f.readline()
    resList = []
    resDict = {}
    while True:
        if not line:
            break
        else:
            line = line.strip()
            word = line.split(spStr)[0].strip()
            tag = line[len(word):].strip()
            if word not in resList:
                resList.append(word)
                resDict[word] = tag
            # else:
            #     f2.write(line+"\n")
            line = f.readline()
    f.close()
    return resList,resDict


def get_line_word_count_Dict(srcTxt,spStr=' '):
    f = codecs.open(srcTxt,'r+','utf-8')
    line = f.readline()
    resList = []
    resDict = {}
    while True:
        if not line:
            break
        else:
            line = line.strip()
            word = line.split(spStr)[0].strip()
            tag = line[len(word):].strip()
            if word not in resList:
                resList.append(word)
                resDict[word] = tag
            else:
                #　重复出现，频数累加
                resDict[word] = str(int(resDict[word])+int(tag))

            line = f.readline()
    f.close()
    return resList,resDict


def py_grep_start_jieb_count(refWordFreTxt,word):              
    text_lineOir = py_grep_eng(refWordFreTxt,"^"+word)
    # log(2,text_lineOir)
    if len(text_lineOir)>0:
        freNum = text_lineOir.split('\t')[1]
    else:
        freNum= 1

    return freNum

if __name__=="__main__":
# 抽离数字-剩下的是spkID,将spkID标志加入list.
# A0001_001


    #writeSpkName()
    #sys.exit()
    
    print(sys.argv[0])
    if len(sys.argv) != 4:
        print(len(sys.argv))
        print('usage: '+ sys.argv[0] +' txtIn txtOUT refWordFreTxt(or 0)')
        sys.exit()
        
    txtIn = sys.argv[1]
    txtOUT = sys.argv[2]
    refWordFreTxt = sys.argv[3]
    
    refList,refDict = get_line_word_count_Dict(refWordFreTxt)



    if  os.path.isfile(refWordFreTxt):
        hasRef = True
    else:
        hasRef = False

    log(0,hasRef)
    
    oriStr = '/b/data/0-ASR-corpus/'
    tarStr = '/c/data/0-ASR-corpus/'
    
    f_in = codecs.open(txtIn,'r+','utf-8')
    f_out = codecs.open(txtOUT,'w+','utf-8')
    
    
    line = f_in.readline()
    if not hasRef :
        while True:
            if not line:
                break
            else:
                new_line = line.split(" ")[0].strip()+" 1\n"
                
                #print('line = '+line)
                #print('new_line = '+new_line)

                f_out.write(new_line)   
                line = f_in.readline()
                #if i00>3:
                #    sys.exit()
    else:
        while True:
            if not line:
                break
            else:
                word = line.split(" ")[0].strip()
                if word in refList:
                    # V2: 
                    # py_grep_start_jieb_count(refWordFreTxt,word)
                    freNum = refDict[word]
                else:
                    freNum=1
                # sys.exit()
                new_line = word+" "+str(freNum)+"\n"
                #print('line = '+line)
                #print('new_line = '+new_line)
                f_out.write(new_line)   
                line = f_in.readline()
                #if i00>3:
                #    sys.exit()       
    f_out.close()
    f_in.close()
    log("end ",'program finished ok!')    

# python3 p6_reNum.py /home/user01/ASR/11-data/engCorpus/voxWAV-TXT_P4 spkList_p5.txt /home/user01/ASR/11-data/engCorpus/voxWAV-TXT_p6_ok










 
