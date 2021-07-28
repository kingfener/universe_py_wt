# coding = utf-8

'''

20191216:
    1, split dict to : pure-CHinese + pure-eng + Mix-of-chi-eng



α->β-->w(omiga) unicode range: 0#03B1 - 0#03C9

    2, add  if ch =="'"  to Eng range.
'''



import sys
import os
import codecs
import re
import numpy as np
from pypinyin import pinyin,lazy_pinyin,Style
from openccpy.opencc import *  # new Fna to Jan


def log(ID, val):
    print('-----> ' + str(ID) + ' = |' + str(val)+"|")
    # print('-----> '+str(ID)+'shape = '+str(val.shape) )
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

def Fan2Jan_openccpy_pure(line):
    #  line  can'yt have fuhao ,like : ,.!? etc     or it will always outut  ilegleMark=1;  
    lineWord = Opencc.to_simple(line+' ')   #  if not add ' ' add tail will err,eg: "畼"
    #lineWord = Opencc.to_simple(line)     #cc.convert(line)
    ilegleMark = 0
    return lineWord[:-1]
# 2  所有全角转半角，和前面正好相反，公式对应：全角 = 半角 + 0xfee0
def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)
# 3  把整个字符串全角转半角，也可以只转部分如数字和字母
def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])

def check_contain_only_Chinese(inStr):
    flag=True
    for ch in inStr:
        if ch==" ":
            continue
        if u'\u4e00'>ch or ch >u'\u9fff':
            flag = False
            break
    return flag
    
def check_contain_only_Chinese(inStr):
    # 基本汉字： 20902字： 4E00-9FA5
    # 基本汉子补充： 74：   9FA6-9FEF
    # 扩展A      : 6582:  3400-4DB5 
    flag=True
    for ch in inStr:
        if ch==" ":
            continue
        if not((u'\u4e00'<=ch and ch <=u'\u9fef') or (u'\u3400'<=ch and ch <=u'\u4db5')):
            flag = False
            # log(1,ch)
            break
    return flag   

def check_contain_only_Chinese_ext(inStr):
    # 基本汉字： 20902字： 4E00-9FA5
    # 基本汉子补充： 74：   9FA6-9FEF
    # 扩展A      : 6582:  3400-4DB5 
    flag=True
    notChiList = []
    for ch in inStr:
        if ch==" ":
            continue
        if not((u'\u4e00'<=ch and ch <=u'\u9fef') or (u'\u3400'<=ch and ch <=u'\u4db5')):
            flag = False
            log(1,ch)
            notChiList.append(ch)
            notChiList.append(ch.encode("Unicode-escape"))
            # break
    return flag,notChiList   



def check_contain_only_eng(inStr):
    flag=True
    for ch in inStr:
        if not (u'\u0041'< ch< u'\u005a' or u'\u0061'< ch <u'\u007a'):
            flag=False
            break
    return flag
    
    
def check_contain_eng_digit(inStr):
    flag=False
    for ch in inStr:
        if ch==" ":
            continue
        if u'\u0041'< ch< u'\u005a' or u'\u0061'< ch <u'\u007a' or  u'\u0030'< ch <u'\u0039':
            flag=True
            break
    return flag
     
def check_contain_only_eng_digit(inStr):
    flag=True
    for ch in inStr:
        try:
            if 0<=int(ch)<=9:
                continue
        except:
            pass
        #if ch =="'" or ch=="[" or ch == "]":
        if ch =="'" :
            continue
        if not(u'\u0041'<= ch<= u'\u005a' or u'\u0061'<= ch <=u'\u007a' or  u'\u0030'<= ch <=u'\u0039'):
            flag=False
            break
    return flag
     
def check_contain_only_eng_digit_alpha(inStr):
#α->β-->w(omiga) unicode range: 0#03B1 - 0#03C9
    flag=True
    for ch in inStr:
        if not(u'\u0041'< ch< u'\u005a' or u'\u0061'< ch <u'\u007a' or  u'\u0030'< ch <u'\u0039' or u'\u03B1'< ch <u'\u03C9' ):
            flag=False
            break
    return flag
     
     

def check_contain_only_Chinese_test(): 
    
    inStr = '㖏郎郎凉凉ー𡟯'
    res = check_contain_only_Chinese(inStr)
    log('res',res)
    log(1,inStr.encode('Unicode-escape'))
    
if __name__=="__main__":

    # check_contain_only_Chinese_test()
    # sys.exit()

    if len(sys.argv) !=2:
        print(len(sys.argv))
        print('usage: '+ sys.argv[0] + ' dictIn')
        sys.exit()
    dictIn = sys.argv[1]

    dict_word_chinese = dictIn[:-4]+"-Chinese.txt"
    dict_word_english = dictIn[:-4]+"-English.txt"
    dict_word_Mix = dictIn[:-4]+"-Mix-CHiEng.txt"

    foldPrepare(dict_word_chinese)


    f_in = codecs.open(dictIn,"r+","utf-8")
    f_out = codecs.open(dict_word_chinese,"w+","utf-8")
    f_out2 = codecs.open(dict_word_english,"w+","utf-8")
    f_outMix = codecs.open(dict_word_Mix,"w+","utf-8")

    line = f_in.readline()

    cnt = 0
    while True:
        res_str = ""
        if not line:
            break
        else:
            line = line.strip()
            line0  = stringQ2B(line)
            line0  = Fan2Jan_openccpy_pure(line0)
            
            
            if 0:
                wordArrSS = line.split("\t")
            else:
                wordArrSS = line.split(" ")
            wordArr = wordArrSS[0].strip()



            # wordArr = "魔高一尺"


            wordArr = stringQ2B(wordArr)
            wordArr = Fan2Jan_openccpy_pure(wordArr)
            
            # log(cnt,wordArr)
            
            res_str = line0+"\n"


            # onCh,notList = check_contain_only_Chinese_ext(wordArr)

            # if onCh:
            if check_contain_only_Chinese(wordArr):
                f_out.write(res_str)
                # log(cnt,"***********only Chi*****")
            elif check_contain_only_eng_digit(wordArr):
                f_out2.write(res_str)
                # log(cnt,"***********only eng*****")
                #log(cnt,wordArr)
                #sys.exit()
            else:
                log(cnt,"***********only mix*****")
                log(cnt,wordArr)
                # f_outMix.write(res_str.strip()+str(notList)+'\n')
                f_outMix.write(res_str)
                
            # sys.exit()    
            if np.mod(cnt,1000)==0:
                f_out.flush()
                f_out2.flush()
                f_outMix.flush()
                log(cnt,res_str)
            cnt = cnt+1
            line = f_in.readline()

    f_in.close()
    f_out.close()
    f_out2.close()
    f_outMix.close()
    print(" finished ok !")
