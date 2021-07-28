# coding=utf-8
'''
    文字 处理类程序
    by @wangtao
'''

import sys
import os
import shutil
import codecs
import re
#import librosa
import subprocess
#import soundfile
#import numpy as np
import opencc   # for Chinese Fanti --> JanTi transform
from openccpy.opencc import *   # new Fna to Jan 
#from langconv import *
import wave
import json
# import jieba


def Fan2Jan_opencc(lineWord,cc=' '): 
    #  line  can'yt have fuhao ,like : ,.!? etc     or it will always outut  ilegleMark=1;  
    if cc==' ':
        cc = opencc.OpenCC('t2s')
    lineWord = cc.convert(lineWord)
    ilegleMark = 0
    for ii in range(0,len(lineWord)):
        ch = lineWord[ii:ii+1]
        if not((u'\u4e00' <= ch and ch <= u'\u9fef') or (u'\u0030' <= ch and ch <= u'\u0039') or (u'\u0061' <= ch and ch <= u'\u007a')or (u'\u0041' <= ch and ch <= u'\u005a') or (ch==' ') or (ch=='\t') or (ch=='\n')  or (ch=='\\')):
            ilegleMark = 1
            print(lineWord)
            print(ch+" is not Chinese .")
            break
            
    return lineWord,ilegleMark

def Fan2Jan_openccpy(line): 
    #  line  can'yt have fuhao ,like : ,.!? etc     or it will always outut  ilegleMark=1;  
    lineWord = Opencc.to_simple(line+' ')   #  if not add ' ' add tail will err,eg: "畼"
    #lineWord = Opencc.to_simple(line)     #cc.convert(line)
    ilegleMark = 0
    
    badList = []
    for ii in range(0,len(lineWord)):
        ch = lineWord[ii:ii+1]
        if not((u'\u4e00' <= ch and ch <= u'\u9fef') or (u'\u3400'<=ch and ch <=u'\u4db5') or (u'\u0030' <= ch and ch <= u'\u0039') or (u'\u0061' <= ch and ch <= u'\u007a')or (u'\u0041' <= ch and ch <= u'\u005a') or (ch==' ') or (ch=='\t') or (ch=='\n')  or (ch=='\\') or (ch==r'[')or (ch==r']') or (ch==r'π') or (ch==r"'")):
            ilegleMark = 1
            print(line)
            print(lineWord)
            print(lineWord[:ii])
            
            print(ch+" is not Chinese .")
            badList.append(line[ii])
            badList.append(ch)
            break
        # if ch==r'[':
        #     print(lineWord)
        #     print("|"+ch+"|")

        #else:
        #    print(100,ch)
        #    print(100,ch.encode('Unicode-escape'))
            
                
    return lineWord[:-1],ilegleMark,badList


def Fan2Jan_openccpy_test():
    line = "畼"
    #line = " 畼续"
    line = "得 继续 招 畼 縣"
    line2 = Fan2Jan_openccpy(line)
    print(1,line)
    print(2,line2)

    line3 = Fan2Jan_opencc(line)
    print(3,line3)
    

# ref https://www.jianshu.com/p/152e081fec1b
# 汉字的判断
# 汉字的unicode编码范围 u4e00 到 u9fa5。
def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False
# 数字0-9的判断
# 数字的unicode编码范围根据全角和半角，有两个不同区域，半角数字 u0030 到 u0039，全角数字 uff10 到 uff19。
def is_number(uchar):
    """判断一个unicode是否是半角数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_Qnumber(uchar):
    """判断一个unicode是否是全角数字"""
    if uchar >= u'\uff10' and uchar <= u'\uff19':
        return True
    else:
        return False

# 大小写字母判断
# 字母的unicode编码根据字母大小写，以及全角和半角共有四个区域。
# 半角大写字母：u0041 - u005a ，半角小写字母：u0061 - u007a ；
# 全角大写字母：uff21 - uff3a ， 全角小写字母：uff41 - uff5a 。
def is_alphabet(uchar):
    """判断一个unicode是否是半角英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def is_Qalphabet(uchar):
    """判断一个unicode是否是全角英文字母"""
    if (uchar >= u'\uff21' and uchar <= u'\uff3a') or (uchar >= u'\uff41' and uchar <= u'\uff5a'):
        return True
    else:
        return False

# 非汉字和数字字母的判断
# 判断除汉字、数字0-9、字母之外的字符。
def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False

# 全角和半角的转换
#
# 全角半角转换需要用到上面的数字、字母等判断。
# 1    所有半角转全角，不是半角范围直接返回，空格半角特殊单独处理，其它半角和全角对应公式：半角 = 全角 - 0xfee0
def B2Q(uchar):
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e: # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020: # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)

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

def stringpartQ2B(ustring):
    """把字符串中数字和字母全角转半角"""
    return "".join([Q2B(uchar) if is_Qnumber(uchar) or is_Qalphabet(uchar) else uchar for uchar in ustring])

# 测试上面的全角半角转换。
def Q2B_test():
    text = "电影《２０１２》讲述了２０１２年１２月２１日的世界末日，主人公Ｊａｃｋ以及世界各国人民挣扎求生的经历，灾难面前，尽现人间百态。"

    print("text原文：", text, sep="\n", end="\n")
    text1 = stringQ2B(text)
    print("全角转半角：", text1, sep="\n", end="\n")
    text2 = stringpartQ2B(text)
    print("数字字母全角转半角：", text2, sep="\n", end="\n")




def eng_sentence_split_ZX(strIN,lexicon):
    text_list=[]
    cur_str = fenci_ZXZDPP_(strIN,lexicon)
    mm = len(cur_str)
    for jj in range(0,mm):
        text_list.append(cur_str[jj])   
    return text_list

def eng_sentence_split_NX(strIN,lexicon):
    text_list=[]
    cur_str = fenci_NXZDPP_(strIN,lexicon)
    mm = len(cur_str)
    for jj in range(0,mm):
        text_list.append(cur_str[jj])   
    return text_list


if __name__== "__main__":

    Fan2Jan_openccpy_test()
    sys.exit()
    import jieba
    print(sys.argv[0])
    for ii in range(0,len(sys.argv)):
        print(ii,sys.argv[ii])
