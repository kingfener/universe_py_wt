# coding=utf-8
'''
    文本 文件处理程序
    by @wangtao
'''


import sys,os,time
import re,codecs,shutil
from typing import Collection

from utils import *
from words_chi_eng import *

def symbol_filter(textStrin):
    ''' 
    delete unnormal symbol。删除 异常 符号
    '''
    fmtSym = re.compile(r'[~-%]')

    textStrin = fmtSym.sub(' ',textStrin)
    
    return textStrin




def chi_eng_sym_check():
    ''' 
    use for check input strs class , 中英文 检测
    
    '''

    pass


def del_symble(line):
    '''
    删除 文本行 重点 所有符号
    '''
    fmtStr = re.compile(r'[,.?!@<>&\*%$#\[\]，。？！@#￥%&-—+~《》（）]')
    line = re.sub(fmtStr,'',line)
    return line

def get_word_list(line):
    word_list = []
    for char in line:
        word_list.append(char)
    return word_list
def write_dict_to_text(word_fre_dict,res_text):
    word_fre_dict = sorted(word_fre_dict.items(),  key=lambda d: d[1], reverse=True)
    cnt = 0
    with open(res_text,'w+',encoding='utf-8') as ff:
        for key,value in word_fre_dict:
            ff.write(str(key)+' '+str(value)+'\n')
            if str(cnt)[-3:] == '000':
               ff.flush() 
            cnt = cnt + 1

def word_fre_get(textIn,res_text):
    '''
    字典 词频 统计import copy
    '''
    fmtStr = re.compile(r'[,、?!@<>&%$#，。？！@#￥%&~《》（）]')
    # fmtStr = re.compile(r'[!]')

    # fmtStr = re.compile(r'[？]')
    word_fre_dict = {}
    cnt = 1
    # stopNum = 100
    stopNum = -1
    logStr = '0000'
    total_line_num = py_wc(textIn)
    # with open(textIn,'r+',encoding='utf-8') as ff:
    with codecs.open(textIn,'r+','utf-8') as ff:
        line = ff.readline().strip()
        # line = ' '.join(line.split('\t')[2:])
        print(line,'----')
        while line:
            line = fmtStr.sub('',line.upper())
            cur_word_list = get_word_list(line)
            # print(cnt,line)
            if len(cur_word_list) == 0:
                continue
            for word in cur_word_list:
                if word not in word_fre_dict:
                    word_fre_dict[word] = 1
                else:
                    word_fre_dict[word] = word_fre_dict[word] + 1
            if str(cnt)[-len(logStr):]==logStr:
                print(cnt,line)
            if (cnt >= stopNum) and (stopNum > 0):
                break
            cnt = cnt + 1
            try:
                line = ff.readline().strip()
            except:
                break
    write_dict_to_text(word_fre_dict,res_text)
    print('textIn: ',textIn)
    
    print('total_line_num : ',cnt)
    print('processed line num: ',cnt)
    print('res_text: ',res_text)

def get_dict_from_text(textIn,topNum=-1):
    res_dict = {}
    cnt = 1
    with codecs.open(textIn,'r+','utf-8') as ff:
        line = ff.readline().strip()
        while line:
            spArr = line.split(' ')
            if len(spArr)>=2:
                res_dict[spArr[0]] = spArr[1]
            if (cnt > topNum) and (topNum > 0):
                break
            cnt = cnt +1
            line = ff.readline().strip()

    # print('res_dict=',res_dict)
    return res_dict

def test_get_dict_from_text():
    textIn = '/media/U1T/open-ASR/LM-text/text-cleand-sorted/general_withDigitFc-word-fre.txt'
    get_dict_from_text(textIn)

def test_word_fre_get():
    textIn = '/media/3tk/1-data-bk/ASR-Chi-tar/2-magicRead/TRANS.txt'
    textIn = '/media/U1T/open-ASR/LM-text/text-cleand-sorted/general_withDigitFc.txt'
    # textIn = '/media/U1T/open-ASR/LM-text/text-cleand-sorted/qaFc.txt'
    # textIn = '/media/U1T/open-ASR/LM-text/text-cleand-sorted/tieba_microBlog_dpdzdpFc.txt'
    textIn = '/media/3B/data/Txt/全国-地址-文本.txt'
    res_text = textIn[:-4]+'-word-fre.txt'
    word_fre_get(textIn,res_text)

def combine_text_dict_1():
    textIn = 'text/general_withDigitFc-word-fre.txt'
    dict1 = get_dict_from_text(textIn)
    textIn = 'text/qaFc-word-fre.txt'
    dict2 = get_dict_from_text(textIn)
    textIn = 'text/tieba_microBlog_dpdzdpFc-word-fre.txt'
    dict3 = get_dict_from_text(textIn)

    res_dict = {}
    for key,value in dict1.items():
        res_dict[key] = int(value)

    for key,value in dict2.items():
        if key in res_dict:
            res_dict[key] = int(res_dict[key])+int(value)
        else:
            res_dict[key] = int(value)

    for key,value in dict3.items():
        if key in res_dict:
            res_dict[key] = int(res_dict[key])+int(value)
        else:
            res_dict[key] = int(value)
    # return res_dict
    res_text = 'text/cmb-3-dict.txt'
    write_dict_to_text(res_dict,res_text)
    print('textIn=',textIn)
    print('res_text=',res_text)


def combine_text_dict():
    textIn = 'text/cmb-3-dict.txt'
    dict1 = get_dict_from_text(textIn)
    textIn = 'text/chiese_address-word-fre.txt'
    dict2 = get_dict_from_text(textIn)

    res_dict = {}
    for key,value in dict1.items():
        res_dict[key] = int(value)

    for key,value in dict2.items():
        if key in res_dict:
            res_dict[key] = int(res_dict[key])+int(value)
        else:
            res_dict[key] = int(value)

    # return res_dict
    res_text = 'text/cmb-4-dict.txt'
    write_dict_to_text(res_dict,res_text)
    print('textIn=',textIn)
    print('res_text=',res_text)

def text_Fan2Jan_openccpy_test():

    src_text = '/home/king/Desktop/asr/universe_py_wt/text/cmb-4-dict-hand.txt'
    res_text = src_text[:-4]+'-jianti.txt'
    cnt = 0
    word_dct = {}
    with codecs.open(src_text,'r+','utf-8') as ff :
        line = ff.readline().strip()
        while line:
            new_line1,_,_ = Fan2Jan_openccpy(line)
            new_line2,_ = Fan2Jan_opencc(line)
            word,nums = new_line2.split(' ')
            num = int(nums)
            if word not in word_dct:
                word_dct[word] = num
            else:
                word_dct[word] =  num if num > word_dct[word] else word_dct[word]

            # if new_line1 == new_line2:
            #     f_out.write(new_line2+'\n')
            # else:
            #     # f_out.write(new_line1+'\n')
            #     f_out.write(new_line2+'\n')

            cnt +=1
            if str(cnt)[-3:] == '000':
                print(cnt,'line     = ',line)   
                # print(cnt,'new_line1= ',new_line1)   
                # print(cnt,'new_line2= ',new_line2)            

            line = ff.readline().strip()
    marklist = sorted(word_dct.items(), key=lambda x:x[1],reverse=True)
    sort_word_dct = dict(marklist)
    cnt = 0
    with codecs.open(res_text,'w+','utf-8') as f_out:
        for key,value in sort_word_dct.items():
            new_line2 = key+' '+str(value)
            f_out.write(new_line2+'\n')
            cnt +=1
            if str(cnt)[-3:] == '000':
                print(cnt,'line     = ',line)   
                print(cnt,'new_line1= ',new_line1)   
                print(cnt,'new_line2= ',new_line2)       

    print('src_text ',src_text)
    print('res_text ',res_text)

    
if __name__=='__main__':
    # test_word_fre_get()
    # test_get_dict_from_text()
    # combine_text_dict_1()
    # combine_text_dict()
    text_Fan2Jan_openccpy_test()
    sys.exit()



