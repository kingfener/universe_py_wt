# -*- coding: utf-8 -*-
'''
20181022: used for caldulate time ; ok
20190711:  add jieba spliet word .
20190712: add wav.scp output.
20190718: english char to - lower .
20190731: 2ed time split words.
20190820: use jieba HMM-False ;
20190821: trans Fan2Jan
20190907: change F2T tool from langconv to opencc.
20190907: Q2B trans.
20190909: add not Chinese line output file ,so can check it later. 
20190923: for segment : re-split word .
20190924:
    1, #content = strrepWT(content, 'ens', '\[c\]')     : only use in buyed AM-data.
    2, content = re.sub(u"[，,。.：:、<>？?！!<<>>《》（）…－＋( )-+·゛-・]"," ",content)   # delte fuhao   add "・"
    3, add  def F2J_3(line):from openccpy.opencc import *转换为简体 s = Opencc.to_simple(line);;;; Fan2Jan_openccpy

    4, keep ileagle Chinese word.
    
20190925: 
    1, keep segment wven only cantain 1 word.
20190926: 
    1, if len()==0; write it to err.txt


20191015:
    1,   content = strrepWT(content, r'"', ' ')
         content = re.sub(u"[‘’】，,。.：:、<>？?！!<<>>《》（）…－＋( )-+·゛-・“”;]"," ",content)   # delte fuhao   
    2, fix : Fan2Jan_openccpy(line)
20191216:
    1, change eng-dict max match len to src Len. 
    2, opt  eng-dict.  word-strip()

20191216:
    1, split dict to : pure-CHinese + pure-eng + Mix-of-chi-eng

    2, add: α->β-->w(omiga) unicode range: 0#03B1 - 0#03C9

    3, add  if ch =="'"  to Eng range.
20200108:
    1, 修复 check_contain_only_Chinese（） 的范围错误。


20200121：
    1， 优化了分词的 结构： 改为： 先将句子 按照中英文，分割成 中分、英文部分。然后分别调用函数进行分词，再组合。 
    2， 独立出英文的分词函数：eng_sentence_split(),
        但是当前的英文分词仍然有问题： 例如： G1361-T0055G1361S0499-0000000-0000362  播放 少女 时代 ig et abo y  ： 正确应该是： i get a boy。
    3, 针对 2中的问题，添加了”最大逆向匹配函数“ fenci_NXZDPP_()

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
#import opencc   # for Chinese Fanti --> JanTi transform
from openccpy.opencc import *   # new Fna to Jan 
#from langconv import *
#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt
import wave
#from mpl_toolkits.mplot3d import Axes3D   
import json
# import jieba
#from PIL import Image  

#import wave,struct,math

'''
def read_audio(path, target_fs=None):
def write_audio(path, audio, sample_rate):
def read_wave_data(file_path):  
def wavSave(Xs,fullname,Fs):
def mfcc_1(signal,sample_rate,showmark):
def enframe(x,win,inc):
def findSegment(express) :# voiceseg是语音段的结构数组，有begin,end, duration
def vad_param1D(dst1,T1,T2,maxsilence,minlen,mFre,freAmpMax,mFreRange,freAmpMaxThr,T22thr,minlenBT1,countTT2Thr,frameTime):
def maxInd(arr):
def frame2time(nFrm,nwlen,inc,Fs):
def cutSilence(X0,Fs,pre_FrmN,pos_FrmN,WlenT,sWlenT,padMark,debugMark,pureName0,seg_sec_len,dstPath,saveMark):
def vad_plot(debugMark,pureName0,X0,X1,X3,cutStartTime,cutEndTime,wlen,Fs,inc,numI,num1t,num2t,frameTime,Ef,T1,T2,freAmpMax,mFre,snr1,XCon,XCon_bkgrnd):
def py_grep(srcTxt,tarStr):
def py_wc(srcTxt):
def foldPrepare(fullfilename0):
def strrepWT(string,orIs,tarS):
def traveseFileFmt3(srcDir,frmStr,fileList): # srcDir = /wav   
def traveseFileFmt2(file_dir,frmStr): 
def removeALL(file_dir):     
def spkIDsplit(instr):  
def getTimeLenSec(file0):
def log(val,ID):
def log2(val,ID):
def rms(y):
def vad_Speech_noise_concate(X0,Fs,Xc,frmEnergy,SF,frameTime,nFrm,XCon1,voiceseg,maxsilence):
def vad_snr_cal(file0,debugMark):    
def batch_vad_snr(srcDir):
def getSetLenStr(tarLen,datain,padStr,before_mark=1):
def get_amplitude_scaling_factor(s, n, snr, method='rms'):
def additive_mixing(s, n):
def add_noise_snr(sig,noise,snr):
def read_audio(path, target_fs=None):
def write_audio(path, audio, sample_rate):
def read_wave_data(file_path):  
def wavSave(Xs,fullname,Fs):
def mfcc_1(signal,sample_rate,showmark):
def enframe(x,win,inc):
def findSegment(express) :# voiceseg是语音段的结构数组，有begin,end, duration
def vad_param1D(dst1,T1,T2,maxsilence,minlen,mFre,freAmpMax,mFreRange,freAmpMaxThr,T22thr,minlenBT1,countTT2Thr,frameTime):
def maxInd(arr):
def frame2time(nFrm,nwlen,inc,Fs):
def cutSilence(X0,Fs,pre_FrmN,pos_FrmN,WlenT,sWlenT,padMark,debugMark,pureName0,seg_sec_len,dstPath,saveMark):
def vad_plot(debugMark,pureName0,X0,X1,X3,cutStartTime,cutEndTime,wlen,Fs,inc,numI,num1t,num2t,frameTime,Ef,T1,T2,freAmpMax,mFre,snr1,XCon,XCon_bkgrnd):
def py_grep(srcTxt,tarStr):
def py_wc(srcTxt):
def foldPrepare(fullfilename0):
def strrepWT(string,orIs,tarS):
def traveseFileFmt3(srcDir,frmStr,fileList): # srcDir = /wav   
def traveseFileFmt2(file_dir,frmStr): 
def removeALL(file_dir):     
def spkIDsplit(instr):  
def getTimeLenSec(file0):
def log(ID,val):
def log2(ID,val):
def rms(y):
def vad_Speech_noise_concate(X0,Fs,Xc,frmEnergy,SF,frameTime,nFrm,XCon1,voiceseg,maxsilence):
def vad_snr_cal(file0,debugMark):    
def batch_vad_snr(srcDir):
def add_mix_snr_main():
def sortedDictValues2(adict): 
def sortedDictValues3(adict): 
def get_seg_trans_cmb(txtFile):
def batch_get_seg_trans_cmb(srcDir,dstDir):
def get_wav_chNum(wavFile):
def get_seg_trans_cmb2(txtFile):
def get_seg_trans_cmb2_sort(txtFile):
def batch_get_seg_trans_cmb2_sort(srcDir, dstDircmb):
def batch_get_seg_trans_sort(srcDir, dstDircmb,dstDir,dstDir2):
def batch_get_seg_cut_wav(srcTxtDir, dstDir_cmb, dstDir_seg, dstDir_segText, srcWavDir, tarWavDir):
def fenci_ZXZDPP_(text,dic):
def get_txt_line_list(srcTxt):
def get_txt_line_list2(srcTxt):
def check_contain_eng(inStr):
def check_contain_eng_digit(inStr):
def check_contain_only_Chinese(inStr):
def check_contain_Chinese(inStr):


'''


def F2J_3(line):
    #from openccpy.opencc import *
    """
    测试转换为简体
    """
    s = Opencc.to_simple(line)
    return s


def getSetLenStr(tarLen,datain,padStr,before_mark=1):
    if before_mark == 1:  # pad before tar str
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
            #return  dataOutStr
    else: # pad after tar str

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
                #dataOutStr = padStr+dataOutStr
                dataOutStr = dataOutStr + padStr
            #return  dataOutStr
    return dataOutStr



def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio = 10. ** (float(snr) / 20.)  # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor

def additive_mixing(s, n):
    """Mix normalized source1 and source2. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      
    Returns:
      mix_audio: ndarray, mixed audio. 
      s: ndarray, pad or truncated and scalered source1. 
      n: ndarray, scaled source2. 
      alpha: float, normalize coefficient. 
    """
    mixed_audio = s + n
    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha

    
def add_noise_snr(sig,noise,snr):
    
    #(speech_audio, _) = read_audio(tarFile, target_fs=Fs)
    #noise_audioOri = noise_audio
    
    noise_audio = noise
    speech_audio = sig
    
    # Repeat noise to the same length as speech. 
    if len(noise_audio) <= len(speech_audio):
        #Nn = len(noise_audio)
        #print('ori-len(noise_audio) ='+str(Nn)) 
        n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
        noise_audio_ex = np.tile(noise_audio, n_repeat)
        noise_audio = noise_audio_ex[0: len(speech_audio)]
    # Truncate noise to the same length as speech. 
    else:
        noise_onset = np.random.randint(0,len(noise_audio)-len(speech_audio))
        noise_offset = noise_onset+len(speech_audio)
        noise_audio = noise_audio[noise_onset:noise_offset]   

    scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
    speech_audio *= scaler
    #print('#############################################################################here 1127')
    
    (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio) 
    
    return mixed_audio, speech_audio, noise_audio, alpha




def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)



#from scipy.io import wavfile
def read_wave_data(file_path):  
    #open a wave file, and return a Wave_read object  
    f = wave.open(file_path,"rb")  
    #read the wave's format infomation,and return a tuple  
    params = f.getparams()  
    #get the info  
    nchannels, sampwidth, framerate, nframes = params[:4]  
    #Reads and returns nframes of audio, as a string of bytes.   
    str_data = f.readframes(nframes)  
    #close the stream  
    f.close()  
    #turn the wave's data to array  
    wave_data = np.fromstring(str_data, dtype = np.short)  
    #for the data is stereo,and format is LRLRLR...  
    #shape the array to n*2(-1 means fit the y coordinate)  
    #wave_data.shape = -1, 2  
    #transpose the data  
    #wave_data = wave_data.T  
    #calculate the time bar  
    #time = np.arange(0, nframes) * (1.0/framerate)  
    return wave_data,framerate
    
def wavSave(Xs,fullname,Fs):
    # 20180203_ for python2 
    # 默认保存16bit，采样率16k;单通道数据。
    # 设置参数
    params = (1, 2, Fs, 0, 'NONE', 'not compressed')
    # 保存文件
    wname = fullname #sys.path[0]+'/musicAddsN.wav'
    wf = wave.open(wname, 'wb')
    # 设置参数
    wf.setparams(params)
    # 设置波形文件 .tostring()将array转换为data
    #print('max(Xs)= '+str(max(Xs)))
    wave_data = Xs.astype(np.short)
    
    data_to_use = wave_data.tostring()
    
    #print('data_to_use = '+str(data_to_use))
    wf.writeframesraw(data_to_use)
    wf.close()
    #print(fullname+'  wav saved.')
    return 0
    
def mfcc_1(signal,sample_rate,showmark):
    # ref   https://www.cnblogs.com/dylancao/p/9790707.html  
    import numpy as np
    import scipy.io.wavfile
    from matplotlib import pyplot as plt
    from scipy.fftpack import dct
    import librosa.display
    #sample_rate,signal=scipy.io.wavfile.read('a1.wav')
    #print(sample_rate,len(signal))
    #读取前3.5s 的数据
    #signal=signal[0:int(3.5*sample_rate)]
    #print(signal)
    #预先处理
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


    frame_size=0.025
    frame_stride=0.01
    frame_length,frame_step=frame_size*sample_rate,frame_stride*sample_rate
    signal_length=len(emphasized_signal)
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    num_frames=int(np.ceil(float(np.abs(signal_length-frame_length))/frame_step))
    
    pad_signal_length=num_frames*frame_step+frame_length
    z=np.zeros((pad_signal_length-signal_length))
    pad_signal=np.append(emphasized_signal,z)


    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

    #加上汉明窗
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **

    #傅立叶变换和功率谱
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    #print(mag_frames.shape)
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    low_freq_mel = 0
    #将频率转换为Mel
    nfilt = 40
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape

    n = np.arange(ncoeff)
    cep_lifter =22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    
    #filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    
    
    
    #showmark = 0
    if showmark:
        plt.figure(figsize=(200,160))
        t = np.arange(0,len(signal))/(sample_rate+0.0)
        plt.subplot(311)
        plt.plot(t,signal)
        plt.subplot(312)
        #plt.plot(np.transpose(mfcc))
        Pxx, freqs, bins, im = plt.specgram(signal, NFFT=int(0.025*sample_rate), Fs=sample_rate, noverlap=int(0.01*sample_rate))
        
        plt.subplot(313)
        librosa.display.specshow(np.transpose(mfcc), x_axis='time')

        print(num_frames)
        print(mfcc.shape)
        plt.show()    
    
    return np.transpose(mfcc)
    
    
    
    
    
    
    
##################################################  VAD start
def enframe(x,win,inc):
    inc = int(inc)
    '''print("inc = ")
    print(inc)'''
    # window or win length
    wlenMark = 0;
    # python2 use bellow
    #if str(type(win))=="<type 'numpy.ndarray'>": # win is an array of win.;
    # python3 use bellow
    if str(type(win))=="<class 'numpy.ndarray'>":
        wlen = len(win)
    else:
        wlen = win
        wlenMark = 1
    # inc judge
    if inc <=0:
        print('*************************************************************')
        print('***** wrong in : ******* enframe(x,win,inc):'+'inc =0,must >0')
        print('*************************************************************')
        return
    # splice to frame
    N = len(x)
    #log(inc,99)
    frmNum = int(np.floor((N-(wlen-inc))/inc))
    #print("frmNum1 = "+ str(frmNum))
    frmNum = int(np.floor((N-(wlen-inc))/inc)+1)
    #print("frmNum2 = "+ str(frmNum))
    pad0num = frmNum*inc+wlen-inc-N
    # pad 0
    padArr = np.zeros(pad0num)
    #print("padArr= "+ str(padArr))
    
    x = np.append(x,padArr)
    N = len(x)
    shiftN = np.arange(0,frmNum*inc+1,inc);
    xFrame = np.zeros([wlen,frmNum])
    frmSE = np.zeros([frmNum,2])
    if wlenMark==1: # rectangle window
        for i in np.arange(0,frmNum):
            xFrame[:,i] = x[shiftN[i]:shiftN[i]+wlen]
            frmSE[i,:] = [shiftN[i],shiftN[i]+wlen-1]
    else:
        for i in np.arange(0,frmNum):
            xFrame[:,i] = x[shiftN[i]:shiftN[i]+wlen]*win;
            frmSE[i,:] = [shiftN[i],shiftN[i]+wlen-1]
    return xFrame,frmNum,frmSE,wlen,padArr
    
def findSegment(express) :# voiceseg是语音段的结构数组，有begin,end, duration
    voicedIndex=express;
    maxNum = int(np.ceil(len(voicedIndex)/2))+1       # +1
    soundSegment = np.zeros([maxNum,3],np.int32);              # start end len;
    k = 0;
    #soundSegment(k).begin = voicedIndex(1);          # 设置第一组有话段的起始位置
    soundSegment[0][0] = voicedIndex[0];              # 设置第一组有话段的起始位置
    #print("soundSegment[0][0]= "+str( soundSegment[0][0]))
    for i in np.arange(0,len(voicedIndex)-1):
        if voicedIndex[i+1]-voicedIndex[i]>1:         #本组有话段结束
            soundSegment[k][1] = voicedIndex[i];      # 设置本组有话段的结束位置
            soundSegment[k+1][0] = voicedIndex[i+1];  # 设置下一组有话段的起始位置  
            k = k+1;
    soundSegment[k][1] = voicedIndex[-1];             # 最后一组有话段的结束位置
    #print("k= "+str( k))
    #print("soundSegment[k][k]= "+str( soundSegment[k][1]))
    #print(soundSegment.shape)
    #% 计算每组有话段的长度
    for i in np.arange(0,k):
        soundSegment[i][2]=soundSegment[i][1]-soundSegment[i][0]+1;
        
    soundSegment0 = np.delete(soundSegment, np.arange(k+1,maxNum),axis=0) 
    #print("soundSegment0= "+str( soundSegment0))
    return soundSegment0
    # 20180621: silence[xn] = 0;  

def vad_param1D(dst1,T1,T2,maxsilence,minlen,mFre,freAmpMax,mFreRange,freAmpMaxThr,T22thr,minlenBT1,countTT2Thr,frameTime):
    fn = len(dst1);  # wt 20170420
    postfN = 3;
    status  = 0;
    count   = np.zeros(int(fn/3)); # not have to big as fn, because 
    silence = np.zeros(int(fn/3));
    # 开始端点检测
    xn = 0;
    x1 = np.zeros(fn); # return a row vertor;
    x2 = np.zeros(fn);
    countTT2 = 0;
    for n in np.arange(1,fn) :
        if status<2 : #case {0,1}                           #0 = 静音, 1 = 可能开始
            if (dst1[n] > T2) and (mFre[n]>=mFreRange[0]) and ((freAmpMax[n]>freAmpMaxThr) or (dst1[n] > T22thr)): # 确信进入语音段
                #x1[xn] = max(n-count[xn]-postfN,0);  
                countTT2 = countTT2+1;
                if countTT2>countTT2Thr:
                    x1[xn] = max(n-count[xn]-postfN,0);
                    status  = 2;
                silence[xn] = 0;               #x1记录无话段帧数；
                count[xn] = count[xn] + 1;
            elif (dst1[n] > T1)and (mFre[n]>=mFreRange[0]) and (mFre[n]<=mFreRange[1]): #可能处于语音段
                status = 1;
                count[xn]  = count[xn] + 1;
                # 添加 连续 xx 帧大于T1,也算 语音帧。
                #x1[xn] = max(n-count[xn]-postfN,0);
                countTT2 = 0;
            else :  
                countTT2 = 0;
                if count[xn]>= minlenBT1:
                    status = 2 #3;
                    count[xn]  = count[xn] + 1;
                    silence[xn] = 0;
                    x1[xn] = max(n-count[xn]-postfN,0);  
                    #countTT2 = 0;
                    if n == fn-1:
                        x2[xn]=min(x1[xn]+count[xn]+postfN,fn-1);
                        status = 0          
                        xn = xn+1 
                        count[xn] = 0
                        silence[xn] = 0
                        x1[xn] = -1
                        x2[xn] = -1
                        countTT2 = 0
                else:
                    status = 0;
                    count[xn] = 0;
                    x1[xn]= -1;
                    x2[xn]= -1;
        elif status==2 :                        # case 2,   #2 = 语音段
            countTT2 = 0;
            if (dst1[n] > T1)and (mFre[n]>=mFreRange[0]) and (mFre[n]<=mFreRange[1]): #dst1[n] > T1:                    # 保持在语音段
                count[xn] = count[xn] + 1;      # count()用来记录有话段帧数；
                silence[xn] = 0;
                if n == fn-1:
                    x2[xn]=min(x1[xn]+count[xn]+postfN,fn-1);
                    status = 0          
                    xn = xn+1 
                    count[xn] = 0
                    silence[xn] = 0
                    x1[xn] = -1
                    x2[xn] = -1
                    countTT2 = 0

            else :                              # 语音将结束
                if n == fn-1:
                    x2[xn]=min(x1[xn]+count[xn]+postfN,fn-1);
                    status = 0          
                    xn = xn+1 
                    count[xn] = 0
                    silence[xn] = 0
                    x1[xn] = -1
                    x2[xn] = -1
                    countTT2 = 0

                silence[xn] = silence[xn]+1;
                if silence[xn] <= maxsilence :   # 静音还不够长，尚未结束
                    count[xn]  = count[xn] + 1;
                elif count[xn] < minlen  :      # 语音长度太短，认为是噪声
                    status  = 0;
                    silence[xn] = 0;
                    count[xn]   = 0;
                else :                          # 语音结束
                    status  = 3;
                    x2[xn]=min(x1[xn]+count[xn]+postfN,fn-1);
        elif (status==3) : 
            status = 0          
            xn = xn+1 
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = -1
            x2[xn] = -1
            countTT2 = 0
    # x1: 0~fn-1; xn: 0~N-1; xn = N-1; N~fn-1 delete;
    x1 = np.delete(x1,np.arange(xn+1,fn),axis=0) # delete row 
    x2 = np.delete(x2,np.arange(xn+1,fn),axis=0) 
    el = len(x1);   # 语音点
    el2 = el;
    if (x1[el-1] == -1):
        el = el-1             # 获得x1的实际长度
    if (el==0):
        return  # can cause err;
    #if (x2[el2-1]==-1):                       #如果x2最后一个值为0，对它设置为fn
    #    #print("Error: Not find endding point!")
    #    x2[el2-1] = fn-1;
    if (x2[el-1]==-1):                       #如果x2最后一个值为0，对它设置为fn
        #print("Error: Not find endding point!")
        x2[el-1] = fn-1;
    '''
    print("*************")
    print("len(x1)= "+ str(len(x1)))  
    print("el2= "+ str(el2)) 
    print("len(x2)= "+ str(len(x2)))  
    print("el= "+ str(el)) 
    print("x2[el-1]= "+ str(x2[el-1]))
    print("fn= "+ str(fn))  
    print("*************")
    '''
    
    eNDfRMnUM = int(x2[el-1])
    eNDfRMnUM2 = int(x2[el-2])
    #print('frameTime[x2[el-1]] = '+str(frameTime[eNDfRMnUM]))
    #print('frameTime[x2[el-2]] = '+str(frameTime[eNDfRMnUM2]))
    SF = np.zeros(fn,np.int32);                 #按x1和x2，对SF和NF赋值
    NF = np.ones(fn,np.int32);
    for i in np.arange(0,el):
        s1 = int(x1[i]);
        s2 = int(x2[i]);
        SF[s1:s2+1] = 1;
        NF[s1:s2+1] = 0;
    #speechIndex = find(SF==1); 
    speechIndex00 = np.where(SF==1)     # 计算寻找SF=1的地址即：帧号；voiceseg
    speechIndex = speechIndex00[0]
    voiceseg = findSegment(speechIndex) # voiceseg 是语音段的结构数组，有begin,end, duration
    #vsl,tt = size(voiceseg);           # voiceseg 的行数 表示语音voiceseg有几段；
    return voiceseg,SF,NF
def maxInd(arr):
    maxPos = 0;
    maxVal = arr[0];
    for i1 in np.arange(1,len(arr)):
        if arr[i1]>arr[maxPos]:
            maxPos = i1;
            maxVal = arr[i1];
    return maxPos,maxVal    
def frame2time(nFrm,nwlen,inc,Fs):
    frmArr = np.arange(0,nFrm)
    #frameTime = (frmArr*inc+nwlen/2.0)/(Fs+0.0)
    frameTime = (frmArr*inc)/(Fs+0.0)
    return frameTime
def cutSilence(X0,Fs,pre_FrmN,pos_FrmN,WlenT,sWlenT,padMark,debugMark,pureName0,seg_sec_len,dstPath,saveMark):
    # 20180718: add CHOOSEid para
    path1,pureName = os.path.split(pureName0)
    #seg_sec_len = 6; # sec, set for mini cut-vad segment time len.

    #pre_FrmN = 2;      # head extend frame Num;
    #pos_FrmN = 8;      # tail extend frame Num;
    IS = 0.25;         # silence time for noise estimate time len at utt begin.
    # normalize to 1 ( use max Value in 1/8~7/8 pos)
    N00 = len(X0);
    Nms = int(N00/8);
    Nme = 7*Nms;
    mm = max(abs(X0[Nms:Nme]))+0.0;
    X1 = X0/mm;
    # enframe
    wlen = int(WlenT*Fs);
    #print('1- wlen='+str(wlen));
    wlen = np.hamming(wlen);
    #print('2- wlen='+str(wlen));

    inc = int(sWlenT*Fs);
    #print('X1 ='+str(X1));
    Y1,nFrm,frmSE,nwlen,padArr = enframe(X1,wlen,inc);  #
    wlen = len(wlen);
    #print("Y1= "+str(Y1))
    if padMark==1 : 
        X0 = np.append(X0,padArr);
    #print("nFrm= "+str(nFrm))
    frameTime = frame2time(nFrm,nwlen,inc,Fs);
    # caldulate energy
    aparam = 2.0;
    eps = np.exp(-12);
    Ef = np.zeros(nFrm);
    freAmpMax = np.zeros(nFrm); # max Amp value arr for each frm;
    mFre = np.zeros(nFrm);      # max frequency arr for each frm;
    df = Fs/(wlen+0.0);
    #print('df= '+str(df));
    ###########
    #mCk = np.zeros([int(wlen/2.0),nFrm]);
    #ff = np.arange(0,wlen)*df;
    for i2 in np.arange(0,nFrm):# i2 in np.arange(0,1):#for 
        xFrm = Y1[:,i2]
        Sp = abs(np.fft.fft(xFrm))  #使用fft函数对余弦波信号进行傅里叶变换。
        Sp = Sp[0:int(wlen/2.0)];
        #mCk[:,i2] = Sp;
        Esum = np.log10(1+sum(Sp*Sp)/aparam)
        prob = Sp/(sum(Sp)+eps)
        H = -sum(prob*np.log10(prob+eps))
        Ef[i2] = np.sqrt(1+abs(Esum/(H+eps)))
        [maxI,freAmpMax[i2]] = maxInd(Sp);
        mFre[i2]= maxI*df;
    Enm = Ef 
    #print("Enm= "+str(Enm))   
    Me = max(Enm)
    # Nis
    NIS = int(np.floor((IS*Fs-nwlen)/inc+1))  # quiet frmNum
    #print("NIS= "+str(NIS))
    eth = np.mean(Enm[0:NIS])
    #eth = min(np.mean(Enm[0:NIS]),1)
    Det = Me -eth
    '''
    coff = 0.1 # coff = 0.06
    T1 = max(coff*Det+eth,1.2)#1.6#0.03*Det+eth
    T2 = max(4*coff*Det+eth,1.4)#1.4#0.08*Det+eth
    '''
    T1 = 1.187 #1.085 # 1.3#1.6#0.03*Det+eth
    T2 = 1.265 #1.12  # 1.4#1.4#0.08*Det+eth
    
    #print("T1= "+ str(T1))
    #print("T2= "+ str(T2))
    
    #maxsilenceTime = 0.105 #0.04#0.02  #0.2  # time 
    #minlenTime = 0.135 #0.2#0.3 #0.2          # time 

    maxsilenceTime = 0.2 #0.2 #0.04#0.02  #0.2  # time 
    minlenTime = 0.1 #0.3 #0.2#0.3 #0.2          # time 

    maxsilence = int(np.floor((maxsilenceTime*Fs-nwlen)/inc+1)) # frm Num
    minlen = int(np.floor((minlenTime*Fs-nwlen)/inc+1))         # corre frm Num  <===>  minlenTime
    #print('minlen = '+str(minlen))
    #maxsilence = 10; # max silence frm Num in VAD;
    #minlen = 4;     # min VAD frm Num;
    freAmpMaxThr = 4 #4 #4; # fft max value's thr;
    minlenBT1 = 8;     # if only > T1, min VAD frm len;
    T22thr = 1.29 # 1.15 ;   # or choose for freAmpMaxThr; (freAmpMax(n)>freAmpMaxThr)) or (dst1(n) > T22) 
    mFreRange = [99,3990]  #[200,2900] # spk frequency range.
    #mFreRange = [200,3500] # spk frequency range.
    countTT2Thr = 2;   # min frm Num > T2;
    [voiceseg,SF,NF] = vad_param1D(Enm,T1,T2,maxsilence,minlen,mFre,freAmpMax,mFreRange,freAmpMaxThr,T22thr,minlenBT1,countTT2Thr,frameTime);
    if len(voiceseg)==0:
        T1 = 0.03*Det+eth;
        T2 = 0.08*Det+eth;
        [voiceseg,SF,NF] = vad_param1D(Enm,T1,T2,maxsilence,minlen,mFre,freAmpMax,mFreRange,freAmpMaxThr,T22thr,minlenBT1,countTT2Thr,frameTime);
    # filter 2ed time;
    vsl = len(voiceseg)
    num1t = np.zeros(vsl);
    num2t = np.zeros(vsl);
    nx1 = voiceseg[0][0]; 
    nx2 = voiceseg[0][1];
    #print("nx1="+str(nx1))
    num1t[0]=frameTime[nx1];
    num2t[0]=frameTime[nx2];
    numI=1;
    interFn = 0; # VAD min  gap frm Num;
    # 相邻段之间的 测次过滤。
    for k in np.arange(1,vsl):
        nx1=voiceseg[k][0]; 
        nx2=voiceseg[k][1];    
        ddFrmNum = nx2-nx1+1;
        if (voiceseg[k][0]-voiceseg[k-1][0]>interFn) and (ddFrmNum >= minlen) : # 针对最后一段帧长做过滤；
            num1t[numI]=frameTime[nx1];
            num2t[numI]=frameTime[nx2];
            numI=numI+1;
    #numI=numI-1;
    num1t = np.delete(num1t,np.arange(numI,vsl),axis=0);
    num2t = np.delete(num2t,np.arange(numI,vsl),axis=0);

    # 20180428: 实现第三条。% 3, 但是无发删除首部的真人噪声。====》 可尝试方法： 如果首部一帧的数值 赋值小于最大值的0.6倍。 则直接抛弃该段。
    FRMsNUM = max(voiceseg[0][0]-pre_FrmN,0)               # 向前扩展 3 帧。
    FRMeNUM = min(voiceseg[numI-1][1]+pos_FrmN,nFrm-1)   # 向后扩展 20 帧。
    cutStartTime = frameTime[FRMsNUM]
    cutEndTime = frameTime[FRMeNUM]
    #print('cutStartTime='+str(cutStartTime)+":::"+'FRMsNUM='+str(FRMsNUM))
    # 根据以上VAD检测的首尾起时间，进行 裁减。
    ss0 = max(int(cutStartTime*Fs),0)
    ss1 = min(int(cutEndTime*Fs),len(X0))
    # xCut = X1(ss0:ss1);
    X3 = X0[ss0:ss1]
    
    (segNum,temp) = np.shape(voiceseg)

    #print(':: segNum= '+str(segNum));

    voicesegInd = np.zeros((segNum,2))
    for ii in range(0,segNum):
        voicesegInd[ii][0] = max(int(frameTime[voiceseg[ii][0]]*Fs),0)
        voicesegInd[ii][1] = min(int(frameTime[voiceseg[ii][1]]*Fs),len(X0))

    if segNum>1:
        ts = int(voiceseg[0][0]);
        te = int(voiceseg[0][1]);
        if ts <= maxsilence:
            ts = 0;
        ts = int(frameTime[ts]*Fs);
        te = int(frameTime[te]*Fs);
        #print('segNum ='+str(segNum));
        #print('ts ='+str(ts));
        #print('te ='+str(te));
        XCon = X0[ts:te]
        for i1 in range(1,segNum-1):
            ts = int(voiceseg[i1][0]);
            te = int(voiceseg[i1][1]);
            ts = int(frameTime[ts]*Fs)
            te = int(frameTime[te]*Fs)
            Nseg = X0[ts:te];
            XCon = np.append(XCon,Nseg);
        #print('len1 = '+str(len(XCon)));
        ts = int(voiceseg[segNum-1][0]);
        te = int(voiceseg[segNum-1][1]);
        if nFrm <= (maxsilence+te):
            #te = nFrm;
            te = N00;  # ori sig totel length
        else:
            te = int(frameTime[te]*Fs)
        ts = int(frameTime[ts]*Fs)
        Nseg = X0[ts:te];
        XCon = np.append(XCon,Nseg);
        #print('len2 = '+str(len(XCon)));
        # ######################
        #'''
        # PART SAVE.
        if 0:
            # more than one segment, cut and save each segment.
            cul_time_sec = 0;
            partNum = 1;
            lock = 0;
            ##################################
            #  only connect 
            ###############################
            for i1 in range(0,segNum):
                ts_frm = int(voiceseg[i1][0]);
                te_frm = int(voiceseg[i1][1]);
                ts_time = frameTime[ts_frm];
                te_time = frameTime[te_frm];

                ts_ind = int(ts_time*Fs);
                te_ind = int(te_time*Fs);
                #print('i1='+str(i1)+":::"+'ts_time='+str(ts_time))
                #print('i1='+str(i1)+":::"+'ts_frm ='+str(ts_frm))
                if lock==0:
                    ts_ind_ok = ts_ind;
                    lock = 1;
                cul_time_sec = cul_time_sec + te_time - ts_time;
                if (cul_time_sec >= seg_sec_len) or (i1==segNum-1):
                    Nseg = X0[ts_ind_ok:te_ind];
                    if saveMark>0:
                        segName = os.path.join(dstPath,pureName[:-4]+'-'+str(partNum)+'.wav')
                        wavSave(Nseg,segName,Fs)
                    #print('part: '+str(partNum));
                    #print('part: '+str(partNum)+' ='+str(Nseg));
                    partNum = partNum+1;
                    cul_time_sec = 0;
                    lock = 0;
        ##################################
        #  cut then connect !
        ###############################
        cul_time_sec = 0;
        partNum = 1;
        lock = 0;
        XConSeg = np.array([]);
        for i1 in range(0,segNum):
            ts_frm = int(voiceseg[i1][0]);
            te_frm = int(voiceseg[i1][1]);
            ts_time = frameTime[ts_frm];
            te_time = frameTime[te_frm];

            ts_ind = int(ts_time*Fs);
            te_ind = int(te_time*Fs);
            #print('i1='+str(i1)+":::"+'ts_time='+str(ts_time))
            #print('i1='+str(i1)+":::"+'ts_frm ='+str(ts_frm))
            if lock==0:
                ts_ind_ok = ts_ind;
                lock = 1;
            cul_time_sec = cul_time_sec + te_time - ts_time;
            Nseg = X0[ts_ind:te_ind];
            XConSeg = np.append(XConSeg,Nseg);
            if (cul_time_sec >= seg_sec_len) or (i1==segNum-1):
                if saveMark>0:
                    segName = os.path.join(dstPath,pureName[:-4]+'-'+str(partNum)+'-pure.wav')
                    wavSave(XConSeg,segName,Fs)
                #print('part: '+str(partNum));
                #print('part: '+str(partNum)+' ='+str(Nseg));
                partNum = partNum+1;
                lock = 0;
                XConSeg = np.array([]);
                cul_time_sec = 0;
    else:  # caution ; if only has one segment,then save it to src-0.wav.
        #print('::<=1, segNum= '+str(segNum));
        ts = int(voiceseg[0][0]);
        te = int(voiceseg[0][1]);
        if ts <= maxsilence:
            ts = 0;
        else:
            ts = int(frameTime[ts]*Fs);
        if nFrm <= (maxsilence+te):
            #te = nFrm;
            te = N00;  # ori sig totel length
        else:
            te = int(frameTime[te]*Fs);
        XCon = X0[ts:te];
        # ######################
        # PART SAVE.
        if saveMark>0:  
            segName = os.path.join(dstPath,pureName[:-4]+'-0'+'.wav')
            wavSave(XCon,segName,Fs)
        
    # cal SNR     
    XCon,XCon_bkgrnd = vad_Speech_noise_concate(X0,Fs,X3,Ef,SF,frameTime,nFrm,XCon,voiceseg,maxsilence)    
    
    if (len(XCon_bkgrnd)!=0) and (len(XCon)!=0):
        
        SigAm = rms(XCon)
        refAm = rms(XCon_bkgrnd)
        #print('SigAm = '+str(SigAm))
        #print('refAm = '+str(refAm))
        snr11 = 20*np.log10(SigAm/(refAm+1e-12)+ 1e-12)
        #log(snr11,11)
        #log(max(XCon),33)
        #log(max(XCon_bkgrnd),44)
        
        XCon0 = XCon/(max(abs(XCon))+0.0)
        XCon_bkgrnd0 = XCon_bkgrnd/(max(abs(XCon))+1e-12)
        
        snr1 = 20*np.log10(rms(XCon0)/(rms(XCon_bkgrnd0)+1e-12)+ 1e-12)
        #avaSig = sum(XCon0**2) / (len(XCon0)+1e-12)
        #avaRef = sum(XCon_bkgrnd0**2)/(len(XCon_bkgrnd0)+1e-12)
        #snr2 = 10*np.log10( avaSig / (avaRef+0.0) + 1e-12)
        
        #log(snr1,1)
        #log(max(XCon0),33)
        #log(max(XCon_bkgrnd0),44)
    else:
        snr1 = -100
    
    # 绘图 ######################  
    rr,Beishu = vad_plot(debugMark,pureName0,X0,X1,X3,cutStartTime,cutEndTime,wlen,Fs,inc,numI,num1t,num2t,frameTime,Ef,T1,T2,freAmpMax,mFre,snr1,XCon,XCon_bkgrnd)

    #return X0,X3,Ef,SF,frameTime,nFrm,XCon,voiceseg,maxsilence,Beishu
    return X0,X3,Ef,SF,frameTime,nFrm,XCon,voicesegInd,maxsilence,Beishu

##################################################    VAD end

def vad_plot(debugMark,pureName0,X0,X1,X3,cutStartTime,cutEndTime,wlen,Fs,inc,numI,num1t,num2t,frameTime,Ef,T1,T2,freAmpMax,mFre,snr1,XCon,XCon_bkgrnd):
    # 绘图 start ######################
    Beishu= 0.0
    if debugMark>0:
        
        path1,pureName = os.path.split(pureName0)
        
        #Fig = plt.figure(1,figsize=(16,7))
        Fig = plt.figure(1,figsize=(18,9))
        # 1st line
        Fs = Fs+0.0;
        t = np.arange(0,len(X1))/Fs;
        #print('t[-1]= '+str(t[-1]))
        plt.subplot(321)
        plt.plot(t,X1,'-k')
        plt.title(pureName+'   len='+str(len(X1))+'    snr= '+'%1.2f'%(snr1));
        #
        plt.plot([cutStartTime,cutStartTime],[-1,1],'--r');
        plt.plot([cutEndTime,cutEndTime],[-1,1],'-r');
        for i2 in np.arange(0,numI):
            plt.plot([num1t[i2],num1t[i2]],[-1,1],'--b');
            plt.plot([num2t[i2],num2t[i2]],[-1,1],'-b');
        #
        plt.subplot(322)
        #log(X1,9999)
        Pxx, freqs, bins, im = plt.specgram(X1, NFFT=wlen, Fs=Fs, noverlap=inc)
                #  (161, 261)
        lowFre = int(7000/8000.0*161)
        rowNUm,colNum =  Pxx.shape
        print(Pxx.shape)
        #log(lowFre,000)
        #log(np.max(Pxx[lowFre:,:]),333)   # 
        #log(np.max(Pxx[:rowNUm-lowFre,:]),444)   # 8k*0.75
        #log2(Pxx,1111)
        Beishu = np.max(Pxx[:rowNUm-lowFre,:])/(np.max(Pxx[lowFre:,:])+1e-12)
        plt.title('%d: %d/%d '%(lowFre,rowNUm,colNum)+'  | %1.12f   '%(np.max(Pxx[lowFre:,:]))  +'  : %1.12f   '%(np.max(Pxx[:rowNUm-lowFre,:]))+'%1.1f'%(Beishu))



        #Pxx, freqs, bins, im =0,0,0,0
        # 2ed Line
        t3 = np.arange(0,len(X3))/Fs;
        plt.subplot(323)
        plt.plot(frameTime,Ef,'-b*')
        plt.plot([0,t[-1]],[T1,T1],'-g');
        plt.plot([0,t[-1]],[T2,T2],'-g');
        plt.plot([cutStartTime,cutStartTime],[0.9,2],'--r');
        plt.plot([cutEndTime,cutEndTime],[0.9,2],'-r');
        plt.ylim(0.8,2);
        for i2 in np.arange(0,numI):
            plt.plot([num1t[i2],num1t[i2]],[0.8,2],'--b');
            plt.plot([num2t[i2],num2t[i2]],[0.8,2],'-b');
        plt.subplot(324)
        #Pxx, freqs, bins, im = plt.specgram(X3, NFFT=wlen, Fs=Fs, noverlap=inc) 
        plt.plot(frameTime,freAmpMax)
        # 3rd line
        # refWav = os.path.join('/home/user01/ASR/11-data/pj/wav2/refCut',pureName);
        #refWav = os.path.join(os.getcwd(),pureName);
        #refWav = os.path.join(path1,pureName);
        #X2ref,Fs2ref = read_wave_data(refWav);
        
        plt.subplot(325)
        t2 = np.arange(0,len(XCon))/(Fs+0.0)
        plt.plot(t2,XCon/(max(XCon)+1e-12),'-k')
        
        t3 = np.arange(0,len(XCon_bkgrnd))/(Fs+0.0)
        
        
        plt.plot(t3+t2[-1]+t2[1],XCon_bkgrnd/(max(abs(XCon))+1e-12),'-b')
        plt.ylim(-1,1)
        plt.plot([t2[-1],t2[-1]],[-1,1],'-r')
        #plt.plot(t3,XCon_bkgrnd/(max(abs(XCon))+0.0)-2,'-b')
        #plt.ylim(-3,1)
        
        plt.title('len(sig)='+str(len(XCon))+'   len(noSpeech)='+str(len(XCon_bkgrnd)))
        
        
        plt.subplot(326)
        #Pxx, freqs, bins, im = plt.specgram(X2ref, NFFT=wlen, Fs=Fs, noverlap=inc)
        plt.plot(frameTime,mFre) 
        ################################################33

        '''
        ################################################33
        fig = plt.figure('3D surface & wire')
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        #ax = fig.add_subplot(1, 2, 1)
        #Pxx, freqs, bins, im = plt.specgram(X1, NFFT=wlen, Fs=Fs, noverlap=inc)
        log2(Pxx,1111)
        print(Pxx.shape)
        #im = Image.imshow(Pxx[120:,:])
        # 生成格点
        x = np.linspace(0, 1, 261)
        y = np.linspace(0, 1, 41)
         
        # x和y是长度为n_grids的array
        # meshgrid会把x和y组合成n_grids*n_grids的array，X和Y对应位置就是所有格点的坐标
        X, Y = np.meshgrid(x, y)
        Z = Pxx[120:,:]
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='jet', rstride=1, cstride=1, lw=0)
        #plt.mesh(Pxx[120:,:])
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        #ax = fig.add_subplot(1, 2, 2)
        #plt.subplot(326)
        log2(Pxx,1111)
        print(Pxx.shape)
        x = np.linspace(0, 1, 261)
        y = np.linspace(0, 1, 120)
        Z = Pxx[:120,:]
        # x和y是长度为n_grids的array
        # meshgrid会把x和y组合成n_grids*n_grids的array，X和Y对应位置就是所有格点的坐标
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='jet', rstride=1, cstride=1, lw=0)
        ax.view_init(0,90)

        x = np.linspace(0, 1, 261)
        y = np.linspace(0, 1, 161)
        X, Y = np.meshgrid(x, y)
        Z = Pxx
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='jet', rstride=1, cstride=1, lw=0)
        #plt.mesh(Pxx[120:,:])
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        #ax = fig.add_subplot(1, 2, 2)
        #plt.subplot(326)
        log2(Pxx,1111)
        print(Pxx.shape)
        x = np.linspace(0, 1, 261)
        y = np.linspace(0, 1, 120)
        Z = Pxx[:120,:]
        # x和y是长度为n_grids的array
        # meshgrid会把x和y组合成n_grids*n_grids的array，X和Y对应位置就是所有格点的坐标
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='jet', rstride=1, cstride=1, lw=0)
        ax.view_init(90,0)
        '''
        # plot
        if debugMark==1:
            plt.draw()#plt.show();
        else:
            plt.show();
        picname = pureName[:-4]+"-%1.2f"%(snr1)+'.png' #'.jpg'
        Fig.savefig(os.path.join(path1,picname))
        plt.close('all')
    # 绘图 end ######################
    return 0,Beishu

def py_grep(srcTxt,tarStr):
    strcmd = 'grep '+tarStr+' '+srcTxt
    #aa = subprocess.check_output(strcmd,shell=True)
    try:
        aa = subprocess.check_output(strcmd,shell=True)
    #print('aa= '+ str(aa))
    except Exception:
        aa= ''
    return aa

def py_wc(srcTxt):

    strcmd = 'wc -l '+srcTxt
    aa = subprocess.check_output(strcmd,shell=True)
    num,name = aa.split()
    #print('aa= '+ str(aa))
    return int(num)

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
                if curRes.split(' ')[0]==tarStr[1:]:
                    aa = curRes
                    # log(100,aa)
                    # log(100,"*****************")
                    # sys.exit()
                    break
    except Exception:
        pass
    # log(2000,aa+'\n')
    return aa

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
     
def traveseFileFmt2(file_dir,frmStr): 
    fileList = []    
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

def log(ID,val):
    print('-----> '+str(ID)+' = '+str(val))    


def log2(ID,val):
    print('-----> '+str(ID)+' = '+str(val))    
    print('-----> '+str(ID)+'type = '+str(type(val)))    
    print('-----> '+str(ID)+'shape = '+str(val.shape) )   

def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))
    

def vad_Speech_noise_concate(X0,Fs,Xc,frmEnergy,SF,frameTime,nFrm,XCon1,voiceseg,maxsilence):
    (segNum,temp) = np.shape(voiceseg)
    N00 = len(X0)
    if segNum>1:
        #print('::>1, segNum= '+str(segNum))
        ts = int(voiceseg[0][0]);
        te = int(voiceseg[0][1]);
        if ts <= maxsilence:
            ts = 0;
        ts = int(frameTime[ts]*Fs);
        te = int(frameTime[te]*Fs);
        #print('segNum ='+str(segNum));
        #print('ts ='+str(ts));
        #print('te ='+str(te));
        XCon = X0[ts:te]
        #print(':::::::::::len(X0) =' + str(len(X0)))
        #print(':::::::::::len_XCon_p0 =' + str(len(X0[ts:te])))
        if ts>0:
            XCon_bkgrnd = X0[0:ts]
            #print('::::::::::::::::::::::len_XCon_bkgrnd_p0 =' + str(len(X0[0:ts])))
        else:
            XCon_bkgrnd = np.array([])
            #print('::::::::::::::::::::::len_XCon_bkgrnd_p0 = 0')
        for i1 in range(1,segNum-1):
            te_old =  int(voiceseg[i1-1][1])
            ie_old_ind = int(frameTime[te_old]*Fs) 
            #print('ie_old_ind =' + str(ie_old_ind))
            ts = int(voiceseg[i1][0])
            te = int(voiceseg[i1][1])
            ts_ind = int(frameTime[ts]*Fs)
            te_ind = int(frameTime[te]*Fs)
            Nseg = X0[ts_ind:te_ind]
            #print(':::::::::::len_XCon_p'+str(i1)+' =' + str(len(Nseg)))
            XCon = np.append(XCon,Nseg)
            
            Nseg_bkGrnd = X0[ie_old_ind:ts_ind]
            #print('::::::::::::::::::::::len_Nseg_bkGrnd_p'+str(i1)+' =' + str(len(Nseg_bkGrnd)))
            
            XCon_bkgrnd = np.append(XCon_bkgrnd,Nseg_bkGrnd)
        #print('len1 = '+str(len(XCon)));
        
        te_old =  int(voiceseg[segNum-2][1])
        ie_old_ind = int(frameTime[te_old]*Fs)
        
        ts_new =  int(voiceseg[segNum-1][0])
        is_new_ind = int(frameTime[ts_new]*Fs)
        Nseg_bkGrnd = X0[ie_old_ind:is_new_ind]
        XCon_bkgrnd = np.append(XCon_bkgrnd,Nseg_bkGrnd)
        #print('::::::::::::::::::::::len_Nseg_bkGrnd_p'+str(segNum-1)+' =' + str(len(Nseg_bkGrnd)))
        ts = int(voiceseg[segNum-1][0]);
        te = int(voiceseg[segNum-1][1]);
        if nFrm <= (maxsilence+te):
            #te = nFrm;
            te = N00;  # ori sig totel length
        else:
            te = int(frameTime[te]*Fs)    
                 
            te_old =  int(voiceseg[segNum-1][1])
            ie_old_ind = int(frameTime[te_old]*Fs)
            Nseg_bkGrnd = X0[ie_old_ind:N00]
            #print('::::::::::::::::::::::len_Nseg_bkGrnd_p'+str(segNum-1)+' =' + str(len(Nseg_bkGrnd)))
            XCon_bkgrnd = np.append(XCon_bkgrnd,Nseg_bkGrnd)   
        ts = int(frameTime[ts]*Fs)
        Nseg = X0[ts:te]
        #print(':::::::::::len_XCon_p'+str(segNum-1)+' =' + str(len(Nseg)))
        XCon = np.append(XCon,Nseg)  
        
    else:  # caution ; if only has one segment,then save it to src-0.wav.
        #print('::<=1, segNum= '+str(segNum))
        XCon_bkgrnd = np.array([])
        
        ts = int(voiceseg[0][0])
        te = int(voiceseg[0][1])
        if ts <= maxsilence:
            ts = 0
        else:
            ts = int(frameTime[ts]*Fs)
        if nFrm <= (maxsilence+te):
            #te = nFrm;
            te = N00;  # ori sig totel length
        else:
            te = int(frameTime[te]*Fs)

        #print('segNum ='+str(segNum));
        #print('ts ='+str(ts));
        #print('te ='+str(te));
        XCon = X0[ts:te]
        
        if (ts>0) and (te<N00):
            ts = int(voiceseg[0][0]);
            te = int(voiceseg[0][1]);
            ts = int(frameTime[ts]*Fs);
            te = int(frameTime[te]*Fs);
            Nseg_bkGrnd = X0[0:ts]
            XCon_bkgrnd = np.append(XCon_bkgrnd,Nseg_bkGrnd) 
             
            Nseg_bkGrnd = X0[te:N00]
            XCon_bkgrnd = np.append(XCon_bkgrnd,Nseg_bkGrnd) 
            
        elif(ts>0) and (te==N00) :
            ts = int(voiceseg[0][0]);
            te = int(voiceseg[0][1]);
            ts = int(frameTime[ts]*Fs);
            te = int(frameTime[te]*Fs);
            Nseg_bkGrnd = X0[0:ts]
            XCon_bkgrnd = np.append(XCon_bkgrnd,Nseg_bkGrnd) 
        elif(ts==0) and (te<N00) :
            Nseg_bkGrnd = X0[te:N00]
            XCon_bkgrnd = np.append(XCon_bkgrnd,Nseg_bkGrnd) 
        # ######################
    return XCon,XCon_bkgrnd
    
def vad_snr_cal(file0,debugMark):    
    
    #file0 = '/media/user01/_new6T/1-data/data_njgeneral/wav-filter-ok/wav-min2Sec-total2Hour-1321/A0038_0148.wav'
    #X0,Fs = read_audio(file0, target_fs=None)
    X0,Fs = read_wave_data(file0)
    path1,pureName = os.path.split(file0); 
    
    saveMark = 0
    dstDir = os.path.join(path1,pureName[:-4]+'-vad-seg')
    if saveMark >0:
        foldPrepare(dstDir)
    
    N00 = len(X0)
    pre_FrmN = 2;      # head extend frame Num;
    pos_FrmN = 8;      # tail extend frame Num;
    padMark = 0 ;      # 0: not padd with 0; 1 pad with 0;
    WlenT = 0.02;      # 0.03; #0.05    unit : ssaveMark
    sWlenT = 0.01;     #  0.015; #0.05  unit : s
    #debugMark = 2;
    maxPersegSecLen = 10  # define max time of per Seg in seconds.
    seg_sec_len = maxPersegSecLen/2.0
    X,Xc,frmEnergy,SF,frameTime,nFrm,XCon1,voiceseg,maxsilence,Beishu = cutSilence(X0,Fs,pre_FrmN,pos_FrmN,WlenT,sWlenT,padMark,debugMark,file0,seg_sec_len,dstDir,saveMark)
    XCon,XCon_bkgrnd = vad_Speech_noise_concate(X,Fs,Xc,frmEnergy,SF,frameTime,nFrm,XCon1,voiceseg,maxsilence)
    
    # PART SAVE.  
    #segName = os.path.join(dstDir,pureName[:-4]+'-VAD-speech'+'.wav')
    #write_audio(segName,XCon,Fs)
    #segName_non = os.path.join(dstDir,pureName[:-4]+'-VAD-noSpeech'+'.wav')
    #write_audio(segName_non,XCon_bkgrnd,Fs)
    if saveMark>0:
        segName = os.path.join(dstDir,pureName[:-4]+'-VAD-speech-2'+'.wav')
        wavSave(XCon,segName,Fs)
        segName_non = os.path.join(dstDir,pureName[:-4]+'-VAD-noSpeech-2'+'.wav')
        wavSave(XCon_bkgrnd,segName_non,Fs)
    ###############################   snr 
    if (len(XCon_bkgrnd) !=0) and (len(XCon_bkgrnd)!=0):
        SigAm = rms(XCon)
        refAm = rms(XCon_bkgrnd)
        snr1 = 20*np.log10(SigAm/(refAm+1e-12)+ 1e-12)

        XCon0 = XCon/(max(abs(XCon))+1e-12)
        XCon_bkgrnd0 = XCon_bkgrnd/(max(abs(XCon))+1e-12)

        snr1 = 20*np.log10(rms(XCon0)/(rms(XCon_bkgrnd0)+1e-12)+ 1e-12)

        '''
        log(snr2,2222)
        log(snr1,1111)
        log(XCon,333)
        log(XCon_bkgrnd,444)
        '''
    else:
        snr1 = -100   
    #avaSig = sum(XCon**2) / (len(XCon)+0.0)
    #avaRef = sum(XCon_bkgrnd**2)/(len(XCon_bkgrnd)+ 1e-12)
    #snr2 = 10*np.log10( avaSig / avaRef + 1e-12)
    return snr1,X0,XCon,XCon_bkgrnd,Beishu


def batch_vad_snr(srcDir):
    # s2: travese file through dir; get file list ; make parent dir and file list;
    #fileList = traveseFileFmt(srcDir,'.txt')
    frmStr = '.wav'
    fileList = traveseFileFmt2(srcDir,frmStr)
    timeAll = 0
    debugMark = 1
    ii = 0
    NN = len(fileList)
    for ii in range(0,NN):
        file0 = fileList[ii]
        log(file0,'--->>>>> :')
        vad_snr_cal(file0,debugMark)
        print(">>>  %d/%d"%(ii,NN))
        #break



def add_mix_snr_main():
    # s1: check paras;
    print(sys.argv[0])
    if len(sys.argv) != 5:
        print(len(sys.argv))
        print('usage: '+ sys.argv[0] +'srcDir noiseDir dstDir detailTXT')
        sys.exit()
    srcDir = sys.argv[1]
    noiseDir = sys.argv[2]
    dstDir = sys.argv[3]
    detailTXT = sys.argv[4]
    foldPrepare(dstDir)
    foldPrepare(detailTXT)
    
    #foldPrepare(dstFile)
    # s2: travese file through dir; get file list ; make parent dir and file list;
    #fileList = traveseFileFmt(srcDir,'.txt')
    frmStr = '.wav'
    fileListSRC = traveseFileFmt2(srcDir,frmStr)
    fileListNOISE = traveseFileFmt2(noiseDir,frmStr)
    
    f1 = codecs.open(detailTXT,'w+','utf-8')
    
    Nsrc = len(fileListSRC)
    Nnoise = len(fileListNOISE)
    
    print(Nsrc)
    print(Nnoise)
    for ii in range(0,Nsrc):
        sigWav = fileListSRC[ii]
        newName = strrepWT(sigWav,srcDir,dstDir)
        #i = i+1
        foldPrepare(newName)
        #  
        sig,fs = soundfile.read(sigWav)
        # randn
        curNum = np.random.randint(0,Nnoise)
        curNoise = fileListNOISE[curNum]
        noise,sr = read_audio(curNoise, target_fs=fs)
        #
        
        snr = np.random.choice([0,5,8,10,15])
        mixed_audio, s, n, alpha = add_noise_snr(sig,noise,snr)
        soundfile.write(newName,mixed_audio,fs)
        
        strdetail = os.path.basename(sigWav)+' : '+os.path.basename(curNoise) +' : '+ str(snr)+' dB\n'

        print(ii)
        f1.write(strdetail)
        
    f1.close()
    print(' end !')
#  python3 python3 clu_txt_im.py /home/user01/ASR/4-预处理-分析程序/Data-asr/py-score-V4/txt_prepare/chengYu-ok 3.txt

#又一个按照key值排序，貌似比上一个速度要快点 
def sortedDictValues2(adict): 
    keys = adict.keys() 
    keys.sort() 
    return [dict[key] for key in keys] 

#还是按key值排序，据说更快。。。而且当key为tuple的时候照样适用 
def sortedDictValues3(adict): 
    keys = adict.keys() 
    keys.sort() 
    return map(adict.get, keys) 

##############
def get_seg_trans_cmb(txtFile):
                                    
    f2 = codecs.open(txtFile,'r+')
    jsonData = f2.readline()
    f2.close()
    
    text = json.loads(jsonData)

    ss = text['ch01']
    jsonData = str(ss)
    aa = jsonData.split('},')
    

    tranRes = list()
    for ii in range(0,len(aa)):
        #print(str(ii)+'='+str(aa[ii])+'\n')
        str1 = str(aa[ii])
        ssaa = str1.split(',')
        timeStart00 = ssaa[4]
        timeStart = timeStart00.split(':')[1]
        timeEnd = ssaa[2]   
        markText = ssaa[7]
        tt = str(markText.split(':')[1])
        if len(tt)>4:
            cur = timeStart[3:-1]
            #print('cur = '+str(cur))
            tranRes.append({'startTime':float(cur),'transCon':tt[3:-1]})
            
    tranRes.sort(key=lambda x:x['startTime'],reverse=False)
    strOK =''
    for ii in range(0,len(tranRes)):
        ss = tranRes[ii]
        strOK = strOK+' '+str(ss['transCon'])
    strOK2 = strOK.decode('unicode-escape')
    
    return strOK2 

def batch_get_seg_trans_cmb(srcDir,dstDir):

    frmStr = '.txt'
    fileListSRC = traveseFileFmt2(srcDir,frmStr)
    Nsrc = len(fileListSRC)
    for iijj in range(0,Nsrc):
        sigWav = fileListSRC[iijj]
        #print('######################  *** '+str(iijj))
        #print(sigWav)
        newName = strrepWT(sigWav,srcDir,dstDir)
        foldPrepare(newName)
        f1 = codecs.open(newName,'w+','utf-8')
        strOK = get_seg_trans_cmb(sigWav)
        f1.write(strOK)
        f1.close()

def get_wav_chNum(wavFile):
    from scipy.io import wavfile
    Fs, X = wavfile.read(wavFile)
    channel_num = len(X.shape)
    return channel_num
##############
def get_seg_trans_cmb2(txtFile):
     
    with open(txtFile) as f:
        text=json.load(f)
    pureNmae = os.path.basename(txtFile)[:-4]
    uttId = text['uttId']
    ch01 = text['ch01']
    segList = list()
    seg_text_list = list()
    utt_seg_id = 1
    for ii in range(0,len(ch01)):
        currr = ch01[ii]
        # print('%d----------->currr ='%ii)
        # print('currr[uttIdNum]     '+str(currr['uttIdNum']))
        # print('currr[noiseSign]    '+str(currr['noiseSign']))
        # print('currr[timeEnd]      '+str(currr['timeEnd']))
        # print('currr[sex]          '+str(currr['sex']))
        # print('currr[timeStart]    '+str(currr['timeStart']))
        # print('currr[channelNumber]'+str(currr['channelNumber']))
        # print('currr[role]         '+str(currr['role']))
        # print(currr['markText'])
        # print('len = '+str(len(currr['markText'])))
        # print('currr[localism]     '+str(currr['localism']))
        # print('currr[isMultiSpk]   '+str(currr['isMultiSpk']))
        # print('currr[noiseType]    '+str(currr['noiseType'])+'\n')

        # filter target txt:
        if (str(currr['noiseSign'])=='VoiceSegments') and (len(currr['markText']) > 0):
            seg_single = str(pureNmae)+'-'+str(utt_seg_id)+' '+str(pureNmae)+' '+str(currr['timeStart'])+' '+str(currr['timeEnd'])
            seg_single = str(pureNmae) + '-' + str(utt_seg_id) + ' ' + str(pureNmae) + ' ' + str(
                currr['timeStart']) + ' ' + str(currr['timeEnd'])+ ' '+currr['markText']
            seg_single_text = str(pureNmae) + '-' + str(utt_seg_id) + ' '+currr['markText']
            segList.append(seg_single)
            seg_text_list.append(seg_single_text)
            utt_seg_id = utt_seg_id + 1
    return segList, seg_text_list


def get_seg_trans_cmb2_sort(txtFile):
    with open(txtFile) as f:
        text = json.load(f)
    pureNmae = os.path.basename(txtFile)[:-4]
    uttId = text['uttId']
    ch01 = text['ch01']
    segList = list()
    seg_text_list = list()
    temp_list = list()
    utt_seg_id = 1
    for ii in range(0, len(ch01)):
        currr = ch01[ii]
        # filter target txt:
        if (str(currr['noiseSign']) == 'VoiceSegments') and (len(currr['markText']) > 0) and (float(currr['timeEnd']) > float(currr['timeStart'])):
            seg_single = {'utt':str(pureNmae), 'timeStart':float(currr['timeStart']), 'timeEnd':float(currr['timeEnd']),'markText':currr['markText']}
            temp_list.append(seg_single)
            # print('---timeStart= |'+str(currr['timeStart'])+'|-----')
    temp_list.sort(key=lambda x: x['timeStart'],reverse = False)  # reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）
    markText_all = ''

    sorted_res_dict_list = list()
    for ii in range(0,len(temp_list)):
        ss = temp_list[ii]
        # standard
        s_seg = str(ss['utt']) + '-' + str(utt_seg_id) + ' ' + str(ss['utt']) + ' ' + str(ss['timeStart']) + ' ' + str(
            ss['timeEnd'])
        # debug
        s_seg = str(ss['utt'])+'-'+str(utt_seg_id)+' '+str(ss['utt'])+' '+str(ss['timeStart'])+' '+str(ss['timeEnd'])+' '+ss['markText']
        s_seg_text = str(ss['utt'])+'-'+str(utt_seg_id)+' '+ss['markText']
        segList.append(s_seg)
        seg_text_list.append(s_seg_text)
        markText_all = markText_all+' ' + ss['markText']

        seg_single = {'utt': str(ss['utt']), 'timeStart': float(ss['timeStart']), 'timeEnd': float(ss['timeEnd']),'markText': ss['markText'], 'utt_seg_id':utt_seg_id}
        sorted_res_dict_list.append(seg_single)

        utt_seg_id = utt_seg_id + 1
    # strOK2 = strOK.decode('unicode-escape')
    return segList, seg_text_list, markText_all, sorted_res_dict_list


def batch_get_seg_trans_cmb2_sort(srcDir, dstDircmb):
    # foldPrepare(dstFile)
    # s2: travese file through dir; get file list ; make parent dir and file list;
    # fileList = traveseFileFmt(srcDir,'.txt')
    frmStr = '.txt'
    fileListSRC = traveseFileFmt2(srcDir, frmStr)

    Nsrc = len(fileListSRC)
    for iijj in range(0, Nsrc):
        sigWav = fileListSRC[iijj]
        # print('######################  *** '+str(iijj))
        # print(sigWav)
        newName3 = strrepWT(sigWav, srcDir, dstDircmb)
        foldPrepare(newName3)
        f3 = codecs.open(newName3, 'w+', 'utf-8')
        segCon, seg_text_list, markText_all, sorted_res_dict_list = get_seg_trans_cmb2_sort(sigWav)
        f3.write(markText_all + '\n')
        f3.close()
        # sys.exit()

def batch_get_seg_trans_sort(srcDir, dstDircmb,dstDir,dstDir2):
    # dstDir = dstDir-seg
    # dstDir2 = dstDir2-seg-text
    frmStr = '.txt'
    fileListSRC = traveseFileFmt2(srcDir, frmStr)

    Nsrc = len(fileListSRC)
    for iijj in range(0, Nsrc):
        sigWav = fileListSRC[iijj]
        # print('######################  *** '+str(iijj))
        # print(sigWav)
        newName0 = strrepWT(sigWav, srcDir, dstDir)
        newName = newName0[:-4] + '-seg.txt'
        newName02 = strrepWT(sigWav, srcDir, dstDir2)
        newName2 = newName02[:-4] + '-seg_text.txt'
        newName3 = strrepWT(sigWav, srcDir, dstDircmb)
        foldPrepare(newName3)
        foldPrepare(newName)
        f1 = codecs.open(newName, 'w+', 'utf-8')
        f2 = codecs.open(newName2, 'w+', 'utf-8')
        f3 = codecs.open(newName3, 'w+', 'utf-8')
        segCon, seg_text_list, markText_all, sorted_res_dict_list = get_seg_trans_cmb2_sort(sigWav)
        for ii in range(0, len(segCon)):
            f1.write(segCon[ii] + '\n')
            f2.write(seg_text_list[ii] + '\n')
        f3.write(markText_all + '\n')
        f1.close()
        f2.close()
        f3.close()
        # sys.exit()


def batch_get_seg_cut_wav(srcTxtDir, dstDir_cmb, dstDir_seg, dstDir_segText, srcWavDir, tarWavDir):
    foldPrepare(dstDir_cmb)
    foldPrepare(dstDir_seg)
    foldPrepare(dstDir_segText)
    foldPrepare(tarWavDir)

    #foldPrepare(dstFile)
    # s2: travese file through dir; get file list ; make parent dir and file list;
    #fileList = traveseFileFmt(srcTxtDir,'.txt')
    frmStr = '.txt'
    fileListSRC = traveseFileFmt2(srcTxtDir, frmStr)

    fileListWAV = traveseFileFmt2(srcWavDir, '.wav')
    assert len(fileListWAV) == len(fileListSRC), ' src txt and wav len dont match !'

    Nsrc = len(fileListSRC)
    for iijj in range(0,Nsrc):
        sigTxt = fileListSRC[iijj]
        sigWav = fileListWAV[iijj]
        # print('######################  *** '+str(iijj))
        # print(sigTxt)
        newName0 = strrepWT(sigTxt,srcTxtDir,dstDir_seg)
        newName = newName0[:-4]+'-seg.txt'
        newName02 = strrepWT(sigTxt, srcTxtDir, dstDir_segText)
        newName2 = newName02[:-4] + '-seg_text.txt'
        newName3 = strrepWT(sigTxt, srcTxtDir, dstDir_cmb)
        foldPrepare(newName3)
        foldPrepare(newName)
        f1 = codecs.open(newName,'w+','utf-8')
        f2 = codecs.open(newName2, 'w+', 'utf-8')
        f3 = codecs.open(newName3, 'w+', 'utf-8')
        segCon, seg_text_list, markText_all, sorted_res_dict_list = get_seg_trans_cmb2_sort(sigTxt)

        # wav read
        data,Fs = soundfile.read(sigWav)
        for ii in range(0, len(segCon)):
            f1.write(segCon[ii]+'\n')
            f2.write(seg_text_list[ii]+'\n')


            ss = sorted_res_dict_list[ii]
            st = ss['timeStart']
            et = ss['timeEnd']
            seg_id = ss['utt_seg_id']
            pureName = str(ss['utt'])
            sNum = max(int(Fs*st),0)
            eNum = min(int(Fs*et),len(data))

            cur_seg_wav = data[sNum:eNum]
            cur_wav_file_name0 = strrepWT(sigWav, srcWavDir, tarWavDir)
            cur_wav_file_name = strrepWT(cur_wav_file_name0, pureName, pureName+'_'+str(seg_id))

            cur_wav_file_name = os.path.join(os.path.join(os.path.split(cur_wav_file_name)[0], pureName), pureName+'_'+str(seg_id)+'.wav')
            foldPrepare(cur_wav_file_name)
            soundfile.write(cur_wav_file_name, cur_seg_wav, Fs)
            # seg-txt write
            cur_file_name_txt = cur_wav_file_name[:-4]+'.txt'
            f0 = codecs.open(cur_file_name_txt, 'w+', 'utf-8')
            f0.write(ss['markText'])
            f0.close()
        f3.write(markText_all+'\n')
        f1.close()
        f2.close()
        f3.close()

        print('---->  batch_get_seg_cut_wav  end ok')


def fenci_ZXZDPP_(text,dic):
    MMresult = []
    text_length = len(text)

    index = 0
    ww = text_length #7  # 每次取七个字符来匹配
    while text_length > index:
        for enP in range(ww + index, index, -1):
            piece = text[index:enP]
            #print(piece)  # 输出每次尝试匹配的字符串
            if piece in dic:
                index = enP - 1
                break
        index = index + 1
        # MMresult.append(piece + '--')
        MMresult.append(piece)
    return MMresult

def fenci_NXZDPP_(text,dic):
    MMresult = []
    text_length = len(text)

    index = text_length
    ww = text_length #7  # 每次取七个字符来匹配  index = -1 index-1=-2; index-w=-w-1
    while index >0:
        for stP in range(max(index-ww,0), index, 1):
            #print(size, ':', ww + index, index, -1)  # 输出遍历索引的过程
            piece = text[stP:index]
            #print(piece)  # 输出每次尝试匹配的字符串
            if piece in dic:
                index = stP+1
                break
        index = index - 1
        # MMresult.append(piece + '--')
        MMresult.append(piece)
    NNreslt=[]
    for ii in range(len(MMresult)-1,-1,-1):
        NNreslt.append(MMresult[ii])
    return NNreslt



def get_txt_line_list(srcTxt):
    f = codecs.open(srcTxt,'r+','utf-8')
    line = f.readline()
    resList = []
    while True:
        if not line:
            break
        else:
            line = line.strip()
            resList.append(line)
            line = f.readline()
    f.close()
    return resList

def get_txt_line_list2(srcTxt):
    f = codecs.open(srcTxt,'r+','utf-8')
    line = f.readline()
    resList = []
    while True:
        if not line:
            break
        else:
            line = line.strip()
            line = line.split(' ')[0].strip()
            resList.append(line)
            line = f.readline()
    f.close()
    return resList


def check_contain_eng(inStr):
    flag=False
    for ch in inStr:
        if u'\u0041'< ch< u'\u005a' or u'\u0061'< ch <u'\u007a':
            flag=True
            break
    return flag
def check_contain_eng_digit(inStr):
    flag=False
    for ch in inStr:
        if u'\u0041'< ch< u'\u005a' or u'\u0061'< ch <u'\u007a' or  u'\u0030'< ch <u'\u0039':
            flag=True
            break
    return flag
  
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
    
        try:
            if 0<=int(ch)<=9:
                continue
        except:
            pass
        #if ch =="'" or ch=="[" or ch == "]":
        if ch =="'" :
            continue
        if not(u'\u0041'< ch< u'\u005a' or u'\u0061'< ch <u'\u007a' or  u'\u0030'< ch <u'\u0039' or u'\u03B1'< ch <u'\u03C9' ):
            flag=False
            break
    return flag
     
          
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
        #    log(100,ch)
        #    log(100,ch.encode('Unicode-escape'))
            
                
    return lineWord[:-1],ilegleMark,badList


def Fan2Jan_openccpy_test():
    line = "畼"
    #line = " 畼续"
    #line = "得 继续 招 畼"
    line2 = Fan2Jan_openccpy(line)
    log(11,line)
    log(12,line2)

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

    #Fan2Jan_openccpy_test()
    #sys.exit()

    import jieba
    # s1: check paras;
    print(sys.argv[0])
    if len(sys.argv) <4:
        print(len(sys.argv))
        print('usage: '+ sys.argv[0] + ' srcTxt resTxt lexicon1Txt lexicon2_eng_Txt(op)')
        sys.exit()
    
    srcTxt = sys.argv[1]
    resTxt = sys.argv[2]
    user_dict = sys.argv[3]
    # 备注： 添加到 user_dict: user_dict.txt 中的词，必须一个词一行，词的字之间不能有空格或其他符号。并且尾部不能添加符任何号。
    #jieba.load_userdict(user_dict)
    #lexicon = get_txt_line_list(user_dict)
    jieba.set_dictionary(user_dict)   # set use fict as first dict.:   aa 1\n bb 1\n...
    print('----use user-dict in '+str(user_dict)+'\n\n\n')
    useDictMark = 0
    if len(sys.argv) > 4:
        lexicon2_eng_Txt = sys.argv[4]
        lexicon = get_txt_line_list2(lexicon2_eng_Txt)
        useDictMark = 1
    log(0,"hhlll")
    foldPrepare(resTxt)
    resTxt_err = resTxt[:-4]+'-err-bad-line.txt'
    foldPrepare(resTxt_err)
    
    #resTxt_1 = resTxt[:-4]+'-1word.txt'
    #foldPrepare(resTxt_1)
    
    
    f_resTxt = codecs.open(resTxt,'w+','utf-8')
    f_resTxt_err = codecs.open(resTxt_err,'w+','utf-8')
    #f_resTxt_1 = codecs.open(resTxt_1,'w+','utf-8')
    
    
    f_srcTxt = codecs.open(srcTxt,'r+','utf-8')
    #Nsrc =  py_wc(srcTxt)
    sigTxt = f_srcTxt.readline()
    cnt = 0 
    #cc = opencc.OpenCC('t2s')
    cnt_err = 0
    #log(999,'hehehe1')
    
    # 中英文混合翻译的切分 匹配
    pattern = re.compile(r'[a-z]*[\']*[a-z]+[\']*')

    while True:
        if not sigTxt:
            break
        else:
            pureName = ''
            if 1:
                pureName = sigTxt.split('\t')[0]+' '
                sigTxt = sigTxt[len(pureName)-1:]
                sigTxt = sigTxt.strip()
                # log(101,sigTxt)
                    
            if 0 : # len(sigTxt)<2:
                log('-----------err\t :',pureName+'\t'+sigTxt)
                
                f_resTxt_1.write(pureName+'\t'+sigTxt+'\n')
                
                sigTxt = f_srcTxt.readline()
                continue
            # jieba
            #print(str(type(sigTxt)))
            if type(sigTxt)=='''<class 'bytes'>''':
                content = sigTxt.encode('utf-8') 
                print('bytes input ')
            else:
                content = sigTxt.strip()
            
            #log(cnt,content)
            content = stringQ2B(content)
            '''
            log(cnt,content)
            sigTxt = f_srcTxt.readline()
            cnt = cnt+1
            
            continue
            '''
            
            content = strrepWT(content, r'ー', '一') 
            content = strrepWT(content, r'↙', ' ') 
            content = strrepWT(content, r'↗', ' ') 
            #content = strrepWT(content, 'ens', '\[c\]')  
            content = strrepWT(content, r'X﹏X', ' ')   
            #content = re.sub(u"[，,。.：:、<>？?！!<<>>《》（）…－＋( )-+·゛-↙╭╯╰╮﹏↗ ]"," ",content)   # delte fuhao   
            #content = re.sub(u"[，,。.：:、<>？?！!<<>>《》（）…－＋( )-+·゛-・]"," ",content)   # delte fuhao   
            
            content = strrepWT(content, r'"', ' ')
            content = re.sub(u"[~‘’【】，,。.：:、<>？?！!<<>>《》（）…－＋( )-+·゛-・“”;]"," ",content)   # delte fuhao   
            
            content = re.sub(r' +','',content) 
            # log(102,content)
            # fan2Jan
            #content = Converter('zh-hans').convert(content)
            #content,ilegleMark = Fan2Jan_opencc(content,cc)
            
            content,ilegleMark,badList = Fan2Jan_openccpy(content)
            
            #log(cnt,ilegleMark)          
            if ilegleMark==1:
            
                f_resTxt_err.write(sigTxt+'   :'+str(badList)+'\n')
                cnt_err = cnt_err +1
                
                if 0:
                    if np.mod(cnt_err,1000)==0:
                        f_resTxt_err.flush()
                    
                    sigTxt = f_srcTxt.readline()
                    cnt = cnt+1
                    continue
            
            # del head and tail \n \r
            content = content.strip()    # delete fuhao in head and tail,like \n,\r.
            if len(content)==0:
                log('-----------err\t :',pureName+'\t'+sigTxt)
                f_resTxt_err.write(sigTxt+'   :'+str(badList)+'\n')
                sigTxt = f_srcTxt.readline()
                continue
            content = content.lower()    # change to lowe format
            
            # pattern
            matRes = pattern.findall(content)
            spRes = pattern.split(content)
            cur_markText_ok = ''
            cur_markText_ok1 = ''
            # 分词后 复原
            # for jj in range(0,len(spRes)-1):
            #     cur_markText_ok1 = cur_markText_ok1 +spRes[jj]+matRes[jj]
            # cur_markText_ok1 = cur_markText_ok1+spRes[-1]
            # log('cur_markText_ok1',cur_markText_ok1)
            for jj in range(0,len(spRes)-1):
                Chi = ' '.join(jieba.lcut(spRes[jj],cut_all=False,HMM=False))
                Eng = ' '.join(eng_sentence_split_ZX(matRes[jj],lexicon))
                # Eng = ' '.join(eng_sentence_split_NX(matRes[jj],lexicon))
                cur_markText_ok = cur_markText_ok +' '+Chi+' '+Eng
            Chi = ' '.join(jieba.lcut(spRes[-1],cut_all=False,HMM=False))
            cur_markText_ok = cur_markText_ok+' '+Chi
            
            if 0:
                segs_5 = jieba.lcut(content,cut_all=False,HMM=False)
                log(103,segs_5)
                # judge if ori txt-line have English:
                # content11 = re.sub(r'\[n\]\[c\]\[l\]','',content)
                #log(1,content)
                content11 = strrepWT(content, '\[c\]', '')
                content11 = strrepWT(content11, '\[n\]', '')
                content11 = strrepWT(content11, '\[l\]', '')
                log(104,content11)
                #log(111,content11)
                haveEng = check_contain_only_Chinese(content11)
                #log(111222,haveEng)
                strIN="let'splaytheguitar"
                resList = eng_sentence_split(strIN,lexicon)
                log('strIN',strIN)
                log('resList',' '.join(resList))
                sys.exit()
                
                if not haveEng:
                    if useDictMark ==1:    
                        text = segs_5
                        text_list= []
                        NN = len(text)
                        for ii in range(0,NN):
                            cur_str = text[ii]
                            haveEngCur = check_contain_only_Chinese(cur_str)
                            if not haveEngCur:
                                if cur_str not in lexicon:
                                    cur_str = fenci_ZXZDPP_(cur_str, lexicon)
                                    mm = len(cur_str)
                                    for jj in range(0,mm):
                                        text_list.append(cur_str[jj])
                                else:
                                    text_list.append(cur_str)
                            else:
                                text_list.append(cur_str)       
                        segs_5 = text_list 
                #log(3,segs_5)
                cur_markText_ok = ' '.join(segs_5) #.encode('utf-8')
            # log(105,cur_markText_ok)
            cur_markText_ok = strrepWT(cur_markText_ok, '\[ c \]', '[C]')
            cur_markText_ok = strrepWT(cur_markText_ok, '\[ l \]', '[L]')
            cur_markText_ok = strrepWT(cur_markText_ok, '\[ n \]', '[N]')
            # log(106,cur_markText_ok)


            str_cur_txt =  cur_markText_ok
            str_cur_txt = str_cur_txt.strip()
            #og(4, str_cur_txt)
            str_cur_txt = re.sub(r' +',' ',str_cur_txt)
            str_cur_txt = pureName+str_cur_txt+'\n'
            f_resTxt.write(str_cur_txt)
            
            cnt = cnt + 1
            if str(cnt)[-3:]=="000":
                f_resTxt.flush()
                #f_resTxt_1.flush()
                f_resTxt_err.flush()
                log(cnt, sigTxt)
                log(cnt, cur_markText_ok+'\n')
                #sys.exit()
            sigTxt = f_srcTxt.readline()



    f_resTxt.close()
    f_srcTxt.close()
    f_resTxt_err.close()
    #f_resTxt_1.close()
    print(' end !')
'''
eg:
taitan wt:
cd /media/user01/_new6T/4-code/Data-pre/py-lm-jieba-split
python3 py_jieba_split_v4.py test_no_space test_no_space-sp.txt lexicon-jieba-0725-su-fre.txt lexicon-jieba-0725-eng-su-fre.txt 


python3 wt_aishell_to_segments.py ai-test/transcript/aishell_transcrip
t_v0.8.txt ai-test/wav/ ai-test/seg_cmb/


python3 wt_aishell_to_segments_add_spk_jieba.py ai-test/transcript/aishell_transcript_v0.8.txt ai-test/wav/ ai-test/seg_cmb_jieba/


txt-seg-txt :
    3ca89280-b89d-4752-97c7-b58086ece4e0-seg_text.txt
        3ca89280-b89d-4752-97c7-b58086ece4e0-1 哎 你 老师 我想问一下 违反自然管理办法 他的处罚怎么处罚
        3ca89280-b89d-4752-97c7-b58086ece4e0-2 哦 可以 浦东的
        3ca89280-b89d-4752-97c7-b58086ece4e0-3 浦东
        3ca89280-b89d-4752-97c7-b58086ece4e0-4 哎
        3ca89280-b89d-4752-97c7-b58086ece4e0-5 哎 哎 要什么 什么处罚 我想问一下
        3ca89280-b89d-4752-97c7-b58086ece4e0-6 哎
        3ca89280-b89d-4752-97c7-b58086ece4e0-7 好好 谢谢谢谢谢谢

tar-ok
    3ca89280-b89d-4752-97c7-b58086ece4e0_1.txt
    3ca89280-b89d-4752-97c7-b58086ece4e0_1.wav
    3ca89280-b89d-4752-97c7-b58086ece4e0_2.txt
    3ca89280-b89d-4752-97c7-b58086ece4e0_2.wav


segments:
 
    P05_S02_DINING.L-0291234-0292406 P05_S02.L 02912.34 02924.06
    P05_S02_DINING.L-0292496-0293128 P05_S02.L 02924.96 02931.28


text:

    P05_S02_DINING.L-0291234-0292406 [laughs] yes here now this is the next phase [laughs] check [laughs] yeah that works
    P05_S02_DINING.L-0292496-0293128 we've got this abhi there's f- the fruit i'll get out how am i do we have serving spoons


'''


##############






