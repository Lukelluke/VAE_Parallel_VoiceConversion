# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import random

from tqdm import tqdm
import sys
import librosa
import numpy as np
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow import PrefetchData
from audio import read_wav, preemphasis, amp2db
from hparam import hparam as hp
from utils import normalize_0_1
import pyworld as world


class DataFlow(RNGDataFlow):
    """
    data_path：原始说话人的音频路径；
    """
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.wav_files = glob.glob(data_path)
        # glob.glob(): 匹配所有的符合条件的文件，并将其以 list 的形式返回(所有文件名)
        self.wav_feats = dict()

    # __call__ 让一个类实例也可以变成一个 可调用对象
    def __call__(self, n_prefetch=1000, n_thread=1):
        df = self
        df = BatchData(df, self.batch_size)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df


class Net1DataFlow(DataFlow):

    def get_data(self):
        while True:
            source_file = random.choice(self.wav_files)

            # random.choice( seq  )  :返回随机项
            # seq -- 可以是一个列表，元组或字符串。
            yield get_mel(fpath=source_file)


def get_mel(fpath:str):
    wav, sr = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float64)  # 返回音频信号值 & 采样率
    # f0, timeaxis = pw.harvest(wav, hp.SR)
    # sp = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)  # sp:频谱包络；pw.cheaptrick 谐波频谱包络估计算法
    # # win size800，hop200，fft 1024
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80,  n_fft=1024, hop_length=200)
    # shape=(n_mels, t) = (80, t)
    mel = mel.T  # (t, 80) ：（帧，维度）
    return mel
