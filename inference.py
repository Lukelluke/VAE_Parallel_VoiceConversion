from model import Graph
import tensorflow as tf
from hyperparams import hyperparams
import numpy as np
import tqdm
from audio_taco import inv_mel_spectrogram
from utils import get_sp
from utils import get_f0
from utils import get_mel
import hparam as hparams
import pyworld as pw
from audio_taco import melspectrogram, preemphasis
import hparam as hparams
import librosa
import os

hp = hyperparams()


"""
合成要sp和f0
sp是网络预测的

找好以下内容：
原始的sp，ap，f0，编码过的sp
目标人的sp，ap，f0，编码过的sp
"""

# librosa.feature.inverse.mel_to_audio()

# def synthesis(ori_path, aim_sp, aim_spkid,mean_f0):
#     print('synthesizing ...')
#     wav, _ = librosa.load(ori_path, sr=hp.SR, mono=True, dtype=np.float64)
#     f0, timeaxis = pw.harvest(wav, hp.SR)
#     sp_per_timeaxis_before = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)  # 1024 压缩到 513 维
#
#     ap = pw.d4c(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)
#     aim_decoded_sp = pw.decode_spectral_envelope(aim_sp, hp.SR, fft_size=hp.N_FFT)  # 转换/解码 后的sp：
#
#     synwav = pw.synthesize(mean_f0, aim_decoded_sp, ap, hp.SR)
#     print(f'synthesize done. path : ./convert_to_{aim_spkid}_test1.wav')
#     librosa.output.write_wav(f'./convert_to_{aim_spkid}_test1.wav', synwav, sr=hp.SR)
#
#     wav, _ = librosa.load(f'./convert_to_{aim_spkid}_test1.wav', sr=hp.SR, mono=True, dtype=np.float64)
#     f0, timeaxis = pw.harvest(wav, hp.SR)
#
#     aim_feat = get_sp(f'./convert_to_{aim_spkid}_test1.wav')  # 合成出来的sp
#     print('合成出来的sp:')
#     print(aim_feat)
#     f0, timeaxis = pw.harvest(wav, hp.SR)
#     sp_per_timeaxis_afetr = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)  # 舍近求远，转换后的sp：写进wav再提出来会损失


def synthesis(ori_path, aim_mel):
    aim_spkid = 'spk2'
    print('synthesizing ...')
    wav, sr = librosa.load(ori_path, sr=hp.SR, mono=True, dtype=np.float64)
    """
    ` hp.SR = 22050
    ` mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80,  n_fft=1024, hop_length=200)
    
    def mel_to_audio(M, sr=22050, n_fft=2048, hop_length=512, win_length=None,
                 window='hann', center=True, pad_mode='reflect', power=2.0, n_iter=32,
                 length=None, dtype=np.float32, **kwargs):
    """
    # synwav = librosa.feature.inverse.mel_to_audio(aim_mel, sr=hp.SR, n_fft=1024, hop_length=200)
    synwav = inv_mel_spectrogram(aim_mel, hparams)
    synwav = synwav / np.abs(synwav).max()
    print(f'synthesize done. path : ./convert_to_{aim_spkid}_test1.wav')
    librosa.output.write_wav(f'./convert_to_{aim_spkid}_test1.wav', synwav, sr=hp.SR)



def main():
    print("line 69")
    fpath = './paralleldata/VCC2SF1/SF1|10001.wav'
    fpath = fpath.strip()
    print(fpath)  # ./data/convert/TMM1/M10007.wav

    new_wav_for_infer, sr = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float32)  # 返回音频信号值 & 采样率

    ori_mel = melspectrogram(new_wav_for_infer, hparams)  # 斌的：原始说话人的sp，接下来用np.random.normal生成同样形状的
    ori_mel = ori_mel.T
    print('源说话人的mel.shape：')
    print(ori_mel.shape)  # sp 是二维的！！！

    mode = 'infer'
    G = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        try:
            print(f'Try to load trained model in {hp.MODEL_DIR} ...')
            saver.restore(sess, tf.train.latest_checkpoint(hp.MODEL_DIR))
        except:
            raise Exception(f'Load trained model failed in {hp.MODEL_DIR}, please check ...')
        finally:
            # ori_mel = np.reshape(ori_mel, (-1, hp.CODED_DIM))  # 斌的：原始说话人的sp
            ori_mel_batch = np.expand_dims(ori_mel, axis=0)   # 模型训练的时候是三维度的  # ???;变成【1，None，80】

            # aim_spkid_batch = np.array([[aim_spkid]])
            for j in tqdm.tqdm(range(1)):
                aim_out = sess.run(G.aim_out, {G.ori_mel: ori_mel_batch})

            aim_out = np.array(aim_out, dtype=np.float64)  # 转换出来的 mel，三维的（第一个是batch=1）
            predict_mel = np.reshape(aim_out, (-1, hp.CODED_DIM))  # [None, 80]，转换成二维mel
            print("line 103 predict_mel.shape = "+str(predict_mel.shape))
            predict_mel = predict_mel.T
            # ori_new_f0 = np.random.normal(source_mean_f0, 1.0, predict_mel.shape[0])

            print('Sp predicted done.')
            synthesis(fpath, aim_mel=predict_mel, )

if __name__ == '__main__':
    print("line105")
    main()
