import pyworld as pw
from hyperparams import hyperparams
import tensorflow as tf
import numpy as np
import librosa
hp = hyperparams()

def get_sp(fpath: str): # 原始人的sp
    wav, _ = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float64)  # librosa.load 返回音频信号值 & 采样率
                                                                         # mono=False声音保持原通道数
    f0, timeaxis = pw.harvest(wav, hp.SR)
    sp = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)  # sp:频谱包络；pw.cheaptrick 谐波频谱包络估计算法
    # sp = pw.cheaptrick(wav, mean_f0, timeaxis, hp.SR, fft_size=hp.N_FFT)
    coded_sp = pw.code_spectral_envelope(sp, hp.SR, hp.CODED_DIM)
    # 将频谱包络sp再 ？压缩？;返回值是：ndarray
    # pw.code_spectral_envelope ：减小频谱包络 和 非周期性的 尺寸 。
    # https://blog.csdn.net/weixin_32393347/article/details/88623256
    coded_sp = coded_sp.T  # ndarray 的 转置矩阵
    return np.array(coded_sp)

# 0314_黄圣杰

def get_f0(fpath:str):  # 求原始/目标人的均值，然后再拿出去做差
    wav, _ = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float64)  # librosa.load 返回音频信号值 & 采样率
    f0, timeaxis = pw.harvest(wav, hp.SR)  # f0是一维数组，每帧会有一个f0
    # return sum(f0) / sum(timeaxis)
    return f0
    # total_f0 = 0
    # num = 0
    # for i in range(timeaxis):
    #     total_f0 += f0
    #     num = num+1
    # average_f0 = total_f0/num
    # return average_f0





def get_mel(fpath:str):
    # print("进入 get_mel 函数")
    # print("fpath = "+fpath)
    fpath = fpath.strip()  # 防止输入进来的路径，还带着回车符号'\n'
    wav, sr = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float64)  # 返回音频信号值 & 采样率
    # print("librosa 成功载入音频")
    f0, timeaxis = pw.harvest(wav, hp.SR)
    sp = pw.cheaptrick(wav, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)  # sp:频谱包络；pw.cheaptrick 谐波频谱包络估计算法
    # win size800，hop200，fft 1024
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80,  n_fft=1024, hop_length=200)
    # shape=(n_mels, t) = (80, t)
    return mel




def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)  # step = 1.0
    # 数据类型转换 cast(x, dtype, name=None)
    # 第一个参数 x: 待转换的数据（张量）
    # 第二个参数 dtype： 目标数据类型
    # 第三个参数 name： 可选参数，定义操作的名称
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def control_weight(global_step, function_type='logistic'):
    # KL散度的权重
    steps = tf.cast(global_step + 1, dtype=tf.float32)
    if function_type == 'logistic':
        return 1/(1 + tf.exp(-hp.K * (steps - hp.X0)))
    elif function_type == 'linear':
        return tf.math.minimum(1, steps/hp.X0)  # 似乎需要 1. 才能运行'linear'情况
    else:
        raise Exception('No Supported VAE LOSS WEIGHT FUNCTION.')




# tmp = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
# print("1 ******")
# print(learning_rate_decay(0.001, tmp,))  # Tensor("mul_1:0", shape=(), dtype=float32)
# print("2 ******")
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print("3 ******")
#     print(sess.run(tmp))  # 0
#     print("4 ******")
#     print(sess.run(learning_rate_decay(0.001, tmp, )))  # 2.5e-07 = 0.001 / 4000
