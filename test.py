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

def test():
    fpath = './paralleldata/VCC2SF1/SF1|10001.wav'
    aim_spkid = 'spk2'
    new_wav_for_infer, sr = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float32)  # 返回音频信号值 & 采样率
    ori_mel = melspectrogram(new_wav_for_infer, hparams)  # 康的
    # ori_mel = ori_mel.T

    synwav = inv_mel_spectrogram(ori_mel, hparams)
    synwav = synwav / np.abs(synwav).max()  # 04.23这句话的意思是，将语音的范围控制在【-1，1】之间
    print(f'synthesize done. path : ./convert_to_{aim_spkid}_test1.wav')
    librosa.output.write_wav(f'./convert_to_{aim_spkid}_test22222.wav', synwav, sr=hp.SR)


if __name__ == '__main__':
    test()


# def synthesis(ori_path, aim_mel):
#     aim_spkid = 'spk2'
#     print('synthesizing ...')
#     wav, sr = librosa.load(ori_path, sr=hp.SR, mono=True, dtype=np.float64)
#     """
#     ` hp.SR = 22050
#     ` mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80,  n_fft=1024, hop_length=200)
#
#     def mel_to_audio(M, sr=22050, n_fft=2048, hop_length=512, win_length=None,
#                  window='hann', center=True, pad_mode='reflect', power=2.0, n_iter=32,
#                  length=None, dtype=np.float32, **kwargs):
#     """
#     # synwav = librosa.feature.inverse.mel_to_audio(aim_mel, sr=hp.SR, n_fft=1024, hop_length=200)
#     synwav = inv_mel_spectrogram(aim_mel, hparams)
#     print(f'synthesize done. path : ./convert_to_{aim_spkid}_test1.wav')
#     librosa.output.write_wav(f'./convert_to_{aim_spkid}_test1.wav', synwav, sr=hp.SR)


# def main():
#     print("line 69")
#     # spkid是代表你想转化到哪个人  fpath是原始音频
#     fpath = './paralleldata/VCC2SF1/SF1|10001.wav'
#
#     print(fpath)  # ./data/convert/TMM1/M10007.wav
#     fpath = fpath.strip()
#     new_wav_for_infer, sr = librosa.load(fpath, sr=hp.SR, mono=True, dtype=np.float32)  # 返回音频信号值 & 采样率
#
#     ori_mel = melspectrogram(new_wav_for_infer, hparams)  # 斌的：原始说话人的sp，接下来用np.random.normal生成同样形状的
#     ori_mel = ori_mel.T
#     print('源说话人的mel.shape：')
#     print(ori_mel.shape)  # sp 是二维的！！！
#
#     mode = 'infer'
#     G = Graph(mode=mode)
#     print('{} graph loaded.'.format(mode))
#     saver = tf.train.Saver()
#     with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
#         try:
#             print(f'Try to load trained model in {hp.MODEL_DIR} ...')
#             saver.restore(sess, tf.train.latest_checkpoint(hp.MODEL_DIR))
#         except:
#             raise Exception(f'Load trained model failed in {hp.MODEL_DIR}, please check ...')
#         finally:
#             ori_mel = np.reshape(ori_mel, (-1, hp.CODED_DIM))  # 斌的：原始说话人的sp
#             ori_mel_batch = np.expand_dims(ori_mel, axis=0)  # 模型训练的时候是三维度的  # ???;变成【1，None，80】
#
#             # aim_spkid_batch = np.array([[aim_spkid]])
#             for j in tqdm.tqdm(range(1)):
#                 aim_out = sess.run(G.aim_out, {G.ori_mel: ori_mel_batch})
#
#             aim_out = np.array(aim_out, dtype=np.float64)  # 转换出来的 mel，三维的（第一个是batch=1）
#             predict_mel = np.reshape(aim_out, (-1, hp.CODED_DIM))  # [None, 80]，转换成二维mel
#             print("line 103 predict_mel.shape = " + str(predict_mel.shape))
#             predict_mel = predict_mel.T
#             # ori_new_f0 = np.random.normal(source_mean_f0, 1.0, predict_mel.shape[0])
#
#             print('Sp predicted done.')
#             synthesis(fpath, aim_mel=predict_mel, )
#

# if __name__ == '__main__':
#     main()

# import tensorflow as tf
#
# with tf.variable_scope('V1', reuse=None):
#     a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
#     a2 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
# with tf.variable_scope('V2', reuse=True):
#     a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
#     a4 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(a1.name)
#     print(a2.name)
#     print(a3.name)
#     print(a4.name)

# import tensorflow as tf
#
# output_file = './paralleldata/train_data/0.tfrecord'
# for example in tf.python_io.tf_record_iterator(output_file):
#         print(tf.train.Example.FromString(example))
#
# with tf.Session() as sess:
#     dataset = tf.data.TFRecordDataset(output_file)  # 加载TFRecord文件
#     dataset = dataset.map(parse_fn2)  # 解析data到Tensor
#     dataset = dataset.repeat(1)  # 重复N epochs
#     dataset = dataset.batch(3)  # batch size
#
#     iterator = dataset.make_one_shot_iterator()
#     next_data = iterator.get_next()
#
#     while True:
#         try:
#             position, label = sess.run(next_data)
#             print(position)
#             print(label)
#         except tf.errors.OutOfRangeError:
#             break

# def func1():
#     try:
#         return 1
#
#     # print(2)
#     finally:
#         return 2
#
#
# def func2():
#     try:
#         raise ValueError()
#     except:
#         return 1
#     finally:
#         return 3
#
#
# print(func1())
# print(func2())

# # new_mel = librosa.feature.melspectrogram(y=wav[24001:48000], sr=sr, n_mels=80, hop_length=240)  # n_mels 默认128,改为80维
# # win size800，hop200，fft 1024
# import tensorflow as tf
# import glob
#
# # tmp = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
# # print(tmp)
#
# data_path = 'paralleldata/VCC2TM1/*'
#
# # for name in glob.glob('paralleldata/VCC2TM1/*'):
#     # print("********")
#
# wav_files = glob.glob(data_path)
# wav_feats = dict()
# print(type(wav_files))
# print(wav_feats)
#
# def foo(num):
#     print("starting...")
#     while num<10:
#         num=num+1
#         yield num
# for n in foo(0):
#     print(n)
#     # for循环本身的原理就是调用next函数
#     # 所以不再需要调用 next() 或者 send() 函数
#     # print(type(foo(0)))  # <class 'generator'>
#     print('*******')
# # g = foo(0)
# # print(next(g))
# # print(next(foo(0)))
#
# # def foo():
# #     print("starting...")
# #     while True:
# #         res = yield 4
# #         print("res:",res)
# # g = foo()
# # print(next(g))
# # print("*"*20)
# # print(next(g))
#
# # class Fib(object):
# #     def __init__(self):
# #         pass
# #
# #     def __call__(self, num):
# #         a, b = 0, 1;
# #         self.l = []
# #
# #         for i in range(num):
# #             self.l.append(a)
# #             a, b = b, a + b
# #         return self.l
# #
# #     def __str__(self):
# #         return str(self.l)
# #
# #     __rept__ = __str__
# #
# #
# # f = Fib()
# # print(type(f(10).__repr__()))
# # print(f(10).__repr__())
