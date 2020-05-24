import codecs  # 专门用作编码转换
from hyperparams import hyperparams
from utils import get_sp
from utils import get_mel
from audio_taco import melspectrogram, preemphasis
import hparam as hparams
import random
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import librosa
import os
hp = hyperparams()


source2target={'SF1': 'TM1', }
"""
匹配表： {源说话人：目标说话人}
在想，如果有多对（源和目标都换人了，但是也是平行语料），整个网络还能行吗？
"""
"""
在get_batch = 16 的时候，
16个 ori_mel 们，会自动地按照最大的ori_mel尺寸 来补齐
16个 aim_mel 们，也会自动按照最大的aim_mel尺寸 来补齐
但是ori_mel 尺寸和 aim_mel 尺寸并不一定一样
所以在 train 中，get_batch 之后，需要手动pad
：可以在图内进行：用tf.pad ; 也可以在图外进行： np.pad，或者用np.tile之类的补齐方法，再查查；
"""

def process(args):
    (tfid, dataset) = args
    # 这个tfid，是指系统里面的cpu数目，也就是多进程的数目。。从handle（）函数看出来的
    writer = tf.python_io.TFRecordWriter(os.path.join(hp.PARALLEL_TRAIN_DATASET_PATH, f'{tfid}.tfrecord'))
    # print('******')
    for i in tqdm(dataset):
        # ori_spkid, ori_fpath, aim_spkid, aim_fpath, target_G, target_D_fake, target_D_real = i[:7]

        ori_spkid, ori_fpath, aim_spkid, aim_fpath = i[:4]
        ori_spkid = np.array(ori_spkid)  # 强制把各种 int、元组tuple、列表list 都转变为 numpy.ndarray 格式！！！
        aim_spkid = np.array(aim_spkid)
        # ori_mel = get_mel(ori_fpath)
        # aim_mel = get_mel(aim_fpath)
        #  hp.SR = 16000
        ori_fpath = ori_fpath.strip()
        aim_fpath = aim_fpath.strip()
        source_wav, sr = librosa.load(ori_fpath, sr=hp.SR, mono=True, dtype=np.float32)  # 返回音频信号值 & 采样率
        aim_wav, sr = librosa.load(aim_fpath, sr=hp.SR, mono=True, dtype=np.float32)  # 返回音频信号值 & 采样率
        source_wav = preemphasis(source_wav, hparams.preemphasis)  # 预加重，k = 0.97; 记住，出wav之后，要重新去加重
        aim_wav = preemphasis(aim_wav, hparams.preemphasis)  # def inv_preemphasis(wav, k, inv_preemphasize=True):

        ori_mel = melspectrogram(source_wav, hparams)  # 新的方法提取(输入的是 wav 才对)
        aim_mel = melspectrogram(aim_wav, hparams)
        print("line 45:ori_mel.shape = "+str(ori_mel.shape))
        print("line 46:aim_mel.shape = " + str(aim_mel.shape))
        """
        line 45:ori_mel.shape = (80, 286)
        line 46:ori_mel.shape = (80, 286)   
        """
        ori_mel = ori_mel.T
        aim_mel = aim_mel.T
        print("line 60:ori_mel.shape = " + str(ori_mel.shape))
        print("line 61:aim_mel.shape = " + str(aim_mel.shape))
        ori_mel_shape = np.array(ori_mel.shape)
        aim_mel_shape = np.array(aim_mel.shape)
        example = tf.train.Example(features=tf.train.Features(feature={

            'ori_spkid': tf.train.Feature(int64_list=tf.train.Int64List(value=ori_spkid.reshape(-1))),
            'ori_mel': tf.train.Feature(float_list=tf.train.FloatList(value=ori_mel.reshape(-1))),
            'ori_mel_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=ori_mel_shape.reshape(-1))),

            'aim_spkid': tf.train.Feature(int64_list=tf.train.Int64List(value=aim_spkid.reshape(-1))),
            'aim_mel': tf.train.Feature(float_list=tf.train.FloatList(value=aim_mel.reshape(-1))),
            'aim_mel_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=aim_mel_shape.reshape(-1))),

        }))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()



# tf.saver.save()




def handle(dataset):
    # 这块用来写tfrecord文件的
    # 关于多进程 https://www.liaoxuefeng.com/wiki/897692888725344/923056295693632
    if hp.MULTI_PROCESS:
        cpu_nums = mp.cpu_count()  # 统计cpu核数
        thread_nums = int(cpu_nums * hp.CPU_RATE)  # hp.CPU_RATE=1，这是在算最多能有多少子进程同时进行
        splits = [(i, dataset[i::thread_nums])
                  for i in range(thread_nums)]  # 【起点：终点：步长】
        # splits 列表里面，保存着一个个的元组 tuple : (i, dataset[::] )
        pool = mp.Pool(thread_nums)  # 进程池！
        pool.map(process, splits)
        pool.close()  # close后进程池不能再 apply / map 任务
        pool.join()
        # pool.join()是用来等待进程池中的worker进程执行完毕，防止主进程在worker进程结束前结束。
        # 但pool.join()必须使用在pool.close()或者pool.terminate()之后。
        # 其中close()跟terminate()的区别在于close()会等待池中的worker进程执行结束再关闭pool,而terminate()则是直接关闭。
    else:
        splits = (0, dataset)
        process(splits)

def main():
    lines = codecs.open(hp.TOTAL_CSV_PATH, 'r').readlines()  # readlines() 把所有的行 都读取进来，source & target
    # 现在的 source.csv 路径 = self.TOTAL_CSV_PATH = './paralleldata/total.csv'
    # lines = SF1|10001.wav......所有行
    # source_lines = codecs.open('./paralleldata/total.csv', 'r').readlines()  # 还没用上
    # hp.SOURCE_CSV_PATH = './paralleldata/source.csv'

    # spk 和 id 的对应关系 ，这两行貌似不需要了，在"多对一"的时候可以用上
    spk2id = {}
    id2spk = {}
    id2fpath = {}  # id2fpath{ } 字典 ，里面的 value值 是 一个 列表！list！这样才能.append（）！！！
    cnt = 0
    train_dataset = []  # 训练集，是一个列表
    test_dataset = []  # 忘记问说，这个用来干什么的了；
    for line in lines:
        # TMM1 | M10035.wav  或者   SF1|10001.wav
        spk, fname = line.strip().split('|')  # 对应说话人名字 & 语音序号名字 ；先不分了，因为音频和csv对应上了
        fpath = os.path.join(hp.TOTAL_WAVS_PATH, line)  # 这里拼接用的是 line：SF1|10001.wav ！！！
        # TOTAL_WAVS_PATH = './parallel/TOTAL'
        # 所有说话人的音频路径 fpath = './parallel/TOTAL/SF1|10001.wav'
        # 拼接成每个音频的路径 & 名字；   # hp.WAVS_PATH
        if spk not in spk2id.keys():
            cnt += 1
            spk2id[spk] = cnt  # 字典添加 {键：值}  对  {SF1:1 , TM1:2}
            id2spk[cnt] = spk
            id2fpath[cnt] = []  # Good ！！！ {1:[一号人的所有文件路径], 2:[二号人的所有文件路径] }
            # 注意！id2fpath{ } 字典 ，里面的值是 一个列表！list！这样才能.append（）！！！
            # 之前没遇见过这个人，就开创新空白列表

        id2fpath[ spk2id[spk] ].append(fpath)  # 这里存着每个人，所有的文件路径 集合列表；通过"键"来访问
        # id2fpath{1:【fpath1，fpath2，fpath3， ...】} ；fpath = './parallel/VCC2SF1/ SF1|10001.wav'
        # 然后不管之前有没有出现过这个人，都要把这个人对应的文件路径添加进去
    # if cnt != hp.SPK_NUM:
    if cnt != hp.PARALLEL_SPK_NUM:
        # PARALLEL_SPK_NUM = 2
        raise Exception('Hyperparams SPK_NUM is not correct. Please cheak again.')

    # 准备训练数据：
    num = 0  # 用来标记说，源说话人的语音序号：10001 对应 1；
    source_lines = codecs.open(hp.SOURCE_CSV_PATH, 'r').readlines()  # 这里读取源说话人信息；

    # for line in lines[:int(len(lines) * hp.TRAIN_RATE)]:
    #     spk, fname = line.strip().split('|')

    for line in source_lines[:int(len(source_lines) * hp.TRAIN_RATE)]:
        # TRAIN_RATE = 1
        # line = SF1|10006.wav
        spk, fname = line.strip().split('|')  # 这里的spk，表示数据的 source 说话人
        line = line.strip()  # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        fpath = os.path.join(hp.TOTAL_WAVS_PATH, line)  # 源说话人的音频位置（源和目标是放在一起的：TOTAL里面）
        # print(fpath)
        # 注意，这里的line，后面带有 回车'\n'，要记得处理掉，不然路径里面有回车符号，找不到音频文件
        # TOTAL_WAVS_PATH = './parallel/TOTAL'
        num += 1
        # fpath = './parallel/TOTAL/SF1|10006.wav'

        # aim_rand_spkid = random.randint(1, cnt)
        aim_spk = source2target[spk]  # aim_spk = TM1  ：源 对应的 目标 名字；
        aim_spkid = spk2id[aim_spk]  # aim_spkid = 2
        # aim_path = id2fpath[aim_rand_spkid][random.randint(0, len(id2fpath[aim_rand_spkid]) - 1)]  # !!!
        aim_path = id2fpath[aim_spkid][num-1]  # 与当前 source语音  对应的 target语音 路径
        # 第二个【】里面参数是位置序号；第一个【】里面是字典的 键key 值
        """
        source2target={'SF1': ['TM1','2','3'] }
        print(source2target['SF1'][0])          # >> TM1
        """
        train_dataset.append([spk2id[spk], fpath, aim_spkid, aim_path,])
        # [源说话人 id=1， 源说话人的某个音频路径 fpath，  目标说话人 id=2，  与当前 source语音 对应的 target语音路径]

    handle(train_dataset)

    # preprocess test dataset
    # 准备测试集
    # source_lines
    for line in lines[int(len(lines) * hp.TRAIN_RATE):]:
        # lines[最后位置：]，所以这时候，lines 没有值
        spk, fpath = line.strip().split('|')
        aim_rand_spkid = random.randint(1, cnt)
        test_dataset.append([spk, fpath, aim_rand_spkid])  # 一个列表
    with open('{}/test_dataset.txt'.format(hp.TEST_DATASET_PATH), 'w') as f:
        # TEST_DATASET_PATH = './parallel/test_data'
        for i in test_dataset:
            f.write(i[0] + '|' + i[1] + '|' + i[2])  # test_dataset【】【】是一个二维列表【n】【3】
            f.write('\n')

    print(spk2id)  # {'SF1': 1, 'TM1': 2}
    # print(aim_spkid)

if __name__ == '__main__':
    main()
