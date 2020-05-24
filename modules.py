from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
from hyperparams import hyperparams
from networks import encoder, decoder, conv1d, bn, prenet, gru

hp = hyperparams()

"""
1.generator()、discriminator()、
2.get_next_batch()、speaker_embedding()、
1.fast_lstm_3_layers():现在暂时没用了；
"""


def get_next_batch():
    """
    这块就是从tfrecord文件中读取已经保存的数据；
    """
    # 获取指定目录下的所有tfrecord文件
    # #加上r让字符串不转义
    tfrecords = glob.glob(f'{hp.TRAIN_DATASET_PATH}/*.tfrecord')
    # print("line23: tfrecords = "+str(tfrecords))
    """
    tf.train.string_input_producer(
        string_tensor,
        num_epochs=None,  # NUM_EPOCHS = 150；从string_tensor中产生 num_epochs 次字符串；如果未指定，则可以无限次循环遍历字符串
        shuffle=True,     # shuffle：布尔值。如果为true，则在每个epoch内随机打乱顺序
        seed=None,
        capacity=32,
        shared_name=None,
        name=None,
        cancel_op=None )
    输出字符串到一个输入管道队列
    ：从TFRecords文件中读取数据， 首先需要用tf.train.string_input_producer（）生成一个解析队列。
    之后调用 tf.TFRecordReader 的 tf.parse_single_example 解析器
    https://blog.csdn.net/tefuirnever/article/details/90271862
    """
    # 输出字符串到一个输入管道队列
    filename_queue = tf.train.string_input_producer(tfrecords, shuffle=True, num_epochs=hp.NUM_EPOCHS)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 解析器首先读取解析队列，返回serialized_example对象
    # 之后调用tf.parse_single_example操作将 Example 协议缓冲区(protocol buffer)解析为张量。
    features = tf.parse_single_example(
        serialized_example,
        features={
            'ori_spkid': tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'ori_mel': tf.VarLenFeature(dtype=tf.float32),
            'ori_mel_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),

            'aim_spkid': tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
            'aim_mel': tf.VarLenFeature(dtype=tf.float32),
            'aim_mel_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
        }
    )
    # tf.sparse_tensor_to_dense 将 SparseTensor 转换为稠密张量.(即理解为，稀疏矩阵，填充上默认值)
    features['ori_mel'] = tf.sparse_tensor_to_dense(features['ori_mel'])
    features['aim_mel'] = tf.sparse_tensor_to_dense(features['aim_mel'])
    ori_spk = features['ori_spkid']
    ori_mel = tf.reshape(features['ori_mel'], features['ori_mel_shape'])
    aim_spk = features['aim_spkid']
    aim_mel = tf.reshape(features['aim_mel'], features['aim_mel_shape'])
    # self.CODED_DIM = 60  # 压缩成60维

    ori_mel = tf.reshape(ori_mel, [-1, hp.CODED_DIM])
    aim_mel = tf.reshape(aim_mel, [-1, hp.CODED_DIM])  # 80 维度 mel
    ori_spk_batch, ori_mel_batch, aim_spk_batch, aim_mel_batch = tf.train.batch([ori_spk, ori_mel, aim_spk, aim_mel],
                                                                                batch_size=hp.BATCH_SIZE,
                                                                                capacity=100,
                                                                                num_threads=10,
                                                                                dynamic_pad=True,
                                                                                allow_smaller_final_batch=False)
    """
    是说在这里，get_next_batch（）函数，返回之前，就可以做 pad 操作吗？
    """
    # tf.shape(ori_mel_batch)[1]
    max_frame = tf.maximum(tf.shape(ori_mel_batch)[1], tf.shape(aim_mel_batch)[1])  # 最大帧值
    gap_frame = max_frame - tf.minimum(tf.shape(ori_mel_batch)[1], tf.shape(aim_mel_batch)[1])  # 帧值 之差

    # print(tf.math.subtract(max_frame, tf.shape(aim_mel_batch)[1]))
    padded = tf.zeros([tf.shape(aim_mel_batch)[0], tf.subtract(max_frame, tf.shape(aim_mel_batch)[1]),
                                                                tf.shape(aim_mel_batch)[2]], dtype=tf.float32)
    # a = padded
    aim_mel_batch = tf.concat((aim_mel_batch, padded), axis=1)
    # concated_1 = aim_mel_batch
    padded = tf.zeros([tf.shape(ori_mel_batch)[0], tf.subtract(max_frame, tf.shape(ori_mel_batch)[1]),
                                                                tf.shape(ori_mel_batch)[2]], dtype=tf.float32)
    # b = padded
    # padded = tf.zeros_like([1, tf.math.subtract(max_frame, tf.shape(ori_mel_batch)[1]), 1], dtype=tf.float32)
    ori_mel_batch = tf.concat((ori_mel_batch, padded), axis=1)

    # concated_2 = ori_mel_batch

    # padded = tf.zeros_like([0，差值，0])
    # aim_mel_batch = tf.concat((aim_mel_batch, padded), axis=1)

    # aim_mel_batch = tf.pad(aim_mel_batch, [[0, 0], [0, tf.math.subtract(max_frame, tf.shape(aim_mel_batch)[1])], [0, 0]], "CONSTANT")
    # ori_mel_batch = tf.pad(ori_mel_batch, [[0, 0], [0, tf.math.subtract(max_frame, tf.shape(ori_mel_batch)[1])], [0, 0]], "CONSTANT")

    # return ori_spk_batch, ori_mel_batch, aim_spk_batch, aim_mel_batch, a, b,concated_1,concated_2,max_frame
    return ori_spk_batch, ori_mel_batch, aim_spk_batch, aim_mel_batch



def speaker_embedding(inputs, spk_num, num_units, zero_pad=True, scope="speaker_embedding", reuse=None):
    '''Embeds a given tensor.

    输入的inputs，是表示说话人序号，所以是用 int 类型；
    num_units = 256，就是把说话人，表示成一个256维度的 embedding 向量

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      spk_num: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[spk_num, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)  # [起始行：结束行，起始列：结束列]
    return tf.nn.embedding_lookup(lookup_table, inputs)


# 先不用 G 和 D，只用Encoder和Decoder
# 还是先用G，并去掉embeding部分内容就可以了
# def generator(speaker_embedding, inputs, is_training=True, scope_name='generator', reuse=None):
#     '''Generate features.
#     Args:
#       speaker_embedding: A `Tensor` with type `float32` contains speaker information. [N, E]
#       inputs: A `Tensor` with type `float32` contains speech features.
#       is_training: Boolean, whether to train or inference.
#       scope_name: Optional scope for `variable_scope`.
#       reuse: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#     Returns:
#       A decoded `Tensor` with aim speaker.
#       vae mu vector.
#       vae log_var vector.
#     '''
#     with tf.variable_scope(scope_name, reuse=reuse):
#         sample, mu, log_var = encoder(inputs, is_training=is_training, scope='vae_encoder') # [N, T, E]
#         #speaker_embedding = tf.expand_dims(speaker_embedding, axis=1) # [N, 1, E]
#         speaker_embedding = tf.tile(speaker_embedding, [1, tf.shape(sample)[1], 1]) # [N, T, E]
#         # tf.tile() 用来对张量(Tensor)进行扩展的,表示每一维度，拓展复制几次；
#         encoded = tf.concat((speaker_embedding, sample), axis=-1) # [N, T, E+G]
#         outputs = decoder(encoded, is_training=is_training, scope='vae_decoder')
#         return outputs, mu, log_var # [N, T, C]

# 输入进来 ori_mel 谱【16，N，80】其中 N 和同一batch 的 aim_mel 是一样的
def generator(inputs, is_training=True, scope_name='generator', reuse=None):
    with tf.variable_scope(scope_name, reuse=reuse):
        sample, mu, log_var = encoder(inputs, is_training=is_training, scope='vae_encoder')  # [N, T, E]
        # speaker_embedding = tf.tile(speaker_embedding, [1, tf.shape(sample)[1], 1]) # [N, T, E]
        # tf.tile() 用来对张量(Tensor)进行扩展的,表示每一维度，拓展复制几次；
        # encoded = tf.concat((speaker_embedding, sample), axis=-1) # [N, T, E+G]
        outputs = decoder(sample, is_training=is_training, scope='vae_decoder')
        return outputs, mu, log_var  # [N, T, C]


# 输入：tmp = tf.placeholder(name='ori_feat', shape=[16, 100, 80], dtype=tf.float32)
# 输出：shape=(16, 100, 60)
# 因为CODED_DIM = 60  # 压缩成60维

# (<tf.Tensor 'generator/vae_decoder/dense_2/Tanh:0' shape=(16, 100, 60) dtype=float32>,
# <tf.Tensor 'generator/vae_encoder/mean/BiasAdd:0' shape=(16, 100, 16) dtype=float32>,
# <tf.Tensor 'generator/vae_encoder/log_var/BiasAdd:0' shape=(16, 100, 16) dtype=float32>)

#
# def discriminator(inputs, scope_name='discriminator', reuse=None):
#     '''Discriminator features.
#
#     Args:
#       inputs: A `Tensor` with type `float32` contains speech features. [N, T, F]
#       scope_name: Optional scope for `variable_scope`.
#       reuse: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#
#     Returns:
#         A softmax
#     '''
#     with tf.variable_scope(scope_name, reuse=reuse):
#         out = lstm_3_layers(inputs, num_units=hp.CODED_DIM * 2, bidirection=False)  # [N, C]
#         out = tf.layers.dense(out, units=hp.SPK_NUM * 2, activation=tf.nn.tanh, name='dense1')  # [N, L*2]
#         out = tf.layers.dense(out, units=hp.SPK_NUM * 2, activation=tf.nn.sigmoid, name='dense2')  # [N, L]
#         return out
#
#
# def fast_lstm_3_layers(inputs, num_units=None, bidirection=False, scope="lstm_3_layers", reuse=tf.AUTO_REUSE):
#     '''
#     :param inputs: A 3-d tensor. [N, T, C]
#     :param num_units: An integer. The last hidden units.
#     :param bidirection: A boolean. If True, bidirectional results are concatenated.
#     :param scope: A string. scope name.
#     :param reuse: Boolean. whether to reuse the weights of a previous layer.
#     :return: if bidirection is True, A 2-d tensor. [N, num_units * 2]
#              else, A 2-d tensor. [N, num_units]
#     '''
#     with tf.variable_scope(scope, reuse=reuse):
#         if not num_units:
#             num_units = inputs.get_shape().as_list[-1]
#         with tf.variable_scope('lstm_1'):
#             lstm_1 = tf.keras.layers.CuDNNLSTM(units=num_units, return_sequences=True, return_state=True)
#         with tf.variable_scope('lstm_2'):
#             lstm_2 = tf.keras.layers.CuDNNLSTM(units=num_units, return_sequences=True, return_state=True)
#         with tf.variable_scope('lstm_3'):
#             lstm_3 = tf.keras.layers.CuDNNLSTM(units=num_units, return_sequences=False, return_state=True)
#         out = lstm_1(inputs)
#         out = lstm_2(out[0])
#         out = lstm_3(out[0])
#         return out[0]
#
#
# def lstm_3_layers(inputs, num_units=None, bidirection=False, scope="lstm", reuse=tf.AUTO_REUSE):
#     '''
#     :param inputs: A 3-d tensor. [N, T, C]
#     :param num_units: An integer. The last hidden units.
#     :param bidirection: A boolean. If True, bidirectional results are concatenated.
#     :param scope: A string. scope name.
#     :param reuse: Boolean. whether to reuse the weights of a previous layer.
#     :return: if bidirection is True, A 2-d tensor. [N, num_units * 2]
#              else, A 2-d tensor. [N, num_units]
#     '''
#     with tf.variable_scope(scope, reuse=reuse):
#         if not num_units:
#             num_units = inputs.get_shape().as_list[-1]
#         # cellls = [tf.nn.rnn_cell.LSTMCell(size) for size in [num_units, num_units, num_units]]
#         cellls = [tf.nn.rnn_cell.LSTMCell(size) for size in [num_units, num_units, num_units]]
#         multi_cell = tf.nn.rnn_cell.MultiRNNCell(cellls)
#         if bidirection:
#             bw_cells = [tf.nn.rnn_cell.LSTMCell(size) for size in [num_units, num_units, num_units]]
#             multi_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells)
#             outputs, final_state = tf.nn.dynamic_rnn(multi_cell, multi_bw_cell, inputs=inputs, dtype=tf.float32)
#             # outputs shape : top lstm outputs, ([N, T, num_units], [N, T, num_units])
#             # lstm final_state : multi final state stack together, ([N, 2, num_units], [N, 2, num_units])
#             return tf.concat(final_state, axis=2)[-1][0]
#         outputs, final_state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=inputs, dtype=tf.float32)
#         # outputs shape : top lstm outputs, [N, T, num_units]
#         # lstm final_state : multi final state stack together, [N, 2, num_units]
#         return final_state[-1][0]


# if __name__ == "__main__":
#     ori_spk_batch, ori_mel_batch, aim_spk_batch, aim_mel_batch= get_next_batch()
#     sess = tf.Session()
#     sess.run(tf.initialize_local_variables())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     try:
#         while not coord.should_stop():
#             ori_mel, aim_mel = sess.run([ori_mel_batch, aim_mel_batch])
#
#             print("1 ori_mel_batch.shape = "+str(ori_mel.shape)+"  aim_mel_batch.shape = " + str(aim_mel.shape))
#
#
#     except tf.errors.OutOfRangeError:
#         print("complete")
#     finally:
#         coord.request_stop()
#     coord.join(threads)
#     sess.close()



# tmp = tf.placeholder(name='ori_feat', shape=[16, 100, 80], dtype=tf.float32)
# # print("242")
# # print(generator(tmp))


# (<tf.Tensor 'generator/vae_decoder/dense_2/Tanh:0' shape=(16, 100, 80) dtype=float32>,
#  decoder 出来的结果，拿去和aim_mel做对比：【 KL损失 】
# <tf.Tensor 'generator/vae_encoder/mean/BiasAdd:0' shape=(16, 100, 16) dtype=float32>,
# <tf.Tensor 'generator/vae_encoder/log_var/BiasAdd:0' shape=(16, 100, 16) dtype=float32>)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # print(get_next_batch())
#     # sess.run(get_next_batch())
#     # a, b, c, d = get_next_batch()
#     #     # tf.print(a)
#     # print(sess.run(get_next_batch()))
#     a, b, c = sess.run(generator(), feed_dict={inputs: tmp})
#     print(a.shape)
#     print("end")
    # (<tf.Tensor 'batch:0' shape=(16, 1) dtype=int64>,
    # <tf.Tensor 'batch:1' shape=(16, ?, 60) dtype=float32>,
    # <tf.Tensor 'batch:2' shape=(16, 1) dtype=int64>,
    # <tf.Tensor 'batch:3' shape=(16, ?, 60) dtype=float32>)




#
# if __name__ == "__main__":
#     ori_spk_batch, ori_mel_batch, aim_spk_batch, aim_mel_batch = get_next_batch()
#     sess = tf.Session()
#     sess.run(tf.initialize_local_variables())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     try:
#         while not coord.should_stop():
#             ori_mel = sess.run(ori_mel_batch)
#             aim_mel = sess.run(aim_mel_batch)
#             print("1 ori_mel_batch.shape = "+str(ori_mel.shape[1])+"  aim_mel_batch.shape = " + str(aim_mel.shape[1]))
#
#             max_frame = max(ori_mel.shape[1], aim_mel.shape[1])  # 最大帧值
#             gap_frame = max_frame - min(ori_mel.shape[1], aim_mel.shape[1])  # 帧值 之差
#             print("max_frame = "+str(max_frame)+" gap_frame = "+str(gap_frame))
#
#             # aim_mel = np.pad(aim_mel, ([[0, 0], [0, gap_frame], [0, 0]]), 'constant')
#             # def pad(array, pad_width, mode='constant', **kwargs):
#             if ori_mel.shape[1] == max_frame:
#                 aim_mel = np.pad(aim_mel, ([[0, 0], [0, gap_frame], [0, 0]]), 'constant')
#                 print("reshape aim_mel")
#             else:
#                 print("这里！ori_mel_batch.shape[1] = " + str(ori_mel.shape[1]))
#                 ori_mel = np.pad(ori_mel, ([[0, 0], [0, gap_frame], [0, 0]]), 'constant')
#                 print("reshape ori_mel")
#
#             print("2 ori_mel_batch.shape = " + str(ori_mel.shape[1]) + "  aim_mel_batch.shape = " + str(aim_mel.shape[1]))
#
#
#     except tf.errors.OutOfRangeError:
#         print("complete")
#     finally:
#         coord.request_stop()
#     coord.join(threads)
#     sess.close()
"""
ori_mel_batch.shape = (16, 547, 80)  aim_mel_batch.shape = (16, 743, 80)
ori_mel_batch.shape = (16, 890, 80)  aim_mel_batch.shape = (16, 750, 80)
ori_mel_batch.shape = (16, 668, 80)  aim_mel_batch.shape = (16, 663, 80)
ori_mel_batch.shape = (16, 737, 80)  aim_mel_batch.shape = (16, 756, 80)
ori_mel_batch.shape = (16, 817, 80)  aim_mel_batch.shape = (16, 743, 80)
ori_mel_batch.shape = (16, 565, 80)  aim_mel_batch.shape = (16, 663, 80)
ori_mel_batch.shape = (16, 737, 80)  aim_mel_batch.shape = (16, 750, 80)
ori_mel_batch.shape = (16, 890, 80)  aim_mel_batch.shape = (16, 728, 80)
ori_mel_batch.shape = (16, 890, 80)  aim_mel_batch.shape = (16, 750, 80)

ori_mel_batch.shape = 817  aim_mel_batch.shape = 663
817
ori_mel_batch.shape = 788  aim_mel_batch.shape = 756
788
ori_mel_batch.shape = 737  aim_mel_batch.shape = 648
737
ori_mel_batch.shape = 817  aim_mel_batch.shape = 750
817
ori_mel_batch.shape = 822  aim_mel_batch.shape = 756

法1：在train.py里面，使用同一个 batch 数据之前，把两个mel 数据pad对齐一下，再送进 G，生成模型（）
法2：在get_next_batch（）函数返回之前，处理好尺寸；
"""



"""
1 ori_mel_batch.shape = 890  aim_mel_batch.shape = 608
max_frame = 890 gap_frame = 282
reshape aim_mel
2 ori_mel_batch.shape = 890  aim_mel_batch.shape = 890
1 ori_mel_batch.shape = 802  aim_mel_batch.shape = 756
max_frame = 802 gap_frame = 46
reshape aim_mel
2 ori_mel_batch.shape = 802  aim_mel_batch.shape = 802
1 ori_mel_batch.shape = 817  aim_mel_batch.shape = 743
max_frame = 817 gap_frame = 74
reshape aim_mel
2 ori_mel_batch.shape = 817  aim_mel_batch.shape = 817
1 ori_mel_batch.shape = 788  aim_mel_batch.shape = 648
max_frame = 788 gap_frame = 140
reshape aim_mel
2 ori_mel_batch.shape = 788  aim_mel_batch.shape = 788
1 ori_mel_batch.shape = 822  aim_mel_batch.shape = 756
max_frame = 822 gap_frame = 66
reshape aim_mel
2 ori_mel_batch.shape = 822  aim_mel_batch.shape = 822
"""