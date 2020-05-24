import tensorflow as tf
from hyperparams import hyperparams

hp = hyperparams()

"""
1.Encoder + Decoder
2. conve1d、gru、bn（batch_normalization）、 prenet
"""

def encoder(inputs, is_training=True, scope='vae_encoder', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        prenet_out = prenet(inputs, is_training=is_training)
        out = conv1d(prenet_out, filters=hp.CODED_DIM * 2, size=3, scope="conv1d_1")  # (N, T, C*2)
        # #一维卷积结果=【batch数目, 一图中卷积次数, 卷积核数目】
        out = bn(out, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")
        out = conv1d(out, filters=hp.CODED_DIM * 2, size=3, scope="conv1d_2")  # (N, T, C*2)
        out = bn(out, is_training=is_training, scope="conv1d_2")
        out = gru(out, num_units=hp.CODED_DIM * 2, bidirection=False, scope="gru_1")  # (N, T, C*2)
        out = gru(out, num_units=hp.EMBED_SIZE, bidirection=False, scope="gru_2")  # (N, T, E)
        # mu 和 log_var 一样
        mu = tf.layers.dense(out, units=hp.VAE_GAUSSION_UNITS, name='mean')  # [N, T, G]
        log_var = tf.layers.dense(out, units=hp.VAE_GAUSSION_UNITS, name='log_var')  # [N, T, G]
        std = tf.sqrt(tf.exp(log_var))  # 标准差（用来给下面乘）
        z = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0)  # [N, T, G] 随机采样的一个0/1标准正态分布
        sample = mu + z * std  # [N, T, G]
        return sample, mu, log_var

# (<tf.Tensor 'vae_encoder/add:0' shape=(16, 100, 16) dtype=float32>,
# <tf.Tensor 'vae_encoder/mean/BiasAdd:0' shape=(16, 100, 16) dtype=float32>,
# <tf.Tensor 'vae_encoder/log_var/BiasAdd:0' shape=(16, 100, 16) dtype=float32>)
# 16 是由于  超参数中：self.VAE_GAUSSION_UNITS = 16


def decoder(inputs, is_training=True, scope='vae_decoder', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        prenet_out = prenet(inputs, is_training=is_training)  # [N, T, E//2]
        out = conv1d(prenet_out, filters=hp.CODED_DIM * 2, size=3, scope="conv1d_1")  # (N, T, C*2)
        out = bn(out, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")
        out = conv1d(out, filters=hp.CODED_DIM * 2, size=3, scope="conv1d_2")  # (N, T, C*2)
        out = bn(out, is_training=is_training, scope="conv1d_2")
        out = gru(out, num_units=hp.CODED_DIM * 2, bidirection=False, scope="gru_1")  # (N, T, C*2)
        out = gru(out, num_units=hp.CODED_DIM * 2, bidirection=False, scope="gru_2")  # (N, T, C*2)
        out = tf.layers.dense(out, units=hp.CODED_DIM * 2, activation=tf.nn.tanh, name='dense_1')  # (N, T, C*2)
        out = tf.layers.dense(out, units=hp.CODED_DIM, activation=tf.nn.tanh, name='dense_2')  # (N, T, C)
        return out

#  Tensor("vae_decoder/dense_2/Tanh:0", shape=(16, 100, 60), dtype=float32)

def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    '''
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "activation": activation_fn,
                  "use_bias": use_bias, "reuse": reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs


def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    '''Applies a GRU.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results
        are concatenated.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        cell = tf.contrib.rnn.GRUCell(num_units)
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs


def prenet(inputs, num_units=None, is_training=True, scope="prenet", reuse=None):
    '''Prenet for Encoder and Decoder1.

    这个学习的是tacotron 的prenet：
    功能是作为bottleneck layer来增加泛化能力和加速收敛
    https://zhuanlan.zhihu.com/p/101064153

    Args:
      inputs: A 2D or 3D tensor.
      num_units: A list of two integers. or None.
      is_training: A python boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, num_units/2].
    '''
    if num_units is None:
        num_units = [hp.EMBED_SIZE, hp.EMBED_SIZE // 2]

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=hp.DROPOUT_RATE, training=is_training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=hp.DROPOUT_RATE, training=is_training, name="dropout2")
    return outputs


def bn(inputs,
       is_training=True,
       activation_fn=None,
       scope="bn",
       reuse=None):
    '''Applies batch normalization.
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension. Note that this is different from the native
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py`
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      is_training: Whether or not the layer is in training mode.
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
    # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               fused=True,
                                               reuse=reuse)
        # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:  # fallback to naive batch norm
        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)
    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs


# tmp= tf.placeholder(name='ori_feat', shape=[16, 100, 80], dtype=tf.float32)
# print("line202")
# t1, t2, t3 = encoder(tmp)
# print(t1,t2,t3)
# # (<tf.Tensor 'vae_encoder/add:0' shape=(16, 100, 16) dtype=float32>,
# # <tf.Tensor 'vae_encoder/mean/BiasAdd:0' shape=(16, 100, 16) dtype=float32>,
# # <tf.Tensor 'vae_encoder/log_var/BiasAdd:0' shape=(16, 100, 16) dtype=float32>)
# # 16 是由于  超参数中：self.VAE_GAUSSION_UNITS = 16
# print("line218")
# print(decoder(t1))
# #  Tensor("vae_decoder/dense_2/Tanh:0", shape=(16, 100, 60), dtype=float32)
# # 因为CODED_DIM = 60  # 压缩成60维
