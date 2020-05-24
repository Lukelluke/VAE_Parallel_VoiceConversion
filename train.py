from model import Graph
import tensorflow as tf
from hyperparams import hyperparams
import os
hp = hyperparams()




def main():
    mode = 'train'
    G = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()  # 训练网络之后保存训练好的模型，以及在程序中读取已保存好的模型
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)  # 这里的1
    # https://blog.csdn.net/c20081052/article/details/82345454
    # per_process_gpu_memory_fraction指定了每个GPU进程中使用显存的上限，
    # 但它只能均匀地作用于所有GPU，无法对不同GPU设置不同的上限。
    with tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(hp.LOG_DIR, sess.graph)
        """
        # tf.summary.FileWritter(path, sess.graph)
        # 指定一个文件用来保存图。 LOG_DIR = './logs'
        gpu_options：每个GPU使用显存上限（集体定义，不能单独定义）
        log_device_placement=False : 是否打印设备分配日志
        allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
        """
        try:
            print(f'Try to load trained model in {hp.MODEL_DIR} ...')
            # MODEL_DIR = './models'
            saver.restore(sess, tf.train.latest_checkpoint(hp.MODEL_DIR))
            # 已经训练好的模型的重新倒入到sess 里面去，然后下面就可以直接使用 sess 来运行了（sess.run....）
        except:
            print('Load trained model failed, start training with initializer ...')
            sess.run(tf.global_variables_initializer())
        finally:
            # finally，就是不管前面报错与否，都会进行的环节，包括return，也会先return finally里面的return语句；
            # 下面几句的 局部初始化local_variables_initializer 以及start_queue_runners，是因为要用到 tf.train.batch 函数

            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    """
                    这里可以插入数据尺寸的后期处理
                    """
                    steps = 1
                    G_loss = 0
                    D_loss = 0
                    if steps % 5 != 0:
                        _, reconstruction_loss, cycle_loss, kl_loss, GAN_G_loss, G_loss, summary, steps = sess.run([G.G_train_op,
                                                                                                                    G.reconstruction_loss,
                                                                                                                    G.cycle_loss,
                                                                                                                    G.kl_loss,
                                                                                                                    G.GAN_G_loss,
                                                                                                                    G.G_loss,
                                                                                                                    G.merged,
                                                                                                                    G.global_step])
                        print('train mode \t steps : {} \t '
                              'reconstruction_loss : {} \t '
                              'cycle_loss : {} \t '
                              'kl_loss : {} \t '
                              'GAN_G_loss : {} \t '
                              'G_total_loss : {}'.format(steps,
                                                         reconstruction_loss,
                                                         cycle_loss,
                                                         kl_loss,
                                                         GAN_G_loss,
                                                         G_loss))
                    else:
                        _, D_fake_loss, D_real_loss, D_loss, summary, steps = sess.run([G.D_train_op,
                                                                                        G.D_fake_loss,
                                                                                        G.D_real_loss,
                                                                                        G.D_loss,
                                                                                        G.merged,
                                                                                        G.global_step])
                        print('train mode \t steps : {} \t '
                              'D_fake_loss : {} \t '
                              'D_real_loss : {} \t '
                              'D_total_loss : {}'.format(steps,
                                                         D_fake_loss,
                                                         D_real_loss,
                                                         D_loss))
                    writer.add_summary(summary=summary, global_step=steps)
                    # 可以调用tf.summary.FileWriter其add_summary（）方法将训练过程数据保存在filewriter指定的文件中

                    if steps % (hp.PER_STEPS + 1) == 0:
                        saver.save(sess, os.path.join(hp.MODEL_DIR, 'model_%.3fGlos_%.3fDlos_%dsteps' % (G_loss,
                                                                                                         D_loss,
                                                                                                         steps)))

            except tf.errors.OutOfRangeError:
                print('Training Done.')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
