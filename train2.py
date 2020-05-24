from model import Graph
import tensorflow as tf
from hyperparams import hyperparams
import os
hp = hyperparams()


def main():
    mode = 'train'
    G = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)  # 这里的1
    with tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(hp.LOG_DIR, sess.graph)
        try:
            print(f'Try to load trained model in {hp.MODEL_DIR} ...')
            # MODEL_DIR = './models'
            saver.restore(sess, tf.train.latest_checkpoint(hp.MODEL_DIR))
            # 已经训练好的模型的重新倒入到sess 里面去，然后下面就可以直接使用 sess 来运行了（sess.run....）
        except:
            print('Load trained model failed, start training with initializer ...')
            sess.run(tf.global_variables_initializer())
        finally:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    # 上面这局也是固定格式，配合 tf.train.batch()
                    steps = 1       # 一步一步来？？？
                    G_loss = 0      # 初始化
                    # D_loss = 0
                    # if steps % 5 != 0:
                    _, kl_loss, G_loss, summary, steps = sess.run([G.G_train_op, G.kl_loss, G.G_loss, G.merged, G.global_step])

                    print('train mode \t steps : {} \t '
                          'kl_loss : {} \t '
                          'G_total_loss : {}'.format(steps,
                                                     kl_loss,
                                                     G_loss))

                    writer.add_summary(summary=summary, global_step=steps)
                    saver.save(sess, os.path.join(hp.MODEL_DIR, 'model_%.3fGlos_%dsteps' % (G_loss, steps)))
                    # else:
                    #     _, kl_loss, G_loss, summary, steps = sess.run(
                    #         [G.G_train_op, G.kl_loss, G.G_loss, G.merged, G.global_step])
                    #     print('train mode \t steps : {} \t '
                    #           'kl_loss : {} \t '
                    #           'G_total_loss : {}'.format(steps,
                    #                                      kl_loss,
                    #                                      G_loss))
                    #     writer.add_summary(summary=summary, global_step=steps)
                    #
                    #     saver.save(sess, os.path.join(hp.MODEL_DIR, 'model_%.3fGlos_%dsteps' % (G_loss, steps)))
                    # else:
                    #     summary, steps = sess.run([G.merged, G.global_step])
                    #
                    #     print('train mode \t steps : {} '.format(steps,))

                    # writer.add_summary(summary=summary, global_step=steps)
                    # if steps % (hp.PER_STEPS + 1) == 0:
                    #     # hp.PER_STEPS = 1000，总共只有760+step，所以不行，改为100
                    #     saver.save(sess, os.path.join(hp.MODEL_DIR, 'model_%.3fGlos_%dsteps' % (G_loss, steps)))

            except tf.errors.OutOfRangeError:
                print('Training Done.')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
