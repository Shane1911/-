# -*- coding:utf-8 -*-
# 此程序用于验证，所用数据集为验证集
import tensorflow as tf
import cv2 as cv
import readTFRescord
import netForward
import ceshi_3
import resNetForward
import netBackward
import time
import numpy as np


# 测试集文件存放路径
testFile = "D:\\Desktop\\python3\\paper_2\\data\\picture\\0hp\\test\\test.tfrecords"
BATCH_SIZE = 1000


def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 64, 64, 1])
        y_ = tf.placeholder(tf.float32, [None, 10])
        x_data = tf.reshape(x, [-1, 64, 64, 1])
        y = ceshi_3.forward(x_data)

        saver = tf.train.Saver()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        ans = tf.argmax(y, 1)
        lab = tf.argmax(y_, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        data_batch, label_batch = readTFRescord.read_and_decode(testFile, BATCH_SIZE)

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(netBackward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                d, ys = sess.run([data_batch, label_batch])

                noi = abs(np.random.normal(0, 64, [BATCH_SIZE, 64, 64, 1]))  # 噪声的形状要与数据匹配
                noise = np.array(noi, dtype='uint8')
                datas = cv.add(d, noise)

                labels = readTFRescord.onehot(ys)

                start_time = time.time()
                accuracy_score = sess.run(accuracy, feed_dict={x: datas, y_: labels})
                ans = sess.run(ans,feed_dict={x: datas, y_: labels})
                lab = sess.run(lab,feed_dict={x: datas, y_: labels})
                for i in range(BATCH_SIZE):
                    if ans[i] != lab[i]:
                        print("output = %d ; label = %d" % (ans[i], lab[i]))
                end_time = time.time()
                print('time:', (end_time - start_time))
                print('The total accuracy is:%f' % accuracy_score)

                coord.request_stop()
                coord.join(threads)

            else:
                print('No checkpoint file found')
                return


def main():
    test()


if __name__ == '__main__':
    main()









