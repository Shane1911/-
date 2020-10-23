# -*- coding:utf-8 -*-
import tensorflow as tf
import cv2 as cv
import netForward
import ceshi_3
import resNetForward
import readTFRescord
import time
import numpy as np
import os
from matplotlib import pyplot as plt

LEARNING_RATE = 0.001
BATCH_SIZE = 32
STEPS = 501
MODEL_SAVE_PATH="D:\\Desktop\\python3\\paper_2\\save\\improve_net\\-6db\\model\\"  # 模型的存储路径
MODEL_NAME="bearing"  # 模型的名字
TFRecords_file = "D:\\Desktop\\python3\\paper_2\\data\\picture\\1hp\\train\\train.tfrecords"  # TFRecord数据的存放位置


def backward():
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None, 64, 64, 1])
        y_ = tf.placeholder(tf.float32, [None, 10])
        x_data = tf.reshape(x, [-1, 64, 64, 1])  # 注意此处要改变输入数据的形状

    y = ceshi_3.forward(x_data)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    data_batch, label_batch = readTFRescord.read_and_decode(TFRecords_file, BATCH_SIZE)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("D:\\Desktop\\python3\\paper_2\\save\\improve_net\\-6db\\log\\", sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        c = []
        a = []
        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            d, ys = sess.run([data_batch, label_batch])

            noi = abs(np.random.normal(0, 64, [BATCH_SIZE, 64, 64, 1]))  # 噪声的形状要与数据匹配
            noise = np.array(noi,dtype='uint8')
            datas = cv.add(d,noise)
            labels = readTFRescord.onehot(ys)

            _, lossValue, step = sess.run([train_step, loss, global_step],
                                          feed_dict={x: datas, y_: labels})
            accuracyScore = sess.run(accuracy,
                                      feed_dict={x: datas, y_: labels})
            a.append(accuracyScore)
            c.append(lossValue)

            if i % 50 == 0:
                print('Step is:%d and the loss value is:%f' % (step,lossValue))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)
                end_time = time.time()
                print('time:',(end_time-start_time))
                start_time = end_time

                accuracyScore = sess.run(accuracy,
                                          feed_dict={x: datas, y_: labels})
                print('The accuracy is:%f' % accuracyScore)

            summary = sess.run(merged, feed_dict={x: datas, y_: labels})
            writer.add_summary(summary,i)

        coord.request_stop()
        coord.join(threads)

    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(c)
    plt.show()


def main():
    backward()


if __name__ == '__main__':
    main()














