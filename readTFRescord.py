# -*- coding:utf-8 -*-
# 读取TFRecord文件，并mini-batch
# 独热编码时注意类别个数
# 注意20行中数据点的个数
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])  # 打开队列
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)  # 读取文件
    data_features = tf.parse_single_example(
        serialized_example,
        features={
            'label':tf.FixedLenFeature([],tf.int64),
            'data_raw':tf.FixedLenFeature([],tf.string),
        })
    data = tf.decode_raw(data_features['data_raw'],tf.uint8)  # 解码data
    data = tf.reshape(data,[64,64,1])  # 整理输出形状注意数据点的个数
    label = tf.cast(data_features['label'],tf.int32)  # 解码标签

    data_batch,label_batch = tf.train.shuffle_batch([data,label],
                                                    batch_size=batch_size,
                                                    num_threads=64,
                                                    capacity=200,
                                                    min_after_dequeue=100)
    return data_batch, tf.reshape(label_batch,[batch_size])  # label_batch


def onehot(labels):  # 独热编码
    '''one-hot编码'''
    n_sample = len(labels)  # 样本的个数
    n_class = 10  # 分类的个数
    onehot_labels = np.zeros([n_sample,n_class])  # 全为0
    onehot_labels[np.arange(n_sample),labels] = 1  # 将类别转化为1
    return onehot_labels



# ## 以下代码主要用于验证
#
# data,label = read_and_decode("D:\\Desktop\\python3\\paper_2\\data\\fuliye\\test\\test.tfrecords",1)
#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     d,l = sess.run([data,label])
#     pic=np.array(d)
#     print(pic.shape)
#     print(l)
#
#     coord.request_stop()
#     coord.join(threads)

























