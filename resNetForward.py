# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers


def weight_variable(shape):  # 可用于任何形状
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape):  # 可用于任何形状
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,w):  # 卷积步长为1
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def conv(x,output,name,stride=1,relu=True):
    input = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"weight",shape=[3, 3, input, output],
                                 initializer=layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.0,shape=[output]))
        conv = tf.nn.conv2d(x,kernel,strides=[1,stride,stride,1],padding="SAME") + bias
        bn = batch_norm(conv)
        if relu == True:
            relu = tf.nn.relu(bn)
        else:
            relu = bn
        return relu


def conv_first(x,output):
    input = x.get_shape()[-1].value
    kernal = tf.Variable(tf.truncated_normal([1, 1, input, output]))
    conv = tf.nn.conv2d(x, kernal, strides=[1,1,1,1], padding='SAME')
    return conv


def max_pool_2x2(x):  # 池化步长为2，结果图片尺寸减半
    tf.nn.max_pool(x,ksize=[1,2,2,1],strides=2,padding="same")
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def batch_norm(x):  # 此处增加了Batch Normalization
    fc_mean, fc_var = tf.nn.moments(x, axes=[0], )
    scale = tf.Variable(tf.ones([x.get_shape()[-1]]))
    shift = tf.Variable(tf.zeros([x.get_shape()[-1]]))
    epsilon = 0.001
    z = tf.nn.batch_normalization(x, fc_mean, fc_var, shift,scale, epsilon)
    return z


def forward(x):  # 输入图片尺寸 64 x 64 x 1
    with tf.name_scope("forward"):
        # 第一层卷积
        with tf.name_scope("conv1"):
            w1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
            b1 = tf.Variable(tf.constant(0.0, shape=[32]))
            c1 = conv2d(x, w1) + b1  # 64 x 64 x 32
            bn1 = batch_norm(c1)
            relu1 = tf.nn.relu(bn1)  # 64 x 64 x 32
            # 第二层池化
            pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  # 32 x 32 x 32
        # 以下是残差单元
        # unit1
        with tf.name_scope("unit_1"):  # ****有问题****
            shortcut1_1 = tf.nn.max_pool(pool1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  # 16 x 16 x 32
            res1_1 = conv(pool1, 32,"res1", stride=2)  # 16 x 16 x 32
            res1_2 = conv(res1_1, 32,"res2", relu=False)  # 16 x 16 x 32
            res_add1 = shortcut1_1 + res1_2  # 16 x 16 x 32
            output1 = tf.nn.relu(res_add1)  # 16 x 16 x 32
        # unit2
        with tf.name_scope("unit_2"):
            res2_1 = conv(output1,32,"res1")  # 16 x 16 x 32
            res2_2 = conv(res2_1,32,"res2",relu=False)  # 16 x 16 x 32
            res_add2 = output1 + res2_2  # 16 x 16 x 32
            output2 = tf.nn.relu(res_add2)  # 16 x 16 x 32
        # unit3
        with tf.name_scope("unit_3"):
            shortcut3_1 = tf.nn.max_pool(output2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  # 8 x 8 x 32
            shortcut3_2 = conv_first(shortcut3_1,64)
            res3_1 = conv(output2,64,"res1",stride=2)  # 8 x 8 x 64
            res3_2 = conv(res3_1,64,"res2",relu=False)  # 8 x 8 x 64
            res_add3 = shortcut3_2 + res3_2  # 8 x 8 x 64
            output3 = tf.nn.relu(res_add3)  # 8 x 8 x 64
        # unit4
        with tf.name_scope("unit_4"):
            res4_1 = conv(output3,64,"res1")  # 8 x 8 x 64
            res4_2 = conv(res4_1,64,"res2",relu=False)  # 8 x 8 x 64
            res_add4 = output3 + res4_2  # 8 x 8 x 64
            output4 = tf.nn.relu(res_add4)  # 8 x 8 x 64
        # unit5
        with tf.name_scope("unit_5"):
            shortcut5_1 = tf.nn.max_pool(output4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  # 4 x 4 x 64
            shortcut5_2 = conv_first(shortcut5_1,128)
            res5_1 = conv(output4,128,"res1",stride=2)  # 4 x 4 x 128
            res5_2 = conv(res5_1,128,"res2",relu=False)  # 4 x 4 x 128
            res_add5 = shortcut5_2 + res5_2  # 4 x 4 x 128
            output5 = tf.nn.relu(res_add5)  # 4 x 4 x 128
        # unit6
        with tf.name_scope("unit_6"):
            res6_1 = conv(output5,128,"res1")  # 4 x 4 x 128
            res6_2 = conv(res6_1,128,"res2",relu=False)  # 4 x 4 x 128
            res_add6 = output5 + res6_2  # 4 x 4 x 128
            output6 = tf.nn.relu(res_add6)  # 4 x 4 x 128
        # unit7
        with tf.name_scope("unit_7"):
            shortcut7_1 = tf.nn.max_pool(output6,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  # 2 x 2 x 128
            shortcut7_2 = conv_first(shortcut7_1,256)
            res7_1 = conv(output6,256,"res1",stride=2)  # 2 x 2 x 256
            res7_2 = conv(res7_1,256,"res2",relu=False)  # 2 x 2 x 256
            res_add7 = shortcut7_2 + res7_2  # 2 x 2 x 256
            output7 = tf.nn.relu(res_add7)  # 2 x 2 x 256
        # unit8
        with tf.name_scope("unit_8"):
            res8_1 = conv(output7,256,"res1")  # 2 x 2 x 256
            res8_2 = conv(res8_1,256,"res2",relu=False)  # 2 x 2 x 256
            res_add8 = output7 + res8_2  # 2 x 2 x 256
            output8 = tf.nn.relu(res_add8)  # 2 x 2 x 256
        # 平均池化
        pool2 = tf.nn.max_pool(output8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 1 x 1 x 256
        # 最终分类
        h_pool2_flat = tf.reshape(pool2,[-1,1 * 1 * 256])
        w_fc = tf.Variable(tf.truncated_normal([1 * 1 * 256, 10]))
        b_fc = tf.Variable(tf.constant(0.0,shape=[10]))
        y_conv = tf.add(tf.matmul(h_pool2_flat,w_fc),b_fc)
        y = batch_norm(y_conv)
        return y  # 1 x 10

'''
if __name__ == "__main__":
    import cv2 as cv
    src = cv.imread('D:\Desktop\MATLAB\garbor.jpg',0)
    x = src.astype(np.float32)
    x = x.reshape(1,64,64,1)
    y = forward(x)
    print("最终输出结果：")
    print(y)
    print("End!")
'''





























