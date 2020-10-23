# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2 as cv

file_dir = "D:\\Desktop\\python3\\paper_2\\data\\picture\\3hp\\test\\"  # 图片文件位置，该文件夹中包含每一类的图片文件的文件夹
datas = []  # 用于存放每张图片路径的列表
temp = []  # 用于存放子文件夹

for root,sub_folders,files in os.walk(file_dir):  # 将所有文件列出来
    for name in files:
        datas.append(os.path.join(root,name))  # 将每一张图片文件的路径加入列表
    for name in sub_folders:
        temp.append(os.path.join(root,name))  # 将每一个子文件夹的路径加入列表

labels = []  # 用于存放标签
for one_folder in temp:
    n_data = len(os.listdir(one_folder))  # 统计子文件夹的个数
    letter = one_folder.split('\\')[-1]  # 取出子文件夹的名字
    if letter == '000normal':  # 以下是对每一个子文件夹里面的文件附上标签
        labels = np.append(labels, n_data * [0])
    elif letter == '007ball':
        labels = np.append(labels, n_data * [1])
    elif letter == '007inner':
        labels = np.append(labels, n_data * [2])
    elif letter == '007outer':
        labels = np.append(labels, n_data * [3])
    elif letter == '014ball':
        labels = np.append(labels, n_data * [4])
    elif letter == '014inner':
        labels = np.append(labels, n_data * [5])
    elif letter == '014outer':
        labels = np.append(labels, n_data * [6])
    elif letter == '021ball':
        labels = np.append(labels, n_data * [7])
    elif letter == '021inner':
        labels = np.append(labels, n_data * [8])
    else:
        labels = np.append(labels, n_data * [9])
# 打乱顺序
temp = np.array([datas,labels])  # 将数据的路径组成的列表和标签列表合起来
temp = temp.transpose()  # 转置，方便下面顺序的打乱
np.random.shuffle(temp)  # 然后打乱顺序，注意此处文件的标签是对应的，打乱的是不同种类的顺序

data_list = list(temp[:,0])  # 取出第一列为数据路径的列表
label_list = list(temp[:,1])  # 取出第二列为标签数据的列表，数据和标签一一对应
label_list = [int(float(i)) for i in label_list]  # 将标签转化为整形


#  2
# 转化为TFRecord格式
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


save_dir = "D:\\Desktop\\python3\\paper_2\\data\\picture\\3hp\\test\\"  # 存放TFRecord文件的位置
name = "test"  # 保存的文件名

fileneme = os.path.join(save_dir, name + '.tfrecords')  # 路径拼接
n_samples = len(label_list)  # 样本个数
writer = tf.python_io.TFRecordWriter(fileneme)
print('\ntransform start........')
for i in np.arange(0, n_samples):
    if i % 100 == 0:  # 方便观察进度
        print(i)
    try:
        data = cv.imread(data_list[i],2)  # 加载所有的图片文件
        data_raw = data.tostring()  # 将其转化为字符串
        label = int(label_list[i])  # 加载标签，标签与图片对应
        example = tf.train.Example(features=tf.train.Features(feature={  # 写入文件
            'label': int64_feature(label),
            'data_raw': bytes_feature(data_raw)}))
        writer.write(example.SerializeToString())
    except IOError as e:
        print('Could not read file')
writer.close()
print('transform done!')

















