# -*- coding:utf-8 -*-
# 将一个mat数据类型的数组转化为一系列txt文件
#修改时注意第 8行的文件，9行的字典key，11行的循环次数，13行的存放位置，其他的不要改
import numpy as np
import scipy.io as sio
import codecs

data_mat = sio.loadmat(r'D:\Desktop\bearing data\3hp\021outer\237.mat')# mat文件路径
data1 = data_mat['X237_DE_time'] # 读取mat字典中的一列数据

for i in range(0,100000,100): # 900为9个,901为10个   循环次数为生成的txt文件个数
    data2 = data1[i:i+2048:1] # 每次循环提取2048个点
    f = codecs.open(r'D:\Desktop\bearing data\txt\3hp\021outer\%d.txt'% (i/100), mode='w+') #  注意i/100   写入txt文件的位置
    for j in data2:  # 循环次数为data2中数据的个数
        k = str(j)   # 将数据转化为字符串，不然无法写入
        k = k.replace("[", "").replace("]", "")  #将转化为字符串后的括号去掉
        f.write(k + '\r\n')  # 将数据（字符串）写入文件，末尾换行
    f.close()  # 关闭文件


# 显示出来看一下
# b=np.loadtxt('D:\\Desktop\\python3\\bearing\\data\\normal\\txt\\3.txt',dtype=np.float32)
# plt.plot(b)
# plt.show()



