# Tensorflow+Spyder+Opencv环境搭建
## 1.Tensorflow的安装
使用python的包管理器**pip**安装**tensorflow**: 

```` pip install tensorflow 
````
## 2.OpenCV的安装和使用

Opencv的安装参见官网
在tensorflow环境中使用Opencv，解决import cv2的ImportError。
按照Opencv官网的源码编译安装方式，在_usr_local_lib_python2.7/site-packages文件夹下找到cv2.so这个动态链接库，将其复制到_home_username_anaconda2_envs_tensorflow_lib_python2.7_site-packages/目录下解决此问题。
参考：[Tensorflow+Spyder+Opencv环境搭建 - lx_ros的博客 - CSDN博客](https://blog.csdn.net/lx_ros/article/details/78804626)

在conda命令行中输入：
````
conda install –c https://conda.binstar.org/menpo opencv
````
即可完成opencv的安装和配置

