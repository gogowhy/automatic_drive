import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

# 彩色图像转为黑白图像
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
# 高斯滤波
def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
 
# sobel 算子求梯度，并阈值二值化
def abs_sobel_thresh(img,orient = 'x',sobel_kernel = 3,thresh = (0,255)):
    gray = grayscale(img)
    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel))
    
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

# 色彩空间转换，RGB->HLS，并针对亮度进行二值化
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <thresh[1])] = 1
    return binary_output

# 畸变矫正
def undistort(img):
    cal_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = cal_pickle['mtx']
    dist = cal_pickle['dist']
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist

# 读入图片
img_test = plt.imread('test_images/straight_lines1.jpg')
plt.figure(figsize = (10,10))
# 去畸变
undist = undistort(img_test)
plt.subplot(221)
plt.imshow(undist)
plt.title('Undistorted Iamge')
cv2.imwrite('./output_images/undist.jpg',255 * undist)
# 边缘图像(梯度)
x_sobel = abs_sobel_thresh(undist,thresh = (22,100))
plt.subplot(222)
plt.imshow(x_sobel,cmap = 'gray')
plt.title('x_sobel Gradients Image')
cv2.imwrite('./output_images/x_sobel.jpg',255 * x_sobel)
# 图像高亮度部分
color_transforms = hls_select(undist,thresh=(150,255))
plt.subplot(223)
plt.imshow(color_transforms,cmap = 'gray')
plt.title('Color Thresh Image')
cv2.imwrite('./output_images/color_transforms.png',255 * color_transforms)

color_x_sobel = np.zeros_like(x_sobel)
color_x_sobel[ (color_transforms == 1) | (x_sobel) == 1 ] = 1
plt.subplot(224)
plt.imshow(color_x_sobel,cmap = 'gray')
plt.title('color and granient image')
cv2.imwrite('./output_images/color_x_sobel.png', 255 * color_x_sobel)