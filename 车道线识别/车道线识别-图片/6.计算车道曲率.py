import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def curvature(left_fit,right_fit,binary_warped,print_data = True):
    ploty = np.linspace(0,binary_warped.shape[0] -1 , binary_warped.shape[0])
    y_eval = np.max(ploty)
    #y_eval就是曲率，这里是选择最大的曲率
    
    ym_per_pix = 30/720#在y维度上 米/像素
    xm_per_pix = 3.7/700#在x维度上 米/像素
    
    #确定左右车道
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #定义新的系数在米
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    #最小二乘法拟合
    
    #计算新的曲率半径
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) 
    / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) 
    / np.absolute(2*right_fit_cr[0])
    
    #计算中心点，线的中点是左右线底部的中间
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix#转换成米
    
    if print_data == True:
        #现在的曲率半径已经转化为米了
        print(left_curverad, 'm', right_curverad, 'm', center, 'm')
 
    return left_curverad, right_curverad, center


new_path = os.path.join("test_images/","*.jpg")
for infile in glob.glob(new_path):
    #读图
    img = plt.imread(infile)
    #畸变
    undist = undistort(img)
    #sobel算子
    x_sobel = abs_sobel_thresh(undist,thresh = (22,100))
    #hls颜色阈值
    color_transforms = hls_select(undist,thresh=(90,255))
    #sobel加hls
    color_x_sobel = np.zeros_like(x_sobel)
    color_x_sobel[ (color_transforms == 1) | (x_sobel) == 1 ] = 1
    #弯曲图像（warped）
    print()
    print('Image name = ',infile)
    warped_img,unpersp, Minv = warp(color_x_sobel)
    #画线
    find_line_imgae,left_fit,right_fit = find_lines(warped_img)
    #算曲率
    curvature(left_fit,right_fit,find_line_imgae)
