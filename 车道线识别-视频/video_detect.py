#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:28:54 2019

@author: wang
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# 从原图像获取二值后的图像并且转换到俯视角度
def Thresh_Warp(img, M):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 2.0)
    # 使用大津算法进行阈值分割
    retval, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # 透视变换
    warp = cv2.warpPerspective(binary_img, M, (img.shape[1], img.shape[0]), flags = cv2.INTER_LINEAR)
    return warp

def fit_lines(binary_img):
    # 计算图像下半部分每一列的灰度值之和（白色点个数 * 255）
    histogram = np.sum(binary_img[binary_img.shape[0] // 2:, :], axis = 0)
    out_img = np.dstack((binary_img, binary_img, binary_img)) # 扩充到 3 * 3
    midpoint = histogram.shape[0] // 2 # 中点
    left_base = np.argmax(histogram[:midpoint]) # 左边基准列（左半边中像素灰度值最大..)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # 右基准列
    nwindows = 9
    window_height = np.int(binary_img.shape[0] / nwindows) # 将图像纵向切成n个windows
    ## == Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero() # 像素不为0的点坐标
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = left_base
    rightx_current = rightx_base
    margin = 70 # 从当前左右中心向外延伸搜索的宽度
    minpix = 55 # 这个值很小，所以基本上每一个窗口都会要重新计算 left / rightx_current
    maxpix = 7000
    left_lane_inds=[]
    right_lane_inds = []

    for window in range(nwindows):
        #识别窗口边界在x和y(左、右)
        win_y_low = binary_img.shape[0] - (window + 1) * window_height #将图像切成9份，一份一份的统计白点数量
        #print('win_y_low',win_y_low)
        win_y_high = binary_img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        #print('win_xleft_low',win_xleft_low)
        win_xleft_high = leftx_current + margin
        #print('win_xleft_high = ',win_xleft_high)
        win_xright_low = rightx_current - margin
        #print('win_xright_low = ',win_xright_low)
        win_xright_high = rightx_current + margin
        #print('win_xright_high = ',win_xright_high)
        #把网格画在可视化图像上
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,0),10)#通过确定对角线 画矩形
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,0,0),10)

        #识别非零像素窗口内的x和y
        good_left_inds = (  (nonzeroy >= win_y_low)  & (nonzeroy < win_y_high)
                              & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                              & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        #添加这些指标列表
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #如果上面大于minpix，重新定位下一个窗口的平均位置
        if ((len(good_left_inds) > minpix) & (len(good_left_inds) < maxpix)):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if ((len(good_right_inds) > minpix) & (len(good_right_inds) < maxpix)):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    #连接索引的数组
    left_lane_inds = np.concatenate(left_lane_inds)
    #把list改成numpy格式而已
    right_lane_inds = np.concatenate(right_lane_inds)

    #提取左和右线像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #针对找出的白点，使用多项式进行拟合（二次多项式已经有效地进行拟合）
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #画图
    ploty = np.array(range(0, binary_img.shape[0], 1))
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty +left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 +right_fit[1] * ploty + right_fit[2]
    #这步的意思是把曲线拟合出来，
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    out_img[ploty.astype(int), left_fitx.astype(int)] = [0, 255, 255]
    out_img[ploty.astype(int), right_fitx.astype(int)] = [0, 255, 255]
    cv2.imshow("out_img", out_img)
    return left_fit, right_fit, out_img

def fit_lines_continous(binary_warped, left_fit, right_fit, nwindows = 9):
    nonzero = binary_warped.nonzero() # 所有白点集合
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    #  利用上一个时刻的拟合的车道线来确定此次搜索的范围（在上次的线周围2 * margin宽度）
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                    (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                    (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    ##  左右车道点坐标集合
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    if len(leftx) == 0 or len(lefty) ==0 or len(rightx)==0 or len(righty) == 0:
        return False, [], [], []


    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] -1,binary_warped.shape[0]) #用此来创建等差数列
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty +left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 +right_fit[1] * ploty + right_fit[2]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))]) # 126
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    # print(left_line_window1.shape, left_line_window2.shape)
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))]) # 126
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    cv2.imshow("out_img", out_img)
    return left_fit, right_fit, out_img


def warp_perspective_back(img, warped, left_fit, right_fit, Minv): # 切换视角回去
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))

    ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])
    #添加新的多项式在X轴Y轴
    # 左右车道线
    left_fitx = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]

    #把X和Y变成可用的形式
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    #np.transpose 转置
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    #向上/向下翻转阵列。
    pts = np.hstack((pts_left, pts_right))
    #填充图像
    cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))
    #透视变换
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    #叠加图层
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result

def main():
    srcMat = np.float32([[370, 660], [580, 460], [740, 460], [1270, 660]])
    # srcMat = np.float32([[740, 460], [1270, 660], [370, 660], [580, 460]])
    dstMat = np.float32([[346, 720], [346, 0], [1266, 0], [1266, 720]])
    M = cv2.getPerspectiveTransform(srcMat, dstMat) # 透视变换
    invM = cv2.getPerspectiveTransform(dstMat, srcMat) # 透视反变换
    cap = cv2.VideoCapture("project_video.mp4")
    flag = 0
    while cap.isOpened():
        ret, frame = cap.read()
        binary_img = Thresh_Warp(frame, M)
        if flag == 0:
            left_fit, right_fit, out_img = fit_lines(binary_img)
            flag = 1
        elif flag == 1:
            left_fit, right_fit, out_img = fit_lines_continous(binary_img, left_fit, right_fit, nwindows = 9)
            print(str(left_fit) +  " " + str(right_fit) )
        result = warp_perspective_back(frame, binary_img, left_fit, right_fit, invM)
        cv2.imshow("result", result)
        if (0xFF & cv2.waitKey(5) == 27):
            break
    cap.release()
    cv2.destroyAllWindows()

main()
