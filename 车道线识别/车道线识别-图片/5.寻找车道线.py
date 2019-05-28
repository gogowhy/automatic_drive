import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

def find_lines(img,print = True):
    #假设您已经创建了一个被扭曲的二进制图像，称为“binary_warped”
    #图像下半部分车道线点数量的直方图
    histogram = np.sum(img[img.shape[0] //2:,:],axis = 0)
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(img.shape[1]), histogram)
    plt.show()
    #创建一个输出图像来绘制和可视化结果
    out_img = np.dstack((img,img,img))*255
    #找出直方图的左半边和右半边的峰值, 这些将是左行和右行的起点
    midpoint = np.int(histogram.shape[0] // 4)
    leftx_base = np.argmax(histogram[:midpoint])
    #np.argmax 是返回最大值所在的位置
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #这里是要返回右边HOG值最大所在的位置，所以要加上midpoint
 
    #选择滑动窗口的数量
    nwindows = 9
    #设置窗口的高度
    window_height = np.int(img.shape[0] // nwindows)
    #确定所有的x和y位置非零像素在图像,这里就是把img图像中非0元素（就是不是黑的地方就找出来，一行是x，一行是y）
    nonzero = img.nonzero()
    #返回numpy数组中非零的元素
    #对于二维数组b2，nonzero(b2)所得到的是一个长度为2的元组。http://www.cnblogs.com/1zhk/articles/4782812.html
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #为每个窗口当前位置更新
    leftx_current = leftx_base
    rightx_current = rightx_base
    #设置窗口的宽度
    margin = 100
    #设置最小数量的像素发现重定位窗口
    minpix = 50
    #创建空的列表接收左和右车道像素指数
    left_lane_inds = []
    right_lane_inds = []
 
    #遍历窗口
    for window in range(nwindows):
        #识别窗口边界在x和y(左、右)
        win_y_low = img.shape[0] - (window + 1) * window_height #将图像切成9份，一份一份的统计白点数量
        #print('win_y_low',win_y_low)
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        #print('win_xleft_low',win_xleft_low)
        win_xleft_high = leftx_current + margin
        #print('win_xleft_high = ',win_xleft_high)
        win_xright_low = rightx_current - margin
        #print('win_xright_low = ',win_xright_low)
        win_xright_high = rightx_current + margin
        #print('win_xright_high = ',win_xright_high)
        #把网格画在可视化图像上
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)#通过确定对角线 画矩形
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
 
        #识别非零像素窗口内的x和y
        good_left_inds = (  (nonzeroy >= win_y_low)  & (nonzeroy < win_y_high)  
                              & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
 
        good_right_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                              & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
 
        #添加这些指标列表
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #如果上面大于minpix，重新定位下一个窗口的平均位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
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
    ploty = np.linspace(0,img.shape[0] -1,img.shape[0]) #用此来创建等差数列
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty +left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 +right_fit[1] * ploty + right_fit[2]
    #这步的意思是把曲线拟合出来，
 
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    if print == True:
        plt.figure(figsize=(8,8))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.show()
    return out_img,left_fit,right_fit

find_line_imgae,left_fit,right_fit = find_lines(warped_img)
