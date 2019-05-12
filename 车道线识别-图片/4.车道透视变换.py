import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 画出二值图像并标出透视变换的基准点
plt.imshow(color_x_sobel,cmap = 'gray')
plt.plot(800,510,'x')
plt.plot(1150,700,'x')
plt.plot(270,700,'x')
plt.plot(510,510,'x')

def warp(img):
    img_size = (img.shape[1],img.shape[0])
    src = np.float32( [ [800,510],[1150,700],[270,700],[510,510]] )
    dst = np.float32( [ [1100,200],[1100,700],[200,700],[200,200]] )
    M = cv2.getPerspectiveTransform(src,dst)
    #返回透视变换的映射矩阵，就是这里的M。对于投影变换，我们只需要知道四个点，
    #通过cv2.getPerspectiveTransform求得变换矩阵.之后使用cv2.warpPerspective获得矫正后的图片。
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img,M,img_size,flags = cv2.INTER_LINEAR)
    #主要作用：对图像进行透视变换，就是变形
    #https://blog.csdn.net/qq_18343569/article/details/47953843
    unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
    return warped, unpersp, Minv

warped_img,unpersp, Minv = warp(color_x_sobel)

plt.figure(figsize = (15,15))
plt.subplot(121)   #绘制一行两列的子图
plt.imshow(warped_img)
plt.plot(1100,200,'x')
plt.plot(1100,700,'x')
plt.plot(200,700,'x')
plt.plot(200,200,'x')
plt.title('warped image')
 
plt.subplot(122)
plt.imshow(unpersp)
plt.plot(800,510,'x')
plt.plot(1150,700,'x')
plt.plot(270,700,'x')
plt.plot(510,510,'x')
plt.title('original image')

#cv2.imshow("warped", 255 * warped_img)
#cv2.imshow("original", 255 * unpersp)
#cv2.waitKey(0)