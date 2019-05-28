import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# 用于相机畸变矫正的棋盘为一个 6 * 9 的棋盘
objp = np.zeros((6*9,3), np.float32)#构建一个72行，3列的零矩阵
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)#把数组变成网格的顺序

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
 
# 图片列表
images = glob.glob('camera_cal/calibration*.jpg')

# 逐张图片寻找所有角点
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    print('number:',fname,'ret = ',ret)
 
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
 
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        plt.figure(figsize = (8,8))
        plt.imshow(img)
        plt.show()