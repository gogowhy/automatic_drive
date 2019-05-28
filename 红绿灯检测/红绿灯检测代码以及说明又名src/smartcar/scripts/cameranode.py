#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import cv2
import os
import sys
import glob
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class camera:
    def __init__(self, device, width, height, rates):
        currentpath, _ = os.path.split(os.path.abspath(sys.argv[0]))
        self.calibrationPath = os.path.join(currentpath, 'calib_pics')
        self.testImgPath = os.path.join(currentpath, 'cc.jpg')
        
        self.camMat = []
        self.camDistortion = []

        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.pub = rospy.Publisher('images', Image, queue_size=1)
        rospy.init_node('camera', anonymous=True)
        self.rate = rospy.Rate(rates)

        self.cvb = CvBridge()        

    def calibrate(self):
        ##== termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        ##== obh points, like (0,0,0),(1,0,0),...(6,6,0)
        objp = np.zeros([7*7, 3], np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
        
        ##== obj points and img points
        objPoints = []
        imgPoints = []
        gray = []
        ##== paths of training imgs
        imgPaths = glob.glob(self.calibrationPath + '/*.jpg')
        if imgPaths == []:
            print 'invalid calibration path'
            return -1
        for path in imgPaths:
            img = cv2.imread(path)
            ##== use gray imgs rather than true color imgs
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ##== find corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
            
            ## if found
            if ret == True:
                objPoints.append(objp)
                
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgPoints.append(corners2)

                #== draw and show img corners
                #img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
                #cv2.imshow('img', img)
                #cv2.waitKey(500)
        #cv2.destroyAllWindows()

        ##== calibration
        ret, cameraMatrix, cameraDistortion, rotation, translation = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
        
        if cameraMatrix != []:
            self.camMat = cameraMatrix
            self.camDistortion = cameraDistortion
            print 'CALIBRATION SUCCESSED!'
            print cameraMatrix
            print cameraDistortion
        else:
            print 'CALIBRATION FAILED!'
        return 0

    def spin(self):
        while not rospy.is_shutdown():
            ret, img = self.cap.read()
            if ret == True:
                dst = cv2.undistort(img, self.camMat, self.camDistortion, None, self.camMat)
                #cv2.imshow("undistort",dst)
                #cv2.waitKey(1)
                self.pub.publish(self.cvb.cv2_to_imgmsg(dst))
            self.rate.sleep()

        self.cap.release()

if __name__ == '__main__':
    device = rospy.get_param('device', 1)
    width = rospy.get_param('width', 1280)
    height = rospy.get_param('height', 720)
    rates = rospy.get_param('rates', 10)
    try:
        cam = camera(device, width, height, rates)
        cam.calibrate()
        cam.spin()
    except rospy.ROSInterruptException:
        pass
