#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class rplidarDetector:
    def __init__(self):
        rospy.init_node('rplidar_detection', anonymous=True)
        rospy.Subscriber('/scan', LaserScan, self.callback)
        self.pub = rospy.Publisher('lidar_vel', Twist, queue_size=1)
        print 'LiDAR is OK'

    def road_detection(self, msg):
        #雷达正前方为720
        ############################ 找右侧距离最近的点（30-60度范围内）
        right_min = 25
        right_min_index = 480//4
        for i in range(480//4, 660//4):
            if msg.ranges[i] < right_min and msg.ranges[i] > 0.05:
                right_min = msg.ranges[i]
                right_min_index = i

        ############################ 找左侧距离最近的点（30-60度范围内）
        left_min = 25
        left_min_index = 960//4
        for i in range(960//4, 780//4,-1):
            if msg.ranges[i] < left_min and msg.ranges[i] > 0.05:
                left_min = msg.ranges[i]
                left_min_index = i

        ################################ 判断左右哪边离障碍物更近
        offset = left_min - right_min
        return offset

    def callback(self, msg):
        offset = self.road_detection(msg)
        #print('offset')
       # print(offset * 0.6)
        twist = Twist()
        twist.linear.x = 0.3
        twist.angular.z = offset * 1.8
        self.pub.publish(twist)

if __name__ == '__main__':
    try:
        detector = rplidarDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
