import cv2
import time
import rospy
import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mtx = np.array([[ 700.9994359,0,623.22280831], [0.,698.56813824,345.23773648], [0.,0.,1.]])
dist = np.array([[-0.32839342,0.1374803,-0.00050684,0.00093653,-0.03155774]])

rospy.init_node('takephoto',anonymous=True)
rate=rospy.Rate(2)

while(1):
    # get a frame
    ret, img = cap.read()
    # show a frame
    if ret == True:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imshow("undistort",img)
        cv2.imwrite("img_" + str(time.time())+".jpg", img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    rate.sleep()

cap.release()
cv2.destroyAllWindows()
