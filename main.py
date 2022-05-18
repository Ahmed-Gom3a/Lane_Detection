
import cv2
import pipeline
import globals
import sys

globals.init()
# capture frames from a camera
cap=cv2.VideoCapture(sys.argv[1])
while(cap.isOpened()):
    # reads frames from a camera
    ret,frame=cap.read()

    if ret==1:
        #Call the pipeline in a single the captured frame from the video
        out_frame=pipeline.PIPELINE_lane_finding(frame,1)
        cv2.imshow("output",out_frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    else:
        print("[ERROR] Can't open the video")
        break
