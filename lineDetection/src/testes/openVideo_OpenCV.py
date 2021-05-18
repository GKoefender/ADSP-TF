import numpy as np
import cv2

###########################################################

video = cv2.VideoCapture('/home/gustavo/Vídeos/2.mp4')
while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture('/home/gustavo/Vídeos/2.mp4')
        continue

    gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(orig_frame,(x1,y1),(x2,y2),(0,255,0),2)
    
    
    #cv2.imshow("mask", mask)
    #cv2.imshow("temp", temp)
    cv2.imshow("orig_frame", orig_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
