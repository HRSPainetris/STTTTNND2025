import cv2
import numpy as np
import time

# Camera 1: LEFT
cam1_url = "rtsp://khkt2024left:khkt2024@ndc!@192.168.0.100:554/stream1"
cap1 = cv2.VideoCapture(cam1_url)
cap1.set(cv2.CAP_PROP_FPS, 5.0)

# Camera 2: RIGHT
cam2_url = "rtsp://khkt2024right:khkt2024@ndc!@192.168.0.103:554/stream1"
cap2 = cv2.VideoCapture(cam2_url)
cap2.set(cv2.CAP_PROP_FPS, 5.0)

while(True): 
    _, frame2 = cap2.read() 
    # # Get the original frame size
    frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img2 = cv2.resize(frame2, (640, 360), interpolation = cv2.INTER_LINEAR)
      
    # Capture the video frame 
    # by frame 
    _, frame1 = cap1.read() 
    # Get the original frame size
    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img1 = cv2.resize(frame1, (640, 360), interpolation = cv2.INTER_LINEAR)



    # concatenate image Horizontally 
    Hori = np.concatenate((img1, img2), axis=1) 
    cv2.imshow('Cam1+2', Hori)
    print("Cam 1: {} x {}. Cam 2: {} x {}.".format(frame_width1,frame_height1,frame_width2,frame_height2))

    # Display the resulting frame 
    # cv2.imshow('Cam1', frame1) 
    # print("Cam 1: {} x {}.".format(frame_width1,frame_height1))
    # cv2.waitKey(1) 
    # cv2.imshow('Cam2', img2) 
    # print("Cam 2: {} x {}.".format(frame_width2,frame_height2))

    # time.sleep(1)
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object 
# cap1.release() 
cap2.release()
# Destroy all the windows 
cv2.destroyAllWindows() 