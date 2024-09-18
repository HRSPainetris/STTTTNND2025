import os
import sys
import time
import threading
import numpy as np
import cv2

# also acts (partly) like a cv.VideoCapture
class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes        
        self.callback = None
        
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond: # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)
        
# Camera 1: LEFT
cam1_url = "rtsp://khkt2024cam1:khkt2024@ndc!@192.168.200.160:554/stream2"
cap1 = cv2.VideoCapture(cam1_url)

# Camera 2: RIGHT
cam2_url = "rtsp://khkt2024cam2:khkt2024@ndc!@192.168.200.189:554/stream2"
cap2 = cv2.VideoCapture(cam2_url)

while(True):
    cap1.set(cv2.CAP_PROP_FPS, 30)
    # wrap it
    frame1 = FreshestFrame(cap1)

    cap2.set(cv2.CAP_PROP_FPS, 30)
    # wrap it
    frame2 = FreshestFrame(cap2)

    # _, frame2 = cap2.read() 
    # # Get the original frame size
    frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # img2 = cv2.resize(frame2, (720, 540), interpolation = cv2.INTER_LINEAR)
      
    # Capture the video frame 
    # by frame 
    # _, frame1 = cap1.read() 
    # Get the original frame size
    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # img1 = cv2.resize(frame1, (720, 540), interpolation = cv2.INTER_LINEAR)



    # concatenate image Horizontally 
    Hori = np.concatenate((frame1, frame2), axis=1) 
    cv2.imshow('Cam1+2', Hori)

    # Display the resulting frame 
    # cv2.imshow('Cam1', frame1) 
    # cv2.waitKey(1) 
    # cv2.imshow('Cam2', frame2) 
    # print("Cam 1: {} x {}.".format(frame_width1,frame_height1))

    print("Cam 1: {} x {}. Cam 2: {} x {}.".format(frame_width1,frame_height1,frame_width2,frame_height2))
    # time.sleep(1)
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object 
cap1.release() 
cap2.release()
# Destroy all the windows 
cv2.destroyAllWindows() 