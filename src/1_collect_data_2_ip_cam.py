import cv2
import threading
import queue
import numpy as np
import os

# Time library
import time
from time import strftime
from datetime import datetime

# Get the parent directory
parent_directory = os.path.dirname(os.getcwd())

# Change the current working directory to the parent directory
os.chdir(parent_directory)
CWD_PATH = os.getcwd()

out_vid_path = os.path.join(CWD_PATH, "Out_Video")

# Make folder for saving the Output Video
folder_name = datetime.now().strftime("%Y%m%d_%H_%M")
folder_path = os.path.join(out_vid_path, str(folder_name))
os.mkdir(folder_path)
print("Folder to save video path: ",folder_path)

# Make the path for saving the video
left_out_vid_path = os.path.join(folder_path, 'left_video.avi')
right_out_vid_path = os.path.join(folder_path, 'right_video.avi')

w_save_size = 640
h_save_size = 480
video_save_size = (w_save_size, h_save_size)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# XVID is more preferable. MJPG results in high size video. X264 gives very small size video
# Python: cv2.VideoWriter([filename, fourcc, fps, frameSize[, isColor]])
left_out_vid = cv2.VideoWriter(left_out_vid_path, fourcc, 30.0, video_save_size)
right_out_vid = cv2.VideoWriter(right_out_vid_path, fourcc, 30.0, video_save_size)

# Buffer-less VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.running = True  # Flag to indicate if the thread should keep running
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                return
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # Discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            self.state=ret

    def read(self):
        return self.q.get(),self.state

    def stop(self):
        self.running = False
        self.t.join()  # Wait for the thread to exit

###################################################
##             COLLECT DATA IP CAMERAS           ##
###################################################   
# Function: Collect data
def collect_data_2_ip_cam():
    global fps

    # LEFT Camera
    left_frame_org, success = left_cam.read()
    # Resize anh cang nho xu ly cang nhanh
    left_frame_resize = cv2.resize(left_frame_org, (w_save_size, h_save_size), interpolation=cv2.INTER_LINEAR)
    left_frame = left_frame_resize
    
    # RIGHT Camera
    right_frame_org, success = right_cam.read()
    right_frame_resize = cv2.resize(right_frame_org, (w_save_size, h_save_size), interpolation=cv2.INTER_LINEAR)        
    right_frame = right_frame_resize
    
    start_time = time.time()
    if not success:
        return
    
    loop_time = time.time() - start_time
    delay = max(1, int((1 / frame_rate - loop_time) * 1000))

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        return

    loop_time2 = time.time() - start_time
    if loop_time2 > 0:
        fps = 0.9 * fps + 0.1 / loop_time2
        print("FPS:", fps)
        
    ###############################
    #      Display the image      #
    ###############################
        
    cv2.putText(left_frame_resize, f"LEFT Camera FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(right_frame_resize, f"RIGHT Camera FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    show_img = np.concatenate((left_frame_resize, right_frame_resize), axis=1)
    cv2.imshow('LEFT and RIGHT Camera', show_img)

    # total_time = time.time() - begin_time
    # print("Total time taken:", total_time, "seconds")
      
    # Save the videos
    left_out_vid.write(left_frame)
    right_out_vid.write(right_frame)  

###################################################
##             INTERFACE WITH CAMERA             ##
###################################################

# IP CAMERAS
try:
    print("[INFO] starting LEFT camera ...")
    left_cam = VideoCapture("rtsp://khkt2024left:khkt2024@ndc!@192.168.0.100:554/stream1")
    print("LEFT camera connect Successfully!")
    print("[INFO] starting RIGHT camera ...")
    right_cam = VideoCapture("rtsp://khkt2024right:khkt2024@ndc!@192.168.0.103:554/stream1")
    time.sleep(1.0)
    print("RIGHT camera connect Successfully!")
except:
    print("Connect not successfully!!!")
    pass
          
begin_time = time.time()
frame_rate = 30
fps = 0

while True:
    '''
    # 1. Use this to collect the image data from both left and right camera
    '''
    collect_data_2_ip_cam()
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
print("[INFO] cleaning up...")
left_cam.cap.release()
right_cam.cap.release()
cv2.destroyAllWindows()
