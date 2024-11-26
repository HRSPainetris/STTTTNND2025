# drive model
# https://drive.google.com/drive/u/0/folders/1bBIL98GZ0IDHQiqZt66s0PVRNEt_A4JF
###################################################
##                     LIBRARY                   ##
###################################################
import os
import numpy as np

# Import Image Processing library
import cv2
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import dlib

# Multi-threading library
import threading
import queue

# Time library
import time
from time import strftime
from datetime import datetime

# Audio playing library
from pygame import mixer
mixer.init()

###################################################
##                     PATH                      ##
###################################################

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to the audio files
audio_path = os.path.join(CWD_PATH,"audio_files")

# Path to the system images
sys_img_path = os.path.join(CWD_PATH, "system_images")

###################################################
##                 VIDEO SAVING                  ##
###################################################
parent_directory = os.path.dirname(CWD_PATH)
os.chdir(parent_directory)
# Path to the input videos
in_vid_path = os.path.join(CWD_PATH, "RUN", "In_videos")
# Path to the output videos
out_vid_path = os.path.join(os.getcwd(), "RUN", "Out_videos")
print("out_vid_path: ", out_vid_path)

# Create folder for saving videos
## TODO: UPDATE THE NAME HERE
vid_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S") # video folder name format: YYYYMMDD_HHMMSS
vid_folder_path = os.path.join(out_vid_path, str(vid_folder_name))
os.mkdir(vid_folder_path)
print("Folder to save video path: ", vid_folder_path)

# Make the path for saving the video
left_out_vid_path = os.path.join(vid_folder_path, 'left_video.avi') # left camera video
right_out_vid_path = os.path.join(vid_folder_path, 'right_video.avi') # right camera video
driver_out_vid_path = os.path.join(vid_folder_path, 'driver_video.avi') # driver camera video

w_save_size = 640
h_save_size = 480
video_save_size = (w_save_size, h_save_size)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# XVID is more preferable. MJPG results in high size video. X264 gives very small size video
# Python: cv2.VideoWriter([filename, fourcc, fps, frameSize[, isColor]])
left_out_vid = cv2.VideoWriter(left_out_vid_path, fourcc, 30.0, video_save_size)
right_out_vid = cv2.VideoWriter(right_out_vid_path, fourcc, 30.0, video_save_size)
driver_out_vid = cv2.VideoWriter(driver_out_vid_path, fourcc, 30.0, video_save_size)

###################################################
##                 DATA SAVING                   ##
###################################################
# TODO 1: Save the data to the CSV file

###################################################
##                 LOAD MODELS                   ##
###################################################
# TODO 2: Load the model for object detection

# TODO 3: Load model for face detection

###################################################
##                     AUDIO                     ##
###################################################
## Phat loa
def phat_loa(audio_file_name):
    audio_file_name = os.path.join(audio_path, audio_file_name)
    if mixer.music.get_busy():
        speaker_wait = 1
        while speaker_wait == 1:
            if not mixer.music.get_busy():
                mixer.music.load(audio_file_name)
                mixer.music.play()
                speaker_wait = 0
    elif not mixer.music.get_busy():
        mixer.music.load(audio_file_name)
        mixer.music.play()
        
## Phat loa va cho den ket thuc
def phat_loa_until_end(audio_file_name):
    audio_file_name = os.path.join(audio_path, audio_file_name)
    if mixer.music.get_busy():
        speaker_wait = 1
        while speaker_wait == 1:
            if not mixer.music.get_busy():
                mixer.music.load(audio_file_name)
                mixer.music.play()
                speaker_wait = 0
                speaking = 1 
                while speaking == 1:
                    if not mixer.music.get_busy():
                        speaking=0            
                        time.sleep(0.05)
    elif not mixer.music.get_busy():
        mixer.music.load(audio_file_name)
        mixer.music.play()
        speaking = 1
        while speaking==1:
            if not mixer.music.get_busy():
                speaking=0            
            time.sleep(0.05)
            
###################################################
##                VISUALIZATION                  ##
###################################################
## SIZE OF IMAGE
w_show_size = 320
h_show_size = 240

system_name_img_width = w_show_size

## LOCATION OF IMAGE
system_name_location = [0, 0]

left_show_location = [0, 240]
right_show_location = [0, 485]
driver_show_location = [0, 760]

info_show_location = [480, 450]
info_img_width = 960

## FUNCTION: show image in a fixed window
def showInFixedWindow(winname, img, x, y):
    img = imutils.resize(img, width = w_show_size)
    cv2.namedWindow(winname, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)  # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)    
    
# FUNCTION: Show the system name image in a fixed window
def visualize_system_name():
    img = cv2.imread(os.path.join(sys_img_path, "system_name.png"))
    img = imutils.resize(img, width=system_name_img_width)
    (h_sysname_img, w_sysname_img) = img.shape[:2]
    print(f"System Name Image Size: {w_sysname_img} x {h_sysname_img}")
    showInFixedWindow('System Name', img, system_name_location[0], system_name_location[1])
    cv2.waitKey(1)

# FUNCTION: Show the notice image in a fixed window
def visualize_info(info_img_name):
    if cv2.getWindowProperty("System Notice", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("System Notice")
    img = cv2.imread(os.path.join(sys_img_path, info_img_name))
    img = imutils.resize(img, width=info_img_width)
    showInFixedWindow('System Notice', img, info_show_location[0], info_show_location[1])
    print("Showing System Notice: ", info_img_name)
    cv2.waitKey(1)
            
# FUNCTION: Show the left camera image in a fixed window
def visualize_left(left_frame):
    showInFixedWindow("Left Camera", left_frame, left_show_location[0], left_show_location[1])

# FUNCTION: Show the right camera image in a fixed window
def visualize_right(right_frame):
    showInFixedWindow("Right Camera", right_frame, right_show_location[0], right_show_location[1])
    
# FUNCTION: Show the driver camera image in a fixed window
def visualize_driver(driver_frame):
    showInFixedWindow("Driver Camera", driver_frame, driver_show_location[0], driver_show_location[1])
    
###################################################
##               VISUALIZE & AUDIO               ##
###################################################
def phat_loa_show_info(info_name):
    visualize_info(info_name + ".png")
    time.sleep(0.5)
    phat_loa_until_end(info_name + ".mp3")
    print("Da phat loa va hien thi thong bao: ", info_name)
    time.sleep(0.5)
    
def phat_loa_no_show_info(info_name):
    time.sleep(0.5)
    phat_loa_until_end(info_name + ".mp3")
    print("Da phat loa va hien thi thong bao: ", info_name)
    time.sleep(0.5)


###################################################
##                LIST CAMERAS                   ##
###################################################
# List all cameras connected to the computer 
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

###################################################
##        BUFFER-LESS VideoCapture               ##
###################################################
# Class to capture video from IP cameras without buffer
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

######################################################
## 2 IP CAMERA VIDEOS (L+R) + 1 USB CAMERA (DRIVER) ##
######################################################
# TODO 4: Process 2 videos from IP cameras and 1 streaming video from USB camera

######################################################
##    2 IP CAMERAS (L+R) + 1 USB CAMERA (DRIVER)    ##
######################################################
# TODO 5: Process 2 streaming videos from IP cameras and 1 streaming video from USB camera
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

######################################################
##    2 USB CAMERAS (L+R) + 1 USB CAMERA (DRIVER)   ##
######################################################
# TODO 6: Process 3 streaming videos from USB cameras
def collect_data_2_usb_cam():
    global fps

    # LEFT Camera
    left_frame_org = left_cam.read()
    # Resize anh cang nho xu ly cang nhanh
    left_frame_resize = cv2.resize(left_frame_org, (w_save_size, h_save_size), interpolation=cv2.INTER_LINEAR)
    left_frame = left_frame_resize
    
    # RIGHT Camera
    right_frame_org = right_cam.read()
    right_frame_resize = cv2.resize(right_frame_org, (w_save_size, h_save_size), interpolation=cv2.INTER_LINEAR)        
    right_frame = right_frame_resize
    
    start_time = time.time()
    
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
###################################################\
# TODO 7: Choosing the image source
'''

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

'''
# # List all USB cameras
# cameras = list_cameras()
# print("Connected cameras:", cameras)

# left_src_cam = 0
# right_src_cam = 1
# try:
#     print("[INFO] starting LEFT camera ...")
#     left_cam = VideoStream(src=left_src_cam).start()
#     print("LEFT camera connect Successfully!")
#     print("[INFO] starting RIGHT camera ...")
#     right_cam = VideoStream(src=right_src_cam).start()
#     time.sleep(1.0)
#     print("RIGHT camera connect Successfully!")
# except:
#     print("Connect not successfully!!!")
#     pass


w_process_size = 320
h_process_size = 240  
          
###################################################
##             INTERFACE WITH PX4                ##
###################################################
# TODO 8: Connect to the PX4 to get the GPS data

###################################################
##                INPUT FROM KEYBOARD            ##
###################################################
# TODO 9: Get the input from the keyboard

###################################################
##                  MAIN FUNCTION                ##
###################################################
# TODO 10: Write the main function to run the system

if __name__ == "__main__":    
    visualize_system_name()
    begin_time = time.time()
    frame_rate = 30
    fps = 0    
    
    ###################################################
    ##       AUDIO AND VISUALIZATION CHECK           ##
    ###################################################
    # DONE 1: Check the audio and visualization
    phat_loa_no_show_info("1_chao_mung")
    # phat_loa_show_info("2_da_ghi_thong_tin")
    # phat_loa_show_info("3_cham_2ben")
    # phat_loa_show_info("4_cham_phai")
    # phat_loa_show_info("5_cham_trai")
    # phat_loa_show_info("6_nguoi_2ben")
    # phat_loa_show_info("7_nguoi_phai")
    # phat_loa_show_info("8_nguoi_trai")
    # phat_loa_show_info("9_bien_bao_2ben")
    # phat_loa_show_info("10_bien_bao_phai")
    # phat_loa_show_info("11_bien_bao_trai")
    # phat_loa_show_info("12_cau_truoc")
    # phat_loa_show_info("13_cau_trai")
    # phat_loa_show_info("14_cau_phai")
    # phat_loa_show_info("15_chieu_cao_tau")
    phat_loa_show_info("16_3h_lien_tuc")
    phat_loa_show_info("17_buon_ngu")
    phat_loa_show_info("18_tap_trung")
    
    
    # while True:
    #     '''
    #     # 1. Use this to collect the image data from both left and right camera
    #     '''
    #     collect_data_2_ip_cam()
    
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # When everything is done, release the capture
    # print("[INFO] cleaning up...")
    # left_cam.cap.release()
    # right_cam.cap.release()
    # cv2.destroyAllWindows()
    
    
    