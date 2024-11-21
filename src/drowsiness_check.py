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
from threading import Thread
import queue

# Time library
import time
from time import strftime
from datetime import datetime

# Audio playing library
from pygame import mixer
mixer.init()

# Math library for calculations
import math

###################################################
##              PARAMETER CONFIGURATION          ##
###################################################
# TODO 12: Set the parameters for the system
# DROWSINESS DETECTION
select_para = 0
EYE_AR_THRESH = 0.22 # Nguong EAR
EYE_AR_CONSEC_FRAMES = 15 # So frame lien tuc de xac dinh buon ngu
# Set_time - Dat truoc thoi gian canh bao tai cong lai lien tuc
SET_HR = 0      # GIO
SET_MIN = 3     # PHUT
SET_SEC = 0     # GIAY

FACE_COUNTER_THRES = 30 # So fame lien tuc co khuon mat de he thong bat dau hoat dong
COUNTER_NON_FACE_THRES_TAP_TRUNG = 30 # So frame lien tuc khong co khuon mat de canh bao TAP TRUNG
COUNTER_NON_FACE_THRES_RESET = 150 # So frame lien tuc khong co khuon mat de reset he thong
time_repeat = 45 # Thoi gian nhac lai canh bao tai xe lai xe lien tuc qua so gio (theo giay (s))
so_lan_khong_tap_trung = 3 # So lan canh bao nhin thang truoc khi roi khoi xe


EAR_COUNTER = 0
COUNTER_FACE = 0
COUNTER_NON_FACE = 0		
flag_canh_bao_tap_trung = 0
flag_canh_bao_3h = 0
flag_drowsy_alarm = 0
hist_size = (4,4)
hist_limit = 1.0
duration = 0

# VIDEO SAVING
w_save_size = 640
h_save_size = 480
video_save_size = (w_save_size, h_save_size)

# VIDEO STREAMING
## SIZE
w_show_size = 320
h_show_size = 240

system_name_img_width = w_show_size


w_process_size = 320
h_process_size = 240  

## LOCATION
system_name_location = [0, 0]

left_show_location = [0, 240]
right_show_location = [0, 485]
driver_show_location = [0, 760]

info_show_location = [480, 450]
info_img_width = 960

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
in_vid_path = os.path.join(os.getcwd(), "RUN", "In_videos")
# Path to the output videos
out_vid_path = os.path.join(os.getcwd(), "RUN", "Out_videos")
print("out_vid_path: ", out_vid_path)
# Path to save the driver images
driver_img_path = os.path.join(os.getcwd(), "RUN", "Driver_images")
print("driver_img_path: ", driver_img_path)

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

# DONE 3: Load model for face detection
# Face detector using HOG method
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(CWD_PATH, "face_detector", "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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
    phat_loa(info_name + ".mp3")
    # phat_loa_until_end(info_name + ".mp3")
    print("Da phat loa va hien thi thong bao: ", info_name)
    time.sleep(0.5)
    
def phat_loa_no_show_info(info_name):
    time.sleep(0.5)
    phat_loa(info_name + ".mp3")
    # phat_loa_until_end(info_name + ".mp3")
    print("Da phat loa va hien thi thong bao: ", info_name)
    time.sleep(0.5)

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
        self.t = Thread(target=self._reader)
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
##                LIST CAMERAS                   ##
###################################################
# List all cameras connected to the computer 
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            return
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

###################################################
##             INTERFACE WITH CAMERA             ##
###################################################
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
# List all USB cameras
# cameras = list_cameras()
# print("Connected cameras:", cameras)

# left_src_cam = 0
# right_src_cam = 2
driver_src_cam = 1
try:
    # print("[INFO] starting LEFT camera ...")
    # left_cam = VideoStream(src=left_src_cam).start()
    # print("LEFT camera connect Successfully!")
    
    # print("[INFO] starting RIGHT camera ...")
    # right_cam = VideoStream(src=right_src_cam).start()
    # time.sleep(1.0)
    # print("RIGHT camera connect Successfully!")
    
    print("[INFO] starting DRIVER camera ...")
    driver_cam = VideoStream(src=driver_src_cam).start()
    time.sleep(1.0)
    print("DRIVER camera connect Successfully!")
        
except:
    print("Connect not successfully!!!")
    pass
         
###################################################
##             INTERFACE WITH PX4                ##
###################################################
# TODO 8: Connect to the PX4 to get the GPS data

###################################################
##                INPUT FROM KEYBOARD            ##
###################################################
# TODO 9: Get the input from the keyboard
def get_keyboard_input_1():
    global select_para
    global EYE_AR_THRESH
    global EYE_AR_CONSEC_FRAMES
    global SET_HR
    global SET_MIN
    global SET_SEC
    global FACE_COUNTER_THRES
    global COUNTER_NON_FACE_THRES_TAP_TRUNG
    global COUNTER_NON_FACE_THRES_RESET
    global time_repeat
    global so_lan_khong_tap_trung   
    
    select_para = int(input(" Ban co muon thay doi thong so khong? -0.Khong -1.Co "))
    if select_para == 0:
        pass
    else:
        EYE_AR_THRESH = float(input("Thiet lap nguong EAR (0.22): "))
        
        EYE_AR_CONSEC_FRAMES = int(input("Thiet lap so frame lien tuc nham mat --> Buon ngu (15): "))
        
        print("Thiet lap thoi gian canh bao tai xe lai xe lien tuc (HH:MM:SS) (00:01:00): ")
        # Set_time
        SET_HR = int(input("Gio (00): "))
        SET_MIN = int(input("Phut (01): "))
        SET_SEC = int(input("Giay (00): "))

        FACE_COUNTER_THRES = int(input("So khung hinh lien tuc co khuon mat de chuong trinh bat dau hoat dong (15): "))
        
        COUNTER_NON_FACE_THRES_TAP_TRUNG = int(input("So khung hinh khong nhan thay mat de canh bao NHIN THANG (15): "))
        
        COUNTER_NON_FACE_THRES_RESET = int(input("So khung hinh nhan biet tai xe ra khoi xe (150): "))
        
        time_repeat = int(input("Thoi gian nhac lai canh bao tai xe lai xe lien tuc qua so gio (theo giay (s)) (45): "))
        
        so_lan_khong_tap_trung = int(input("So lan canh bao nhin thang truoc khi roi khoi xe (3): "))
        
        print("-------------------------------------------------")
        time.sleep(1.0)

def check_get_keyboard_input_1():
    global EYE_AR_THRESH
    global EYE_AR_CONSEC_FRAMES
    global SET_HR
    global SET_MIN
    global SET_SEC
    global FACE_COUNTER_THRES
    global COUNTER_NON_FACE_THRES_TAP_TRUNG
    global COUNTER_NON_FACE_THRES_RESET
    global time_repeat
    global so_lan_khong_tap_trung   
    
    print("Da thiet lap cac thong so!")
    print("EYE_AR_THRESH: ", EYE_AR_THRESH)
    print("EYE_AR_CONSEC_FRAMES: ", EYE_AR_CONSEC_FRAMES)
    print(f"SET_HR: {SET_HR}, SET_MIN: {SET_MIN}, SET_SEC: {SET_SEC}")
    print("FACE_COUNTER_THRES: ", FACE_COUNTER_THRES)
    print("COUNTER_NON_FACE_THRES_TAP_TRUNG: ", COUNTER_NON_FACE_THRES_TAP_TRUNG)
    print("COUNTER_NON_FACE_THRES_RESET: ", COUNTER_NON_FACE_THRES_RESET)
    print("time_repeat: ", time_repeat)
    print("so_lan_khong_tap_trung: ", so_lan_khong_tap_trung)
    print("-------------------------------------------------") 

###################################################
##                DROWSINESS UTILS               ##
###################################################
# TODO 11.1: Write the drowsiness ultilities functions

# Compute and return the euclidean distance between the 2 points
def euclidean_dist(ptA, ptB): 
	return np.linalg.norm(ptA - ptB)

# Calculate the set time in seconds
def cal_set_time(h,m,s): 
    set_time = h * 3600 + m * 60 + s
    return set_time

# Compute the euclidean distances between the two sets of
# vertical eye landmarks (x, y)-coordinates
def eye_aspect_ratio(eye):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])

	# Compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = euclidean_dist(eye[0], eye[3])

	# compute the eye aspect ratio
	EAR = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return EAR

def capture_and_convert_driver_frame():
    driver_frame = driver_cam.read()
    driver_frame = imutils.resize(driver_frame, width=320)    #450, cang nho xu ly cang nhanh
    gray = cv2.cvtColor(driver_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = hist_limit, tileGridSize = hist_size)
    gray = clahe.apply(gray)
    return driver_frame, gray

def nhan_mat_set_time(FACE_COUNTER_THRES):
    FACE_COUNTER = 0
    while FACE_COUNTER < FACE_COUNTER_THRES:        
        driver_frame, gray = capture_and_convert_driver_frame()
        # Face detection and extract the ROI
        rects = detector(gray, 0)
        key = cv2.waitKey(1) & 0xFF
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            cv2.destroyAllWindows()
            vs.stop()
        if len(rects) > 0:
            FACE_COUNTER += 1
        else:
            FACE_COUNTER=0
        driver_frame = print_face_counter_on_driver_frame(driver_frame, FACE_COUNTER)
        visualize_driver(driver_frame)
        # print(FACE_COUNTER)
    else:
        t1 = Thread(phat_loa_no_show_info("1_chao_mung"))
        t1.deamon = True
        t1.start()
        start_time = datetime.now()
        visualize_driver(driver_frame)
        print(f"[WELCOME] Driver start to drive the vehicle at {start_time.strftime('%H:%M:%S')}")
        cv2.imwrite(os.path.join(driver_img_path, f"driver_{start_time.strftime('%Y%m%d_%H%M%S')}.jpg"), driver_frame)       
    return start_time

def draw_eye_contour_red(driver_frame, leftEye, rightEye):
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(driver_frame, [leftEyeHull], -1, (0, 0, 255), 1)
    cv2.drawContours(driver_frame, [rightEyeHull], -1, (0, 0, 255), 1)
    return driver_frame

def draw_eye_contour_green(driver_frame, leftEye, rightEye):
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(driver_frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(driver_frame, [rightEyeHull], -1, (0, 255, 0), 1)
    return driver_frame

def print_face_counter_on_driver_frame(driver_frame, FACE_COUNTER):
    cv2.putText(driver_frame, "FACE COUNTER: {:.0f}".format(FACE_COUNTER), (150, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return driver_frame

def print_EAR_on_driver_frame(driver_frame, EAR, EAR_COUNTER):
    cv2.putText(driver_frame, "EAR: {:.3f}".format(EAR), (220, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(driver_frame, "LOW EAR COUNTER: {:.0f}".format(EAR_COUNTER), (130, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return driver_frame

def show_info_on_driver_frame(driver_frame, duration):
    cv2.putText(driver_frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (10, 230),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    h = math.floor(duration/3600)
    m = math.floor((duration-h*3600)/60)
    s = math.floor(duration - h*3600-m*60)
    string_time = str(h)+':'+str(m)+':'+str(s)
    date_time_obj = datetime.strptime(string_time, '%H:%M:%S')
    # print('Time:', date_time_obj.time())  
    cv2.putText(driver_frame," LAI LIEN TUC: %s" %date_time_obj.time(), (5, 200), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return driver_frame

###################################################
##                DROWSINESS UTILS               ##
###################################################
# TODO 11.2: Write the drowsiness detection function
def drowsiness_detection():
    global start_time
    global flag_canh_bao_tap_trung
    global duration
    global set_time
    global flag_canh_bao_3h
    global time_repeat  
    global COUNTER_NON_FACE
    global lStart, lEnd, rStart, rEnd
    global EAR_COUNTER
    global EYE_AR_CONSEC_FRAMES
    global flag_drowsy_alarm
    
    driver_frame, gray = capture_and_convert_driver_frame()
    # Face detection and extract the ROI
    rects = detector(gray, 0)
    if len(rects) > 0:
        flag_canh_bao_tap_trung = 0
        COUNTER_NON_FACE = 0
        duration = (datetime.now() - start_time).total_seconds()
        # print("Thoi gian canh bao lai xe lien tuc: {}(s)".format(set_time))
        
        if duration >= set_time and flag_canh_bao_3h == 0:
            t2 = Thread(phat_loa_show_info("16_3h_lien_tuc"))
            t2.deamon = True
            t2.start()
            flag_canh_bao_3h = 1

        if duration >= set_time and round(duration- set_time) % time_repeat == 0:
            flag_canh_bao_3h = 0
        
        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = predictor(gray, rect)            
			# Convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = face_utils.shape_to_np(shape)
            # Extract the left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
			# Compute the eye aspect ratio for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            # Calculate the average of eye aspect ratio together for both eyes
            EAR = (leftEAR + rightEAR) / 2.0
            
            # Check whether the eye aspect ratio (EAR) is below the threshold
            if EAR < EYE_AR_THRESH:
                # Increment the blink frame counter
                EAR_COUNTER += 1
                
                # Draw the eyes contour on the frame
                driver_frame = draw_eye_contour_red(driver_frame, leftEye, rightEye)
                
                # If the eyes were closed for a sufficient number of consecutive frames
                if EAR_COUNTER >= EYE_AR_CONSEC_FRAMES and flag_drowsy_alarm == 0:
                    print("Phat hien tai xe dang buon ngu!")
                    flag_drowsy_alarm = 1
                    t3 = Thread(phat_loa_show_info("17_buon_ngu"))
                    t3.deamon = True
                    t3.start()
            else:
                EAR_COUNTER = 0
                flag_drowsy_alarm = 0
                # Draw the eyes contour on the frame
                driver_frame = draw_eye_contour_green(driver_frame, leftEye, rightEye)
            driver_frame = print_EAR_on_driver_frame(driver_frame, EAR, EAR_COUNTER)
    else:
        COUNTER_NON_FACE += 1
        print("COUNTER_NON_FACE: ", COUNTER_NON_FACE)
        
    if COUNTER_NON_FACE > COUNTER_NON_FACE_THRES_TAP_TRUNG and flag_canh_bao_tap_trung < so_lan_khong_tap_trung:
        t4 = Thread(phat_loa_show_info("18_tap_trung"))
        t4.deamon = True
        t4.start()
        flag_canh_bao_tap_trung += 1
        COUNTER_NON_FACE = 0
        print("Tai cong khong tap trung")
        
    elif COUNTER_NON_FACE > COUNTER_NON_FACE_THRES_RESET and flag_canh_bao_tap_trung >= so_lan_khong_tap_trung:
        COUNTER_NON_FACE = 0
        flag_canh_bao_tap_trung = 0
        flag_canh_bao_3h = 0
        print("Tai cong da roi khoi buong lai")
        time.sleep(2.0)
        start_time = nhan_mat_set_time(FACE_COUNTER_THRES)
    
    driver_frame = show_info_on_driver_frame(driver_frame, duration)
    visualize_driver(driver_frame)
    
###################################################
##                  SEND TO ARDUINO              ##
###################################################
# TODO 14: Send the data to the Arduino   

###################################################
##                  MAIN FUNCTION                ##
###################################################
# TODO 10: Write the main function to run the system
# if args["alarm"] > 0:
                                
#     #Send data
#     send3h = "3h        " + str(datetime.now().strftime('%H:%M %d/%m/%Y'))
#     ser.write(send3h.encode())
#     time.sleep(0.1)

if __name__ == "__main__":    
    visualize_system_name()
    begin_time = time.time()
    frame_rate = 30
    fps = 0
    
    # DROWSINESS DETECTION
    set_time = cal_set_time(SET_HR, SET_MIN, SET_SEC)
    print("Set time: ", set_time)
    start_time = nhan_mat_set_time(FACE_COUNTER_THRES)
    
    # get_keyboard_input_1()
    check_get_keyboard_input_1()
    
    ###################################################
    ##       AUDIO AND VISUALIZATION CHECK           ##
    ###################################################
    # DONE 1: Check the audio and visualization
    # phat_loa_no_show_info("1_chao_mung")
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
    # phat_loa_show_info("16_3h_lien_tuc")
    # phat_loa_show_info("17_buon_ngu")
    # phat_loa_show_info("18_tap_trung")
    
    
    while True:
        '''
        # 1. Drowsiness Detection
        '''
        drowsiness_detection()
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    print("[INFO] cleaning up...")
    # left_cam.cap.release()
    # right_cam.cap.release()
    driver_cam.cap.release()
    cv2.destroyAllWindows()
    
    
    