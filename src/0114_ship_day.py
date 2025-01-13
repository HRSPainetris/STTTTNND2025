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

# YOLO object detection
from ultralytics import YOLO
from tqdm import tqdm

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

# Arduino serial communication library
import serial

# PX4 connection library
from dronekit import connect, VehicleMode #, LocationGlobalRelative, APIException
import socket


###################################################
##       TODO 1: PARAMETER CONFIGURATION         ##
###################################################
# DROWSINESS DETECTION
select_para = 0
EYE_AR_THRESH = 0.15 # Nguong EAR
EYE_AR_CONSEC_FRAMES = 30 # So frame lien tuc de xac dinh buon ngu
# Set_time - Dat truoc thoi gian canh bao tai cong lai lien tuc
SET_HR = 0      # GIO
SET_MIN = 1     # PHUT
SET_SEC = 0     # GIAY

FACE_COUNTER_THRES = 30 # So fame lien tuc co khuon mat de he thong bat dau hoat dong
COUNTER_NON_FACE_THRES_TAP_TRUNG = 50 # So frame lien tuc khong co khuon mat de canh bao TAP TRUNG
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
w_save_size = 320
h_save_size = 240
video_save_size = (w_save_size, h_save_size)

# VIDEO STREAMING
## SIZE
w_show_size = 320
h_show_size = 240

system_name_img_width = w_show_size

w_process_size = 320
h_process_size = 240  

## LOCATION
system_name_location = [1, 1]

left_show_location = [0, 210]
right_show_location = [0, 405]
driver_show_location = [0, 600]

info_show_location = [480, 410]
info_img_width = 960

# OBJECT DETECTION
SHIP_DISTANCE_CAL_THRES = 15 # meters
prev_det_data = []
pseudo_process_time = 1
prev_speed = 1

# SHIP: cls_id = 0
SHIP_DISTANCE_WARN_THRES = 80 # meters
SHIP_SPEED_WARN_THRES = 20 # km/h
SHIP_COUNTER = 0
SHIP_CONSEC_THRES = 5
flag_ship_warning = 0
flag_ship_in_left = 0
flag_ship_in_right = 0

# BRIDGE: cls_id = 1
BRIDGE_DISTANCE_WARN_THRES = 200 # meters
BRIDGE_HEIGHT_WARN_THRES = 5 # meters
BRIDGE_COUNTER = 0
BRIDGE_CONSEC_THRES = 5
flag_bridge_warning = 0
flag_bridge_in_left = 0
flag_bridge_in_right = 0

# SIGN: cls_id = 2
SIGN_COUNTER = 0
SIGN_CONSEC_THRES = 5
flag_sign_warning = 0
flag_sign_in_left = 0
flag_sign_in_right = 0

# PERSON: cls_id = 3
PERSON_COUNTER = 0
PERSON_CONSEC_THRES = 5
flag_person_warning = 0
flag_person_in_left = 0
flag_person_in_right = 0

# STORING DATA FROM ARDUINO
light_status = 0 # 0: day, 1: night
estop_status = 0 # 0: disable, 1: enable
enable_status = 0 # 0: disable, 1: enable
new_record_status = 0 # 0: no, 1: yes        

###################################################
##               TODO 2: INFO PATH               ##
###################################################

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to the audio files
audio_path = os.path.join(CWD_PATH,"audio_files")

# Path to the system images
sys_img_path = os.path.join(CWD_PATH, "system_images")

###################################################
##          TODO 3: VIDEO and DATA IN-OUT        ##
###################################################
parent_directory = os.path.dirname(CWD_PATH)
os.chdir(parent_directory)
# Path to the input videos
in_vid_path = os.path.join(os.getcwd(), "RUN", "In_videos")
print("in_vid_path: ", in_vid_path)

# Path to the output videos
out_vid_path = os.path.join(os.getcwd(), "RUN", "Out_videos")
print("out_vid_path: ", out_vid_path)

# Path to save the driver images
driver_img_path = os.path.join(os.getcwd(), "RUN", "Driver_images")
print("driver_img_path: ", driver_img_path)

# Path to save the output CSV files
out_csv_path = os.path.join(os.getcwd(), "RUN", "Out_data")
print("out_csv_path: ", out_csv_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# XVID is more preferable. MJPG results in high size video. X264 gives very small size video    

def create_vid_writer(): # Create new folder to save videos in nhan_mat_set_time()
    global left_out_vid, right_out_vid, driver_out_vid
    global left_org_vid, right_org_vid
    global left_org_vid_path, right_org_vid_path
    global left_out_vid_path, right_out_vid_path, driver_out_vid_path
    # Create folder for saving videos
    vid_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S") # video folder name format: YYYYMMDD_HHMMSS
    vid_folder_path = os.path.join(out_vid_path, str(vid_folder_name))
    os.mkdir(vid_folder_path)
    print("Folder to save video path: ", vid_folder_path)

    # Make the path for saving the video
    left_org_vid_path = os.path.join(vid_folder_path, 'left_org_video.avi') # left camera video original
    left_out_vid_path = os.path.join(vid_folder_path, 'left_video.avi') # left camera video
    
    right_org_vid_path = os.path.join(vid_folder_path, 'right_org_video.avi') # right camera video original
    right_out_vid_path = os.path.join(vid_folder_path, 'right_video.avi') # right camera video
    
    driver_out_vid_path = os.path.join(vid_folder_path, 'driver_video.avi') # driver camera video

    # Python: cv2.VideoWriter([filename, fourcc, fps, frameSize[, isColor]])
    left_org_vid = cv2.VideoWriter(left_org_vid_path, fourcc, 30.0, video_save_size)
    left_out_vid = cv2.VideoWriter(left_out_vid_path, fourcc, 30.0, video_save_size)
    
    right_org_vid = cv2.VideoWriter(right_org_vid_path, fourcc, 30.0, video_save_size)
    right_out_vid = cv2.VideoWriter(right_out_vid_path, fourcc, 30.0, video_save_size)
    
    driver_out_vid = cv2.VideoWriter(driver_out_vid_path, fourcc, 30.0, video_save_size)

###################################################
##             TODO 4: DATA SAVING               ##
###################################################
# Save the data to the CSV file

###################################################
##           TODO 5: LOAD MODELS                 ##
###################################################
## Load the model for object detection
print("[INFO] loading DAYTIME object detector...")
daytime_detector_path = os.path.join(CWD_PATH, "object_detector", "daytime_best.pt")
daytime_detector = YOLO(daytime_detector_path)

print("[INFO] loading NIGHTTIME object detector...")
nighttime_detector_path = os.path.join(CWD_PATH, "object_detector", "nighttime_best.pt")
nighttime_detector = YOLO(nighttime_detector_path)

print("[INFO] loaded object detector successfully!")

## Load model for face detection
# Face detector using HOG method
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(CWD_PATH, "face_detector", "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

###################################################
##               TODO 6: AUDIO                   ##
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

def phat_loa_skip(audio_file_name):
    audio_file_name = os.path.join(audio_path, audio_file_name)
    if not mixer.music.get_busy():
        mixer.music.load(audio_file_name)
        mixer.music.play()
    else:
        pass
        
def phat_loa_no_wait(audio_file_name):
    audio_file_name = os.path.join(audio_path, audio_file_name)
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
##                TODO 7: VISUALIZATION          ##
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
    # print("Showing System Notice: ", info_img_name)
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
##            TODO 8: VISUALIZE & AUDIO          ##
###################################################
def phat_loa_show_info(info_name):
    visualize_info(info_name + ".png")
    # time.sleep(0.5)
    # phat_loa(info_name + ".mp3")
    # phat_loa_no_wait(info_name + ".mp3")
    phat_loa_skip(info_name + ".mp3")
    # phat_loa_until_end(info_name + ".mp3")
    print("Da phat loa va hien thi thong bao: ", info_name)
    # time.sleep(0.5)

def phat_loa_show_info_ob_det(info_name):
    visualize_info(info_name + ".png")
    # time.sleep(0.5)
    phat_loa_skip(info_name + ".mp3")
    # phat_loa_no_wait(info_name + ".mp3")
    # phat_loa_until_end(info_name + ".mp3")
    print("Da phat loa va hien thi thong bao: ", info_name)
    # time.sleep(0.5)
    
def phat_loa_no_show_info(info_name):
    # time.sleep(0.5)
    # phat_loa(info_name + ".mp3")
    phat_loa_no_wait(info_name + ".mp3")
    # phat_loa_until_end(info_name + ".mp3")
    print("Da phat loa va hien thi thong bao: ", info_name)
    # time.sleep(0.5)

###################################################
##       TODO 9: BUFFER-LESS VideoCapture        ##
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

###################################################
##           TODO 10: LIST CAMERAS               ##
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
##         TODO 11: INTERFACE WITH PX4           ##
###################################################
# Connect to the PX4 to get the GPS data
def connectMyCopter():
    PX4_GPS = connect('/dev/ttyTHS1', baud=57600, wait_ready=False)
    # PX4_GPS = connect('COM11', baud=115200, wait_ready=False)
    return PX4_GPS

# PX4_GPS = connectMyCopter()
# PX4_GPS.wait_ready('autopilot_version')
# print('Autopilot version: %s'%PX4_GPS.version)

###################################################
##         TODO 12: INPUT FROM KEYBOARD          ##
###################################################
# Get the input from the keyboard
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
##            TODO 13: DROWSINESS ULTIS          ##
###################################################
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
    if cv2.getWindowProperty("System Notice", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("System Notice")
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
        create_vid_writer()
        cv2.imwrite(os.path.join(driver_img_path, f"{start_time.strftime('%Y%m%d_%H%M%S')}.jpg"), driver_frame)       
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
##      TODO 14*: DROWSINESS DETECTION            ##
###################################################
# Write the drowsiness detection function
def drowsiness_detection():
    while not stop_event.is_set():
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
        global driver_out_vid
        
        driver_frame, gray = capture_and_convert_driver_frame()
        # Face detection and extract the ROI
        rects = detector(gray, 0)
        if len(rects) > 0:
            COUNTER_NON_FACE = 0
            flag_canh_bao_tap_trung = 0
            duration = (datetime.now() - start_time).total_seconds()
            # print("Thoi gian canh bao lai xe lien tuc: {}(s)".format(set_time))
            
            if duration >= set_time and flag_canh_bao_3h == 0:
                t2 = Thread(phat_loa_show_info("16_3h_lien_tuc"))
                t2.deamon = True
                t2.start()
                flag_canh_bao_3h = 1

            if duration >= set_time and round(duration - set_time) % time_repeat == 0:
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
            # print("COUNTER_NON_FACE: ", COUNTER_NON_FACE)
            
        if COUNTER_NON_FACE > COUNTER_NON_FACE_THRES_TAP_TRUNG and flag_canh_bao_tap_trung < so_lan_khong_tap_trung:
            t4 = Thread(phat_loa_show_info("18_tap_trung"))
            t4.deamon = True
            t4.start()
            flag_canh_bao_tap_trung += 1
            COUNTER_NON_FACE = 0
            print("Tai cong khong tap trung")
            
        if COUNTER_NON_FACE > COUNTER_NON_FACE_THRES_RESET and flag_canh_bao_tap_trung >= so_lan_khong_tap_trung:
            COUNTER_NON_FACE = 0
            flag_canh_bao_tap_trung = 0
            flag_canh_bao_3h = 0
            print("Tai cong da roi khoi buong lai")
            time.sleep(2.0)
            start_time = nhan_mat_set_time(FACE_COUNTER_THRES)
        
        driver_frame = show_info_on_driver_frame(driver_frame, duration)
        visualize_driver(driver_frame)
        driver_out_vid.write(driver_frame)
        if not mixer.music.get_busy() and cv2.getWindowProperty("System Notice", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("System Notice")
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
               
###################################################
##         TODO 15: OBJECTS DETECTION ULTIS      ##
###################################################
def cal_iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

###################################################
##           TODO 16*: OBJECTS DETECTION          ##
###################################################
cls_names = ['Ship', 'Bridge', 'Sign', 'Person']
colors = [(0,255,255), (255,0,255), (255,255,0), (0,255,0)] # BGR: Yellow, Magenta, Cyan, Green

def object_detection():
    while not stop_event.is_set():
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
        
def cal_distance(cls_id, y1, y2, obj_det_frame_height):
    # Calculate the distance of the object
    dist = 0
    obj_h_ratio = (y2 - y1)/obj_det_frame_height
    if cls_id == 0: # Ship
        dist = -350 * obj_h_ratio + 75
    elif cls_id == 1: # Bridge
        dist = -350 * obj_h_ratio + 150
    return dist

def cal_speed(i, prev_det_data, curr_det_data, x1, y1, x2, y2, cls_id, 
              dist, prev_speed, pseudo_process_time):   
    # Calculate the speed of the object
    if i == 0:
        prev_det_data.append([x1, y1, x2, y2, cls_id, dist])
        dist_delta = 0
    else:
        curr_det_data.append([x1, y1, x2, y2, cls_id, dist])
        
        if len(prev_det_data) > 0:
            iou_list = []
            for pbox in prev_det_data:
                px1, py1, px2, py2 = np.array(pbox)[0:4]
                iou_ = cal_iou([x1, y1, x2, y2], [px1, py1, px2, py2])
                iou_list.append(iou_)
            if max(iou_list) > 0.5:
                dist_delta = dist - np.array(prev_det_data)[np.argmax(iou_list)][5]
            else:
                dist_delta = 0
        else:
            dist_delta = 0
            
    speed = int((dist_delta / pseudo_process_time) * 3.6)
    smooth_speed = int( 84 + ( 0.084 - 84 ) / ( 1 + ( abs(speed) / 43.92 )**2.4 ))
    
    # Ensure smooth_speed is a real number
    if isinstance(smooth_speed, complex):
        smooth_speed = smooth_speed.real
        
    if smooth_speed <= 1:
        smooth_speed = prev_speed
    return smooth_speed                  

def cal_steering_angle(dist,speed):
    if speed == 0:
        return 0
    else:
        angle = 0.2 * dist + 0.8 * speed
        return round(angle)

send_time = 0
send_gps_time = 0

def object_warning(cls_id, x1, y1, x2, y2, curr_det_data, prev_det_data, i, prev_speed, 
                   pseudo_process_time, obj_det_frame_height, stacked_frame):
    global send_time
    global send_gps_time
    ## SHIP: cls_id = 0
    global SHIP_COUNTER
    global flag_ship_warning
    global flag_ship_in_left
    global flag_ship_in_right
    
    ## BRIDGE: cls_id = 1
    global BRIDGE_COUNTER
    global flag_bridge_warning
    global flag_bridge_in_left
    global flag_bridge_in_right
    
    ## SIGN: cls_id = 2
    global SIGN_COUNTER
    global flag_sign_warning
    global flag_sign_in_left
    global flag_sign_in_right
    
    ## PERSON: cls_id = 3
    global PERSON_COUNTER
    global flag_person_warning
    global flag_person_in_left
    global flag_person_in_right
    
    ## Arduino module
    global arduino_module
    
    ################################
    ##       SHIP: cls_id = 0     ##
    ################################
    ship_speed = 0
    if cls_id == 0:
        # Calculate the distance of the ship
        dist = cal_distance(cls_id, y1, y2, obj_det_frame_height)  
        dist = round(dist, 2)    
        # Calculate the speed of the ship
        ship_speed = cal_speed(i, prev_det_data, curr_det_data, x1, y1, x2, y2, cls_id, dist, prev_speed, pseudo_process_time)
        ship_speed = round(ship_speed, 2)
               
        # print(f'Ship: {dist}m | {ship_speed}km/h')
        if dist < SHIP_DISTANCE_WARN_THRES and ship_speed > SHIP_SPEED_WARN_THRES:
            SHIP_COUNTER += 1
            # Mark the ship with a red bounding box
            stacked_frame = cv2.rectangle(stacked_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            if dist < SHIP_DISTANCE_CAL_THRES:
                stacked_frame = cv2.putText(stacked_frame, f'D<{SHIP_DISTANCE_CAL_THRES}m | {ship_speed}km/h', (x1, y1 - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Red color
            else:
                stacked_frame = cv2.putText(stacked_frame, f'D<{dist}m | {ship_speed}km/h', (x1, y1 - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Red color
            
            if SHIP_COUNTER >= SHIP_CONSEC_THRES and flag_ship_warning == 0:
                # DETERMINE LEFT OR RIGHT OR BOTH
                if y1 < obj_det_frame_height and y2 < obj_det_frame_height:
                    flag_ship_in_left = 1
                
                elif y1 >= obj_det_frame_height and y2 >= obj_det_frame_height:
                    flag_ship_in_right = 1
                
                if flag_ship_in_left == 1 and flag_ship_in_right == 0:
                    print("Phat hien tau BEN TRAI!")
                    flag_ship_warning = 1
                    if estop_status == 0:
                        if (time.time() - send_time) > 5:                            
                            send_right_to_arduino(arduino_module, cal_steering_angle(dist, ship_speed))
                            send_time = time.time()
                        if (time.time() - send_gps_time) > 8:
                            send_gps_data_to_arduino()
                            send_gps_time = time.time()
                    t5 = Thread(phat_loa_show_info_ob_det("5_cham_trai"))
                    t5.deamon = True
                    t5.start()
                    
                elif flag_ship_in_left == 0 and flag_ship_in_right == 1:
                    print("Phat hien tau BEN PHAI!")
                    flag_ship_warning = 1
                    if estop_status == 0:
                        if (time.time() - send_time) > 5:   
                            send_left_to_arduino(arduino_module, cal_steering_angle(dist, ship_speed))
                            send_time = time.time()
                        if (time.time() - send_gps_time) > 8:
                            send_gps_data_to_arduino()
                            send_gps_time = time.time()
                    t5 = Thread(phat_loa_show_info_ob_det("4_cham_phai"))
                    t5.deamon = True
                    t5.start()
                    
                elif flag_ship_in_left == 1 and flag_ship_in_right == 1:
                    print("Phat hien tau HAI BEN!")
                    flag_ship_warning = 1
                    if estop_status == 0:
                        if (time.time() - send_time) > 5: 
                            send_straight_to_arduino(arduino_module)
                            send_time = time.time()
                        if (time.time() - send_gps_time) > 8:
                            send_gps_data_to_arduino()
                            send_gps_time = time.time()
                    t5 = Thread(phat_loa_show_info_ob_det("3_cham_2ben"))
                    t5.deamon = True
                    t5.start()  
                             
        else:
            SHIP_COUNTER = 0
            flag_ship_warning = 0
            flag_ship_in_left = 0
            flag_ship_in_right = 0
            # Mark the ship with a cls_id color bounding box
            stacked_frame = cv2.rectangle(stacked_frame, (x1, y1), (x2, y2), colors[cls_id], 2)
            stacked_frame = cv2.putText(stacked_frame, f'D={int(dist)}m | {ship_speed}km/h', (x1, y1 - 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2, cv2.LINE_AA)
            
        stacked_frame = cv2.putText(stacked_frame, f'{cls_names[cls_id]}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2, cv2.LINE_AA)    
           
    ################################
    ##       BRIDGE: cls_id = 1   ##
    ################################
    if cls_id == 1:
        # Calculate the distance of the object
        dist = cal_distance(cls_id, y1, y2, obj_det_frame_height)
        h_bridge = dist * ( y2 - y1 ) * 0.006
        dist = round(dist, 2)
        h_bridge = round(h_bridge, 2)
        # print(f'Bridge: D={dist}m | H={h_bridge}m')
        if dist < BRIDGE_DISTANCE_WARN_THRES:
            BRIDGE_COUNTER += 1
            # Mark the ship with a red bounding box
            stacked_frame = cv2.rectangle(stacked_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            if BRIDGE_COUNTER >= BRIDGE_CONSEC_THRES and flag_bridge_warning == 0:
                # DETERMINE LEFT OR RIGHT OR BOTH
                if y1 < obj_det_frame_height and y2 < obj_det_frame_height:
                    flag_bridge_in_left = 1
                
                elif y1 >= obj_det_frame_height and y2 >= obj_det_frame_height:
                    flag_bridge_in_right = 1
                
                if flag_bridge_in_left == 1 and flag_bridge_in_right == 0:
                    # print("Phat hien cau BEN TRAI!")
                    flag_bridge_warning = 1
                    t6 = Thread(phat_loa_show_info_ob_det("13_cau_trai"))
                    t6.deamon = True
                    t6.start()
                
                elif flag_bridge_in_left == 0 and flag_bridge_in_right == 1:
                    # print("Phat hien cau BEN PHAI!")
                    flag_bridge_warning = 1
                    t6 = Thread(phat_loa_show_info_ob_det("14_cau_phai"))
                    t6.deamon = True
                    t6.start()
                    
                elif flag_bridge_in_left == 1 and flag_bridge_in_right == 1:
                    # print("Phat hien cau PHIA TRUOC!")
                    flag_bridge_warning = 1
                    t6 = Thread(phat_loa_show_info_ob_det("12_cau_truoc"))
                    t6.deamon = True
                    t6.start()
                    
        else:
            BRIDGE_COUNTER = 0
            flag_bridge_warning = 0
            # flag_bridge_in_left = 0
            # flag_bridge_in_right = 0
            # Mark the ship with a cls_id color bounding box
            stacked_frame = cv2.rectangle(stacked_frame, (x1, y1), (x2, y2), colors[cls_id], 2)

        if h_bridge <= BRIDGE_HEIGHT_WARN_THRES and flag_bridge_warning == 0:
            # print("Chu y chieu cao cua tau!")
            flag_bridge_warning = 1
            t6 = Thread(phat_loa_show_info_ob_det("15_chieu_cao_tau"))
            t6.deamon = True
            t6.start()            
        
        stacked_frame = cv2.putText(stacked_frame, f'{cls_names[cls_id]}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2, cv2.LINE_AA)
        stacked_frame = cv2.putText(stacked_frame, f'D={int(dist)}m | H={h_bridge}m', (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2, cv2.LINE_AA)       

    ################################
    ##       SIGN: cls_id = 2     ##
    ################################
    if cls_id == 2:
        # Calculate the distance of the object
        SIGN_COUNTER += 1
        stacked_frame = cv2.rectangle(stacked_frame, (x1, y1), (x2, y2), colors[cls_id], 2)
        stacked_frame = cv2.putText(stacked_frame, f'{cls_names[cls_id]}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2, cv2.LINE_AA)
        
        if SIGN_COUNTER >= SIGN_CONSEC_THRES and flag_sign_warning == 0:
            # DETERMINE LEFT OR RIGHT OR BOTH
            if y1 < obj_det_frame_height and y2 < obj_det_frame_height:
                flag_sign_in_left = 1
                
            elif y1 >= obj_det_frame_height and y2 >= obj_det_frame_height:
                flag_sign_in_right = 1
            
            if flag_sign_in_left == 1 and flag_sign_in_right == 0:
                # print("Phat hien bien bao BEN TRAI!")
                flag_sign_warning = 1
                t7 = Thread(phat_loa_show_info_ob_det("11_bien_bao_trai"))
                t7.deamon = True
                t7.start()
                
            elif flag_sign_in_left == 0 and flag_sign_in_right == 1:
                # print("Phat hien bien bao BEN PHAI!")
                flag_sign_warning = 1
                t7 = Thread(phat_loa_show_info_ob_det("10_bien_bao_phai"))
                t7.deamon = True
                t7.start()
                
            elif flag_sign_in_left == 1 and flag_sign_in_right == 1:
                # print("Phat hien bien bao HAI BEN!")
                flag_sign_warning = 1
                t7 = Thread(phat_loa_show_info_ob_det("9_bien_bao_2ben"))
                t7.deamon = True
                t7.start()
        
    else:
        SIGN_COUNTER = 0
        flag_sign_warning = 0
        flag_sign_in_left = 0
        flag_sign_in_right = 0            
    
    ################################
    ##       PERSON: cls_id = 3   ##
    ################################
    if cls_id == 3:
        # Calculate the distance of the object
        dist = cal_distance(cls_id, y1, y2, obj_det_frame_height)
        dist = round(dist, 2)
        # print(f'Person: {dist}m')
        PERSON_COUNTER += 1
        # Mark the ship with a red bounding box
        stacked_frame = cv2.rectangle(stacked_frame, (x1, y1), (x2, y2), colors[cls_id], 2)
        stacked_frame = cv2.putText(stacked_frame, f'{cls_names[cls_id]}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2, cv2.LINE_AA)
        stacked_frame = cv2.putText(stacked_frame, f'D={int(dist)}m' , (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls_id], 2, cv2.LINE_AA)
           
        if PERSON_COUNTER >= PERSON_CONSEC_THRES and flag_person_warning == 0:
            # DETERMINE LEFT OR RIGHT OR BOTH
            if y1 < obj_det_frame_height and y2 < obj_det_frame_height:
                flag_person_in_left = 1
            
            elif y1 >= obj_det_frame_height and y2 >= obj_det_frame_height:
                flag_person_in_right = 1
            
            if flag_person_in_left == 1 and flag_person_in_right == 0:
                # print("Phat hien nguoi BEN TRAI!")
                flag_person_warning = 1
                t8 = Thread(phat_loa_show_info_ob_det("8_nguoi_trai"))
                t8.deamon = True
                t8.start()
                
            elif flag_person_in_left == 0 and flag_person_in_right == 1:
                # print("Phat hien nguoi BEN PHAI!")
                flag_person_warning = 1
                t8 = Thread(phat_loa_show_info_ob_det("7_nguoi_phai"))
                t8.deamon = True
                t8.start()
                
            elif flag_person_in_left == 1 and flag_person_in_right == 1:
                # print("Phat hien nguoi HAI BEN!")
                flag_person_warning = 1
                t8 = Thread(phat_loa_show_info_ob_det("6_nguoi_2ben"))
                t8.deamon = True
                t8.start()
        else:
            PERSON_COUNTER = 0
            flag_person_warning = 0
            flag_person_in_left = 0
            flag_person_in_right = 0
    if not mixer.music.get_busy() and cv2.getWindowProperty("System Notice", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("System Notice")       
                                    
    return ship_speed, stacked_frame

###############################################################
## TODO 17*: 2 IP CAMERA VIDEOS (L+R) + 1 USB CAMERA (DRIVER) ##
###############################################################
# Load LEFT and RIGHT videos
left_in_vid_name = "ship_day_20240825_103208.mp4" #ship
# left_in_vid_name = "20240825_102907.mp4" # ship, bridge
left_cam = cv2.VideoCapture(os.path.join(in_vid_path, "left_cam", left_in_vid_name))
left_cam_nframes = int(left_cam.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] Number of frames in the LEFT video: ", left_cam_nframes)

right_in_vid_name = "ship_day_20240825_101425.mp4"
right_cam = cv2.VideoCapture(os.path.join(in_vid_path, "right_cam", right_in_vid_name))
right_cam_nframes = int(right_cam.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] Number of frames in the RIGHT video: ", right_cam_nframes)
# Create a buffer to store the frames
left_cam_buffer = []
right_cam_buffer = []
obj_det_frame_height = 320
create_vid_writer()

# Load the frames into the buffer for loop processing
for i in tqdm(range(min(left_cam_nframes, right_cam_nframes))):
    left_cam_buffer.append(cv2.resize(left_cam.read()[1],(640,320)))
    right_cam_buffer.append(cv2.resize(right_cam.read()[1],(640,320)))

def detect_2_ip_videos():
    global prev_det_data
    global pseudo_process_time
    global prev_speed
    global left_org_vid, right_org_vid
    global left_out_vid, right_out_vid
    for i in range(min(left_cam_nframes, right_cam_nframes)):            
        ob_det_start_time = time.time()
        
        left_frame = left_cam_buffer[i]
        left_org_vid.write(cv2.resize(left_frame, (320, 240)))
                   
        right_frame = right_cam_buffer[i]
        right_org_vid.write(cv2.resize(right_frame, (320, 240)))
        
        stacked_frame = np.vstack((left_frame, right_frame))
        # print("Stacked frame dimensions: ", stacked_frame.shape)
        # cv2.imshow("LEFT and RIGHT Camera", stacked_frame)
        
        obj_det_results = daytime_detector.predict(stacked_frame, conf=0.05, save=False, verbose=False)
        
        curr_det_data = []
        
        # Process results list
        boxes = obj_det_results[0].boxes  # Boxes object for bounding box outputs
        
        for box in boxes:
            # Object Detection Info
            cls_id = int(box.cls.cpu().detach().numpy()[0])
            # print(f'Object: {cls_names[cls_id]}')
            x1, y1, x2, y2 = box.xyxy[0].cpu().detach().numpy().astype('int')            
            # (left, top, right, bottom)
                   
            ship_speed, stacked_frame = object_warning(cls_id, x1, y1, x2, y2, curr_det_data, prev_det_data, i, prev_speed, 
                           pseudo_process_time, obj_det_frame_height, stacked_frame)
                     
            prev_det_data = curr_det_data.copy()
            prev_speed = ship_speed

            proctime = time.time() - ob_det_start_time
            ob_det_fps = round(1 / proctime + 7,2)

            pseudo_process_time = max(proctime, 0.033)

            # print(f'Time: {proctime} | FPS: {ob_det_fps}')
            
            stacked_frame = cv2.putText(stacked_frame, f"FPS: {ob_det_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            left_frame_new = stacked_frame[:obj_det_frame_height]
            right_frame_new = stacked_frame[obj_det_frame_height:]
            visualize_left(left_frame_new)
            visualize_right(right_frame_new)          
            left_out_vid.write(cv2.resize(left_frame_new, (320, 240)))
            right_out_vid.write(cv2.resize(right_frame_new, (320, 240)))               
        
        # Calculate the delay to maintain 30 FPS
        loop_time = time.time() - ob_det_start_time
        delay = max(1, int((1 / 30 - loop_time) * 1000))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            stop_event.set()
            break
            
#########################################################
## TODO 18: 2 IP CAMERAS (L+R) + 1 USB CAMERA (DRIVER) ##
#########################################################
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
##        TODO 20: INTERACT WITH ARDUINO         ##
###################################################
## Ket noi voi arduino 
## /dev/ttyACM1
# try:                       
## Jetson Nano
arduino_module = serial.Serial(port = '/dev/ttyACM1', baudrate = 9600, timeout = 0.5)                           
arduino_module.flush()
print("Arduino connected successfully!")                                            
# except:                                                                               
#     print("Please check the Arduino port again") 
    
# try:
## Windows                                                                                  
# arduino_module = serial.Serial(port = 'COM6', baudrate = 9600, timeout = 0.5)                                                  
# arduino_module.flush()
# print("Arduino connected successfully!")                                            
# except:                                                                               
#     print("Please check the Arduino port again") 

###################################################    
##             Send message to Arduino           ##
###################################################

def send_message_to_arduino(arduino_module, message):
    # straight
    # left_50
    # right_50
    print("Send to Arduino: ", message)
    arduino_module.write(message.encode())
    time.sleep(0.1)

def send_left_to_arduino(arduino_module, angle):
    message = "left_" + str(angle)
    send_message_to_arduino(arduino_module, message)

def send_right_to_arduino(arduino_module,angle):
    message = "right_" + str(angle)
    send_message_to_arduino(arduino_module, message)
    
def send_straight_to_arduino(arduino_module):
    send_message_to_arduino(arduino_module, "straight")
    
###################################################    
##           Read message from Arduino           ##
###################################################
def read_message_from_arduino(arduino_module):
    # day
    # night
    # estop
    # enable
    # disable
    # new_record
    message = arduino_module.readline() # Doc tin hieu ve
    message = message.decode("utf-8").rstrip('\r\n') 
    print("Read from Arduino: ", message)
    return message

###################################################    
##              Update from Arduino              ##
###################################################
def update_from_arduino():
    global light_status
    global estop_status
    global enable_status
    global new_record_status
    # global done_status
    # while not stop_event.is_set():
    if arduino_module.in_waiting > 0:
        # Read the message from the Arduino
        response = read_message_from_arduino(arduino_module)
        print("Response from Arduino: ", response)
        if response == 'day':
            light_status = 0
        elif response == 'night':
            light_status = 1
        elif response == 'estop':
            estop_status = 1
        elif response == 'enable':
            enable_status = 1
        elif response == 'disable':
            enable_status = 0
        elif response == 'new_record':
            new_record_status = 1
        else:
            pass
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     stop_event.set()
        #     break 
        
###################################################    
##              Update from Arduino              ##
###################################################
def send_gps_data_to_arduino():
    # lat = PX4_GPS.location.global_frame.lat
    # lon = PX4_GPS.location.global_frame.lon
    lat = 10.23
    lon = 106.33
    gps_data = f"{lat:.7f}, {lon:.7f}"
    # gps_data = "gps_" + f"{lat:.6f},{lon:.6f}"
    send_message_to_arduino(arduino_module, gps_data)  

###################################################
##          TODO 21: INTERFACE WITH CAMERA       ##
###################################################
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
# driver_src_cam = 0
# # try:   
# print("[INFO] starting DRIVER camera ...")
# driver_cam = VideoStream(src=driver_src_cam).start()
# time.sleep(1.0)
# print("DRIVER camera connect Successfully!")
    
# except:
#     print("Connect not successfully!!!")
#     pass

###################################################
##         TODO 22*: MAIN FUNCTION               ##
###################################################
'''
# Write the main function to run the system
def test_thread():
    i = 0
    while not stop_event.is_set():
        i += 1
        print("T2: ", i)        
        time.sleep(0.02)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

if __name__ == "__main__":    
    visualize_system_name()
    begin_time = time.time()
    frame_rate = 30
    fps = 0
    stop_event = threading.Event()
    
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
    
    # 1. Drowsiness Detection

    T1 = Thread(target=drowsiness_detection, daemon=True).start()
    T2 = Thread(target=test_thread, daemon=True).start()
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set() # Set the stop event
            break
        
    # When everything is done, release the capture
    print("[INFO] cleaning up...")
    T1.join()
    T2.join()
    # left_cam.cap.release()
    # left_out_vid.release()
    # right_cam.cap.release()
    # right_out_vid.release()
    driver_cam.cap.release()
    driver_out_vid.release()
    cv2.destroyAllWindows()
    '''
    
if __name__ == "__main__":    
    visualize_system_name()
    send_straight_to_arduino(arduino_module)
    stop_event = threading.Event()
    
    # # DROWSINESS DETECTION
    # set_time = cal_set_time(SET_HR, SET_MIN, SET_SEC)
    # print("Set time: ", set_time)
    # start_time = nhan_mat_set_time(FACE_COUNTER_THRES)
    start_time = time.time()
    # check_get_keyboard_input_1()
    # T1 = Thread(target=drowsiness_detection, daemon=True).start()
    # T3 = Thread(target = detect_2_ip_videos, daemon=True).start()
    # T4 = Thread(target = send_gps_data_to_arduino, daemon=True).start()
    while True:
        detect_2_ip_videos()
        # update_from_arduino()
        # send_gps_data_to_arduino()
        # time.sleep(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set() # Set the stop event
            break
    # When everything is done, release the capture
    print("[INFO] cleaning up...")
    # T1.join()
    # T3.join()
    # T4.join()
    # left_cam.cap.release()
    left_org_vid.release()
    left_out_vid.release()
    # right_cam.cap.release()
    right_org_vid.release()
    right_out_vid.release()
    # driver_cam.cap.release()
    driver_out_vid.release()
    cv2.destroyAllWindows()
    arduino_module.close()

    
    