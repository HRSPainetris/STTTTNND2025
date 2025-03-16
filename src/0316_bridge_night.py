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

# Get the screen width and height
from screeninfo import get_monitors

# Arduino serial communication library
import serial

# PX4 connection library
from dronekit import connect, VehicleMode #, LocationGlobalRelative, APIException
import socket

# App library
from PyQt5 import QtCore, QtGui, QtWidgets

# Excel saving library
import sys
import pandas as pd

###################################################
##       TODO : PARAMETER CONFIGURATION         ##
###################################################
# VIDEO SAVING
w_save_size = 320
h_save_size = 240
video_save_size = (w_save_size, h_save_size)

# VIDEO STREAMING
## FUNCTION: Get the screen size
def get_screen_size(screen_number = 0):
    monitors = get_monitors()
    if screen_number < len(monitors):
        monitor = monitors[screen_number]
        return int(monitor.width), int(monitor.height)
    else:
        raise ValueError("Screen number out of range")
    
screen_number = 0  # Change this to the index of the screen you want to use
scale_factor = 2.0  # Set this to the scale factor of your screen (e.g., 2.0 for 200%)
screen_width, screen_height = get_screen_size(screen_number)
print(f"Screen width: {screen_width}, Screen height: {screen_height}")

## SIZE
h_show_size = screen_height // 4
w_show_size = int(h_show_size * screen_width/screen_height)
# w_show_size = 320
# h_show_size = 240

system_name_img_width = w_show_size

w_process_size = 320
h_process_size = 240  

## LOCATION
system_name_location = [1, 1]

left_show_location = [0, h_show_size - 30]
right_show_location = [0, (h_show_size - 55) * 2]

info_img_width = int(screen_width / 4)
info_show_location = [int((screen_width - info_img_width) // 2), int((screen_height - 1/2* h_show_size - 20) // 2)]

# OBJECT DETECTION
SHIP_DISTANCE_CAL_THRES = 15 # meters
prev_det_data = []
pseudo_process_time = 1
prev_speed = 1

# SHIP: cls_id = 0
SHIP_DISTANCE_WARN_THRES = 50 # meters
SHIP_SPEED_WARN_THRES = 40 # km/h
SHIP_COUNTER = 0
SHIP_CONSEC_THRES = 5
flag_ship_warning = 0
flag_ship_in_left = 0
flag_ship_in_right = 0
prev_flag_ship_in_left = 0
prev_flag_ship_in_right = 0
wt = 0

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
##               TODO : INFO PATH               ##
###################################################

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to the audio files
audio_path = os.path.join(CWD_PATH,"audio_files")

# Path to the system images
sys_img_path = os.path.join(CWD_PATH, "system_images")

###################################################
##          TODO : VIDEO and DATA IN-OUT        ##
###################################################
parent_directory = os.path.dirname(CWD_PATH)
os.chdir(parent_directory)
# Path to the input videos
in_vid_path = os.path.join(os.getcwd(), "RUN", "In_videos")
print("in_vid_path: ", in_vid_path)

# Path to the output videos
out_vid_path = os.path.join(os.getcwd(), "RUN", "Out_videos")
print("out_vid_path: ", out_vid_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# XVID is more preferable. MJPG results in high size video. X264 gives very small size video    

def create_vid_writer(): # Create new folder to save videos in nhan_mat_set_time()
    global left_org_vid_path, right_org_vid_path
    global left_org_vid, right_org_vid
    global left_out_vid_path, right_out_vid_path
    global left_out_vid, right_out_vid
    # Create folder for saving videos
    vid_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S") # video folder name format: YYYYMMDD_HHMMSS
    vid_folder_path = os.path.join(out_vid_path, str(vid_folder_name))
    if not os.path.exists(vid_folder_path):
        os.mkdir(vid_folder_path)
    print("Folder to save video path: ", vid_folder_path)

    # Make the path for saving the video
    left_org_vid_path = os.path.join(vid_folder_path, 'left_org_video.avi') # left camera video original
    left_out_vid_path = os.path.join(vid_folder_path, 'left_video.avi') # left camera video
    
    right_org_vid_path = os.path.join(vid_folder_path, 'right_org_video.avi') # right camera video original
    right_out_vid_path = os.path.join(vid_folder_path, 'right_video.avi') # right camera video
    
    # Python: cv2.VideoWriter([filename, fourcc, fps, frameSize[, isColor]])
    left_org_vid = cv2.VideoWriter(left_org_vid_path, fourcc, 30.0, video_save_size)
    left_out_vid = cv2.VideoWriter(left_out_vid_path, fourcc, 30.0, video_save_size)
    
    right_org_vid = cv2.VideoWriter(right_org_vid_path, fourcc, 30.0, video_save_size)
    right_out_vid = cv2.VideoWriter(right_out_vid_path, fourcc, 30.0, video_save_size)

###################################################
##             TODO : DATA SAVING               ##
###################################################
# Path to save the output CSV files
out_csv_path = os.path.join(os.getcwd(), "RUN", "Out_data")
print("out_csv_path: ", out_csv_path)

# Create a new CSV file with headers
def create_csv_file():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(out_csv_path, f"{current_time}.csv")
    headers = ["STT", "Thời gian đi", "Thời gian đến","Điểm đi", "Điểm đến"]
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_path, index=False,encoding='utf-8-sig')
    return file_path

# Initialize the CSV file at the start of the program
csv_file_path = create_csv_file()
print(f"CSV file created at: {csv_file_path}")

###################################################
##           TODO : LOAD MODELS                 ##
###################################################
## Load the model for object detection
print("[INFO] loading DAYTIME object detector...")
daytime_detector_path = os.path.join(CWD_PATH, "object_detector", "daytime_best.pt")
daytime_detector = YOLO(daytime_detector_path)

print("[INFO] loading NIGHTTIME object detector...")
nighttime_detector_path = os.path.join(CWD_PATH, "object_detector", "nighttime_best.pt")
nighttime_detector = YOLO(nighttime_detector_path)

print("[INFO] loaded object detector successfully!")

###################################################
##               TODO : AUDIO                   ##
###################################################
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
        
###################################################
##                TODO : VISUALIZATION          ##
###################################################
## FUNCTION: show image in a fixed window
def showInFixedWindow(winname, img, x, y):
    img = imutils.resize(img, width = w_show_size)
    cv2.namedWindow(winname, flags = cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)  # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)
    
def showInFixedWindow_info(winname, img, x, y):
    cv2.namedWindow(winname, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)  # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)      
    
# FUNCTION: Show the system name image in a fixed window
def visualize_system_name():
    img = cv2.imread(os.path.join(sys_img_path, "system_name.png"))
    img = imutils.resize(img, width = system_name_img_width)
    (h_sysname_img, w_sysname_img) = img.shape[:2]
    print(f"System Name Image Size: {w_sysname_img} x {h_sysname_img}")
    showInFixedWindow('System Name', img, system_name_location[0], system_name_location[1])
    cv2.waitKey(1)
    return h_sysname_img, w_sysname_img

# FUNCTION: Show the notice image in a fixed window
def visualize_info(info_img_name):
    if cv2.getWindowProperty("System Notice", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("System Notice")
    img = cv2.imread(os.path.join(sys_img_path, info_img_name))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow_info('System Notice', img, info_show_location[0], info_show_location[1])
    # print("Showing System Notice: ", info_img_name)
    cv2.waitKey(1)
            
# FUNCTION: Show the left camera image in a fixed window
def visualize_left(left_frame):
    showInFixedWindow("Left Camera", left_frame, left_show_location[0], left_show_location[1])

# FUNCTION: Show the right camera image in a fixed window
def visualize_right(right_frame):
    showInFixedWindow("Right Camera", right_frame, right_show_location[0], right_show_location[1])
       
###################################################
##            TODO : VISUALIZE & AUDIO          ##
###################################################
def phat_loa_show_info(info_name):
    visualize_info(info_name + ".png")
    # phat_loa_no_wait(info_name + ".mp3")
    phat_loa_skip(info_name + ".mp3")
    # print("Da phat loa va hien thi thong bao: ", info_name)

def phat_loa_show_info_ob_det(info_name):
    visualize_info(info_name + ".png")
    phat_loa_skip(info_name + ".mp3")
    # phat_loa_no_wait(info_name + ".mp3")
    # print("Da phat loa va hien thi thong bao: ", info_name)  

###################################################
##       TODO : BUFFER-LESS VideoCapture        ##
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
##           TODO : LIST CAMERAS               ##
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
##         TODO : INTERFACE WITH PX4           ##
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
##         TODO : OBJECTS DETECTION ULTIS      ##
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

def get_M(pt_A, pt_B, pt_C, pt_D):
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
    
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    return M, maxWidth, maxHeight

def get_transformed_point(orig_point_coors, transf_matrix):
    p = orig_point_coors
    matrix = transf_matrix
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return (int(px), int(py))

# DEFINE PERSPECTIVE TRANSFORM
# Upper
y1_lim = 150
pt_A1 = [300, y1_lim]
pt_B1 = [-640, 320]
pt_C1 = [1280, 320]
pt_D1 = [340, y1_lim]

#Lower
y2_lim = 490
pt_A2 = [300, y2_lim]
pt_B2 = [-640, 640]
pt_C2 = [1280, 640]
pt_D2 = [340, y2_lim]

M1, maxWidth1, maxHeight1 = get_M(pt_A1, pt_B1, pt_C1, pt_D1)
M2, maxWidth2, maxHeight2 = get_M(pt_A2, pt_B2, pt_C2, pt_D2)

###################################################
##           TODO : * OBJECTS DETECTION          ##
###################################################
cls_names = ['Ship', 'Bridge', 'Sign', 'Person']
colors = [(0,255,255), (255,0,255), (255,255,0), (0,255,0)] # BGR: Yellow, Magenta, Cyan, Green

def object_detection():
    while not stop_event.is_set():
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
        
def cal_distance(cls_id, y1, y2, obj_det_frame_height, stacked_frame):

    # Calculate the distance of the object
    dist, dist_size, dist_pers = 0, 0, 0

    obj_h_ratio = (y2 - y1)/obj_det_frame_height
    if cls_id == 0: # Ship
        dist_size = -175 * obj_h_ratio + 75
    elif cls_id == 1: # Bridge
        dist_size = -350 * obj_h_ratio + 150
    elif cls_id == 3: # Person
        dist_size = -100 * obj_h_ratio + 75

    out1 = cv2.warpPerspective(stacked_frame, M1, (maxWidth1, maxHeight1), flags=cv2.INTER_LINEAR)
    out2 = cv2.warpPerspective(stacked_frame, M2, (maxWidth2, maxHeight2), flags=cv2.INTER_LINEAR)

    # Distance Estimation based on Perspective Transform 
    pt = [320,y2]
    if y2 < 320:
        tr_coors = get_transformed_point(pt, M1)
        pt_y = tr_coors[1]
        roi_max_y = out1.shape[0]
        offset1 = 180
        dist_pers = (1 - pt_y / roi_max_y) * offset1
    else:
        tr_coors = get_transformed_point(pt, M2)
        pt_y = tr_coors[1]
        roi_max_y = out2.shape[0]
        offset2 = 200
        dist_pers = (1 - pt_y / roi_max_y) * offset2

    # Final Distance Estimation
    dist = 0.5 * dist_size + 0.5 * dist_pers

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
    global prev_flag_ship_in_left
    global prev_flag_ship_in_right
    global wt
    
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
        dist = cal_distance(cls_id, y1, y2, obj_det_frame_height, stacked_frame)  
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
                    
                if (prev_flag_ship_in_left == 1 and flag_ship_in_right == 1) or (flag_ship_in_left == 1 and prev_flag_ship_in_right == 1):
                    print("Phát hiện tàu HAI BÊN -> ĐI THẲNG")
                    flag_ship_warning = 1
                    wt = time.time()
                    # if estop_status == 0:
                    #     if (time.time() - send_time) > 5: 
                    #         send_straight_to_arduino(arduino_module)
                    #         send_time = time.time()
                    #     if (time.time() - send_gps_time) > 8:
                    #         send_gps_data_to_arduino()
                    #         send_gps_time = time.time()
                    t5 = Thread(phat_loa_show_info_ob_det("3_cham_2ben"))
                    t5.deamon = True
                    t5.start()
                
                elif flag_ship_in_left == 1: # and flag_ship_in_right == 0:
                    print("Phát hiện tàu BÊN TRÁI -> RẼ PHẢI")
                    flag_ship_warning = 1
                    wt = time.time()
                    # if estop_status == 0:
                    #     if (time.time() - send_time) > 5:                            
                    #         send_right_to_arduino(arduino_module, cal_steering_angle(dist, ship_speed))
                    #         send_time = time.time()
                    #     if (time.time() - send_gps_time) > 8:
                    #         send_gps_data_to_arduino()
                    #         send_gps_time = time.time()
                    t5 = Thread(phat_loa_show_info_ob_det("5_cham_trai"))
                    t5.deamon = True
                    t5.start()
                    prev_flag_ship_in_left = 1
                    
                elif flag_ship_in_right == 1: # and flag_ship_in_left == 0
                    print("Phát hiện tàu BÊN PHẢI -> RẼ TRÁI")
                    flag_ship_warning = 1
                    wt = time.time()
                    # if estop_status == 0:
                    #     if (time.time() - send_time) > 5:   
                    #         send_left_to_arduino(arduino_module, cal_steering_angle(dist, ship_speed))
                    #         send_time = time.time()
                    #     if (time.time() - send_gps_time) > 8:
                    #         send_gps_data_to_arduino()
                    #         send_gps_time = time.time()
                    t5 = Thread(phat_loa_show_info_ob_det("4_cham_phai"))
                    t5.deamon = True
                    t5.start()
                    prev_flag_ship_in_right = 1
                    
            if (time.time() - wt > 5):
                flag_ship_warning = 0
                # flag_ship_in_left = 0
                # flag_ship_in_right = 0
                # prev_flag_ship_in_left = 0
                # prev_flag_ship_in_right = 0
                                               
        else:
            SHIP_COUNTER = 0
            flag_ship_warning = 0
            flag_ship_in_left = 0
            flag_ship_in_right = 0
            prev_flag_ship_in_left = 0
            prev_flag_ship_in_right = 0
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
        dist = cal_distance(cls_id, y1, y2, obj_det_frame_height, stacked_frame)
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
                    # print("Phát hiện cầu BÊN TRÁI!")
                    flag_bridge_warning = 1
                    t6 = Thread(phat_loa_show_info_ob_det("13_cau_trai"))
                    t6.deamon = True
                    t6.start()
                
                elif flag_bridge_in_left == 0 and flag_bridge_in_right == 1:
                    # print("Phát hiện cầu BÊN PHẢI!")
                    flag_bridge_warning = 1
                    t6 = Thread(phat_loa_show_info_ob_det("14_cau_phai"))
                    t6.deamon = True
                    t6.start()
                    
                elif flag_bridge_in_left == 1 and flag_bridge_in_right == 1:
                    # print("Phát hiện cầu PHÍA TRƯỚC!")
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
            # print("Chú ý chiều cao của tàu!")
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
                # print("Phát hiện biển báo BÊN TRÁI!")
                flag_sign_warning = 1
                t7 = Thread(phat_loa_show_info_ob_det("11_bien_bao_trai"))
                t7.deamon = True
                t7.start()
                
            elif flag_sign_in_left == 0 and flag_sign_in_right == 1:
                # print("Phát hiện biển báo BÊN PHẢI!")
                flag_sign_warning = 1
                t7 = Thread(phat_loa_show_info_ob_det("10_bien_bao_phai"))
                t7.deamon = True
                t7.start()
                
            elif flag_sign_in_left == 1 and flag_sign_in_right == 1:
                # print("Phát hiện biển báo HAI BÊN!")
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
        dist = cal_distance(cls_id, y1, y2, obj_det_frame_height, stacked_frame)
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
                # print("Phát hiện người BÊN TRÁI!")
                flag_person_warning = 1
                t8 = Thread(phat_loa_show_info_ob_det("8_nguoi_trai"))
                t8.deamon = True
                t8.start()
                
            elif flag_person_in_left == 0 and flag_person_in_right == 1:
                # print("Phát hiện người BÊN PHẢI!")
                flag_person_warning = 1
                t8 = Thread(phat_loa_show_info_ob_det("7_nguoi_phai"))
                t8.deamon = True
                t8.start()
                
            elif flag_person_in_left == 1 and flag_person_in_right == 1:
                # print("Phát hiện người HAI BÊN!")
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
##              TODO : * 2 IP CAMERA VIDEOS (L+R)            ##
###############################################################
# Load LEFT and RIGHT videos
left_in_vid_name = "bridge_night_250114.mp4" # bridge
# left_in_vid_name = "20240825_102907.mp4" # ship, bridge
left_cam = cv2.VideoCapture(os.path.join(in_vid_path, "left_cam", left_in_vid_name))
left_cam_nframes = int(left_cam.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] Number of frames in the LEFT video: ", left_cam_nframes)

right_in_vid_name = "bridge_night_250114.mp4"
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
        
        obj_det_results = nighttime_detector.predict(stacked_frame, conf=0.05, save=False, verbose=False)
        
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

###################################################
##        TODO : INTERACT WITH ARDUINO         ##
###################################################
## Ket noi voi arduino 
## /dev/ttyACM1
# try:                       
## Jetson Nano
# arduino_module = serial.Serial(port = '/dev/ttyACM1', baudrate = 9600, timeout = 0.5)                           
# arduino_module.flush()
# print("Arduino connected successfully!")                                            
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
##         TODO : ** APP FUNCTION                ##
###################################################
class Ui_MainWindow(object):
    def setupUi(self, Trip_Information):
        Trip_Information.setObjectName("Trip_Information")
        Trip_Information.resize(w_sysname_img, h_sysname_img)
        
        self.centralwidget = QtWidgets.QWidget(Trip_Information)
        self.centralwidget.setObjectName("centralwidget")

        # Create a vertical layout
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        # Title label
        self.title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.verticalLayout.addWidget(self.title)

        # Create a horizontal layout for checkboxes
        self.horizontalLayout_checkboxes = QtWidgets.QHBoxLayout()
        self.horizontalLayout_checkboxes.setObjectName("horizontalLayout_checkboxes")

        self.trongtinh_check = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(12)
        self.trongtinh_check.setFont(font)
        self.trongtinh_check.setObjectName("trongtinh_check")
        self.horizontalLayout_checkboxes.addWidget(self.trongtinh_check)

        self.lientinh_check = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(12)
        self.lientinh_check.setFont(font)
        self.lientinh_check.setObjectName("lientinh_check")
        self.horizontalLayout_checkboxes.addWidget(self.lientinh_check)

        self.verticalLayout.addLayout(self.horizontalLayout_checkboxes)

        # Create a horizontal layout for combo boxes
        self.horizontalLayout_comboboxes = QtWidgets.QHBoxLayout()
        self.horizontalLayout_comboboxes.setObjectName("horizontalLayout_comboboxes")

        self.from_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(12)
        self.from_label.setFont(font)
        self.from_label.setObjectName("from_label")
        self.horizontalLayout_comboboxes.addWidget(self.from_label)

        self.from_box = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(12)
        self.from_box.setFont(font)
        self.from_box.setObjectName("from_box")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.from_box.addItem("")
        self.horizontalLayout_comboboxes.addWidget(self.from_box)

        self.to_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(12)
        self.to_label.setFont(font)
        self.to_label.setObjectName("to_label")
        self.horizontalLayout_comboboxes.addWidget(self.to_label)

        self.to_box = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(12)
        self.to_box.setFont(font)
        self.to_box.setObjectName("to_box")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.to_box.addItem("")
        self.horizontalLayout_comboboxes.addWidget(self.to_box)

        self.verticalLayout.addLayout(self.horizontalLayout_comboboxes)

        # Create a horizontal layout for buttons
        self.horizontalLayout_buttons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_buttons.setObjectName("horizontalLayout_buttons")

        self.newtrip_button = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.newtrip_button.setFont(font)
        self.newtrip_button.setObjectName("newtrip_button")
        self.horizontalLayout_buttons.addWidget(self.newtrip_button)

        self.OK_button = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.OK_button.setFont(font)
        self.OK_button.setObjectName("OK_button")
        self.horizontalLayout_buttons.addWidget(self.OK_button)

        self.verticalLayout.addLayout(self.horizontalLayout_buttons)

        Trip_Information.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Trip_Information)
        self.statusbar.setObjectName("statusbar")
        Trip_Information.setStatusBar(self.statusbar)

        self.retranslateUi(Trip_Information)
        QtCore.QMetaObject.connectSlotsByName(Trip_Information)
        
        self.newtrip_button.setEnabled(False)
        self.OK_button.setEnabled(False)
        self.from_box.setEnabled(False)
        self.to_box.setEnabled(False)
        
        # Start the detect_2_ip_videos function in a new thread
        self.detect_thread = threading.Thread(target=self.run_infinite_loop, daemon=True)
        self.detect_thread.start()

    def retranslateUi(self, Trip_Information):
        _translate = QtCore.QCoreApplication.translate
        Trip_Information.setWindowTitle(_translate("Trip_Information", "Trip_Information"))
        self.OK_button.setText(_translate("Trip_Information", "OK"))
        self.from_box.setItemText(0, _translate("Trip_Information", "Bến Tre"))
        self.from_box.setItemText(1, _translate("Trip_Information", "An Giang"))
        self.from_box.setItemText(2, _translate("Trip_Information", "Bà Rịa Vũng Tàu"))
        self.from_box.setItemText(3, _translate("Trip_Information", "Bạc Liêu"))
        self.from_box.setItemText(4, _translate("Trip_Information", "Bình Dương"))
        self.from_box.setItemText(5, _translate("Trip_Information", "Bình Định"))
        self.from_box.setItemText(6, _translate("Trip_Information", "Bình Thuận"))
        self.from_box.setItemText(7, _translate("Trip_Information", "Cà Mau"))
        self.from_box.setItemText(8, _translate("Trip_Information", "Cần Thơ"))
        self.from_box.setItemText(9, _translate("Trip_Information", "Đà Nẵng"))
        self.to_box.setItemText(0, _translate("Trip_Information", "Bạc Liêu"))
        self.to_box.setItemText(1, _translate("Trip_Information", "An Giang"))
        self.to_box.setItemText(2, _translate("Trip_Information", "Bà Rịa Vũng Tàu"))
        self.to_box.setItemText(3, _translate("Trip_Information", "Bạc Liêu"))
        self.to_box.setItemText(4, _translate("Trip_Information", "Bình Dương"))
        self.to_box.setItemText(5, _translate("Trip_Information", "Bình Định"))
        self.to_box.setItemText(6, _translate("Trip_Information", "Bình Thuận"))
        self.to_box.setItemText(7, _translate("Trip_Information", "Cà Mau"))
        self.to_box.setItemText(8, _translate("Trip_Information", "Cần Thơ"))
        self.to_box.setItemText(9, _translate("Trip_Information", "Đà Nẵng"))
        self.trongtinh_check.setText(_translate("Trip_Information", "TRONG TỈNH"))
        self.lientinh_check.setText(_translate("Trip_Information", "LIÊN TỈNH"))
        self.title.setText(_translate("Trip_Information", "KHAI BÁO THÔNG TIN HẢI TRÌNH"))
        self.to_label.setText(_translate("Trip_Information", "Từ:"))
        self.from_label.setText(_translate("Trip_Information", "Đến:"))
        self.newtrip_button.setText(_translate("Trip_Information", "HÀNH TRÌNH MỚI"))
        
        self.trongtinh_check.stateChanged.connect(self.toggleComboBoxes)
        self.lientinh_check.stateChanged.connect(self.toggleComboBoxes)
        self.OK_button.clicked.connect(self.OK_click)
        self.newtrip_button.clicked.connect(self.newtrip_click)
        
    def toggleComboBoxes(self, state):
        sender = QtWidgets.QApplication.instance().sender()
        if sender == self.trongtinh_check and state == QtCore.Qt.Checked:
            self.lientinh_check.setChecked(False)
            self.from_box.setEnabled(False)
            self.to_box.setEnabled(False)
            self.OK_button.setEnabled(True)
        elif sender == self.lientinh_check and state == QtCore.Qt.Checked:
            self.trongtinh_check.setChecked(False)
            self.from_box.setEnabled(True)
            self.to_box.setEnabled(True)
            self.OK_button.setEnabled(True)
        else:
            self.from_box.setEnabled(False)
            self.to_box.setEnabled(False)

    def OK_click(self):
        global start_value, end_value, depart_time, new_data
        if self.trongtinh_check.isChecked():
            start_value = "Bến Tre"
            end_value = "Bến Tre"
        else:
            start_value = self.from_box.currentText()
            end_value = self.to_box.currentText()
            
        # Save the information to the CSV file
        depart_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Create a new DataFrame with the new data
        new_data = pd.DataFrame({
            "STT": [len(pd.read_csv(csv_file_path)) + 1],
            "Thời gian đi": [depart_time],
            "Điểm đi": [start_value],
            "Điểm đến": [end_value]
        })
    
        print(f"Từ: {start_value} -> Đến: {end_value} | Khởi hành lúc: {depart_time}")
        msg = QtWidgets.QMessageBox()
        msg.setGeometry(msg_box_x, msg_box_y, w_msg_box, h_msg_box)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        
        # Set the font for the message box
        font = QtGui.QFont()
        font.setPointSize(12)
        msg.setFont(font)
        
        # Format the message text
        msg_text = (f"<b>Từ:</b> {start_value}<br>"
                f"<b>Đến:</b> {end_value}<br>"
                f"<b>Khởi hành lúc:</b> {depart_time}")
        msg.setText(msg_text)
        
        msg.setWindowTitle("Bắt đầu hải trình")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.buttonClicked.connect(msg.close)
        QtCore.QTimer.singleShot(8000, msg.close)
        msg.exec_()

        # Deactivate all widgets except newtrip_button
        self.from_box.setEnabled(False)
        self.to_box.setEnabled(False)
        self.trongtinh_check.setEnabled(False)
        self.lientinh_check.setEnabled(False)
        self.OK_button.setEnabled(False)
        self.newtrip_button.setEnabled(True)
    
    def newtrip_click(self):
        global start_value, end_value, depart_time, new_data
        self.from_box.setEnabled(False)
        self.to_box.setEnabled(False)
        self.trongtinh_check.setEnabled(True)
        self.lientinh_check.setEnabled(True)
        self.trongtinh_check.setChecked(False)
        self.lientinh_check.setChecked(False)
        self.OK_button.setEnabled(True)
        self.newtrip_button.setEnabled(False) 
        
        # Save the arrival time to the CSV file
        arrive_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        new_data["Thời gian đến"] = [arrive_time]
        
        # Reorder the columns to match the header in the CSV file
        new_data = new_data[["STT", "Thời gian đi", "Thời gian đến", "Điểm đi", "Điểm đến"]]
        
        # Append the new data to the existing CSV file with UTF-8 encoding
        new_data.to_csv(csv_file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        print(f"Từ: {start_value} -> Đến: {end_value} | Khởi hành lúc: {depart_time} | Kết thúc lúc: {arrive_time}")
        msg = QtWidgets.QMessageBox()
        msg.setGeometry(msg_box_x, msg_box_y, w_msg_box, h_msg_box)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        
        # Set the font for the message box
        font = QtGui.QFont()
        font.setPointSize(12)
        msg.setFont(font)
        
        # Format the message text
        msg_text = (f"<b>Từ:</b> {start_value}<br>"
            f"<b>Đến:</b> {end_value}<br>"
            f"<b>Khởi hành lúc:</b> {depart_time}<br>"
            f"<b>Kết thúc lúc:</b> <br> {arrive_time}")
        msg.setText(msg_text)
        
        msg.setWindowTitle("Kết thúc hải trình")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.buttonClicked.connect(msg.close)
        QtCore.QTimer.singleShot(8000, msg.close)
        msg.exec_()
        
    def run_infinite_loop(self):
        while True:      
            detect_2_ip_videos()
            # update_from_arduino()
            # send_gps_data_to_arduino()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set() # Set the stop event
                break

###################################################
##         TODO 22*: MAIN FUNCTION               ##
###################################################   
if __name__ == "__main__":
    h_sysname_img, w_sysname_img = visualize_system_name()
    w_app = w_sysname_img
    h_app = h_sysname_img - 100
    app_location_x = system_name_location[0] + w_sysname_img + 10
    app_location_y = system_name_location[1] + 50
    # w_msg_box = 450
    # h_msg_box = 100
    w_msg_box = w_app
    h_msg_box = h_app
    msg_box_x = app_location_x
    msg_box_y = app_location_y
    app = QtWidgets.QApplication(sys.argv)
    Trip_Information = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(Trip_Information)
    Trip_Information.setGeometry(app_location_x, app_location_y, w_app, h_app)
    Trip_Information.show()
    stop_event = threading.Event()
    sys.exit(app.exec_())
    
    # When everything is done, release the capture
    print("[INFO] cleaning up...")

    left_org_vid.release()
    left_out_vid.release()
    right_org_vid.release()
    right_out_vid.release()
    cv2.destroyAllWindows()
    # arduino_module.close()

    
    